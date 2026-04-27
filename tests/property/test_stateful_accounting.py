"""
Stateful property tests for MemoryBackend quota accounting.

Uses Hypothesis RuleBasedStateMachine to generate random sequences of
consume / try_acquire / refund / set_max_capacity operations, verifying
invariants against a deterministic shadow model with frozen time.
"""

import asyncio
import enum
import warnings
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._limiter_backends._memory._backend import MemoryBackend
from token_throttle._limiter_backends._memory._bucket import MemoryBucket
from token_throttle._limiter_backends._memory._sync_backend import SyncMemoryBackend

FROZEN_TIME = 1_000_000.0
METRIC = "tokens"
WINDOW = 60
LIMIT = 1000.0

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

amounts = st.floats(
    min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False
)
small_amounts = st.floats(
    min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False
)
max_cap_values = st.floats(
    min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(limit: float = LIMIT) -> PerModelConfig:
    quota = Quota(metric=METRIC, limit=limit, per_seconds=WINDOW)
    return PerModelConfig(model_family="test", quotas=UsageQuotas([quota]))


def _make_bucket(limit: float = LIMIT) -> MemoryBucket:
    return MemoryBucket(
        metric=METRIC, per_seconds=WINDOW, limit=limit, model_family="test"
    )


def _make_sync_backend(limit: float = LIMIT) -> SyncMemoryBackend:
    config = _make_config(limit)
    bucket = _make_bucket(limit)
    return SyncMemoryBackend(buckets=[bucket], limit_config=config)


# ---------------------------------------------------------------------------
# 2a. SingleBucketAccountingMachine
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class SingleBucketAccountingMachine(RuleBasedStateMachine):
    """
    Shadow-model test for a single-bucket SyncMemoryBackend with frozen time.

    Shadow tracks:
    - shadow_raw_stored: the raw value stored in bucket.capacity (None = fresh)
    - shadow_max_capacity: current max_capacity
    """

    def __init__(self):
        super().__init__()
        self.backend: SyncMemoryBackend | None = None
        self.bucket: MemoryBucket | None = None
        self.shadow_raw_stored: float | None = None
        self.shadow_max_capacity: float = LIMIT
        self.total_consumed: float = 0.0
        self.total_refunded: float = 0.0
        self.accounting_baseline: float = LIMIT
        self._time_patcher = patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        )
        self._mock_time = self._time_patcher.start()
        self._mock_time.time.return_value = FROZEN_TIME
        self._mock_time.monotonic.return_value = FROZEN_TIME

    def _shadow_readable(self) -> float:
        """What get_capacity() should return — mirrors calculate_capacity() logic."""
        if self.shadow_raw_stored is None:
            return self.shadow_max_capacity
        return min(self.shadow_max_capacity, self.shadow_raw_stored)

    @initialize()
    def init_backend(self):
        self.backend = _make_sync_backend()
        self.bucket = self.backend._buckets[0]
        self.shadow_raw_stored = None
        self.shadow_max_capacity = LIMIT
        self.total_consumed = 0.0
        self.total_refunded = 0.0
        self.accounting_baseline = LIMIT

    @rule(amount=amounts)
    def consume(self, amount):
        readable_before = self._shadow_readable()
        self.backend.consume_capacity(frozen_usage({METRIC: amount}))
        self.shadow_raw_stored = readable_before - amount
        self.total_consumed += amount

    @rule(amount=amounts)
    def try_acquire(self, amount):
        readable = self._shadow_readable()
        if amount > self.shadow_max_capacity:
            with pytest.raises(ValueError, match="exceeds bucket max capacity"):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )
            return

        if amount <= readable:
            # Should succeed (no wait needed, capacity is sufficient)
            self.backend.wait_for_capacity(frozen_usage({METRIC: amount}), timeout=0.0)
            # After successful acquire: capacity = max(0, readable - amount)
            self.shadow_raw_stored = max(0.0, readable - amount)
            self.total_consumed += amount
        else:
            # Should fail with timeout since capacity is insufficient and time is frozen
            with pytest.raises(TimeoutError):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )

    @rule(
        reserved=amounts,
        actual_fraction=st.floats(
            min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False
        ),
    )
    def refund(self, reserved, actual_fraction):
        actual = reserved * actual_fraction
        readable_before = self._shadow_readable()
        self.backend.refund_capacity(
            frozen_usage({METRIC: reserved}),
            frozen_usage({METRIC: actual}),
        )
        refund_amount = max(reserved - actual, -self.shadow_max_capacity)
        self.shadow_raw_stored = min(
            readable_before + refund_amount, self.shadow_max_capacity
        )
        self.total_refunded += refund_amount

    @rule(value=max_cap_values)
    def set_max_capacity(self, value):
        old_readable = self._shadow_readable()
        self.backend.set_max_capacity(METRIC, WINDOW, value)
        self.shadow_max_capacity = value
        new_readable = self._shadow_readable()
        # Track capacity gained from raising the ceiling
        gain = new_readable - old_readable
        if gain > 0:
            self.accounting_baseline += gain

    @invariant()
    def capacity_within_max(self):
        if self.bucket is None:
            return
        actual = self.bucket.get_capacity(FROZEN_TIME).amount
        assert actual <= self.shadow_max_capacity + 1e-9, (
            f"Capacity {actual} exceeded max {self.shadow_max_capacity}"
        )

    @invariant()
    def capacity_matches_shadow(self):
        if self.bucket is None:
            return
        actual = self.bucket.get_capacity(FROZEN_TIME).amount
        expected = self._shadow_readable()
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"Capacity mismatch: actual={actual}, shadow={expected}, "
            f"raw_stored={self.shadow_raw_stored}, max_cap={self.shadow_max_capacity}"
        )

    @invariant()
    def total_accounting(self):
        """No operation should create capacity from nothing.

        With frozen time, capacity + consumed - refunded <= baseline.
        The baseline starts at LIMIT and increases when set_max_capacity
        raises the readable ceiling. Capping (refund hitting max,
        set_max_capacity lowering cap) can only *lose* capacity,
        so the balance can only be ≤ the baseline.
        """
        if self.bucket is None:
            return
        actual = self.bucket.get_capacity(FROZEN_TIME).amount
        balance = actual + self.total_consumed - self.total_refunded
        assert balance <= self.accounting_baseline + 1e-9, (
            f"Total accounting violation: capacity={actual}, "
            f"consumed={self.total_consumed}, refunded={self.total_refunded}, "
            f"balance={balance}, baseline={self.accounting_baseline}"
        )

    def teardown(self):
        self._time_patcher.stop()


StatefulSingleBucket = SingleBucketAccountingMachine.TestCase
StatefulSingleBucket.settings = hypothesis_settings(
    max_examples=200, stateful_step_count=50, deadline=None
)


# ---------------------------------------------------------------------------
# 2b. MultiWindowAccountingMachine
# ---------------------------------------------------------------------------

SHORT_WINDOW = 60
SHORT_LIMIT = 100.0
LONG_WINDOW = 3600
LONG_LIMIT = 1000.0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class MultiWindowAccountingMachine(RuleBasedStateMachine):
    """
    Shadow-model test for a two-window SyncMemoryBackend with frozen time.

    Key invariant: try_acquire succeeds iff BOTH windows have sufficient capacity.
    """

    def __init__(self):
        super().__init__()
        self.backend: SyncMemoryBackend | None = None
        self.short_bucket: MemoryBucket | None = None
        self.long_bucket: MemoryBucket | None = None
        # Shadow per window
        self.shadow_short_raw: float | None = None
        self.shadow_long_raw: float | None = None
        self.short_max = SHORT_LIMIT
        self.long_max = LONG_LIMIT
        self._time_patcher = patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        )
        self._mock_time = self._time_patcher.start()
        self._mock_time.time.return_value = FROZEN_TIME
        self._mock_time.monotonic.return_value = FROZEN_TIME

    def _readable(self, raw: float | None, max_cap: float) -> float:
        if raw is None:
            return max_cap
        return min(max_cap, raw)

    def _short_readable(self) -> float:
        return self._readable(self.shadow_short_raw, self.short_max)

    def _long_readable(self) -> float:
        return self._readable(self.shadow_long_raw, self.long_max)

    @initialize()
    def init_backend(self):
        short_q = Quota(metric=METRIC, limit=SHORT_LIMIT, per_seconds=SHORT_WINDOW)
        long_q = Quota(metric=METRIC, limit=LONG_LIMIT, per_seconds=LONG_WINDOW)
        config = PerModelConfig(
            model_family="test", quotas=UsageQuotas([short_q, long_q])
        )
        short_b = MemoryBucket(
            metric=METRIC,
            per_seconds=SHORT_WINDOW,
            limit=SHORT_LIMIT,
            model_family="test",
        )
        long_b = MemoryBucket(
            metric=METRIC,
            per_seconds=LONG_WINDOW,
            limit=LONG_LIMIT,
            model_family="test",
        )
        self.backend = SyncMemoryBackend(buckets=[short_b, long_b], limit_config=config)
        self.short_bucket = short_b
        self.long_bucket = long_b
        self.shadow_short_raw = None
        self.shadow_long_raw = None
        self.short_max = SHORT_LIMIT
        self.long_max = LONG_LIMIT

    @rule(amount=small_amounts)
    def consume(self, amount):
        short_r = self._short_readable()
        long_r = self._long_readable()
        self.backend.consume_capacity(frozen_usage({METRIC: amount}))
        self.shadow_short_raw = short_r - amount
        self.shadow_long_raw = long_r - amount

    @rule(amount=small_amounts)
    def try_acquire(self, amount):
        short_r = self._short_readable()
        long_r = self._long_readable()

        # Exceeds the smaller window's max → ValueError
        if amount > self.short_max:
            with pytest.raises(ValueError, match="exceeds bucket max capacity"):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )
            return

        # All-or-nothing: both windows must have enough
        if amount <= short_r and amount <= long_r:
            self.backend.wait_for_capacity(frozen_usage({METRIC: amount}), timeout=0.0)
            self.shadow_short_raw = max(0.0, short_r - amount)
            self.shadow_long_raw = max(0.0, long_r - amount)
        else:
            with pytest.raises(TimeoutError):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )

    @rule(
        reserved=small_amounts,
        actual_fraction=st.floats(
            min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False
        ),
    )
    def refund(self, reserved, actual_fraction):
        actual = reserved * actual_fraction
        short_r = self._short_readable()
        long_r = self._long_readable()
        self.backend.refund_capacity(
            frozen_usage({METRIC: reserved}),
            frozen_usage({METRIC: actual}),
        )
        raw_refund = reserved - actual
        self.shadow_short_raw = min(
            short_r + max(raw_refund, -self.short_max), self.short_max
        )
        self.shadow_long_raw = min(
            long_r + max(raw_refund, -self.long_max), self.long_max
        )

    @invariant()
    def short_capacity_matches_shadow(self):
        if self.short_bucket is None:
            return
        actual = self.short_bucket.get_capacity(FROZEN_TIME).amount
        expected = self._short_readable()
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"Short window mismatch: actual={actual}, shadow={expected}"
        )

    @invariant()
    def long_capacity_matches_shadow(self):
        if self.long_bucket is None:
            return
        actual = self.long_bucket.get_capacity(FROZEN_TIME).amount
        expected = self._long_readable()
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"Long window mismatch: actual={actual}, shadow={expected}"
        )

    @invariant()
    def capacities_within_max(self):
        if self.short_bucket is None:
            return
        short_actual = self.short_bucket.get_capacity(FROZEN_TIME).amount
        long_actual = self.long_bucket.get_capacity(FROZEN_TIME).amount
        assert short_actual <= self.short_max + 1e-9
        assert long_actual <= self.long_max + 1e-9

    def teardown(self):
        self._time_patcher.stop()


StatefulMultiWindow = MultiWindowAccountingMachine.TestCase
StatefulMultiWindow.settings = hypothesis_settings(
    max_examples=200, stateful_step_count=50, deadline=None
)


# ---------------------------------------------------------------------------
# 2c. Sync/async parity property test
# ---------------------------------------------------------------------------


class Op(enum.Enum):
    CONSUME = "consume"
    REFUND = "refund"
    ACQUIRE = "acquire"


@st.composite
def ops_strategy(draw):
    """Generate a list of (Op, ...) tuples."""
    ops = []
    n = draw(st.integers(min_value=1, max_value=20))
    for _ in range(n):
        op_type = draw(st.sampled_from(Op))
        if op_type == Op.CONSUME:
            amount = draw(
                st.floats(
                    min_value=0.1,
                    max_value=200.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            ops.append((Op.CONSUME, amount))
        elif op_type == Op.REFUND:
            reserved = draw(
                st.floats(
                    min_value=0.1,
                    max_value=200.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            actual = draw(
                st.floats(
                    min_value=0.0,
                    max_value=400.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            ops.append((Op.REFUND, reserved, actual))
        else:
            # ACQUIRE: small amounts that may or may not fit in current capacity.
            # Uses timeout=0.0 so it either succeeds instantly or raises TimeoutError.
            amount = draw(
                st.floats(
                    min_value=0.1,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            ops.append((Op.ACQUIRE, amount))
    return ops


def _run_sync_ops(backend, ops):
    """Execute ops on a sync backend, returning which ACQUIRE ops succeeded."""
    acquire_results = []
    for op in ops:
        if op[0] == Op.CONSUME:
            backend.consume_capacity(frozen_usage({METRIC: op[1]}))
        elif op[0] == Op.REFUND:
            backend.refund_capacity(
                frozen_usage({METRIC: op[1]}),
                frozen_usage({METRIC: op[2]}),
            )
        elif op[0] == Op.ACQUIRE:
            try:
                backend.wait_for_capacity(frozen_usage({METRIC: op[1]}), timeout=0.0)
                acquire_results.append(True)
            except (TimeoutError, ValueError):
                acquire_results.append(False)
    return acquire_results


async def _run_async_ops(backend, ops):
    """Execute ops on an async backend, returning which ACQUIRE ops succeeded."""
    acquire_results = []
    for op in ops:
        if op[0] == Op.CONSUME:
            await backend.consume_capacity(frozen_usage({METRIC: op[1]}))
        elif op[0] == Op.REFUND:
            await backend.refund_capacity(
                frozen_usage({METRIC: op[1]}),
                frozen_usage({METRIC: op[2]}),
            )
        elif op[0] == Op.ACQUIRE:
            try:
                await backend.await_for_capacity(
                    frozen_usage({METRIC: op[1]}), timeout=0.0
                )
                acquire_results.append(True)
            except (TimeoutError, ValueError):
                acquire_results.append(False)
    return acquire_results


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@hypothesis_settings(max_examples=500, deadline=None)
@given(ops=ops_strategy())
def test_sync_async_parity(ops):
    """Identical op sequences on sync and async backends must produce identical final capacity."""
    sync_config = _make_config()
    sync_bucket = _make_bucket()
    async_config = _make_config()
    async_bucket = _make_bucket()

    with (
        patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        ) as sync_mock,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        sync_mock.time.return_value = FROZEN_TIME
        sync_mock.monotonic.return_value = FROZEN_TIME
        sync_backend = SyncMemoryBackend(
            buckets=[sync_bucket], limit_config=sync_config
        )
        sync_acquire_results = _run_sync_ops(sync_backend, ops)

    with (
        patch("token_throttle._limiter_backends._memory._backend.time") as async_mock,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        async_mock.time.return_value = FROZEN_TIME
        async_mock.monotonic.return_value = FROZEN_TIME
        async_backend = MemoryBackend(buckets=[async_bucket], limit_config=async_config)
        async_acquire_results = asyncio.run(_run_async_ops(async_backend, ops))

    sync_cap = sync_bucket.get_capacity(FROZEN_TIME).amount
    async_cap = async_bucket.get_capacity(FROZEN_TIME).amount

    assert sync_acquire_results == async_acquire_results, (
        f"Acquire results diverged: sync={sync_acquire_results}, async={async_acquire_results}"
    )
    assert sync_cap == pytest.approx(async_cap, abs=1e-12), (
        f"Sync/async divergence: sync={sync_cap}, async={async_cap}"
    )


# ---------------------------------------------------------------------------
# 2d. MultiMetricAccountingMachine
# ---------------------------------------------------------------------------

TOKENS_METRIC = "tokens"
TOKENS_LIMIT = 1000.0
TOKENS_WINDOW = 60
REQUESTS_METRIC = "requests"
REQUESTS_LIMIT = 100.0
REQUESTS_WINDOW = 60

multi_metric_amounts = st.floats(
    min_value=0.1, max_value=40.0, allow_nan=False, allow_infinity=False
)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class MultiMetricAccountingMachine(RuleBasedStateMachine):
    """
    Shadow-model test for a two-metric SyncMemoryBackend with frozen time.

    Two independent metrics: "tokens" (limit=1000, window=60)
    and "requests" (limit=100, window=60).

    Key invariants:
    - Each metric's capacity matches its shadow
    - Each metric's capacity ≤ its max_capacity
    - Operations on one metric don't affect the other
    """

    def __init__(self):
        super().__init__()
        self.backend: SyncMemoryBackend | None = None
        self.tokens_bucket: MemoryBucket | None = None
        self.requests_bucket: MemoryBucket | None = None
        self.shadow_tokens_raw: float | None = None
        self.shadow_requests_raw: float | None = None
        self.tokens_max: float = TOKENS_LIMIT
        self.requests_max: float = REQUESTS_LIMIT
        self._time_patcher = patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        )
        self._mock_time = self._time_patcher.start()
        self._mock_time.time.return_value = FROZEN_TIME
        self._mock_time.monotonic.return_value = FROZEN_TIME

    def _readable(self, raw: float | None, max_cap: float) -> float:
        if raw is None:
            return max_cap
        return min(max_cap, raw)

    def _tokens_readable(self) -> float:
        return self._readable(self.shadow_tokens_raw, self.tokens_max)

    def _requests_readable(self) -> float:
        return self._readable(self.shadow_requests_raw, self.requests_max)

    @initialize()
    def init_backend(self):
        tokens_q = Quota(
            metric=TOKENS_METRIC, limit=TOKENS_LIMIT, per_seconds=TOKENS_WINDOW
        )
        requests_q = Quota(
            metric=REQUESTS_METRIC, limit=REQUESTS_LIMIT, per_seconds=REQUESTS_WINDOW
        )
        config = PerModelConfig(
            model_family="test", quotas=UsageQuotas([tokens_q, requests_q])
        )
        tokens_b = MemoryBucket(
            metric=TOKENS_METRIC,
            per_seconds=TOKENS_WINDOW,
            limit=TOKENS_LIMIT,
            model_family="test",
        )
        requests_b = MemoryBucket(
            metric=REQUESTS_METRIC,
            per_seconds=REQUESTS_WINDOW,
            limit=REQUESTS_LIMIT,
            model_family="test",
        )
        self.backend = SyncMemoryBackend(
            buckets=[tokens_b, requests_b], limit_config=config
        )
        self.tokens_bucket = tokens_b
        self.requests_bucket = requests_b
        self.shadow_tokens_raw = None
        self.shadow_requests_raw = None
        self.tokens_max = TOKENS_LIMIT
        self.requests_max = REQUESTS_LIMIT

    @rule(tokens_amount=multi_metric_amounts, requests_amount=multi_metric_amounts)
    def consume(self, tokens_amount, requests_amount):
        tokens_r = self._tokens_readable()
        requests_r = self._requests_readable()
        self.backend.consume_capacity(
            frozen_usage(
                {TOKENS_METRIC: tokens_amount, REQUESTS_METRIC: requests_amount}
            )
        )
        self.shadow_tokens_raw = tokens_r - tokens_amount
        self.shadow_requests_raw = requests_r - requests_amount

    @rule(tokens_amount=multi_metric_amounts, requests_amount=multi_metric_amounts)
    def try_acquire(self, tokens_amount, requests_amount):
        tokens_r = self._tokens_readable()
        requests_r = self._requests_readable()

        # Exceeds any metric's max → ValueError
        if tokens_amount > self.tokens_max or requests_amount > self.requests_max:
            with pytest.raises(ValueError, match="exceeds bucket max capacity"):
                self.backend.wait_for_capacity(
                    frozen_usage(
                        {TOKENS_METRIC: tokens_amount, REQUESTS_METRIC: requests_amount}
                    ),
                    timeout=0.0,
                )
            return

        if tokens_amount <= tokens_r and requests_amount <= requests_r:
            self.backend.wait_for_capacity(
                frozen_usage(
                    {TOKENS_METRIC: tokens_amount, REQUESTS_METRIC: requests_amount}
                ),
                timeout=0.0,
            )
            self.shadow_tokens_raw = max(0.0, tokens_r - tokens_amount)
            self.shadow_requests_raw = max(0.0, requests_r - requests_amount)
        else:
            with pytest.raises(TimeoutError):
                self.backend.wait_for_capacity(
                    frozen_usage(
                        {TOKENS_METRIC: tokens_amount, REQUESTS_METRIC: requests_amount}
                    ),
                    timeout=0.0,
                )

    @rule(
        tokens_reserved=multi_metric_amounts,
        requests_reserved=multi_metric_amounts,
        actual_fraction=st.floats(
            min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False
        ),
    )
    def refund(self, tokens_reserved, requests_reserved, actual_fraction):
        tokens_actual = tokens_reserved * actual_fraction
        requests_actual = requests_reserved * actual_fraction
        tokens_r = self._tokens_readable()
        requests_r = self._requests_readable()
        self.backend.refund_capacity(
            frozen_usage(
                {TOKENS_METRIC: tokens_reserved, REQUESTS_METRIC: requests_reserved}
            ),
            frozen_usage(
                {TOKENS_METRIC: tokens_actual, REQUESTS_METRIC: requests_actual}
            ),
        )
        tokens_refund = max(tokens_reserved - tokens_actual, -self.tokens_max)
        requests_refund = max(requests_reserved - requests_actual, -self.requests_max)
        self.shadow_tokens_raw = min(tokens_r + tokens_refund, self.tokens_max)
        self.shadow_requests_raw = min(requests_r + requests_refund, self.requests_max)

    @rule(value=max_cap_values)
    def set_max_capacity_tokens(self, value):
        self.backend.set_max_capacity(TOKENS_METRIC, TOKENS_WINDOW, value)
        self.tokens_max = value

    @rule(value=max_cap_values)
    def set_max_capacity_requests(self, value):
        self.backend.set_max_capacity(REQUESTS_METRIC, REQUESTS_WINDOW, value)
        self.requests_max = value

    @invariant()
    def tokens_capacity_matches_shadow(self):
        if self.tokens_bucket is None:
            return
        actual = self.tokens_bucket.get_capacity(FROZEN_TIME).amount
        expected = self._tokens_readable()
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"Tokens mismatch: actual={actual}, shadow={expected}"
        )

    @invariant()
    def requests_capacity_matches_shadow(self):
        if self.requests_bucket is None:
            return
        actual = self.requests_bucket.get_capacity(FROZEN_TIME).amount
        expected = self._requests_readable()
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"Requests mismatch: actual={actual}, shadow={expected}"
        )

    @invariant()
    def capacities_within_max(self):
        if self.tokens_bucket is None:
            return
        tokens_actual = self.tokens_bucket.get_capacity(FROZEN_TIME).amount
        requests_actual = self.requests_bucket.get_capacity(FROZEN_TIME).amount
        assert tokens_actual <= self.tokens_max + 1e-9
        assert requests_actual <= self.requests_max + 1e-9

    @invariant()
    def metrics_independent(self):
        """Verify cross-metric isolation: each metric tracks its own shadow."""
        if self.tokens_bucket is None:
            return
        tokens_actual = self.tokens_bucket.get_capacity(FROZEN_TIME).amount
        requests_actual = self.requests_bucket.get_capacity(FROZEN_TIME).amount
        tokens_expected = self._tokens_readable()
        requests_expected = self._requests_readable()
        # If metrics aren't independent, one would drift from its shadow
        assert tokens_actual == pytest.approx(tokens_expected, abs=1e-9)
        assert requests_actual == pytest.approx(requests_expected, abs=1e-9)

    def teardown(self):
        self._time_patcher.stop()


StatefulMultiMetric = MultiMetricAccountingMachine.TestCase
StatefulMultiMetric.settings = hypothesis_settings(
    max_examples=200, stateful_step_count=50, deadline=None
)


# ---------------------------------------------------------------------------
# 2e. Acquire-refund conservation property test
# ---------------------------------------------------------------------------


@st.composite
def acquire_refund_pairs(draw):
    """Generate a list of (acquire_amount, actual_fraction) pairs."""
    pairs = []
    n = draw(st.integers(min_value=1, max_value=15))
    for _ in range(n):
        # Small amounts so acquires can succeed on a 1000-capacity bucket
        amount = draw(
            st.floats(
                min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False
            )
        )
        actual_fraction = draw(
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        )
        pairs.append((amount, actual_fraction))
    return pairs


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@hypothesis_settings(max_examples=500, deadline=None)
@given(pairs=acquire_refund_pairs())
def test_acquire_refund_conservation(pairs):
    """After each acquire+refund pair, capacity is conserved.

    For each pair: acquire `amount`, then refund with reserved=amount,
    actual=amount*fraction. Final capacity should equal
    pre_acquire - actual (capped at max_capacity).
    """
    config = _make_config()
    bucket = _make_bucket()

    with (
        patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        ) as mock_time,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        mock_time.time.return_value = FROZEN_TIME
        mock_time.monotonic.return_value = FROZEN_TIME
        backend = SyncMemoryBackend(buckets=[bucket], limit_config=config)

        for amount, actual_fraction in pairs:
            pre_acquire = bucket.get_capacity(FROZEN_TIME).amount

            if amount > LIMIT:
                # Can't acquire more than max capacity
                continue

            if amount > pre_acquire:
                # Not enough capacity — skip (frozen time, no refill)
                continue

            backend.wait_for_capacity(frozen_usage({METRIC: amount}), timeout=0.0)
            actual = amount * actual_fraction
            backend.refund_capacity(
                frozen_usage({METRIC: amount}),
                frozen_usage({METRIC: actual}),
            )

            post_refund = bucket.get_capacity(FROZEN_TIME).amount

            # Conservation: what we actually used is `actual`, so capacity should
            # drop by `actual` from pre_acquire (capped at max_capacity)
            expected = min(pre_acquire - actual, LIMIT)
            assert post_refund == pytest.approx(expected, abs=1e-9), (
                f"Conservation violation: pre={pre_acquire}, amount={amount}, "
                f"actual={actual}, post={post_refund}, expected={expected}"
            )

            # Capacity must never exceed max
            assert post_refund <= LIMIT + 1e-9
