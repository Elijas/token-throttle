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

amounts = st.floats(min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False)
small_amounts = st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False)
max_cap_values = st.floats(min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False)


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

    @rule(amount=amounts)
    def consume(self, amount):
        readable_before = self._shadow_readable()
        self.backend.consume_capacity(frozen_usage({METRIC: amount}))
        self.shadow_raw_stored = readable_before - amount

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
            self.backend.wait_for_capacity(
                frozen_usage({METRIC: amount}), timeout=0.0
            )
            # After successful acquire: capacity = max(0, readable - amount)
            self.shadow_raw_stored = max(0.0, readable - amount)
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
        refund_amount = reserved - actual
        self.shadow_raw_stored = min(
            readable_before + refund_amount, self.shadow_max_capacity
        )

    @rule(value=max_cap_values)
    def set_max_capacity(self, value):
        self.backend.set_max_capacity(METRIC, WINDOW, value)
        self.shadow_max_capacity = value

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
        self.backend = SyncMemoryBackend(
            buckets=[short_b, long_b], limit_config=config
        )
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
            self.backend.wait_for_capacity(
                frozen_usage({METRIC: amount}), timeout=0.0
            )
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
        refund_amount = reserved - actual
        self.shadow_short_raw = min(short_r + refund_amount, self.short_max)
        self.shadow_long_raw = min(long_r + refund_amount, self.long_max)

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
        else:
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
    return ops


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@hypothesis_settings(max_examples=100, deadline=None)
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

        for op in ops:
            if op[0] == Op.CONSUME:
                sync_backend.consume_capacity(frozen_usage({METRIC: op[1]}))
            elif op[0] == Op.REFUND:
                sync_backend.refund_capacity(
                    frozen_usage({METRIC: op[1]}),
                    frozen_usage({METRIC: op[2]}),
                )

    with (
        patch(
            "token_throttle._limiter_backends._memory._backend.time"
        ) as async_mock,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        async_mock.time.return_value = FROZEN_TIME
        async_mock.monotonic.return_value = FROZEN_TIME
        async_backend = MemoryBackend(
            buckets=[async_bucket], limit_config=async_config
        )

        async def run_async():
            for op in ops:
                if op[0] == Op.CONSUME:
                    await async_backend.consume_capacity(
                        frozen_usage({METRIC: op[1]})
                    )
                elif op[0] == Op.REFUND:
                    await async_backend.refund_capacity(
                        frozen_usage({METRIC: op[1]}),
                        frozen_usage({METRIC: op[2]}),
                    )

        asyncio.run(run_async())

    sync_cap = sync_bucket.get_capacity(FROZEN_TIME).amount
    async_cap = async_bucket.get_capacity(FROZEN_TIME).amount

    assert sync_cap == pytest.approx(async_cap, abs=1e-12), (
        f"Sync/async divergence: sync={sync_cap}, async={async_cap}"
    )
