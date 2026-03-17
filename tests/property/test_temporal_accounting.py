"""
Temporal property tests for MemoryBackend quota accounting.

Uses Hypothesis RuleBasedStateMachine to generate random sequences of
operations WITH advancing time, testing the interaction between time-based
refill (stored + elapsed * rate) and mutations (consume, refund, set_max_capacity).

The existing stateful tests in test_stateful_accounting.py all use FROZEN_TIME,
so the refill path through calculate_capacity() is never exercised under random
operation sequences.  These tests close that gap.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INITIAL_TIME = 1_000_000.0
METRIC = "tokens"
WINDOW = 60
LIMIT = 1000.0

SHORT_WINDOW = 60
SHORT_LIMIT = 100.0
LONG_WINDOW = 3600
LONG_LIMIT = 1000.0

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

time_deltas = st.floats(
    min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False
)
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


def _make_config(limit: float = LIMIT, window: int = WINDOW) -> PerModelConfig:
    quota = Quota(metric=METRIC, limit=limit, per_seconds=window)
    return PerModelConfig(model_family="test", quotas=UsageQuotas([quota]))


def _make_bucket(limit: float = LIMIT, window: int = WINDOW) -> MemoryBucket:
    return MemoryBucket(
        metric=METRIC, per_seconds=window, limit=limit, model_family="test"
    )


def _make_sync_backend(limit: float = LIMIT, window: int = WINDOW) -> SyncMemoryBackend:
    config = _make_config(limit, window)
    bucket = _make_bucket(limit, window)
    return SyncMemoryBackend(buckets=[bucket], limit_config=config)


# ---------------------------------------------------------------------------
# 1a. TimeAdvancingAccountingMachine (single bucket)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TimeAdvancingAccountingMachine(RuleBasedStateMachine):
    """
    Shadow-model test for a single-bucket SyncMemoryBackend with advancing time.

    Unlike SingleBucketAccountingMachine (frozen time), this machine advances
    a mutable mock clock between operations so that the lazy refill path
    (stored + elapsed * rate) in calculate_capacity() is exercised.
    """

    def __init__(self):
        super().__init__()
        self.backend: SyncMemoryBackend | None = None
        self.bucket: MemoryBucket | None = None
        self.current_time: float = INITIAL_TIME

        # Shadow state — mirrors bucket.capacity / bucket.last_checked
        self.shadow_stored: float | None = None
        self.shadow_last_checked: float | None = None
        self.shadow_max_capacity: float = LIMIT
        self.shadow_rate: float = LIMIT / WINDOW

        # Conservation accounting
        self.total_consumed: float = 0.0
        self.total_refunded: float = 0.0
        self.accounting_baseline: float = LIMIT
        self.cumulative_max_refill: float = 0.0

        self._time_patcher = patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        )
        self._mock_time = self._time_patcher.start()
        self._mock_time.time.side_effect = lambda: self.current_time
        self._mock_time.monotonic.side_effect = lambda: self.current_time

    def _shadow_readable(self) -> float:
        """Mirror calculate_capacity() logic with the shadow's stored state."""
        if self.shadow_stored is None:
            return self.shadow_max_capacity
        time_passed = max(0.0, self.current_time - self.shadow_last_checked)
        return min(
            self.shadow_max_capacity,
            self.shadow_stored + time_passed * self.shadow_rate,
        )

    @initialize()
    def init_backend(self):
        self.backend = _make_sync_backend()
        self.bucket = self.backend._buckets[0]
        self.current_time = INITIAL_TIME
        self.shadow_stored = None
        self.shadow_last_checked = None
        self.shadow_max_capacity = LIMIT
        self.shadow_rate = LIMIT / WINDOW
        self.total_consumed = 0.0
        self.total_refunded = 0.0
        self.accounting_baseline = LIMIT
        self.cumulative_max_refill = 0.0

    @rule(delta=time_deltas)
    def advance_time(self, delta):
        """Tick the clock forward.  No backend call — refill is lazy."""
        self.cumulative_max_refill += delta * self.shadow_rate
        self.current_time += delta

    @rule(amount=amounts)
    def consume(self, amount):
        readable = self._shadow_readable()
        self.backend.consume_capacity(frozen_usage({METRIC: amount}))
        self.shadow_stored = readable - amount
        self.shadow_last_checked = self.current_time
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
            self.backend.wait_for_capacity(
                frozen_usage({METRIC: amount}), timeout=0.0
            )
            self.shadow_stored = max(0.0, readable - amount)
            self.shadow_last_checked = self.current_time
            self.total_consumed += amount
        else:
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
        readable = self._shadow_readable()
        self.backend.refund_capacity(
            frozen_usage({METRIC: reserved}),
            frozen_usage({METRIC: actual}),
        )
        refund_amount = reserved - actual
        self.shadow_stored = min(
            readable + refund_amount, self.shadow_max_capacity
        )
        self.shadow_last_checked = self.current_time
        self.total_refunded += refund_amount

    @rule(value=max_cap_values)
    def set_max_capacity(self, value):
        old_readable = self._shadow_readable()
        self.backend.set_max_capacity(METRIC, WINDOW, value)
        self.shadow_max_capacity = value
        self.shadow_rate = value / WINDOW
        new_readable = self._shadow_readable()
        gain = new_readable - old_readable
        if gain > 0:
            self.accounting_baseline += gain

    @invariant()
    def capacity_matches_shadow(self):
        if self.bucket is None:
            return
        actual = self.bucket.get_capacity(self.current_time).amount
        expected = self._shadow_readable()
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"Capacity mismatch: actual={actual}, shadow={expected}, "
            f"stored={self.shadow_stored}, last_checked={self.shadow_last_checked}, "
            f"time={self.current_time}, max_cap={self.shadow_max_capacity}"
        )

    @invariant()
    def capacity_within_max(self):
        if self.bucket is None:
            return
        actual = self.bucket.get_capacity(self.current_time).amount
        assert actual <= self.shadow_max_capacity + 1e-9, (
            f"Capacity {actual} exceeded max {self.shadow_max_capacity}"
        )

    @invariant()
    def conservation(self):
        """No operation should create capacity from nothing.

        capacity + consumed - refunded <= baseline + cumulative_max_refill.
        The baseline starts at LIMIT and increases when set_max_capacity
        raises the readable ceiling.  cumulative_max_refill is the integral
        of rate * dt across all time advances (an upper bound on actual
        refill, since capping can only lose capacity).
        """
        if self.bucket is None:
            return
        actual = self.bucket.get_capacity(self.current_time).amount
        balance = actual + self.total_consumed - self.total_refunded
        budget = self.accounting_baseline + self.cumulative_max_refill
        assert balance <= budget + 1e-6, (
            f"Conservation violation: capacity={actual}, "
            f"consumed={self.total_consumed}, refunded={self.total_refunded}, "
            f"balance={balance}, budget={budget}"
        )

    def teardown(self):
        self._time_patcher.stop()


StatefulTimeAdvancing = TimeAdvancingAccountingMachine.TestCase
StatefulTimeAdvancing.settings = hypothesis_settings(
    max_examples=200, stateful_step_count=50, deadline=None
)


# ---------------------------------------------------------------------------
# 1b. TimeAdvancingMultiWindowMachine (two windows)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TimeAdvancingMultiWindowMachine(RuleBasedStateMachine):
    """
    Shadow-model test for a two-window SyncMemoryBackend with advancing time.

    Two quotas for the same metric: 60s window (limit=100) and 3600s window
    (limit=1000).  Time advances between operations, creating asymmetric
    refill rates (short window refills ~6x faster per unit time).

    Key invariant: try_acquire succeeds iff BOTH windows have sufficient
    capacity (including refill).
    """

    def __init__(self):
        super().__init__()
        self.backend: SyncMemoryBackend | None = None
        self.short_bucket: MemoryBucket | None = None
        self.long_bucket: MemoryBucket | None = None
        self.current_time: float = INITIAL_TIME

        # Short window shadow
        self.short_stored: float | None = None
        self.short_last_checked: float | None = None
        self.short_max: float = SHORT_LIMIT
        self.short_rate: float = SHORT_LIMIT / SHORT_WINDOW

        # Long window shadow
        self.long_stored: float | None = None
        self.long_last_checked: float | None = None
        self.long_max: float = LONG_LIMIT
        self.long_rate: float = LONG_LIMIT / LONG_WINDOW

        self._time_patcher = patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        )
        self._mock_time = self._time_patcher.start()
        self._mock_time.time.side_effect = lambda: self.current_time
        self._mock_time.monotonic.side_effect = lambda: self.current_time

    def _readable(
        self,
        stored: float | None,
        last_checked: float | None,
        max_cap: float,
        rate: float,
    ) -> float:
        if stored is None:
            return max_cap
        time_passed = max(0.0, self.current_time - last_checked)
        return min(max_cap, stored + time_passed * rate)

    def _short_readable(self) -> float:
        return self._readable(
            self.short_stored, self.short_last_checked,
            self.short_max, self.short_rate,
        )

    def _long_readable(self) -> float:
        return self._readable(
            self.long_stored, self.long_last_checked,
            self.long_max, self.long_rate,
        )

    @initialize()
    def init_backend(self):
        short_q = Quota(metric=METRIC, limit=SHORT_LIMIT, per_seconds=SHORT_WINDOW)
        long_q = Quota(metric=METRIC, limit=LONG_LIMIT, per_seconds=LONG_WINDOW)
        config = PerModelConfig(
            model_family="test", quotas=UsageQuotas([short_q, long_q])
        )
        short_b = MemoryBucket(
            metric=METRIC, per_seconds=SHORT_WINDOW,
            limit=SHORT_LIMIT, model_family="test",
        )
        long_b = MemoryBucket(
            metric=METRIC, per_seconds=LONG_WINDOW,
            limit=LONG_LIMIT, model_family="test",
        )
        self.backend = SyncMemoryBackend(
            buckets=[short_b, long_b], limit_config=config
        )
        self.short_bucket = short_b
        self.long_bucket = long_b
        self.current_time = INITIAL_TIME

        self.short_stored = None
        self.short_last_checked = None
        self.short_max = SHORT_LIMIT
        self.short_rate = SHORT_LIMIT / SHORT_WINDOW

        self.long_stored = None
        self.long_last_checked = None
        self.long_max = LONG_LIMIT
        self.long_rate = LONG_LIMIT / LONG_WINDOW

    @rule(delta=time_deltas)
    def advance_time(self, delta):
        self.current_time += delta

    @rule(amount=small_amounts)
    def consume(self, amount):
        short_r = self._short_readable()
        long_r = self._long_readable()
        self.backend.consume_capacity(frozen_usage({METRIC: amount}))
        self.short_stored = short_r - amount
        self.short_last_checked = self.current_time
        self.long_stored = long_r - amount
        self.long_last_checked = self.current_time

    @rule(amount=small_amounts)
    def try_acquire(self, amount):
        short_r = self._short_readable()
        long_r = self._long_readable()

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
            self.short_stored = max(0.0, short_r - amount)
            self.short_last_checked = self.current_time
            self.long_stored = max(0.0, long_r - amount)
            self.long_last_checked = self.current_time
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
        self.short_stored = min(short_r + refund_amount, self.short_max)
        self.short_last_checked = self.current_time
        self.long_stored = min(long_r + refund_amount, self.long_max)
        self.long_last_checked = self.current_time

    @invariant()
    def short_capacity_matches_shadow(self):
        if self.short_bucket is None:
            return
        actual = self.short_bucket.get_capacity(self.current_time).amount
        expected = self._short_readable()
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"Short window mismatch: actual={actual}, shadow={expected}"
        )

    @invariant()
    def long_capacity_matches_shadow(self):
        if self.long_bucket is None:
            return
        actual = self.long_bucket.get_capacity(self.current_time).amount
        expected = self._long_readable()
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"Long window mismatch: actual={actual}, shadow={expected}"
        )

    @invariant()
    def capacities_within_max(self):
        if self.short_bucket is None:
            return
        short_actual = self.short_bucket.get_capacity(self.current_time).amount
        long_actual = self.long_bucket.get_capacity(self.current_time).amount
        assert short_actual <= self.short_max + 1e-9
        assert long_actual <= self.long_max + 1e-9

    def teardown(self):
        self._time_patcher.stop()


StatefulTimeAdvancingMultiWindow = TimeAdvancingMultiWindowMachine.TestCase
StatefulTimeAdvancingMultiWindow.settings = hypothesis_settings(
    max_examples=200, stateful_step_count=50, deadline=None
)


# ---------------------------------------------------------------------------
# 1c. Time-advancing sync/async parity property test
# ---------------------------------------------------------------------------


class Op(enum.Enum):
    CONSUME = "consume"
    REFUND = "refund"
    ACQUIRE = "acquire"
    ADVANCE_TIME = "advance_time"


@st.composite
def temporal_ops_strategy(draw):
    """Generate a list of (Op, ...) tuples including time advances."""
    ops = []
    n = draw(st.integers(min_value=1, max_value=20))
    for _ in range(n):
        op_type = draw(st.sampled_from(Op))
        if op_type == Op.CONSUME:
            amount = draw(
                st.floats(
                    min_value=0.1, max_value=200.0,
                    allow_nan=False, allow_infinity=False,
                )
            )
            ops.append((Op.CONSUME, amount))
        elif op_type == Op.REFUND:
            reserved = draw(
                st.floats(
                    min_value=0.1, max_value=200.0,
                    allow_nan=False, allow_infinity=False,
                )
            )
            actual = draw(
                st.floats(
                    min_value=0.0, max_value=400.0,
                    allow_nan=False, allow_infinity=False,
                )
            )
            ops.append((Op.REFUND, reserved, actual))
        elif op_type == Op.ACQUIRE:
            amount = draw(
                st.floats(
                    min_value=0.1, max_value=100.0,
                    allow_nan=False, allow_infinity=False,
                )
            )
            ops.append((Op.ACQUIRE, amount))
        else:
            delta = draw(
                st.floats(
                    min_value=0.0, max_value=60.0,
                    allow_nan=False, allow_infinity=False,
                )
            )
            ops.append((Op.ADVANCE_TIME, delta))
    return ops


def _run_sync_temporal_ops(backend, ops, clock):
    """Execute ops on a sync backend with a mutable clock."""
    acquire_results = []
    for op in ops:
        if op[0] == Op.ADVANCE_TIME:
            clock[0] += op[1]
        elif op[0] == Op.CONSUME:
            backend.consume_capacity(frozen_usage({METRIC: op[1]}))
        elif op[0] == Op.REFUND:
            backend.refund_capacity(
                frozen_usage({METRIC: op[1]}),
                frozen_usage({METRIC: op[2]}),
            )
        elif op[0] == Op.ACQUIRE:
            try:
                backend.wait_for_capacity(
                    frozen_usage({METRIC: op[1]}), timeout=0.0
                )
                acquire_results.append(True)
            except (TimeoutError, ValueError):
                acquire_results.append(False)
    return acquire_results


async def _run_async_temporal_ops(backend, ops, clock):
    """Execute ops on an async backend with a mutable clock."""
    acquire_results = []
    for op in ops:
        if op[0] == Op.ADVANCE_TIME:
            clock[0] += op[1]
        elif op[0] == Op.CONSUME:
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
@hypothesis_settings(max_examples=100, deadline=None)
@given(ops=temporal_ops_strategy())
def test_temporal_sync_async_parity(ops):
    """Identical op sequences with time advancement on sync and async backends
    must produce identical final capacity and identical acquire results."""
    sync_config = _make_config()
    sync_bucket = _make_bucket()
    async_config = _make_config()
    async_bucket = _make_bucket()

    # Mutable clock shared via single-element list so lambdas capture by ref
    sync_clock = [INITIAL_TIME]
    async_clock = [INITIAL_TIME]

    with (
        patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        ) as sync_mock,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        sync_mock.time.side_effect = lambda: sync_clock[0]
        sync_mock.monotonic.side_effect = lambda: sync_clock[0]
        sync_backend = SyncMemoryBackend(
            buckets=[sync_bucket], limit_config=sync_config
        )
        sync_acquire_results = _run_sync_temporal_ops(sync_backend, ops, sync_clock)

    with (
        patch(
            "token_throttle._limiter_backends._memory._backend.time"
        ) as async_mock,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        async_mock.time.side_effect = lambda: async_clock[0]
        async_mock.monotonic.side_effect = lambda: async_clock[0]
        async_backend = MemoryBackend(
            buckets=[async_bucket], limit_config=async_config
        )
        async_acquire_results = asyncio.run(
            _run_async_temporal_ops(async_backend, ops, async_clock)
        )

    # Both clocks should have advanced identically
    assert sync_clock[0] == pytest.approx(async_clock[0], abs=1e-12)

    sync_cap = sync_bucket.get_capacity(sync_clock[0]).amount
    async_cap = async_bucket.get_capacity(async_clock[0]).amount

    assert sync_acquire_results == async_acquire_results, (
        f"Acquire results diverged: sync={sync_acquire_results}, "
        f"async={async_acquire_results}"
    )
    assert sync_cap == pytest.approx(async_cap, abs=1e-12), (
        f"Sync/async capacity divergence: sync={sync_cap}, async={async_cap}"
    )
