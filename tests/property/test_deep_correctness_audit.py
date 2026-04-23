"""
Deep correctness audit property tests for quota accounting.

Four stateful machines + two parametric tests targeting five coverage gaps:

A. Speedometer (consume_capacity) -> time advance -> refund_capacity lifecycle
B. Per-window independent set_max_capacity with consume
C. Fractional/tiny limits (3.7, 0.5, 17.31) - irrational-looking rates
D. Long sequences (250 steps) - float accumulation drift detection
E. consume -> set_max_capacity -> refund 3-way chain
"""

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
    precondition,
    rule,
)

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._limiter_backends._memory._bucket import MemoryBucket
from token_throttle._limiter_backends._memory._sync_backend import SyncMemoryBackend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INITIAL_TIME = 1_000_000.0
FROZEN_TIME = 1_000_000.0
METRIC = "tokens"

# Machine 1: Speedometer + Refund Lifecycle (two windows)
M1_SHORT_WINDOW = 60
M1_SHORT_LIMIT = 100.0
M1_LONG_WINDOW = 3600
M1_LONG_LIMIT = 1000.0

# Machine 2: Consume -> SetMax -> Refund chain
M2_WINDOW = 60
M2_LIMIT = 200.0

# Machine 4: Long sequence stress test
M4_WINDOW = 60
M4_LIMIT = 1000.0

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

time_deltas = st.floats(
    min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False
)

# Machine 1
m1_consume_amounts = st.floats(
    min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False
)
m1_acquire_amounts = st.floats(
    min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False
)
m1_max_cap_values = st.floats(
    min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False
)

# Machine 2
m2_consume_amounts = st.floats(
    min_value=10.0, max_value=400.0, allow_nan=False, allow_infinity=False
)
m2_max_cap_values = st.floats(
    min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)
m2_acquire_amounts = st.floats(
    min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False
)

# Machine 3
m3_consume_amounts = st.floats(
    min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False
)
m3_acquire_amounts = st.floats(
    min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False
)

# Machine 4
m4_amounts = st.floats(
    min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False
)
m4_max_cap_values = st.floats(
    min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False
)


# ===========================================================================
# Machine 1: SpeedometerRefundLifecycleMachine (Gaps A + B)
# ===========================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class SpeedometerRefundLifecycleMachine(RuleBasedStateMachine):
    """
    Two-window system (60s + 3600s, same metric) using consume_capacity
    exclusively to drive capacity deeply negative. Tests the speedometer ->
    time advance -> refund lifecycle with independent per-window
    set_max_capacity.
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
        self.short_max: float = M1_SHORT_LIMIT
        self.short_rate: float = M1_SHORT_LIMIT / M1_SHORT_WINDOW

        # Long window shadow
        self.long_stored: float | None = None
        self.long_last_checked: float | None = None
        self.long_max: float = M1_LONG_LIMIT
        self.long_rate: float = M1_LONG_LIMIT / M1_LONG_WINDOW

        # Lifecycle tracking: consumes that can be refunded later
        self.outstanding_consumes: list[float] = []

        # Per-window conservation
        self.short_consumed: float = 0.0
        self.short_refunded: float = 0.0
        self.short_baseline: float = M1_SHORT_LIMIT
        self.short_cumulative_refill: float = 0.0

        self.long_consumed: float = 0.0
        self.long_refunded: float = 0.0
        self.long_baseline: float = M1_LONG_LIMIT
        self.long_cumulative_refill: float = 0.0

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
            self.short_stored,
            self.short_last_checked,
            self.short_max,
            self.short_rate,
        )

    def _long_readable(self) -> float:
        return self._readable(
            self.long_stored,
            self.long_last_checked,
            self.long_max,
            self.long_rate,
        )

    @initialize()
    def init_backend(self):
        short_q = Quota(
            metric=METRIC, limit=M1_SHORT_LIMIT, per_seconds=M1_SHORT_WINDOW
        )
        long_q = Quota(metric=METRIC, limit=M1_LONG_LIMIT, per_seconds=M1_LONG_WINDOW)
        config = PerModelConfig(
            model_family="test", quotas=UsageQuotas([short_q, long_q])
        )
        short_b = MemoryBucket(
            metric=METRIC,
            per_seconds=M1_SHORT_WINDOW,
            limit=M1_SHORT_LIMIT,
            model_family="test",
        )
        long_b = MemoryBucket(
            metric=METRIC,
            per_seconds=M1_LONG_WINDOW,
            limit=M1_LONG_LIMIT,
            model_family="test",
        )
        self.backend = SyncMemoryBackend(buckets=[short_b, long_b], limit_config=config)
        self.short_bucket = short_b
        self.long_bucket = long_b
        self.current_time = INITIAL_TIME

        self.short_stored = None
        self.short_last_checked = None
        self.short_max = M1_SHORT_LIMIT
        self.short_rate = M1_SHORT_LIMIT / M1_SHORT_WINDOW

        self.long_stored = None
        self.long_last_checked = None
        self.long_max = M1_LONG_LIMIT
        self.long_rate = M1_LONG_LIMIT / M1_LONG_WINDOW

        self.outstanding_consumes = []

        self.short_consumed = 0.0
        self.short_refunded = 0.0
        self.short_baseline = M1_SHORT_LIMIT
        self.short_cumulative_refill = 0.0

        self.long_consumed = 0.0
        self.long_refunded = 0.0
        self.long_baseline = M1_LONG_LIMIT
        self.long_cumulative_refill = 0.0

    @rule(delta=time_deltas)
    def advance_time(self, delta):
        self.short_cumulative_refill += delta * self.short_rate
        self.long_cumulative_refill += delta * self.long_rate
        self.current_time += delta

    @rule(amount=m1_consume_amounts)
    def consume(self, amount):
        """Speedometer consume, tracked for lifecycle refund."""
        short_r = self._short_readable()
        long_r = self._long_readable()
        self.backend.consume_capacity(frozen_usage({METRIC: amount}))
        self.short_stored = short_r - amount
        self.short_last_checked = self.current_time
        self.long_stored = long_r - amount
        self.long_last_checked = self.current_time
        self.short_consumed += amount
        self.long_consumed += amount
        self.outstanding_consumes.append(amount)

    @precondition(lambda self: len(self.outstanding_consumes) > 0)
    @rule(
        actual_fraction=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    def refund_outstanding(self, actual_fraction):
        """Refund against a prior consume — lifecycle-tracked."""
        reserved = self.outstanding_consumes.pop()
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
        self.short_refunded += refund_amount
        self.long_refunded += refund_amount

    @rule(value=m1_max_cap_values)
    def set_max_capacity_short(self, value):
        """Set max capacity for short window only (Gap B)."""
        old_readable = self._short_readable()
        if self.short_stored is not None and self.short_last_checked is not None:
            time_passed = max(0.0, self.current_time - self.short_last_checked)
            self.short_stored = self.short_stored + time_passed * self.short_rate
            self.short_last_checked = self.current_time
        self.backend.set_max_capacity(METRIC, M1_SHORT_WINDOW, value)
        self.short_max = value
        self.short_rate = value / M1_SHORT_WINDOW
        new_readable = self._short_readable()
        gain = new_readable - old_readable
        if gain > 0:
            self.short_baseline += gain

    @rule(value=m1_max_cap_values)
    def set_max_capacity_long(self, value):
        """Set max capacity for long window only (Gap B)."""
        old_readable = self._long_readable()
        if self.long_stored is not None and self.long_last_checked is not None:
            time_passed = max(0.0, self.current_time - self.long_last_checked)
            self.long_stored = self.long_stored + time_passed * self.long_rate
            self.long_last_checked = self.current_time
        self.backend.set_max_capacity(METRIC, M1_LONG_WINDOW, value)
        self.long_max = value
        self.long_rate = value / M1_LONG_WINDOW
        new_readable = self._long_readable()
        gain = new_readable - old_readable
        if gain > 0:
            self.long_baseline += gain

    @rule(amount=m1_consume_amounts)
    def consume_then_immediate_refund(self, amount):
        """Compound rule: consume then immediately refund full amount."""
        # Consume
        short_r = self._short_readable()
        long_r = self._long_readable()
        self.backend.consume_capacity(frozen_usage({METRIC: amount}))
        self.short_stored = short_r - amount
        self.short_last_checked = self.current_time
        self.long_stored = long_r - amount
        self.long_last_checked = self.current_time
        self.short_consumed += amount
        self.long_consumed += amount

        # Immediate full refund (reserved=amount, actual=0)
        short_r2 = self._short_readable()
        long_r2 = self._long_readable()
        self.backend.refund_capacity(
            frozen_usage({METRIC: amount}),
            frozen_usage({METRIC: 0.0}),
        )
        self.short_stored = min(short_r2 + amount, self.short_max)
        self.short_last_checked = self.current_time
        self.long_stored = min(long_r2 + amount, self.long_max)
        self.long_last_checked = self.current_time
        self.short_refunded += amount
        self.long_refunded += amount

    @rule(amount=m1_acquire_amounts)
    def try_acquire(self, amount):
        short_r = self._short_readable()
        long_r = self._long_readable()

        if amount > self.short_max or amount > self.long_max:
            with pytest.raises(ValueError, match="exceeds bucket max capacity"):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )
            return

        if amount <= short_r and amount <= long_r:
            self.backend.wait_for_capacity(frozen_usage({METRIC: amount}), timeout=0.0)
            self.short_stored = max(0.0, short_r - amount)
            self.short_last_checked = self.current_time
            self.long_stored = max(0.0, long_r - amount)
            self.long_last_checked = self.current_time
            self.short_consumed += amount
            self.long_consumed += amount
        else:
            with pytest.raises(TimeoutError):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )

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

    @invariant()
    def short_conservation(self):
        if self.short_bucket is None:
            return
        actual = self.short_bucket.get_capacity(self.current_time).amount
        balance = actual + self.short_consumed - self.short_refunded
        budget = self.short_baseline + self.short_cumulative_refill
        assert balance <= budget + 1e-6, (
            f"Short conservation violation: capacity={actual}, "
            f"consumed={self.short_consumed}, refunded={self.short_refunded}, "
            f"balance={balance}, budget={budget}"
        )

    @invariant()
    def long_conservation(self):
        if self.long_bucket is None:
            return
        actual = self.long_bucket.get_capacity(self.current_time).amount
        balance = actual + self.long_consumed - self.long_refunded
        budget = self.long_baseline + self.long_cumulative_refill
        assert balance <= budget + 1e-6, (
            f"Long conservation violation: capacity={actual}, "
            f"consumed={self.long_consumed}, refunded={self.long_refunded}, "
            f"balance={balance}, budget={budget}"
        )

    def teardown(self):
        self._time_patcher.stop()


StatefulSpeedometerRefundLifecycle = SpeedometerRefundLifecycleMachine.TestCase
StatefulSpeedometerRefundLifecycle.settings = hypothesis_settings(
    max_examples=200, stateful_step_count=60, deadline=None
)


# ===========================================================================
# Machine 2: ConsumeSetMaxRefundChainMachine (Gap E)
# ===========================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class ConsumeSetMaxRefundChainMachine(RuleBasedStateMachine):
    """
    Single bucket, small limit (200). Tests the specific 3-step chain:
    consume -> set_max_capacity -> refund. Small limit makes negative
    capacity common.
    """

    def __init__(self):
        super().__init__()
        self.backend: SyncMemoryBackend | None = None
        self.bucket: MemoryBucket | None = None
        self.current_time: float = INITIAL_TIME

        self.shadow_stored: float | None = None
        self.shadow_last_checked: float | None = None
        self.shadow_max: float = M2_LIMIT
        self.shadow_rate: float = M2_LIMIT / M2_WINDOW

        self.outstanding_consumes: list[float] = []

        self.total_consumed: float = 0.0
        self.total_refunded: float = 0.0
        self.accounting_baseline: float = M2_LIMIT
        self.cumulative_max_refill: float = 0.0

        self._time_patcher = patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        )
        self._mock_time = self._time_patcher.start()
        self._mock_time.time.side_effect = lambda: self.current_time
        self._mock_time.monotonic.side_effect = lambda: self.current_time

    def _shadow_readable(self) -> float:
        if self.shadow_stored is None:
            return self.shadow_max
        time_passed = max(0.0, self.current_time - self.shadow_last_checked)
        return min(self.shadow_max, self.shadow_stored + time_passed * self.shadow_rate)

    @initialize()
    def init_backend(self):
        config = PerModelConfig(
            model_family="test",
            quotas=UsageQuotas(
                [Quota(metric=METRIC, limit=M2_LIMIT, per_seconds=M2_WINDOW)]
            ),
        )
        bucket = MemoryBucket(
            metric=METRIC,
            per_seconds=M2_WINDOW,
            limit=M2_LIMIT,
            model_family="test",
        )
        self.backend = SyncMemoryBackend(buckets=[bucket], limit_config=config)
        self.bucket = bucket
        self.current_time = INITIAL_TIME

        self.shadow_stored = None
        self.shadow_last_checked = None
        self.shadow_max = M2_LIMIT
        self.shadow_rate = M2_LIMIT / M2_WINDOW

        self.outstanding_consumes = []

        self.total_consumed = 0.0
        self.total_refunded = 0.0
        self.accounting_baseline = M2_LIMIT
        self.cumulative_max_refill = 0.0

    @rule(delta=time_deltas)
    def advance_time(self, delta):
        self.cumulative_max_refill += delta * self.shadow_rate
        self.current_time += delta

    @rule(amount=m2_consume_amounts)
    def consume(self, amount):
        readable = self._shadow_readable()
        self.backend.consume_capacity(frozen_usage({METRIC: amount}))
        self.shadow_stored = readable - amount
        self.shadow_last_checked = self.current_time
        self.total_consumed += amount
        self.outstanding_consumes.append(amount)

    @rule(value=m2_max_cap_values)
    def set_max_capacity(self, value):
        old_readable = self._shadow_readable()
        if self.shadow_stored is not None and self.shadow_last_checked is not None:
            time_passed = max(0.0, self.current_time - self.shadow_last_checked)
            self.shadow_stored = self.shadow_stored + time_passed * self.shadow_rate
            self.shadow_last_checked = self.current_time
        self.backend.set_max_capacity(METRIC, M2_WINDOW, value)
        self.shadow_max = value
        self.shadow_rate = value / M2_WINDOW
        new_readable = self._shadow_readable()
        gain = new_readable - old_readable
        if gain > 0:
            self.accounting_baseline += gain

    @precondition(lambda self: len(self.outstanding_consumes) > 0)
    @rule(
        actual_fraction=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    def refund(self, actual_fraction):
        reserved = self.outstanding_consumes.pop()
        actual = reserved * actual_fraction
        readable = self._shadow_readable()
        self.backend.refund_capacity(
            frozen_usage({METRIC: reserved}),
            frozen_usage({METRIC: actual}),
        )
        refund_amount = reserved - actual
        self.shadow_stored = min(readable + refund_amount, self.shadow_max)
        self.shadow_last_checked = self.current_time
        self.total_refunded += refund_amount

    @rule(
        consume_amount=m2_consume_amounts,
        new_max=m2_max_cap_values,
        actual_fraction=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    def consume_then_set_max_then_refund(
        self, consume_amount, new_max, actual_fraction
    ):
        """Compound: consume -> set_max_capacity -> refund (Gap E)."""
        # 1. Consume
        readable = self._shadow_readable()
        self.backend.consume_capacity(frozen_usage({METRIC: consume_amount}))
        self.shadow_stored = readable - consume_amount
        self.shadow_last_checked = self.current_time
        self.total_consumed += consume_amount

        # 2. Set max capacity (changes the rules for the refund)
        old_readable = self._shadow_readable()
        if self.shadow_stored is not None and self.shadow_last_checked is not None:
            time_passed = max(0.0, self.current_time - self.shadow_last_checked)
            self.shadow_stored = self.shadow_stored + time_passed * self.shadow_rate
            self.shadow_last_checked = self.current_time
        self.backend.set_max_capacity(METRIC, M2_WINDOW, new_max)
        self.shadow_max = new_max
        self.shadow_rate = new_max / M2_WINDOW
        new_readable_after_set = self._shadow_readable()
        gain = new_readable_after_set - old_readable
        if gain > 0:
            self.accounting_baseline += gain

        # 3. Refund (capped by the NEW max, not the old one)
        actual = consume_amount * actual_fraction
        readable2 = self._shadow_readable()
        self.backend.refund_capacity(
            frozen_usage({METRIC: consume_amount}),
            frozen_usage({METRIC: actual}),
        )
        refund_amount = consume_amount - actual
        self.shadow_stored = min(readable2 + refund_amount, self.shadow_max)
        self.shadow_last_checked = self.current_time
        self.total_refunded += refund_amount

    @rule(amount=m2_acquire_amounts)
    def try_acquire(self, amount):
        readable = self._shadow_readable()
        if amount > self.shadow_max:
            with pytest.raises(ValueError, match="exceeds bucket max capacity"):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )
            return

        if amount <= readable:
            self.backend.wait_for_capacity(frozen_usage({METRIC: amount}), timeout=0.0)
            self.shadow_stored = max(0.0, readable - amount)
            self.shadow_last_checked = self.current_time
            self.total_consumed += amount
        else:
            with pytest.raises(TimeoutError):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )

    @invariant()
    def capacity_matches_shadow(self):
        if self.bucket is None:
            return
        actual = self.bucket.get_capacity(self.current_time).amount
        expected = self._shadow_readable()
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"Capacity mismatch: actual={actual}, shadow={expected}"
        )

    @invariant()
    def capacity_within_max(self):
        if self.bucket is None:
            return
        actual = self.bucket.get_capacity(self.current_time).amount
        assert actual <= self.shadow_max + 1e-9, (
            f"Capacity {actual} exceeded max {self.shadow_max}"
        )

    @invariant()
    def conservation(self):
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

    @invariant()
    def negative_debt_preserved(self):
        """set_max_capacity must NOT clamp negative stored capacity."""
        if self.bucket is None or self.bucket.capacity is None:
            return
        if self.shadow_stored is not None and self.shadow_stored < 0:
            assert self.bucket.capacity == pytest.approx(
                self.shadow_stored, abs=1e-9
            ), (
                f"Negative debt was clamped: bucket.capacity={self.bucket.capacity}, "
                f"shadow_stored={self.shadow_stored}"
            )

    def teardown(self):
        self._time_patcher.stop()


StatefulConsumeSetMaxRefundChain = ConsumeSetMaxRefundChainMachine.TestCase
StatefulConsumeSetMaxRefundChain.settings = hypothesis_settings(
    max_examples=300, stateful_step_count=60, deadline=None
)


# ===========================================================================
# Machine 3: FractionalLimitsMachine (Gap C)
# ===========================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class FractionalLimitsMachine(RuleBasedStateMachine):
    """
    Single bucket with fractional limits and odd windows to stress float
    precision. Limits like 3.7/37s give rates like 0.1, while 17.31/97s
    gives ~0.1784... — irrational-looking rates.
    """

    def __init__(self):
        super().__init__()
        self.backend: SyncMemoryBackend | None = None
        self.bucket: MemoryBucket | None = None
        self.current_time: float = INITIAL_TIME

        self.shadow_stored: float | None = None
        self.shadow_last_checked: float | None = None
        self.shadow_max: float = 0.0
        self.shadow_rate: float = 0.0
        self.window: int = 0

        self.total_consumed: float = 0.0
        self.total_refunded: float = 0.0
        self.accounting_baseline: float = 0.0
        self.cumulative_max_refill: float = 0.0

        self._time_patcher = patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        )
        self._mock_time = self._time_patcher.start()
        self._mock_time.time.side_effect = lambda: self.current_time
        self._mock_time.monotonic.side_effect = lambda: self.current_time

    def _shadow_readable(self) -> float:
        if self.shadow_stored is None:
            return self.shadow_max
        time_passed = max(0.0, self.current_time - self.shadow_last_checked)
        return min(self.shadow_max, self.shadow_stored + time_passed * self.shadow_rate)

    @initialize(
        limit=st.sampled_from([0.5, 3.7, 7.77, 17.31, 33.333, 99.99]),
        window=st.sampled_from([37, 60, 97]),
    )
    def init_backend(self, limit, window):
        self.window = window
        config = PerModelConfig(
            model_family="test",
            quotas=UsageQuotas([Quota(metric=METRIC, limit=limit, per_seconds=window)]),
        )
        bucket = MemoryBucket(
            metric=METRIC,
            per_seconds=window,
            limit=limit,
            model_family="test",
        )
        self.backend = SyncMemoryBackend(buckets=[bucket], limit_config=config)
        self.bucket = bucket
        self.current_time = INITIAL_TIME

        self.shadow_stored = None
        self.shadow_last_checked = None
        self.shadow_max = limit
        self.shadow_rate = limit / window

        self.total_consumed = 0.0
        self.total_refunded = 0.0
        self.accounting_baseline = limit
        self.cumulative_max_refill = 0.0

    @rule(delta=time_deltas)
    def advance_time(self, delta):
        self.cumulative_max_refill += delta * self.shadow_rate
        self.current_time += delta

    @rule(amount=m3_consume_amounts)
    def consume(self, amount):
        readable = self._shadow_readable()
        self.backend.consume_capacity(frozen_usage({METRIC: amount}))
        self.shadow_stored = readable - amount
        self.shadow_last_checked = self.current_time
        self.total_consumed += amount

    @rule(amount=m3_acquire_amounts)
    def try_acquire(self, amount):
        readable = self._shadow_readable()
        if amount > self.shadow_max:
            with pytest.raises(ValueError, match="exceeds bucket max capacity"):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )
            return

        if amount <= readable:
            self.backend.wait_for_capacity(frozen_usage({METRIC: amount}), timeout=0.0)
            self.shadow_stored = max(0.0, readable - amount)
            self.shadow_last_checked = self.current_time
            self.total_consumed += amount
        else:
            with pytest.raises(TimeoutError):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )

    @rule(
        reserved=m3_consume_amounts,
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
        self.shadow_stored = min(readable + refund_amount, self.shadow_max)
        self.shadow_last_checked = self.current_time
        self.total_refunded += refund_amount

    @rule(
        value=st.sampled_from([0.5, 3.7, 7.77, 17.31, 33.333, 99.99]),
    )
    def set_max_capacity(self, value):
        old_readable = self._shadow_readable()
        if self.shadow_stored is not None and self.shadow_last_checked is not None:
            time_passed = max(0.0, self.current_time - self.shadow_last_checked)
            self.shadow_stored = self.shadow_stored + time_passed * self.shadow_rate
            self.shadow_last_checked = self.current_time
        self.backend.set_max_capacity(METRIC, self.window, value)
        self.shadow_max = value
        self.shadow_rate = value / self.window
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
            f"max={self.shadow_max}, window={self.window}"
        )

    @invariant()
    def capacity_within_max(self):
        if self.bucket is None:
            return
        actual = self.bucket.get_capacity(self.current_time).amount
        assert actual <= self.shadow_max + 1e-9, (
            f"Capacity {actual} exceeded max {self.shadow_max}"
        )

    @invariant()
    def conservation(self):
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


StatefulFractionalLimits = FractionalLimitsMachine.TestCase
StatefulFractionalLimits.settings = hypothesis_settings(
    max_examples=200, stateful_step_count=50, deadline=None
)


# ===========================================================================
# Machine 4: LongSequenceStressMachine (Gap D)
# ===========================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class LongSequenceStressMachine(RuleBasedStateMachine):
    """
    Single bucket, frozen time, 250 steps. Uses tight (exact ==) conservation
    invariant to detect float accumulation drift over many operations.

    Tracks clipped-by-cap for refund overshoot and set_max_capacity losses
    so the exact balance equation holds.
    """

    def __init__(self):
        super().__init__()
        self.backend: SyncMemoryBackend | None = None
        self.bucket: MemoryBucket | None = None

        self.shadow_stored: float | None = None
        self.shadow_max: float = M4_LIMIT

        self.total_consumed: float = 0.0
        self.total_refunded: float = 0.0
        self.total_clipped_by_cap: float = 0.0
        self.accounting_baseline: float = M4_LIMIT

        self._time_patcher = patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        )
        self._mock_time = self._time_patcher.start()
        self._mock_time.time.return_value = FROZEN_TIME
        self._mock_time.monotonic.return_value = FROZEN_TIME

    def _shadow_readable(self) -> float:
        if self.shadow_stored is None:
            return self.shadow_max
        return min(self.shadow_max, self.shadow_stored)

    @initialize()
    def init_backend(self):
        config = PerModelConfig(
            model_family="test",
            quotas=UsageQuotas(
                [Quota(metric=METRIC, limit=M4_LIMIT, per_seconds=M4_WINDOW)]
            ),
        )
        bucket = MemoryBucket(
            metric=METRIC,
            per_seconds=M4_WINDOW,
            limit=M4_LIMIT,
            model_family="test",
        )
        self.backend = SyncMemoryBackend(buckets=[bucket], limit_config=config)
        self.bucket = bucket

        self.shadow_stored = None
        self.shadow_max = M4_LIMIT

        self.total_consumed = 0.0
        self.total_refunded = 0.0
        self.total_clipped_by_cap = 0.0
        self.accounting_baseline = M4_LIMIT

    @rule(amount=m4_amounts)
    def consume(self, amount):
        readable = self._shadow_readable()
        self.backend.consume_capacity(frozen_usage({METRIC: amount}))
        self.shadow_stored = readable - amount
        self.total_consumed += amount

    @rule(amount=m4_amounts)
    def try_acquire(self, amount):
        readable = self._shadow_readable()
        if amount > self.shadow_max:
            with pytest.raises(ValueError, match="exceeds bucket max capacity"):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )
            return

        if amount <= readable:
            self.backend.wait_for_capacity(frozen_usage({METRIC: amount}), timeout=0.0)
            self.shadow_stored = max(0.0, readable - amount)
            self.total_consumed += amount
        else:
            with pytest.raises(TimeoutError):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )

    @rule(
        reserved=m4_amounts,
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
        new_raw = readable + refund_amount
        capped = min(new_raw, self.shadow_max)
        clipped = new_raw - capped
        if clipped > 0:
            self.total_clipped_by_cap += clipped
        self.shadow_stored = capped
        self.total_refunded += refund_amount

    @rule(value=m4_max_cap_values)
    def set_max_capacity(self, value):
        old_readable = self._shadow_readable()
        self.backend.set_max_capacity(METRIC, M4_WINDOW, value)
        self.shadow_max = value
        new_readable = self._shadow_readable()
        change = new_readable - old_readable
        if change > 0:
            self.accounting_baseline += change
        elif change < 0:
            self.total_clipped_by_cap += -change

    @invariant()
    def capacity_matches_shadow(self):
        if self.bucket is None:
            return
        actual = self.bucket.get_capacity(FROZEN_TIME).amount
        expected = self._shadow_readable()
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"Capacity mismatch: actual={actual}, shadow={expected}"
        )

    @invariant()
    def tight_conservation(self):
        """Exact conservation under frozen time with set_max_capacity.

        Every unit of capacity is either:
        - in the bucket (readable)
        - consumed and not refunded (consumed - refunded)
        - lost to cap-clipping (clipped)
        Total must exactly equal accounting_baseline.
        """
        if self.bucket is None:
            return
        actual = self.bucket.get_capacity(FROZEN_TIME).amount
        balance = (
            actual
            + self.total_consumed
            - self.total_refunded
            + self.total_clipped_by_cap
        )
        assert balance == pytest.approx(self.accounting_baseline, abs=1e-9), (
            f"Tight conservation violation: capacity={actual}, "
            f"consumed={self.total_consumed}, refunded={self.total_refunded}, "
            f"clipped={self.total_clipped_by_cap}, balance={balance}, "
            f"expected={self.accounting_baseline}"
        )

    def teardown(self):
        self._time_patcher.stop()


StatefulLongSequenceStress = LongSequenceStressMachine.TestCase
StatefulLongSequenceStress.settings = hypothesis_settings(
    max_examples=100, stateful_step_count=250, deadline=None
)


# ===========================================================================
# Parametric Tests
# ===========================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@hypothesis_settings(max_examples=200, deadline=None)
@given(
    limit=st.sampled_from([100.0, 500.0, 1000.0]),
    consume_amount=st.floats(
        min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False
    ),
    time_delta=st.floats(
        min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False
    ),
)
def test_consume_time_advance_refund_roundtrip(limit, consume_amount, time_delta):
    """
    Gap A roundtrip: consume X -> advance T -> refund(reserved=X, actual=0).

    Final capacity should == min(readable_before_refund + X, max).
    """
    clock = [INITIAL_TIME]

    with (
        patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        ) as mock_time,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        mock_time.time.side_effect = lambda: clock[0]
        mock_time.monotonic.side_effect = lambda: clock[0]

        config = PerModelConfig(
            model_family="test",
            quotas=UsageQuotas([Quota(metric=METRIC, limit=limit, per_seconds=60)]),
        )
        bucket = MemoryBucket(
            metric=METRIC, per_seconds=60, limit=limit, model_family="test"
        )
        backend = SyncMemoryBackend(buckets=[bucket], limit_config=config)

        # 1. Consume X
        backend.consume_capacity(frozen_usage({METRIC: consume_amount}))

        # 2. Advance time
        clock[0] += time_delta

        # 3. Record readable BEFORE refund
        readable_before_refund = bucket.get_capacity(clock[0]).amount

        # 4. Refund (reserved=X, actual=0)
        backend.refund_capacity(
            frozen_usage({METRIC: consume_amount}),
            frozen_usage({METRIC: 0.0}),
        )

        # 5. Verify
        final = bucket.get_capacity(clock[0]).amount
        expected = min(readable_before_refund + consume_amount, limit)
        assert final == pytest.approx(expected, abs=1e-9), (
            f"Roundtrip failed: limit={limit}, consumed={consume_amount}, "
            f"time_delta={time_delta}, readable_before_refund={readable_before_refund}, "
            f"final={final}, expected={expected}"
        )
        assert final <= limit + 1e-12


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@hypothesis_settings(max_examples=200, deadline=None)
@given(
    limit=st.sampled_from([100.0, 500.0, 1000.0]),
    overshoot=st.floats(
        min_value=0.1, max_value=2000.0, allow_nan=False, allow_infinity=False
    ),
    new_max=st.floats(
        min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False
    ),
)
def test_set_max_preserves_negative_debt(limit, overshoot, new_max):
    """
    Consume large amount -> capacity goes deeply negative -> set_max_capacity
    -> raw stored must be unchanged (NOT clamped to 0 or -new_max).
    """
    consume_amount = limit + overshoot  # guarantees negative capacity

    with (
        patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        ) as mock_time,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        mock_time.time.return_value = FROZEN_TIME
        mock_time.monotonic.return_value = FROZEN_TIME

        config = PerModelConfig(
            model_family="test",
            quotas=UsageQuotas([Quota(metric=METRIC, limit=limit, per_seconds=60)]),
        )
        bucket = MemoryBucket(
            metric=METRIC, per_seconds=60, limit=limit, model_family="test"
        )
        backend = SyncMemoryBackend(buckets=[bucket], limit_config=config)

        # Consume large amount -> deeply negative
        backend.consume_capacity(frozen_usage({METRIC: consume_amount}))
        stored_before = bucket.capacity
        assert stored_before == pytest.approx(limit - consume_amount, abs=1e-9)
        assert stored_before < 0, f"Expected negative capacity, got {stored_before}"

        # Set max capacity
        backend.set_max_capacity(METRIC, 60, new_max)

        # Raw stored must be unchanged
        assert bucket.capacity == pytest.approx(stored_before, abs=1e-9), (
            f"set_max_capacity clamped stored: "
            f"before={stored_before}, after={bucket.capacity}, new_max={new_max}"
        )

        # Readable is capped at new_max (but negative stored stays negative)
        readable = bucket.get_capacity(FROZEN_TIME).amount
        assert readable == pytest.approx(min(new_max, stored_before), abs=1e-9)
