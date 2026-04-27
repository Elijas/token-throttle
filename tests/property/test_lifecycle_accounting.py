"""
Lifecycle property tests for quota accounting correctness.

Three stateful machines targeting specific invariant gaps:

1. NegativeCapacitySetMaxMachine  - set_max_capacity interactions with negative
   capacity and time advancement (Gaps 3, 4)
2. ReservationTrackingMachine     - acquire-refund lifecycle tracking with
   conservation (Gap 2)
3. FullLifecycleMachine           - complete operation cycle on two-window quotas
   with lifecycle-tracked reservations (Gap 5)
"""

from unittest.mock import patch

import pytest
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
METRIC = "tokens"

# Machine 1: small limit to frequently drive capacity negative
M1_WINDOW = 60
M1_LIMIT = 200.0

# Machine 2: moderate limit
M2_WINDOW = 60
M2_LIMIT = 500.0

# Machine 3: two windows
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
# Machine 1: large amounts relative to limit to drive deeply negative
m1_amounts = st.floats(
    min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False
)
m1_small_amounts = st.floats(
    min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False
)
m1_max_cap_values = st.floats(
    min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)

m2_amounts = st.floats(
    min_value=1.0, max_value=250.0, allow_nan=False, allow_infinity=False
)

m3_amounts = st.floats(
    min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False
)
m3_max_cap_values = st.floats(
    min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False
)


# ---------------------------------------------------------------------------
# Machine 1: NegativeCapacitySetMaxMachine
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class NegativeCapacitySetMaxMachine(RuleBasedStateMachine):
    """
    Test set_max_capacity interactions with negative capacity and time advancement.

    Uses a small LIMIT (200) with large consume amounts (50-500) to frequently
    drive capacity deeply negative. Exercises the interaction between
    set_max_capacity (which changes rate and max but NOT stored) and deeply
    negative stored capacity recovering through time-based refill.
    """

    def __init__(self):
        super().__init__()
        self.backend: SyncMemoryBackend | None = None
        self.bucket: MemoryBucket | None = None
        self.current_time: float = INITIAL_TIME

        # Shadow state
        self.shadow_stored: float | None = None
        self.shadow_last_checked: float | None = None
        self.shadow_max_capacity: float = M1_LIMIT
        self.shadow_rate: float = M1_LIMIT / M1_WINDOW

        # Conservation accounting
        self.total_consumed: float = 0.0
        self.total_refunded: float = 0.0
        self.accounting_baseline: float = M1_LIMIT
        self.cumulative_max_refill: float = 0.0

        self._time_patcher = patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        )
        self._mock_time = self._time_patcher.start()
        self._mock_time.time.side_effect = lambda: self.current_time
        self._mock_time.monotonic.side_effect = lambda: self.current_time

    def _shadow_readable(self) -> float:
        if self.shadow_stored is None:
            return self.shadow_max_capacity
        time_passed = max(0.0, self.current_time - self.shadow_last_checked)
        return min(
            self.shadow_max_capacity,
            self.shadow_stored + time_passed * self.shadow_rate,
        )

    @initialize()
    def init_backend(self):
        config = PerModelConfig(
            model_family="test",
            quotas=UsageQuotas(
                [Quota(metric=METRIC, limit=M1_LIMIT, per_seconds=M1_WINDOW)]
            ),
        )
        bucket = MemoryBucket(
            metric=METRIC,
            per_seconds=M1_WINDOW,
            limit=M1_LIMIT,
            model_family="test",
        )
        self.backend = SyncMemoryBackend(buckets=[bucket], limit_config=config)
        self.bucket = bucket
        self.current_time = INITIAL_TIME
        self.shadow_stored = None
        self.shadow_last_checked = None
        self.shadow_max_capacity = M1_LIMIT
        self.shadow_rate = M1_LIMIT / M1_WINDOW
        self.total_consumed = 0.0
        self.total_refunded = 0.0
        self.accounting_baseline = M1_LIMIT
        self.cumulative_max_refill = 0.0

    @rule(delta=time_deltas)
    def advance_time(self, delta):
        self.cumulative_max_refill += delta * self.shadow_rate
        self.current_time += delta

    @rule(amount=m1_amounts)
    def consume(self, amount):
        readable = self._shadow_readable()
        self.backend.consume_capacity(frozen_usage({METRIC: amount}))
        self.shadow_stored = readable - amount
        self.shadow_last_checked = self.current_time
        self.total_consumed += amount

    @rule(amount=m1_small_amounts)
    def try_acquire(self, amount):
        readable = self._shadow_readable()
        if amount > self.shadow_max_capacity:
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
        reserved=m1_small_amounts,
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
        refund_amount = max(reserved - actual, -self.shadow_max_capacity)
        self.shadow_stored = min(readable + refund_amount, self.shadow_max_capacity)
        self.shadow_last_checked = self.current_time
        self.total_refunded += refund_amount

    @rule(value=m1_max_cap_values)
    def set_max_capacity(self, value):
        old_readable = self._shadow_readable()
        if self.shadow_stored is not None and self.shadow_last_checked is not None:
            time_passed = max(0.0, self.current_time - self.shadow_last_checked)
            self.shadow_stored = self.shadow_stored + time_passed * self.shadow_rate
            self.shadow_last_checked = self.current_time
        self.backend.set_max_capacity(METRIC, M1_WINDOW, value)
        self.shadow_max_capacity = value
        self.shadow_rate = value / M1_WINDOW
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
        """No operation should create capacity from nothing."""
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


StatefulNegativeCapacitySetMax = NegativeCapacitySetMaxMachine.TestCase
StatefulNegativeCapacitySetMax.settings = hypothesis_settings(
    max_examples=200, stateful_step_count=50, deadline=None
)


# ---------------------------------------------------------------------------
# Machine 2: ReservationTrackingMachine
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class ReservationTrackingMachine(RuleBasedStateMachine):
    """
    Lifecycle-tracked acquire-refund with conservation.

    Unlike existing tests where refunds are arbitrary (not tied to prior
    acquires), this machine tracks outstanding reservations from successful
    acquires and only refunds against actual reservations. This makes the
    conservation invariant much tighter.
    """

    def __init__(self):
        super().__init__()
        self.backend: SyncMemoryBackend | None = None
        self.bucket: MemoryBucket | None = None
        self.current_time: float = INITIAL_TIME

        # Shadow state
        self.shadow_stored: float | None = None
        self.shadow_last_checked: float | None = None
        self.shadow_max_capacity: float = M2_LIMIT
        self.shadow_rate: float = M2_LIMIT / M2_WINDOW

        # Lifecycle tracking
        self.outstanding_reservations: list[float] = []

        # Conservation accounting
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
            return self.shadow_max_capacity
        time_passed = max(0.0, self.current_time - self.shadow_last_checked)
        return min(
            self.shadow_max_capacity,
            self.shadow_stored + time_passed * self.shadow_rate,
        )

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
        self.shadow_max_capacity = M2_LIMIT
        self.shadow_rate = M2_LIMIT / M2_WINDOW
        self.outstanding_reservations = []
        self.total_consumed = 0.0
        self.total_refunded = 0.0
        self.accounting_baseline = M2_LIMIT
        self.cumulative_max_refill = 0.0

    @rule(delta=time_deltas)
    def advance_time(self, delta):
        self.cumulative_max_refill += delta * self.shadow_rate
        self.current_time += delta

    @rule(amount=m2_amounts)
    def consume(self, amount):
        """Speedometer-style consume (no reservation tracking)."""
        readable = self._shadow_readable()
        self.backend.consume_capacity(frozen_usage({METRIC: amount}))
        self.shadow_stored = readable - amount
        self.shadow_last_checked = self.current_time
        self.total_consumed += amount

    @rule(amount=m2_amounts)
    def try_acquire(self, amount):
        """Acquire with lifecycle tracking — successful acquires are recorded."""
        readable = self._shadow_readable()
        if amount > self.shadow_max_capacity:
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
            self.outstanding_reservations.append(amount)
        else:
            with pytest.raises(TimeoutError):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )

    @precondition(lambda self: len(self.outstanding_reservations) > 0)
    @rule(
        actual_fraction=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    def refund_outstanding(self, actual_fraction):
        """Refund against an actual outstanding reservation."""
        reserved = self.outstanding_reservations.pop()
        actual = reserved * actual_fraction
        readable = self._shadow_readable()
        self.backend.refund_capacity(
            frozen_usage({METRIC: reserved}),
            frozen_usage({METRIC: actual}),
        )
        refund_amount = max(reserved - actual, -self.shadow_max_capacity)
        self.shadow_stored = min(readable + refund_amount, self.shadow_max_capacity)
        self.shadow_last_checked = self.current_time
        self.total_refunded += refund_amount

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
        assert actual <= self.shadow_max_capacity + 1e-9, (
            f"Capacity {actual} exceeded max {self.shadow_max_capacity}"
        )

    @invariant()
    def conservation(self):
        """Conservation is tight because refunds are bounded by actual acquisitions."""
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


StatefulReservationTracking = ReservationTrackingMachine.TestCase
StatefulReservationTracking.settings = hypothesis_settings(
    max_examples=200, stateful_step_count=50, deadline=None
)


# ---------------------------------------------------------------------------
# Machine 3: FullLifecycleMachine
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class FullLifecycleMachine(RuleBasedStateMachine):
    """
    Complete operation cycle on two-window quotas with lifecycle-tracked
    reservations.

    Two windows: SHORT (60s, limit=100) and LONG (3600s, limit=1000), same
    metric. Exercises acquire + consume + refund + set_max_capacity + time
    advance together. Lifecycle-tracked reservations (acquire pushes, refund
    pops). All-or-nothing acquire semantics verified against both window shadows.
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

        # Lifecycle tracking
        self.outstanding_reservations: list[float] = []

        # Per-window conservation accounting
        self.short_consumed: float = 0.0
        self.short_refunded: float = 0.0
        self.short_baseline: float = SHORT_LIMIT
        self.short_cumulative_refill: float = 0.0

        self.long_consumed: float = 0.0
        self.long_refunded: float = 0.0
        self.long_baseline: float = LONG_LIMIT
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
        self.current_time = INITIAL_TIME

        self.short_stored = None
        self.short_last_checked = None
        self.short_max = SHORT_LIMIT
        self.short_rate = SHORT_LIMIT / SHORT_WINDOW

        self.long_stored = None
        self.long_last_checked = None
        self.long_max = LONG_LIMIT
        self.long_rate = LONG_LIMIT / LONG_WINDOW

        self.outstanding_reservations = []

        self.short_consumed = 0.0
        self.short_refunded = 0.0
        self.short_baseline = SHORT_LIMIT
        self.short_cumulative_refill = 0.0

        self.long_consumed = 0.0
        self.long_refunded = 0.0
        self.long_baseline = LONG_LIMIT
        self.long_cumulative_refill = 0.0

    @rule(delta=time_deltas)
    def advance_time(self, delta):
        self.short_cumulative_refill += delta * self.short_rate
        self.long_cumulative_refill += delta * self.long_rate
        self.current_time += delta

    @rule(amount=m3_amounts)
    def consume(self, amount):
        short_r = self._short_readable()
        long_r = self._long_readable()
        self.backend.consume_capacity(frozen_usage({METRIC: amount}))
        self.short_stored = short_r - amount
        self.short_last_checked = self.current_time
        self.long_stored = long_r - amount
        self.long_last_checked = self.current_time
        self.short_consumed += amount
        self.long_consumed += amount

    @rule(amount=m3_amounts)
    def try_acquire(self, amount):
        short_r = self._short_readable()
        long_r = self._long_readable()

        # The backend raises ValueError if usage exceeds ANY bucket's max_capacity
        if amount > self.short_max or amount > self.long_max:
            with pytest.raises(ValueError, match="exceeds bucket max capacity"):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )
            return

        # All-or-nothing: both windows must have enough
        if amount <= short_r and amount <= long_r:
            self.backend.wait_for_capacity(frozen_usage({METRIC: amount}), timeout=0.0)
            self.short_stored = max(0.0, short_r - amount)
            self.short_last_checked = self.current_time
            self.long_stored = max(0.0, long_r - amount)
            self.long_last_checked = self.current_time
            self.short_consumed += amount
            self.long_consumed += amount
            self.outstanding_reservations.append(amount)
        else:
            with pytest.raises(TimeoutError):
                self.backend.wait_for_capacity(
                    frozen_usage({METRIC: amount}), timeout=0.0
                )

    @precondition(lambda self: len(self.outstanding_reservations) > 0)
    @rule(
        actual_fraction=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    def refund_outstanding(self, actual_fraction):
        reserved = self.outstanding_reservations.pop()
        actual = reserved * actual_fraction
        short_r = self._short_readable()
        long_r = self._long_readable()
        self.backend.refund_capacity(
            frozen_usage({METRIC: reserved}),
            frozen_usage({METRIC: actual}),
        )
        raw_refund = reserved - actual
        short_refund = max(raw_refund, -self.short_max)
        long_refund = max(raw_refund, -self.long_max)
        self.short_stored = min(short_r + short_refund, self.short_max)
        self.short_last_checked = self.current_time
        self.long_stored = min(long_r + long_refund, self.long_max)
        self.long_last_checked = self.current_time
        self.short_refunded += short_refund
        self.long_refunded += long_refund

    @rule(value=m3_max_cap_values)
    def set_max_capacity_short(self, value):
        old_readable = self._short_readable()
        if self.short_stored is not None and self.short_last_checked is not None:
            time_passed = max(0.0, self.current_time - self.short_last_checked)
            self.short_stored = self.short_stored + time_passed * self.short_rate
            self.short_last_checked = self.current_time
        self.backend.set_max_capacity(METRIC, SHORT_WINDOW, value)
        self.short_max = value
        self.short_rate = value / SHORT_WINDOW
        new_readable = self._short_readable()
        gain = new_readable - old_readable
        if gain > 0:
            self.short_baseline += gain

    @rule(value=m3_max_cap_values)
    def set_max_capacity_long(self, value):
        old_readable = self._long_readable()
        if self.long_stored is not None and self.long_last_checked is not None:
            time_passed = max(0.0, self.current_time - self.long_last_checked)
            self.long_stored = self.long_stored + time_passed * self.long_rate
            self.long_last_checked = self.current_time
        self.backend.set_max_capacity(METRIC, LONG_WINDOW, value)
        self.long_max = value
        self.long_rate = value / LONG_WINDOW
        new_readable = self._long_readable()
        gain = new_readable - old_readable
        if gain > 0:
            self.long_baseline += gain

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


StatefulFullLifecycle = FullLifecycleMachine.TestCase
StatefulFullLifecycle.settings = hypothesis_settings(
    max_examples=200, stateful_step_count=50, deadline=None
)
