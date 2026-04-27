"""
Correctness audit property tests for quota accounting.

Five targeted tests closing identified coverage gaps:

1. TightConservationMachine     — exact == conservation under frozen time (no set_max_capacity)
2. MultiMetricMultiWindowMachine — 4-bucket system (2 metrics x 2 windows) with time advance
3. SetMaxCapacityRefundInteractionMachine — acquire -> lower max -> refund "trapped capacity" path
4. Floating-point boundary tests — exact-boundary acquire/consume/refund precision
5. CancellationRefundMathTest   — property test for _refund_cancelled_consumption path
"""

import asyncio
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
from token_throttle._limiter_backends._memory._backend import MemoryBackend
from token_throttle._limiter_backends._memory._bucket import MemoryBucket
from token_throttle._limiter_backends._memory._sync_backend import SyncMemoryBackend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FROZEN_TIME = 1_000_000.0
INITIAL_TIME = 1_000_000.0

# Test 1
T1_METRIC = "tokens"
T1_WINDOW = 60
T1_LIMIT = 1000.0

# Test 2
TOKENS_METRIC = "tokens"
REQUESTS_METRIC = "requests"
T2_TOKENS_SHORT_WINDOW = 60
T2_TOKENS_SHORT_LIMIT = 100.0
T2_TOKENS_LONG_WINDOW = 3600
T2_TOKENS_LONG_LIMIT = 1000.0
T2_REQUESTS_SHORT_WINDOW = 60
T2_REQUESTS_SHORT_LIMIT = 20.0
T2_REQUESTS_LONG_WINDOW = 3600
T2_REQUESTS_LONG_LIMIT = 200.0

# Test 3
T3_METRIC = "tokens"
T3_WINDOW = 60
T3_LIMIT = 500.0

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

amounts = st.floats(
    min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False
)
small_amounts = st.floats(
    min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False
)
time_deltas = st.floats(
    min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False
)

# Test 2: amounts sized for the smallest bucket (requests/60s = 20)
t2_token_amounts = st.floats(
    min_value=0.1, max_value=40.0, allow_nan=False, allow_infinity=False
)
t2_request_amounts = st.floats(
    min_value=0.1, max_value=8.0, allow_nan=False, allow_infinity=False
)

# Test 3
t3_amounts = st.floats(
    min_value=10.0, max_value=250.0, allow_nan=False, allow_infinity=False
)
t3_acquire_amounts = st.floats(
    min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False
)


# ===========================================================================
# Test 1: TightConservationMachine
# ===========================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TightConservationMachine(RuleBasedStateMachine):
    """
    Single-bucket with frozen time, no set_max_capacity, no advance_time.

    Under these conditions conservation is exact (==) not just (<=).
    Any capacity lost to cap-clipping (refund overshooting max) is tracked
    explicitly so the equation balances perfectly.
    """

    def __init__(self):
        super().__init__()
        self.backend: SyncMemoryBackend | None = None
        self.bucket: MemoryBucket | None = None
        self.shadow_raw_stored: float | None = None
        self.shadow_max_capacity: float = T1_LIMIT
        self.total_consumed: float = 0.0
        self.total_refunded: float = 0.0
        self.total_clipped_by_cap: float = 0.0
        self._time_patcher = patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        )
        self._mock_time = self._time_patcher.start()
        self._mock_time.time.return_value = FROZEN_TIME
        self._mock_time.monotonic.return_value = FROZEN_TIME

    def _shadow_readable(self) -> float:
        if self.shadow_raw_stored is None:
            return self.shadow_max_capacity
        return min(self.shadow_max_capacity, self.shadow_raw_stored)

    @initialize()
    def init_backend(self):
        config = PerModelConfig(
            model_family="test",
            quotas=UsageQuotas(
                [Quota(metric=T1_METRIC, limit=T1_LIMIT, per_seconds=T1_WINDOW)]
            ),
        )
        bucket = MemoryBucket(
            metric=T1_METRIC,
            per_seconds=T1_WINDOW,
            limit=T1_LIMIT,
            model_family="test",
        )
        self.backend = SyncMemoryBackend(buckets=[bucket], limit_config=config)
        self.bucket = bucket
        self.shadow_raw_stored = None
        self.shadow_max_capacity = T1_LIMIT
        self.total_consumed = 0.0
        self.total_refunded = 0.0
        self.total_clipped_by_cap = 0.0

    @rule(amount=amounts)
    def consume(self, amount):
        readable_before = self._shadow_readable()
        self.backend.consume_capacity(frozen_usage({T1_METRIC: amount}))
        self.shadow_raw_stored = readable_before - amount
        self.total_consumed += amount

    @rule(amount=amounts)
    def try_acquire(self, amount):
        readable = self._shadow_readable()
        if amount > self.shadow_max_capacity:
            with pytest.raises(ValueError, match="exceeds bucket max capacity"):
                self.backend.wait_for_capacity(
                    frozen_usage({T1_METRIC: amount}), timeout=0.0
                )
            return

        if amount <= readable:
            self.backend.wait_for_capacity(
                frozen_usage({T1_METRIC: amount}), timeout=0.0
            )
            self.shadow_raw_stored = max(0.0, readable - amount)
            self.total_consumed += amount
        else:
            with pytest.raises(TimeoutError):
                self.backend.wait_for_capacity(
                    frozen_usage({T1_METRIC: amount}), timeout=0.0
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
            frozen_usage({T1_METRIC: reserved}),
            frozen_usage({T1_METRIC: actual}),
        )
        refund_amount = max(reserved - actual, -self.shadow_max_capacity)
        new_raw = readable_before + refund_amount
        capped = min(new_raw, self.shadow_max_capacity)
        clipped = new_raw - capped
        if clipped > 0:
            self.total_clipped_by_cap += clipped
        self.shadow_raw_stored = capped
        self.total_refunded += refund_amount

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
        """Exact conservation: capacity + consumed - refunded + clipped == LIMIT.

        Under frozen time with no set_max_capacity and no time advance,
        every unit of capacity is either:
        - still in the bucket (capacity)
        - consumed and not refunded (consumed - refunded)
        - lost to cap-clipping on refund (clipped)
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
        assert balance == pytest.approx(T1_LIMIT, abs=1e-9), (
            f"Tight conservation violation: capacity={actual}, "
            f"consumed={self.total_consumed}, refunded={self.total_refunded}, "
            f"clipped={self.total_clipped_by_cap}, balance={balance}, "
            f"expected={T1_LIMIT}"
        )

    def teardown(self):
        self._time_patcher.stop()


StatefulTightConservation = TightConservationMachine.TestCase
StatefulTightConservation.settings = hypothesis_settings(
    max_examples=300, stateful_step_count=60, deadline=None
)


# ===========================================================================
# Test 2: MultiMetricMultiWindowMachine
# ===========================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class MultiMetricMultiWindowMachine(RuleBasedStateMachine):
    """
    4-bucket system: tokens x {60s, 3600s} + requests x {60s, 3600s}.

    Time-advancing, lifecycle-tracked reservations. Exercises the cross-product
    of multiple metrics AND multiple windows that no existing test covers.
    """

    def __init__(self):
        super().__init__()
        self.backend: SyncMemoryBackend | None = None
        self.current_time: float = INITIAL_TIME

        # 4 shadow buckets: (stored, last_checked, max, rate, consumed, refunded, baseline, cumulative_max_refill)
        self.shadows: dict[tuple[str, int], dict] | None = None
        self.buckets_by_key: dict[tuple[str, int], MemoryBucket] | None = None

        # Lifecycle tracking: list of (tokens_amount, requests_amount)
        self.outstanding_reservations: list[tuple[float, float]] = []

        self._time_patcher = patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        )
        self._mock_time = self._time_patcher.start()
        self._mock_time.time.side_effect = lambda: self.current_time
        self._mock_time.monotonic.side_effect = lambda: self.current_time

    def _shadow_readable(self, key: tuple[str, int]) -> float:
        s = self.shadows[key]
        if s["stored"] is None:
            return s["max"]
        time_passed = max(0.0, self.current_time - s["last_checked"])
        return min(s["max"], s["stored"] + time_passed * s["rate"])

    def _init_shadow(self, metric: str, window: int, limit: float) -> dict:
        return {
            "stored": None,
            "last_checked": None,
            "max": limit,
            "rate": limit / window,
            "consumed": 0.0,
            "refunded": 0.0,
            "baseline": limit,
            "cumulative_max_refill": 0.0,
        }

    @initialize()
    def init_backend(self):
        tokens_short_q = Quota(
            metric=TOKENS_METRIC,
            limit=T2_TOKENS_SHORT_LIMIT,
            per_seconds=T2_TOKENS_SHORT_WINDOW,
        )
        tokens_long_q = Quota(
            metric=TOKENS_METRIC,
            limit=T2_TOKENS_LONG_LIMIT,
            per_seconds=T2_TOKENS_LONG_WINDOW,
        )
        requests_short_q = Quota(
            metric=REQUESTS_METRIC,
            limit=T2_REQUESTS_SHORT_LIMIT,
            per_seconds=T2_REQUESTS_SHORT_WINDOW,
        )
        requests_long_q = Quota(
            metric=REQUESTS_METRIC,
            limit=T2_REQUESTS_LONG_LIMIT,
            per_seconds=T2_REQUESTS_LONG_WINDOW,
        )
        config = PerModelConfig(
            model_family="test",
            quotas=UsageQuotas(
                [tokens_short_q, tokens_long_q, requests_short_q, requests_long_q]
            ),
        )
        ts_b = MemoryBucket(
            metric=TOKENS_METRIC,
            per_seconds=T2_TOKENS_SHORT_WINDOW,
            limit=T2_TOKENS_SHORT_LIMIT,
            model_family="test",
        )
        tl_b = MemoryBucket(
            metric=TOKENS_METRIC,
            per_seconds=T2_TOKENS_LONG_WINDOW,
            limit=T2_TOKENS_LONG_LIMIT,
            model_family="test",
        )
        rs_b = MemoryBucket(
            metric=REQUESTS_METRIC,
            per_seconds=T2_REQUESTS_SHORT_WINDOW,
            limit=T2_REQUESTS_SHORT_LIMIT,
            model_family="test",
        )
        rl_b = MemoryBucket(
            metric=REQUESTS_METRIC,
            per_seconds=T2_REQUESTS_LONG_WINDOW,
            limit=T2_REQUESTS_LONG_LIMIT,
            model_family="test",
        )
        self.backend = SyncMemoryBackend(
            buckets=[ts_b, tl_b, rs_b, rl_b], limit_config=config
        )
        self.current_time = INITIAL_TIME

        self.buckets_by_key = {
            (TOKENS_METRIC, T2_TOKENS_SHORT_WINDOW): ts_b,
            (TOKENS_METRIC, T2_TOKENS_LONG_WINDOW): tl_b,
            (REQUESTS_METRIC, T2_REQUESTS_SHORT_WINDOW): rs_b,
            (REQUESTS_METRIC, T2_REQUESTS_LONG_WINDOW): rl_b,
        }
        self.shadows = {
            (TOKENS_METRIC, T2_TOKENS_SHORT_WINDOW): self._init_shadow(
                TOKENS_METRIC, T2_TOKENS_SHORT_WINDOW, T2_TOKENS_SHORT_LIMIT
            ),
            (TOKENS_METRIC, T2_TOKENS_LONG_WINDOW): self._init_shadow(
                TOKENS_METRIC, T2_TOKENS_LONG_WINDOW, T2_TOKENS_LONG_LIMIT
            ),
            (REQUESTS_METRIC, T2_REQUESTS_SHORT_WINDOW): self._init_shadow(
                REQUESTS_METRIC, T2_REQUESTS_SHORT_WINDOW, T2_REQUESTS_SHORT_LIMIT
            ),
            (REQUESTS_METRIC, T2_REQUESTS_LONG_WINDOW): self._init_shadow(
                REQUESTS_METRIC, T2_REQUESTS_LONG_WINDOW, T2_REQUESTS_LONG_LIMIT
            ),
        }
        self.outstanding_reservations = []

    def _consume_shadow(self, metric: str, amount: float):
        for key, s in self.shadows.items():
            if key[0] != metric:
                continue
            readable = self._shadow_readable(key)
            s["stored"] = readable - amount
            s["last_checked"] = self.current_time
            s["consumed"] += amount

    def _refund_shadow(self, metric: str, refund_amount: float):
        for key, s in self.shadows.items():
            if key[0] != metric:
                continue
            readable = self._shadow_readable(key)
            s["stored"] = min(readable + refund_amount, s["max"])
            s["last_checked"] = self.current_time
            s["refunded"] += refund_amount

    @rule(delta=time_deltas)
    def advance_time(self, delta):
        for s in self.shadows.values():
            s["cumulative_max_refill"] += delta * s["rate"]
        self.current_time += delta

    @rule(tokens_amount=t2_token_amounts, requests_amount=t2_request_amounts)
    def consume(self, tokens_amount, requests_amount):
        self.backend.consume_capacity(
            frozen_usage(
                {TOKENS_METRIC: tokens_amount, REQUESTS_METRIC: requests_amount}
            )
        )
        self._consume_shadow(TOKENS_METRIC, tokens_amount)
        self._consume_shadow(REQUESTS_METRIC, requests_amount)

    @rule(tokens_amount=t2_token_amounts, requests_amount=t2_request_amounts)
    def try_acquire(self, tokens_amount, requests_amount):
        # Check if exceeds any bucket's max
        tokens_short_max = self.shadows[(TOKENS_METRIC, T2_TOKENS_SHORT_WINDOW)]["max"]
        requests_short_max = self.shadows[(REQUESTS_METRIC, T2_REQUESTS_SHORT_WINDOW)][
            "max"
        ]

        if tokens_amount > tokens_short_max or requests_amount > requests_short_max:
            with pytest.raises(ValueError, match="exceeds bucket max capacity"):
                self.backend.wait_for_capacity(
                    frozen_usage(
                        {TOKENS_METRIC: tokens_amount, REQUESTS_METRIC: requests_amount}
                    ),
                    timeout=0.0,
                )
            return

        # All-or-nothing across ALL 4 buckets
        all_ok = True
        for key in self.shadows:
            metric = key[0]
            amt = tokens_amount if metric == TOKENS_METRIC else requests_amount
            if amt > self._shadow_readable(key):
                all_ok = False
                break

        if all_ok:
            self.backend.wait_for_capacity(
                frozen_usage(
                    {TOKENS_METRIC: tokens_amount, REQUESTS_METRIC: requests_amount}
                ),
                timeout=0.0,
            )
            for key, s in self.shadows.items():
                metric = key[0]
                amt = tokens_amount if metric == TOKENS_METRIC else requests_amount
                readable = self._shadow_readable(key)
                s["stored"] = max(0.0, readable - amt)
                s["last_checked"] = self.current_time
                s["consumed"] += amt
            self.outstanding_reservations.append((tokens_amount, requests_amount))
        else:
            with pytest.raises(TimeoutError):
                self.backend.wait_for_capacity(
                    frozen_usage(
                        {TOKENS_METRIC: tokens_amount, REQUESTS_METRIC: requests_amount}
                    ),
                    timeout=0.0,
                )

    @precondition(lambda self: len(self.outstanding_reservations) > 0)
    @rule(
        actual_fraction=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    def refund_outstanding(self, actual_fraction):
        tokens_reserved, requests_reserved = self.outstanding_reservations.pop()
        tokens_actual = tokens_reserved * actual_fraction
        requests_actual = requests_reserved * actual_fraction
        self.backend.refund_capacity(
            frozen_usage(
                {TOKENS_METRIC: tokens_reserved, REQUESTS_METRIC: requests_reserved}
            ),
            frozen_usage(
                {TOKENS_METRIC: tokens_actual, REQUESTS_METRIC: requests_actual}
            ),
        )
        self._refund_shadow(TOKENS_METRIC, tokens_reserved - tokens_actual)
        self._refund_shadow(REQUESTS_METRIC, requests_reserved - requests_actual)

    @rule(
        value=st.floats(
            min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False
        )
    )
    def set_max_capacity_tokens(self, value):
        for key in [
            (TOKENS_METRIC, T2_TOKENS_SHORT_WINDOW),
            (TOKENS_METRIC, T2_TOKENS_LONG_WINDOW),
        ]:
            old_readable = self._shadow_readable(key)
            s = self.shadows[key]
            if s["stored"] is not None and s["last_checked"] is not None:
                time_passed = max(0.0, self.current_time - s["last_checked"])
                s["stored"] = s["stored"] + time_passed * s["rate"]
                s["last_checked"] = self.current_time
            self.backend.set_max_capacity(key[0], key[1], value)
            s["max"] = value
            s["rate"] = value / key[1]
            new_readable = self._shadow_readable(key)
            gain = new_readable - old_readable
            if gain > 0:
                s["baseline"] += gain

    @rule(
        value=st.floats(
            min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False
        )
    )
    def set_max_capacity_requests(self, value):
        for key in [
            (REQUESTS_METRIC, T2_REQUESTS_SHORT_WINDOW),
            (REQUESTS_METRIC, T2_REQUESTS_LONG_WINDOW),
        ]:
            old_readable = self._shadow_readable(key)
            s = self.shadows[key]
            if s["stored"] is not None and s["last_checked"] is not None:
                time_passed = max(0.0, self.current_time - s["last_checked"])
                s["stored"] = s["stored"] + time_passed * s["rate"]
                s["last_checked"] = self.current_time
            self.backend.set_max_capacity(key[0], key[1], value)
            s["max"] = value
            s["rate"] = value / key[1]
            new_readable = self._shadow_readable(key)
            gain = new_readable - old_readable
            if gain > 0:
                s["baseline"] += gain

    @invariant()
    def capacity_matches_shadow(self):
        if self.shadows is None:
            return
        for key, bucket in self.buckets_by_key.items():
            actual = bucket.get_capacity(self.current_time).amount
            expected = self._shadow_readable(key)
            assert actual == pytest.approx(expected, abs=1e-9), (
                f"Bucket {key} mismatch: actual={actual}, shadow={expected}"
            )

    @invariant()
    def capacity_within_max(self):
        if self.shadows is None:
            return
        for key, bucket in self.buckets_by_key.items():
            actual = bucket.get_capacity(self.current_time).amount
            assert actual <= self.shadows[key]["max"] + 1e-9, (
                f"Bucket {key}: capacity {actual} exceeded max {self.shadows[key]['max']}"
            )

    @invariant()
    def conservation(self):
        if self.shadows is None:
            return
        for key, bucket in self.buckets_by_key.items():
            s = self.shadows[key]
            actual = bucket.get_capacity(self.current_time).amount
            balance = actual + s["consumed"] - s["refunded"]
            budget = s["baseline"] + s["cumulative_max_refill"]
            assert balance <= budget + 1e-6, (
                f"Bucket {key} conservation violation: capacity={actual}, "
                f"consumed={s['consumed']}, refunded={s['refunded']}, "
                f"balance={balance}, budget={budget}"
            )

    def teardown(self):
        self._time_patcher.stop()


StatefulMultiMetricMultiWindow = MultiMetricMultiWindowMachine.TestCase
StatefulMultiMetricMultiWindow.settings = hypothesis_settings(
    max_examples=200, stateful_step_count=50, deadline=None
)


# ===========================================================================
# Test 3: SetMaxCapacityRefundInteractionMachine
# ===========================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class SetMaxCapacityRefundInteractionMachine(RuleBasedStateMachine):
    """
    Single-bucket targeting the acquire -> lower max_capacity -> refund path.

    When stored > new_max after lowering max_capacity, the bucket has "trapped"
    capacity. set_max_capacity does NOT clamp stored, but get_capacity returns
    min(max, stored + refill). This machine verifies refund capping behavior
    when max has been lowered.
    """

    def __init__(self):
        super().__init__()
        self.backend: SyncMemoryBackend | None = None
        self.bucket: MemoryBucket | None = None
        self.current_time: float = INITIAL_TIME

        self.shadow_stored: float | None = None
        self.shadow_last_checked: float | None = None
        self.shadow_max: float = T3_LIMIT
        self.shadow_rate: float = T3_LIMIT / T3_WINDOW

        self.outstanding_reservations: list[float] = []

        self.total_consumed: float = 0.0
        self.total_refunded: float = 0.0
        self.accounting_baseline: float = T3_LIMIT
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
                [Quota(metric=T3_METRIC, limit=T3_LIMIT, per_seconds=T3_WINDOW)]
            ),
        )
        bucket = MemoryBucket(
            metric=T3_METRIC,
            per_seconds=T3_WINDOW,
            limit=T3_LIMIT,
            model_family="test",
        )
        self.backend = SyncMemoryBackend(buckets=[bucket], limit_config=config)
        self.bucket = bucket
        self.current_time = INITIAL_TIME

        self.shadow_stored = None
        self.shadow_last_checked = None
        self.shadow_max = T3_LIMIT
        self.shadow_rate = T3_LIMIT / T3_WINDOW

        self.outstanding_reservations = []

        self.total_consumed = 0.0
        self.total_refunded = 0.0
        self.accounting_baseline = T3_LIMIT
        self.cumulative_max_refill = 0.0

    @rule(delta=time_deltas)
    def advance_time(self, delta):
        self.cumulative_max_refill += delta * self.shadow_rate
        self.current_time += delta

    @rule(amount=t3_amounts)
    def consume(self, amount):
        readable = self._shadow_readable()
        self.backend.consume_capacity(frozen_usage({T3_METRIC: amount}))
        self.shadow_stored = readable - amount
        self.shadow_last_checked = self.current_time
        self.total_consumed += amount

    @rule(amount=t3_acquire_amounts)
    def try_acquire(self, amount):
        readable = self._shadow_readable()
        if amount > self.shadow_max:
            with pytest.raises(ValueError, match="exceeds bucket max capacity"):
                self.backend.wait_for_capacity(
                    frozen_usage({T3_METRIC: amount}), timeout=0.0
                )
            return

        if amount <= readable:
            self.backend.wait_for_capacity(
                frozen_usage({T3_METRIC: amount}), timeout=0.0
            )
            self.shadow_stored = max(0.0, readable - amount)
            self.shadow_last_checked = self.current_time
            self.total_consumed += amount
            self.outstanding_reservations.append(amount)
        else:
            with pytest.raises(TimeoutError):
                self.backend.wait_for_capacity(
                    frozen_usage({T3_METRIC: amount}), timeout=0.0
                )

    @rule(
        value=st.floats(
            min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False
        )
    )
    def lower_max_capacity(self, value):
        # Bias toward lowering: use min of value and current max
        new_max = min(value, self.shadow_max)
        if new_max <= 0:
            return
        old_readable = self._shadow_readable()
        if self.shadow_stored is not None and self.shadow_last_checked is not None:
            time_passed = max(0.0, self.current_time - self.shadow_last_checked)
            self.shadow_stored = self.shadow_stored + time_passed * self.shadow_rate
            self.shadow_last_checked = self.current_time
        self.backend.set_max_capacity(T3_METRIC, T3_WINDOW, new_max)
        self.shadow_max = new_max
        self.shadow_rate = new_max / T3_WINDOW
        new_readable = self._shadow_readable()
        gain = new_readable - old_readable
        if gain > 0:
            self.accounting_baseline += gain

    @rule(
        value=st.floats(
            min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False
        )
    )
    def raise_max_capacity(self, value):
        # Bias toward raising: use max of value and current max
        new_max = max(value, self.shadow_max)
        old_readable = self._shadow_readable()
        if self.shadow_stored is not None and self.shadow_last_checked is not None:
            time_passed = max(0.0, self.current_time - self.shadow_last_checked)
            self.shadow_stored = self.shadow_stored + time_passed * self.shadow_rate
            self.shadow_last_checked = self.current_time
        self.backend.set_max_capacity(T3_METRIC, T3_WINDOW, new_max)
        self.shadow_max = new_max
        self.shadow_rate = new_max / T3_WINDOW
        new_readable = self._shadow_readable()
        gain = new_readable - old_readable
        if gain > 0:
            self.accounting_baseline += gain

    @precondition(lambda self: len(self.outstanding_reservations) > 0)
    @rule(
        actual_fraction=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    def refund_outstanding(self, actual_fraction):
        reserved = self.outstanding_reservations.pop()
        actual = reserved * actual_fraction
        readable = self._shadow_readable()
        self.backend.refund_capacity(
            frozen_usage({T3_METRIC: reserved}),
            frozen_usage({T3_METRIC: actual}),
        )
        refund_amount = max(reserved - actual, -self.shadow_max)
        self.shadow_stored = min(readable + refund_amount, self.shadow_max)
        self.shadow_last_checked = self.current_time
        self.total_refunded += refund_amount

    @invariant()
    def capacity_matches_shadow(self):
        if self.bucket is None:
            return
        actual = self.bucket.get_capacity(self.current_time).amount
        expected = self._shadow_readable()
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"Capacity mismatch: actual={actual}, shadow={expected}, "
            f"raw_stored={self.shadow_stored}, max={self.shadow_max}"
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
    def raw_stored_not_clamped_by_set_max(self):
        """Verify set_max_capacity does NOT clamp raw stored value.

        The bucket's internal .capacity may exceed .max_capacity after
        lowering max. Only get_capacity() applies the min() clamp.
        """
        if self.bucket is None or self.bucket.capacity is None:
            return
        if self.shadow_stored is not None and self.shadow_stored > self.shadow_max:
            # Raw stored should NOT have been clamped
            assert self.bucket.capacity == pytest.approx(
                self.shadow_stored, abs=1e-9
            ), (
                f"Raw stored was unexpectedly clamped: "
                f"bucket.capacity={self.bucket.capacity}, "
                f"shadow_stored={self.shadow_stored}, max={self.shadow_max}"
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


StatefulSetMaxRefundInteraction = SetMaxCapacityRefundInteractionMachine.TestCase
StatefulSetMaxRefundInteraction.settings = hypothesis_settings(
    max_examples=200, stateful_step_count=50, deadline=None
)


# ===========================================================================
# Test 4: Floating-Point Boundary Tests
# ===========================================================================

# Strategies for limits — use nice round numbers to reduce float noise
boundary_limits = st.sampled_from([100.0, 500.0, 1000.0, 2000.0])
boundary_windows = st.sampled_from([60, 300, 3600])


def _make_fresh_bucket_and_backend(limit: float, window: int = 60):
    config = PerModelConfig(
        model_family="test",
        quotas=UsageQuotas([Quota(metric="tokens", limit=limit, per_seconds=window)]),
    )
    bucket = MemoryBucket(
        metric="tokens", per_seconds=window, limit=limit, model_family="test"
    )
    return bucket, SyncMemoryBackend(buckets=[bucket], limit_config=config)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@hypothesis_settings(max_examples=100, deadline=None)
@given(limit=boundary_limits)
def test_acquire_exact_max_on_fresh_bucket(limit):
    """Fresh bucket (capacity == limit). Acquire exactly `limit` should succeed, capacity == 0."""
    with patch(
        "token_throttle._limiter_backends._memory._sync_backend.time"
    ) as mock_time:
        mock_time.time.return_value = FROZEN_TIME
        mock_time.monotonic.return_value = FROZEN_TIME
        bucket, backend = _make_fresh_bucket_and_backend(limit)

        backend.wait_for_capacity(frozen_usage({"tokens": limit}), timeout=0.0)
        actual = bucket.get_capacity(FROZEN_TIME).amount
        assert actual == 0.0, f"Expected 0.0 after exact-max acquire, got {actual}"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@hypothesis_settings(max_examples=100, deadline=None)
@given(
    limit=boundary_limits,
    amount=st.floats(
        min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False
    ),
)
def test_consume_then_full_refund_roundtrip(limit, amount):
    """Consume X, refund (reserved=X, actual=0). Capacity should return to original."""
    with (
        patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        ) as mock_time,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        mock_time.time.return_value = FROZEN_TIME
        mock_time.monotonic.return_value = FROZEN_TIME
        bucket, backend = _make_fresh_bucket_and_backend(limit)

        original = bucket.get_capacity(FROZEN_TIME).amount
        backend.consume_capacity(frozen_usage({"tokens": amount}))
        backend.refund_capacity(
            frozen_usage({"tokens": amount}),
            frozen_usage({"tokens": 0.0}),
        )
        after_refund = bucket.get_capacity(FROZEN_TIME).amount
        expected = min(original, limit)  # capped at max
        assert after_refund == pytest.approx(expected, abs=1e-9), (
            f"Roundtrip failed: original={original}, amount={amount}, "
            f"after_refund={after_refund}, expected={expected}"
        )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@hypothesis_settings(max_examples=100, deadline=None)
@given(
    limit=boundary_limits,
    amount_fraction=st.floats(
        min_value=0.99, max_value=1.01, allow_nan=False, allow_infinity=False
    ),
)
def test_acquire_boundary_no_partial_state(limit, amount_fraction):
    """Acquire at boundary: either fully succeeds (capacity drops) or fully fails (capacity unchanged)."""
    with patch(
        "token_throttle._limiter_backends._memory._sync_backend.time"
    ) as mock_time:
        mock_time.time.return_value = FROZEN_TIME
        mock_time.monotonic.return_value = FROZEN_TIME
        bucket, backend = _make_fresh_bucket_and_backend(limit)

        cap_before = bucket.get_capacity(FROZEN_TIME).amount
        amount = limit * amount_fraction

        if amount > limit:
            # Exceeds max capacity -> ValueError, state unchanged
            with pytest.raises(ValueError, match="exceeds bucket max capacity"):
                backend.wait_for_capacity(frozen_usage({"tokens": amount}), timeout=0.0)
            cap_after = bucket.get_capacity(FROZEN_TIME).amount
            assert cap_after == pytest.approx(cap_before, abs=1e-9), (
                f"State changed after rejected acquire: before={cap_before}, after={cap_after}"
            )
        elif amount <= cap_before:
            # Should succeed
            backend.wait_for_capacity(frozen_usage({"tokens": amount}), timeout=0.0)
            cap_after = bucket.get_capacity(FROZEN_TIME).amount
            expected = max(0.0, cap_before - amount)
            assert cap_after == pytest.approx(expected, abs=1e-9), (
                f"Capacity after acquire: expected={expected}, actual={cap_after}"
            )
        else:
            # Insufficient capacity -> TimeoutError, state unchanged
            with pytest.raises(TimeoutError):
                backend.wait_for_capacity(frozen_usage({"tokens": amount}), timeout=0.0)
            cap_after = bucket.get_capacity(FROZEN_TIME).amount
            assert cap_after == pytest.approx(cap_before, abs=1e-9), (
                f"State changed after timeout: before={cap_before}, after={cap_after}"
            )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@hypothesis_settings(max_examples=100, deadline=None)
@given(
    limit=boundary_limits,
    consume_amount=st.floats(
        min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False
    ),
    refund_reserved=st.floats(
        min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False
    ),
    actual_fraction=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
def test_refund_never_exceeds_max(
    limit, consume_amount, refund_reserved, actual_fraction
):
    """After arbitrary consume + refund, capacity must never exceed max_capacity."""
    with (
        patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        ) as mock_time,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        mock_time.time.return_value = FROZEN_TIME
        mock_time.monotonic.return_value = FROZEN_TIME
        bucket, backend = _make_fresh_bucket_and_backend(limit)

        backend.consume_capacity(frozen_usage({"tokens": consume_amount}))
        backend.refund_capacity(
            frozen_usage({"tokens": refund_reserved}),
            frozen_usage({"tokens": refund_reserved * actual_fraction}),
        )
        actual = bucket.get_capacity(FROZEN_TIME).amount
        assert actual <= limit + 1e-12, (
            f"Capacity {actual} exceeded max {limit} after refund"
        )


# ===========================================================================
# Test 5: CancellationRefundMathTest
# ===========================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@hypothesis_settings(max_examples=100, deadline=None)
@given(
    limit=boundary_limits,
    pre_consume=st.floats(
        min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False
    ),
    cancel_amount=st.floats(
        min_value=0.1, max_value=200.0, allow_nan=False, allow_infinity=False
    ),
    time_delta=st.floats(
        min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False
    ),
)
def test_cancellation_refund_math(limit, pre_consume, cancel_amount, time_delta):
    """Property test for _refund_cancelled_consumption path.

    1. Optional pre-consume to reach arbitrary state
    2. Consume cancel_amount (simulates pre-cancel consumption)
    3. Advance clock by time_delta (refill happens)
    4. Call _refund_cancelled_consumption(cancel_amount)
    5. Verify: final == min(readable_after_refill + cancel_amount, max_capacity)
       and final <= max_capacity
    """
    clock = [INITIAL_TIME]

    with (
        patch("token_throttle._limiter_backends._memory._backend.time") as mock_time,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        mock_time.time.side_effect = lambda: clock[0]
        mock_time.monotonic.side_effect = lambda: clock[0]

        config = PerModelConfig(
            model_family="test",
            quotas=UsageQuotas([Quota(metric="tokens", limit=limit, per_seconds=60)]),
        )
        bucket = MemoryBucket(
            metric="tokens", per_seconds=60, limit=limit, model_family="test"
        )
        backend = MemoryBackend(buckets=[bucket], limit_config=config)

        loop = asyncio.new_event_loop()
        try:
            # Step 1: optional pre-consume
            if pre_consume > 0:
                loop.run_until_complete(
                    backend.consume_capacity(frozen_usage({"tokens": pre_consume}))
                )

            # Step 2: consume the amount that will be "cancelled"
            loop.run_until_complete(
                backend.consume_capacity(frozen_usage({"tokens": cancel_amount}))
            )

            # Step 3: advance clock
            clock[0] += time_delta

            # Step 4: record readable BEFORE refund
            readable_after_refill = bucket.get_capacity(clock[0]).amount

            # Step 5: call _refund_cancelled_consumption
            loop.run_until_complete(
                backend._refund_cancelled_consumption(
                    frozen_usage({"tokens": cancel_amount})
                )
            )

            # Step 6: verify
            final = bucket.get_capacity(clock[0]).amount
            expected = min(readable_after_refill + cancel_amount, limit)
            assert final == pytest.approx(expected, abs=1e-9), (
                f"Cancellation refund math: final={final}, expected={expected}, "
                f"readable_after_refill={readable_after_refill}, "
                f"cancel_amount={cancel_amount}"
            )
            assert final <= limit + 1e-12, (
                f"Capacity {final} exceeded max {limit} after cancellation refund"
            )
        finally:
            loop.close()
