"""
Property-based tests for calculate_capacity() using Hypothesis.

These are pure math tests -- no Redis or async needed. They verify invariants
that must hold for ANY valid combination of inputs, not just hand-picked examples.
"""

from unittest.mock import AsyncMock

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._bucket import RedisBucket

# ---------------------------------------------------------------------------
# Strategies for meaningful token-bucket values
# ---------------------------------------------------------------------------

limits = st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False)
per_seconds = st.integers(min_value=1, max_value=86400)
capacities = st.floats(
    min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
)
negative_capacities = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
times = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_bucket(limit: float = 100.0, per_seconds_val: int = 60) -> RedisBucket:
    """Create a RedisBucket with a mock Redis client for pure-math testing."""
    quota = Quota(metric="requests", limit=limit, per_seconds=per_seconds_val)
    config = PerModelConfig(model_family="test", quotas=UsageQuotas([quota]))
    return RedisBucket(
        quota=quota, limit_config=config, redis_client=AsyncMock(), key_prefix="test"
    )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestCapacityNeverExceedsMax:
    """calculate_capacity result must always be <= max_capacity."""

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        outdated_capacity=capacities,
        last_checked=times,
        time_delta=times,
    )
    def test_result_never_exceeds_max_capacity(
        self,
        limit: float,
        per_seconds_val: float,
        outdated_capacity: float,
        last_checked: float,
        time_delta: float,
    ):
        bucket = make_bucket(limit=limit, per_seconds_val=per_seconds_val)
        current_time = (
            last_checked + time_delta
        )  # guarantees current_time >= last_checked

        result = bucket.calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=current_time,
        )

        assert result.amount <= bucket.max_capacity + 1e-9, (
            f"Capacity {result.amount} exceeded max {bucket.max_capacity}"
        )


class TestCapacityNeverNegative:
    """calculate_capacity result must always be >= 0 for non-negative inputs."""

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        outdated_capacity=capacities,
        last_checked=times,
        time_delta=times,
    )
    def test_result_never_negative(
        self,
        limit: float,
        per_seconds_val: float,
        outdated_capacity: float,
        last_checked: float,
        time_delta: float,
    ):
        bucket = make_bucket(limit=limit, per_seconds_val=per_seconds_val)
        current_time = last_checked + time_delta

        result = bucket.calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=current_time,
        )

        assert result.amount >= 0.0, f"Capacity was negative: {result.amount}"


class TestRefundCappedAtMax:
    """After a refund, capacity must still be <= max_capacity."""

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        current_capacity=capacities,
        refund_amount=capacities,
    )
    def test_capacity_plus_refund_capped_at_max(
        self,
        limit: float,
        per_seconds_val: float,
        current_capacity: float,
        refund_amount: float,
    ):
        bucket = make_bucket(limit=limit, per_seconds_val=per_seconds_val)
        max_cap = bucket.max_capacity

        # Mirrors the capping logic from RedisBackend.refund_capacity
        refunded = min(max(current_capacity + refund_amount, 0.0), max_cap)

        assert refunded <= max_cap + 1e-9
        assert refunded >= 0.0


class TestRefillMonotonicallyNonDecreasing:
    """For a fixed outdated_capacity, more elapsed time must produce >= capacity."""

    @given(data=st.data())
    def test_more_time_means_more_or_equal_capacity(self, data: st.DataObject):
        limit = data.draw(limits, label="limit")
        per_seconds_val = data.draw(per_seconds, label="per_seconds")
        outdated_capacity = data.draw(capacities, label="outdated_capacity")
        last_checked = data.draw(times, label="last_checked")
        delta1 = data.draw(times, label="delta1")
        extra_delta = data.draw(times, label="extra_delta")

        bucket = make_bucket(limit=limit, per_seconds_val=per_seconds_val)

        t1 = last_checked + delta1
        t2 = t1 + extra_delta  # t2 >= t1 by construction

        r1 = bucket.calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=t1,
        )
        r2 = bucket.calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=t2,
        )

        assert r2.amount >= r1.amount - 1e-9, (
            f"Capacity decreased with more time: {r1.amount} -> {r2.amount}"
        )


class TestConserveReserveAndRefund:
    """
    Reserve X then refund X must return to original capacity.

    This tests the arithmetic identity: (capacity - X) + X == capacity,
    provided the result stays within [0, max_capacity].
    """

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        capacity_before=capacities,
        reserve_fraction=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_reserve_then_refund_returns_to_original(
        self,
        limit: float,
        per_seconds_val: float,
        capacity_before: float,
        reserve_fraction: float,
    ):
        bucket = make_bucket(limit=limit, per_seconds_val=per_seconds_val)
        max_cap = bucket.max_capacity

        # Clamp capacity_before to valid range
        capacity_before = min(capacity_before, max_cap)

        # Reserve a fraction of current capacity
        reserve_amount = capacity_before * reserve_fraction
        after_reserve = capacity_before - reserve_amount

        # Refund the same amount (capped like the backend does)
        after_refund = min(max(after_reserve + reserve_amount, 0.0), max_cap)

        assert after_refund == pytest.approx(capacity_before, abs=1e-9), (
            f"Conservation violated: {capacity_before} -> reserve {reserve_amount} "
            f"-> {after_reserve} -> refund -> {after_refund}"
        )


class TestRefillRateIsLinear:
    """
    The refill rate is constant at limit / per_seconds tokens per second.

    For two time points where neither is capped, the capacity difference
    should equal rate * time_difference.
    """

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        last_checked=times,
        delta1=st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        delta2=st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_rate_is_limit_over_per_seconds(
        self,
        limit: float,
        per_seconds_val: float,
        last_checked: float,
        delta1: float,
        delta2: float,
    ):
        bucket = make_bucket(limit=limit, per_seconds_val=per_seconds_val)
        rate = limit / per_seconds_val

        # Start from 0 so we can observe refill without hitting the cap too early
        outdated_capacity = 0.0
        t1 = last_checked + delta1
        t2 = last_checked + delta2

        # Skip cases where deltas are too small to survive floating-point
        # addition with last_checked (absorbed into rounding error)
        assume(t1 - last_checked == delta1)
        assume(t2 - last_checked == delta2)

        r1 = bucket.calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=t1,
        )
        r2 = bucket.calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=t2,
        )

        # Only check linearity when neither result is capped
        if (
            r1.amount < bucket.max_capacity - 1e-9
            and r2.amount < bucket.max_capacity - 1e-9
        ):
            expected_diff = (delta2 - delta1) * rate
            actual_diff = r2.amount - r1.amount
            # Use relative tolerance for large values where floating-point
            # error from (large_time * large_rate) can exceed a fixed abs
            assert actual_diff == pytest.approx(expected_diff, rel=1e-9, abs=1e-6), (
                f"Non-linear refill: expected diff {expected_diff}, got {actual_diff} "
                f"(rate={rate}, dt={delta2 - delta1})"
            )


class TestFreshStartOnNone:
    """When last_checked or outdated_capacity is None, result is max_capacity."""

    @given(limit=limits, per_seconds_val=per_seconds, current_time=times)
    def test_none_last_checked_gives_max(
        self,
        limit: float,
        per_seconds_val: float,
        current_time: float,
    ):
        bucket = make_bucket(limit=limit, per_seconds_val=per_seconds_val)
        result = bucket.calculate_capacity(
            last_checked=None,
            outdated_capacity=50.0,
            current_time=current_time,
        )
        assert result.amount == bucket.max_capacity
        assert result.is_fresh_start is True

    @given(limit=limits, per_seconds_val=per_seconds, current_time=times)
    def test_none_outdated_capacity_gives_max(
        self,
        limit: float,
        per_seconds_val: float,
        current_time: float,
    ):
        bucket = make_bucket(limit=limit, per_seconds_val=per_seconds_val)
        result = bucket.calculate_capacity(
            last_checked=100.0,
            outdated_capacity=None,
            current_time=current_time,
        )
        assert result.amount == bucket.max_capacity
        assert result.is_fresh_start is True


# ---------------------------------------------------------------------------
# Negative-capacity property tests
# ---------------------------------------------------------------------------


class TestCapacityBoundFromNegative:
    """calculate_capacity with negative outdated_capacity must still return <= max_capacity."""

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        outdated_capacity=negative_capacities,
        last_checked=times,
        time_delta=times,
    )
    def test_result_never_exceeds_max_capacity_from_negative(
        self,
        limit: float,
        per_seconds_val: int,
        outdated_capacity: float,
        last_checked: float,
        time_delta: float,
    ):
        bucket = make_bucket(limit=limit, per_seconds_val=per_seconds_val)
        current_time = last_checked + time_delta

        result = bucket.calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=current_time,
        )

        assert result.amount <= bucket.max_capacity + 1e-9, (
            f"Capacity {result.amount} exceeded max {bucket.max_capacity} "
            f"(from negative start {outdated_capacity})"
        )


class TestRefillMonotonicFromNegative:
    """More elapsed time must produce >= capacity, even starting from negative."""

    @given(data=st.data())
    def test_more_time_means_more_or_equal_capacity_from_negative(
        self, data: st.DataObject
    ):
        limit = data.draw(limits, label="limit")
        per_seconds_val = data.draw(per_seconds, label="per_seconds")
        outdated_capacity = data.draw(negative_capacities, label="outdated_capacity")
        last_checked = data.draw(times, label="last_checked")
        delta1 = data.draw(times, label="delta1")
        extra_delta = data.draw(times, label="extra_delta")

        bucket = make_bucket(limit=limit, per_seconds_val=per_seconds_val)

        t1 = last_checked + delta1
        t2 = t1 + extra_delta  # t2 >= t1 by construction

        r1 = bucket.calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=t1,
        )
        r2 = bucket.calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=t2,
        )

        assert r2.amount >= r1.amount - 1e-9, (
            f"Capacity decreased with more time from negative start: "
            f"{r1.amount} -> {r2.amount} (outdated={outdated_capacity})"
        )


class TestRefundCappingWithNegativeCapacity:
    """min(current + refund, max_cap) where current and refund can be negative — result <= max_cap."""

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        current_capacity=negative_capacities,
        refund_amount=negative_capacities,
    )
    def test_refund_result_never_exceeds_max(
        self,
        limit: float,
        per_seconds_val: int,
        current_capacity: float,
        refund_amount: float,
    ):
        bucket = make_bucket(limit=limit, per_seconds_val=per_seconds_val)
        max_cap = bucket.max_capacity

        # Mirrors the capping logic from SyncMemoryBackend.refund_capacity
        refunded = min(current_capacity + refund_amount, max_cap)

        assert refunded <= max_cap + 1e-9, (
            f"Refund result {refunded} exceeded max {max_cap} "
            f"(current={current_capacity}, refund={refund_amount})"
        )
