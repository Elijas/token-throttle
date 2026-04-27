"""
Tests for RedisBucket.calculate_capacity() — pure token bucket math, no Redis needed.

These tests verify the core token bucket algorithm: refill rates, capacity caps,
fresh starts, edge cases, and error handling.
"""

from unittest.mock import AsyncMock

import pytest

pytest.importorskip("redis", reason="redis package not installed")

import token_throttle._capacity as _cap
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._bucket import (
    CalculatedCapacity,
    RedisBucket,
)


def make_bucket(
    limit: float = 100,
    per_seconds: int = 60,
    metric: str = "requests",
    model_family: str = "test-model",
) -> RedisBucket:
    """Create a RedisBucket with a mock Redis client for pure-math testing."""
    quota = Quota(metric=metric, limit=limit, per_seconds=per_seconds)
    config = PerModelConfig(
        model_family=model_family,
        quotas=UsageQuotas([quota]),
    )
    mock_redis = AsyncMock()
    return RedisBucket(quota=quota, limit_config=config, redis_client=mock_redis)


class TestFreshStart:
    """When last_checked or outdated_capacity is None, bucket treats it as a fresh start."""

    def test_last_checked_none_returns_max_capacity(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        result = bucket.calculate_capacity(
            last_checked=None, outdated_capacity=50.0, current_time=1000.0
        )
        assert result.amount == 100.0
        assert result.is_fresh_start is True

    def test_outdated_capacity_none_returns_max_capacity(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        result = bucket.calculate_capacity(
            last_checked=1000.0, outdated_capacity=None, current_time=1000.0
        )
        assert result.amount == 100.0
        assert result.is_fresh_start is True

    def test_both_none_returns_max_capacity(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        result = bucket.calculate_capacity(
            last_checked=None, outdated_capacity=None, current_time=1000.0
        )
        assert result.amount == 100.0
        assert result.is_fresh_start is True

    def test_fresh_start_returns_calculated_capacity_type(self):
        bucket = make_bucket(limit=50, per_seconds=10)
        result = bucket.calculate_capacity(
            last_checked=None, outdated_capacity=None, current_time=1000.0
        )
        assert isinstance(result, CalculatedCapacity)
        assert result.amount == 50.0
        assert result.is_fresh_start is True


class TestRefill:
    """Tests for the token refill calculation: min(max_capacity, outdated + time * rate)."""

    def test_no_time_passed_capacity_unchanged(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        result = bucket.calculate_capacity(
            last_checked=1000.0, outdated_capacity=40.0, current_time=1000.0
        )
        assert result.amount == pytest.approx(40.0)
        assert result.is_fresh_start is False

    def test_partial_refill_exact_math(self):
        """rate_per_sec = 100/60, time_passed = 30s → refill = 30 * (100/60) = 50."""
        bucket = make_bucket(limit=100, per_seconds=60)
        result = bucket.calculate_capacity(
            last_checked=1000.0, outdated_capacity=20.0, current_time=1030.0
        )
        expected = 20.0 + 30.0 * (100.0 / 60.0)  # 20 + 50 = 70
        assert result.amount == pytest.approx(expected)
        assert result.is_fresh_start is False

    def test_refill_capped_at_max_capacity(self):
        """Even with a long time_passed, capacity never exceeds max_capacity."""
        bucket = make_bucket(limit=100, per_seconds=60)
        result = bucket.calculate_capacity(
            last_checked=1000.0, outdated_capacity=50.0, current_time=2000.0
        )
        # 50 + 1000 * (100/60) = 50 + 1666.67 → capped at 100
        assert result.amount == pytest.approx(100.0)

    def test_zero_outdated_capacity_refills_from_zero(self):
        """Starting from zero, refill = time_passed * rate_per_sec."""
        bucket = make_bucket(limit=60, per_seconds=60)
        # rate = 60/60 = 1.0/s, 10 seconds → refill = 10.0
        result = bucket.calculate_capacity(
            last_checked=1000.0, outdated_capacity=0.0, current_time=1010.0
        )
        assert result.amount == pytest.approx(10.0)

    def test_full_capacity_stays_at_max(self):
        """Already at max_capacity + time → still at max_capacity."""
        bucket = make_bucket(limit=100, per_seconds=60)
        result = bucket.calculate_capacity(
            last_checked=1000.0, outdated_capacity=100.0, current_time=1030.0
        )
        assert result.amount == pytest.approx(100.0)

    def test_rate_calculation_limit_divided_by_per_seconds(self):
        """Verify rate_per_sec = limit / per_seconds drives the refill."""
        bucket = make_bucket(limit=300, per_seconds=60)
        # rate = 300/60 = 5.0/s
        assert bucket._rate_per_sec == pytest.approx(5.0)
        result = bucket.calculate_capacity(
            last_checked=1000.0, outdated_capacity=0.0, current_time=1010.0
        )
        # 0 + 10 * 5.0 = 50
        assert result.amount == pytest.approx(50.0)

    def test_string_inputs_are_converted_to_float(self):
        """Redis returns byte strings; calculate_capacity should convert them."""
        bucket = make_bucket(limit=100, per_seconds=60)
        result = bucket.calculate_capacity(
            last_checked="1000.0", outdated_capacity="40.0", current_time=1030.0
        )
        expected = 40.0 + 30.0 * (100.0 / 60.0)
        assert result.amount == pytest.approx(expected)
        assert result.is_fresh_start is False

    def test_bytes_inputs_are_converted_to_float(self):
        """Redis actually returns bytes; calculate_capacity should handle them."""
        bucket = make_bucket(limit=100, per_seconds=60)
        result = bucket.calculate_capacity(
            last_checked=b"1000.0", outdated_capacity=b"40.0", current_time=1030.0
        )
        expected = 40.0 + 30.0 * (100.0 / 60.0)
        assert result.amount == pytest.approx(expected)
        assert result.is_fresh_start is False


class TestDynamicMaxCapacity:
    """When _max_capacity_cached is set, it overrides the default from quota.limit."""

    def test_cached_max_capacity_affects_fresh_start(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket._max_capacity_cached = 50.0
        result = bucket.calculate_capacity(
            last_checked=None, outdated_capacity=None, current_time=1000.0
        )
        assert result.amount == 50.0
        assert result.is_fresh_start is True

    def test_cached_max_capacity_affects_cap(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket._max_capacity_cached = 30.0
        # rate = 100/60 ≈ 1.667/s, time = 100s → refill = 166.7, but capped at 30
        result = bucket.calculate_capacity(
            last_checked=1000.0, outdated_capacity=10.0, current_time=1100.0
        )
        assert result.amount == pytest.approx(30.0)

    def test_cached_max_capacity_higher_than_default(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket._max_capacity_cached = 200.0
        # rate = 100/60, time = 120s → refill = 120 * (100/60) = 200
        # 0 + 200 = 200, capped at 200 (new max)
        result = bucket.calculate_capacity(
            last_checked=1000.0, outdated_capacity=0.0, current_time=1120.0
        )
        assert result.amount == pytest.approx(200.0)


class TestEdgeCases:
    """Edge cases: precision, extreme rates, clock skew, invalid inputs."""

    def test_sub_millisecond_time_precision(self):
        """Tiny time_passed should produce a small but correct refill."""
        bucket = make_bucket(limit=100, per_seconds=60)
        # rate = 100/60, time = 0.0001s → refill = 0.0001 * (100/60) ≈ 0.001667
        time_passed = 0.0001
        result = bucket.calculate_capacity(
            last_checked=1000.0,
            outdated_capacity=50.0,
            current_time=1000.0 + time_passed,
        )
        expected = 50.0 + time_passed * (100.0 / 60.0)
        assert result.amount == pytest.approx(expected)

    def test_very_large_per_seconds_low_rate(self):
        """per_seconds=86400 (1 day) → rate = 100/86400 ≈ 0.001157/s."""
        bucket = make_bucket(limit=100, per_seconds=86400)
        assert bucket._rate_per_sec == pytest.approx(100.0 / 86400.0)
        result = bucket.calculate_capacity(
            last_checked=1000.0, outdated_capacity=0.0, current_time=1060.0
        )
        # 0 + 60 * (100/86400) ≈ 0.06944
        expected = 60.0 * (100.0 / 86400.0)
        assert result.amount == pytest.approx(expected)

    def test_very_large_limit(self):
        """limit=1_000_000_000 with normal per_seconds."""
        bucket = make_bucket(limit=1_000_000_000, per_seconds=60)
        assert bucket._rate_per_sec == pytest.approx(1_000_000_000.0 / 60.0)
        result = bucket.calculate_capacity(
            last_checked=1000.0, outdated_capacity=0.0, current_time=1001.0
        )
        # 0 + 1 * (1e9/60) ≈ 16,666,666.67
        expected = 1.0 * (1_000_000_000.0 / 60.0)
        assert result.amount == pytest.approx(expected)

    def test_negative_time_passed_clamps_to_zero_with_warning(self):
        """Clock skew (negative time_passed) clamps to 0 and issues RuntimeWarning."""
        _cap._backward_clock_warned = False
        bucket = make_bucket(limit=100, per_seconds=60)
        with pytest.warns(RuntimeWarning, match="Negative time_passed"):
            result = bucket.calculate_capacity(
                last_checked=1010.0, outdated_capacity=40.0, current_time=1000.0
            )
        # time_passed clamped to 0 → capacity unchanged at 40
        assert result.amount == pytest.approx(40.0)
        assert result.is_fresh_start is False

    def test_negative_time_passed_warning_includes_key(self):
        """RuntimeWarning for negative time mentions the bucket's Redis key."""
        _cap._backward_clock_warned = False
        bucket = make_bucket(
            limit=100, per_seconds=60, model_family="my-model", metric="tokens"
        )
        with pytest.warns(RuntimeWarning, match="rate_limiting:my-model:tokens:60"):
            bucket.calculate_capacity(
                last_checked=1010.0, outdated_capacity=40.0, current_time=1000.0
            )

    def test_invalid_last_checked_string_raises_value_error(self):
        """Non-numeric last_checked raises ValueError."""
        bucket = make_bucket(limit=100, per_seconds=60)
        with pytest.raises(ValueError, match="Invalid last_checked or capacity"):
            bucket.calculate_capacity(
                last_checked="not-a-number",
                outdated_capacity=40.0,
                current_time=1000.0,
            )

    def test_invalid_outdated_capacity_string_raises_value_error(self):
        """Non-numeric outdated_capacity raises ValueError."""
        bucket = make_bucket(limit=100, per_seconds=60)
        with pytest.raises(ValueError, match="Invalid last_checked or capacity"):
            bucket.calculate_capacity(
                last_checked=1000.0,
                outdated_capacity="not-a-number",
                current_time=1000.0,
            )
