"""
Tests for MemoryBucket — in-memory token bucket state holder.

These tests verify the wiring between MemoryBucket and the shared
calculate_capacity() function, plus MemoryBucket-specific state management
(set_capacity clamping, set_max_capacity validation).
"""

import pytest

from token_throttle._capacity import CalculatedCapacity
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackend
from token_throttle._limiter_backends._memory._bucket import MemoryBucket
from token_throttle._limiter_backends._memory._sync_backend import SyncMemoryBackend


def make_bucket(
    limit: float = 100,
    per_seconds: int = 60,
    metric: str = "requests",
    model_family: str = "test-model",
) -> MemoryBucket:
    """Create a MemoryBucket for testing."""
    return MemoryBucket(
        metric=metric,
        per_seconds=per_seconds,
        limit=limit,
        model_family=model_family,
    )


class TestFreshStart:
    """When capacity and last_checked are None (initial state), bucket returns max_capacity."""

    def test_initial_state_returns_max_capacity(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        result = bucket.get_capacity(current_time=1000.0)
        assert result.amount == 100.0
        assert result.is_fresh_start is True

    def test_fresh_start_returns_calculated_capacity_type(self):
        bucket = make_bucket(limit=50, per_seconds=10)
        result = bucket.get_capacity(current_time=1000.0)
        assert isinstance(result, CalculatedCapacity)
        assert result.amount == 50.0
        assert result.is_fresh_start is True

    def test_fresh_start_independent_of_current_time(self):
        """Fresh start always returns max_capacity regardless of current_time."""
        bucket = make_bucket(limit=100, per_seconds=60)
        result = bucket.get_capacity(current_time=9999999.0)
        assert result.amount == 100.0
        assert result.is_fresh_start is True


class TestRefill:
    """Tests for the token refill calculation: min(max_capacity, outdated + time * rate)."""

    def test_no_time_passed_capacity_unchanged(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket.set_capacity(40.0, current_time=1000.0)
        result = bucket.get_capacity(current_time=1000.0)
        assert result.amount == pytest.approx(40.0)
        assert result.is_fresh_start is False

    def test_partial_refill_exact_math(self):
        """rate_per_sec = 100/60, time_passed = 30s -> refill = 30 * (100/60) = 50."""
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket.set_capacity(20.0, current_time=1000.0)
        result = bucket.get_capacity(current_time=1030.0)
        expected = 20.0 + 30.0 * (100.0 / 60.0)  # 20 + 50 = 70
        assert result.amount == pytest.approx(expected)
        assert result.is_fresh_start is False

    def test_refill_capped_at_max_capacity(self):
        """Even with a long time_passed, capacity never exceeds max_capacity."""
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket.set_capacity(50.0, current_time=1000.0)
        result = bucket.get_capacity(current_time=2000.0)
        # 50 + 1000 * (100/60) = 50 + 1666.67 -> capped at 100
        assert result.amount == pytest.approx(100.0)

    def test_zero_capacity_refills_from_zero(self):
        """Starting from zero, refill = time_passed * rate_per_sec."""
        bucket = make_bucket(limit=60, per_seconds=60)
        bucket.set_capacity(0.0, current_time=1000.0)
        # rate = 60/60 = 1.0/s, 10 seconds -> refill = 10.0
        result = bucket.get_capacity(current_time=1010.0)
        assert result.amount == pytest.approx(10.0)

    def test_full_capacity_stays_at_max(self):
        """Already at max_capacity + time -> still at max_capacity."""
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket.set_capacity(100.0, current_time=1000.0)
        result = bucket.get_capacity(current_time=1030.0)
        assert result.amount == pytest.approx(100.0)

    def test_rate_calculation_limit_divided_by_per_seconds(self):
        """Verify rate_per_sec = limit / per_seconds drives the refill."""
        bucket = make_bucket(limit=300, per_seconds=60)
        assert bucket._rate_per_sec == pytest.approx(5.0)
        bucket.set_capacity(0.0, current_time=1000.0)
        result = bucket.get_capacity(current_time=1010.0)
        # 0 + 10 * 5.0 = 50
        assert result.amount == pytest.approx(50.0)


class TestDynamicMaxCapacity:
    """When set_max_capacity is called, it affects subsequent calculations."""

    def test_set_max_capacity_affects_fresh_start(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket.set_max_capacity(50.0)
        result = bucket.get_capacity(current_time=1000.0)
        assert result.amount == 50.0
        assert result.is_fresh_start is True

    def test_set_max_capacity_affects_cap(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket.set_capacity(10.0, current_time=1000.0)
        bucket.set_max_capacity(30.0)
        # rate = 100/60 ~ 1.667/s, time = 100s -> refill = 166.7, but capped at 30
        result = bucket.get_capacity(current_time=1100.0)
        assert result.amount == pytest.approx(30.0)

    def test_set_max_capacity_higher_than_default(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket.set_capacity(0.0, current_time=1000.0)
        bucket.set_max_capacity(200.0)
        # rate = 100/60, time = 120s -> refill = 120 * (100/60) = 200
        # 0 + 200 = 200, capped at 200 (new max)
        result = bucket.get_capacity(current_time=1120.0)
        assert result.amount == pytest.approx(200.0)


class TestEdgeCases:
    """Edge cases: precision, extreme rates, clock skew."""

    def test_sub_millisecond_time_precision(self):
        """Tiny time_passed should produce a small but correct refill."""
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket.set_capacity(50.0, current_time=1000.0)
        time_passed = 0.0001
        result = bucket.get_capacity(current_time=1000.0 + time_passed)
        expected = 50.0 + time_passed * (100.0 / 60.0)
        assert result.amount == pytest.approx(expected)

    def test_very_large_per_seconds_low_rate(self):
        """per_seconds=86400 (1 day) -> rate = 100/86400 ~ 0.001157/s."""
        bucket = make_bucket(limit=100, per_seconds=86400)
        assert bucket._rate_per_sec == pytest.approx(100.0 / 86400.0)
        bucket.set_capacity(0.0, current_time=1000.0)
        result = bucket.get_capacity(current_time=1060.0)
        # 0 + 60 * (100/86400) ~ 0.06944
        expected = 60.0 * (100.0 / 86400.0)
        assert result.amount == pytest.approx(expected)

    def test_very_large_limit(self):
        """limit=1_000_000_000 with normal per_seconds."""
        bucket = make_bucket(limit=1_000_000_000, per_seconds=60)
        assert bucket._rate_per_sec == pytest.approx(1_000_000_000.0 / 60.0)
        bucket.set_capacity(0.0, current_time=1000.0)
        result = bucket.get_capacity(current_time=1001.0)
        # 0 + 1 * (1e9/60) ~ 16,666,666.67
        expected = 1.0 * (1_000_000_000.0 / 60.0)
        assert result.amount == pytest.approx(expected)

    def test_negative_time_passed_clamps_to_zero_with_warning(self):
        """Clock skew (negative time_passed) clamps to 0 and issues RuntimeWarning."""
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket.set_capacity(40.0, current_time=1010.0)
        with pytest.warns(RuntimeWarning, match="Negative time_passed"):
            result = bucket.get_capacity(current_time=1000.0)
        # time_passed clamped to 0 -> capacity unchanged at 40
        assert result.amount == pytest.approx(40.0)
        assert result.is_fresh_start is False

    def test_negative_time_passed_warning_includes_bucket_id(self):
        """RuntimeWarning for negative time mentions the bucket's ID."""
        bucket = make_bucket(
            limit=100, per_seconds=60, model_family="my-model", metric="tokens"
        )
        bucket.set_capacity(40.0, current_time=1010.0)
        with pytest.warns(RuntimeWarning, match="memory:my-model:tokens:60"):
            bucket.get_capacity(current_time=1000.0)


class TestSetCapacity:
    """Tests for set_capacity: value clamping and timestamp update."""

    def test_stores_value_and_timestamp(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket.set_capacity(42.0, current_time=1000.0)
        assert bucket.capacity == 42.0
        assert bucket.last_checked == 1000.0

    def test_clamps_negative_to_zero(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket.set_capacity(-10.0, current_time=1000.0)
        assert bucket.capacity == 0.0

    def test_allows_zero(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket.set_capacity(0.0, current_time=1000.0)
        assert bucket.capacity == 0.0


class TestSetMaxCapacity:
    """Tests for set_max_capacity: validation and mutation."""

    def test_rejects_zero(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        with pytest.raises(ValueError, match="max_capacity must be greater than 0"):
            bucket.set_max_capacity(0)

    def test_rejects_negative(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        with pytest.raises(ValueError, match="max_capacity must be greater than 0"):
            bucket.set_max_capacity(-5)

    def test_accepts_positive(self):
        bucket = make_bucket(limit=100, per_seconds=60)
        bucket.set_max_capacity(50.0)
        assert bucket.max_capacity == 50.0


class TestSleepIntervalZero:
    """sleep_interval=0 must be respected, not treated as falsy."""

    def test_memory_backend_sleep_interval_zero(self):
        quota = Quota(metric="requests", limit=100, per_seconds=60)
        config = PerModelConfig(model_family="test", quotas=UsageQuotas([quota]))
        bucket = make_bucket(limit=100, per_seconds=60)
        backend = MemoryBackend(
            buckets=[bucket],
            limit_config=config,
            sleep_interval=0,
        )
        assert backend._sleep_interval == 0

    def test_sync_memory_backend_sleep_interval_zero(self):
        quota = Quota(metric="requests", limit=100, per_seconds=60)
        config = PerModelConfig(model_family="test", quotas=UsageQuotas([quota]))
        bucket = make_bucket(limit=100, per_seconds=60)
        backend = SyncMemoryBackend(
            buckets=[bucket],
            limit_config=config,
            sleep_interval=0,
        )
        assert backend._sleep_interval == 0
