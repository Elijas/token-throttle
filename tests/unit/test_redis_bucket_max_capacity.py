"""Tests for dynamic max_capacity in RedisBucket.

These tests verify that max_capacity can be updated at runtime via Redis,
enabling adaptive rate limiting scenarios where limits change dynamically.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._bucket import (
    MaxCapacityOverrideParseError,
    RedisBucket,
)


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.pipeline.return_value = AsyncMock()
    return mock


@pytest.fixture
def quota():
    """Create a test quota with limit=20."""
    return Quota(metric="requests", limit=20, per_seconds=1)


@pytest.fixture
def limit_config(quota):
    """Create a test limit config with the quota."""
    return PerModelConfig(
        model_family="test/model",
        quotas=UsageQuotas([quota]),
    )


@pytest.fixture
def bucket(mock_redis, quota, limit_config):
    """Create a RedisBucket for testing."""
    return RedisBucket(
        quota=quota,
        limit_config=limit_config,
        redis_client=mock_redis,
        key_prefix="test",
    )


class TestMaxCapacityProperty:
    """Tests for the max_capacity property."""

    def test_returns_default_when_no_cache(self, bucket, quota):
        """max_capacity returns quota.limit when no cached value."""
        assert bucket.max_capacity == quota.limit
        assert bucket.max_capacity == 20.0

    def test_returns_cached_value_when_set(self, bucket):
        """max_capacity returns cached value when available."""
        bucket._max_capacity_cached = 5.0
        assert bucket.max_capacity == 5.0


class TestGetMaxCapacity:
    """Tests for get_max_capacity() async method."""

    def test_fetches_from_redis_when_no_cache(self, bucket, mock_redis):
        """get_max_capacity() fetches from Redis when cache is empty."""
        payload = json.dumps(
            {"configured_max_capacity": 20.0, "override_max_capacity": 15.0}
        ).encode()
        mock_redis.get.return_value = payload

        result = asyncio.run(bucket.get_max_capacity())

        assert result == 15.0
        mock_redis.get.assert_called_once_with(bucket._max_capacity_key)

    def test_returns_cached_value_when_fresh(self, bucket, mock_redis):
        """get_max_capacity() returns cached value without Redis call when fresh."""
        bucket._max_capacity_cached = 10.0
        bucket._max_capacity_cache_time = time.time()  # Fresh cache

        result = asyncio.run(bucket.get_max_capacity())

        assert result == 10.0
        mock_redis.get.assert_not_called()

    def test_refetches_when_cache_stale(self, bucket, mock_redis):
        """get_max_capacity() refetches from Redis when cache is stale."""
        bucket._max_capacity_cached = 10.0
        bucket._max_capacity_cache_time = time.time() - 2.0  # Stale (>1s TTL)
        payload = json.dumps(
            {"configured_max_capacity": 20.0, "override_max_capacity": 8.0}
        ).encode()
        mock_redis.get.return_value = payload

        result = asyncio.run(bucket.get_max_capacity())

        assert result == 8.0
        mock_redis.get.assert_called_once()

    def test_returns_default_when_redis_key_missing(self, bucket, mock_redis, quota):
        """get_max_capacity() returns default when Redis key doesn't exist."""
        mock_redis.get.return_value = None

        result = asyncio.run(bucket.get_max_capacity())

        assert result == quota.limit
        assert result == 20.0

    def test_ignores_legacy_max_capacity_key_from_previous_versions(
        self, bucket, mock_redis, quota
    ):
        """Only the dedicated runtime-override key should affect fresh processes."""

        def get_side_effect(key: str):
            legacy_key = f"{bucket.full_redis_key}:max_capacity"
            if key == legacy_key:
                return b"5.0"
            if key == bucket._max_capacity_key:
                return None
            return None

        mock_redis.get.side_effect = get_side_effect

        result = asyncio.run(bucket.get_max_capacity())

        assert result == quota.limit
        mock_redis.get.assert_called_once_with(bucket._max_capacity_key)

    def test_handles_invalid_redis_value(self, bucket, mock_redis):
        """get_max_capacity() surfaces corrupt Redis override values."""
        mock_redis.get.return_value = b"not-a-number"

        with pytest.raises(MaxCapacityOverrideParseError, match="not valid JSON"):
            asyncio.run(bucket.get_max_capacity())

    def test_raises_on_nan_from_redis(self, bucket, mock_redis):
        """get_max_capacity() surfaces non-canonical NaN."""
        mock_redis.get.return_value = b"nan"

        with pytest.raises(MaxCapacityOverrideParseError, match="not valid JSON"):
            asyncio.run(bucket.get_max_capacity())

    def test_raises_on_inf_from_redis(self, bucket, mock_redis):
        """get_max_capacity() surfaces non-canonical inf."""
        mock_redis.get.return_value = b"inf"

        with pytest.raises(MaxCapacityOverrideParseError, match="not valid JSON"):
            asyncio.run(bucket.get_max_capacity())


class TestSetMaxCapacity:
    """Tests for set_max_capacity() async method."""

    def test_stores_value_in_redis(self, bucket, mock_redis):
        """set_max_capacity() stores the value in Redis."""
        asyncio.run(bucket.set_max_capacity(5.0))

        assert mock_redis.set.await_count == 2
        key, payload = mock_redis.set.await_args_list[1].args
        assert key == bucket._max_capacity_key
        assert mock_redis.set.await_args_list[1].kwargs == {
            "ex": bucket._override_ttl_seconds
        }
        assert json.loads(payload) == {
            "configured_max_capacity": 20.0,
            "override_max_capacity": 5.0,
        }

    def test_updates_cache_immediately(self, bucket, mock_redis):
        """set_max_capacity() updates local cache immediately."""
        asyncio.run(bucket.set_max_capacity(5.0))

        assert bucket._max_capacity_cached == 5.0
        assert bucket._max_capacity_cache_time > 0

    def test_rejects_zero_value(self, bucket):
        """set_max_capacity() raises for zero value."""
        with pytest.raises(ValueError, match="must be finite and greater than 0"):
            asyncio.run(bucket.set_max_capacity(0))

    def test_rejects_negative_value(self, bucket):
        """set_max_capacity() raises for negative value."""
        with pytest.raises(ValueError, match="must be finite and greater than 0"):
            asyncio.run(bucket.set_max_capacity(-5.0))

    def test_rejects_nan(self, bucket):
        """set_max_capacity() raises for NaN."""
        with pytest.raises(ValueError, match="must be finite and greater than 0"):
            asyncio.run(bucket.set_max_capacity(float("nan")))

    def test_rejects_positive_inf(self, bucket):
        """set_max_capacity() raises for positive infinity."""
        with pytest.raises(ValueError, match="must be finite and greater than 0"):
            asyncio.run(bucket.set_max_capacity(float("inf")))

    def test_rejects_negative_inf(self, bucket):
        """set_max_capacity() raises for negative infinity."""
        with pytest.raises(ValueError, match="must be finite and greater than 0"):
            asyncio.run(bucket.set_max_capacity(float("-inf")))

    def test_rejects_boolean(self, bucket):
        """set_max_capacity() raises for boolean values."""
        with pytest.raises(ValueError, match="max_capacity must not be a boolean"):
            asyncio.run(bucket.set_max_capacity(True))


class TestMaxCapacityInCalculations:
    """Tests for max_capacity usage in capacity calculations."""

    def test_calculate_capacity_uses_max_capacity_for_fresh_start(self, bucket):
        """calculate_capacity() uses max_capacity when no prior data."""
        bucket._max_capacity_cached = 5.0

        result = bucket.calculate_capacity(
            last_checked=None,
            outdated_capacity=None,
            current_time=time.time(),
        )

        assert result.is_fresh_start is True
        assert result.amount == 5.0  # Uses cached max_capacity, not default

    def test_calculate_capacity_caps_refill_at_max_capacity(self, bucket):
        """calculate_capacity() caps refilled capacity at max_capacity."""
        bucket._max_capacity_cached = 5.0
        current_time = time.time()

        # Bucket was at 3.0 capacity, 10 seconds ago, rate is 20/s
        # Refill would be 3.0 + (10 * 20) = 203, but capped at max_capacity=5.0
        result = bucket.calculate_capacity(
            last_checked=current_time - 10,
            outdated_capacity=3.0,
            current_time=current_time,
        )

        assert result.is_fresh_start is False
        assert result.amount == 5.0  # Capped at max_capacity


class TestSetMaxCapacityUpdatesRate:
    """Regression: _rate_per_sec must update when max_capacity changes."""

    def test_set_max_capacity_updates_rate_per_sec(self, bucket):
        """set_max_capacity() must recalculate _rate_per_sec.

        Regression: _rate_per_sec was computed once at init (limit/per_seconds)
        and never updated, so the bucket refilled at the old rate.
        """
        # Initial: limit=20, per_seconds=1 -> rate=20.0
        assert bucket._rate_per_sec == pytest.approx(20.0)

        asyncio.run(bucket.set_max_capacity(10.0))

        # New rate should be 10/1 = 10.0
        assert bucket._rate_per_sec == pytest.approx(10.0)

    def test_refill_uses_new_rate_after_set_max_capacity(self, bucket):
        """After set_max_capacity(), refill amount must reflect the new rate.

        Uses a short time delta so the refill is observable below the cap.
        """
        asyncio.run(bucket.set_max_capacity(100.0))
        # New rate = 100/1 = 100.0/s
        current_time = time.time()

        # Bucket at 0 capacity, 0.1s elapsed -> refill = 0.1 * 100 = 10.0
        result = bucket.calculate_capacity(
            last_checked=current_time - 0.1,
            outdated_capacity=0.0,
            current_time=current_time,
        )
        assert result.amount == pytest.approx(10.0, abs=0.01)

    def test_get_max_capacity_updates_rate_on_redis_change(self, bucket, mock_redis):
        """get_max_capacity() must recalculate _rate_per_sec when Redis value differs."""
        # Simulate another process changing max_capacity in Redis
        payload = json.dumps(
            {"configured_max_capacity": 20.0, "override_max_capacity": 50.0}
        ).encode()
        mock_redis.get.return_value = payload
        bucket._max_capacity_cache_time = 0.0  # Force cache miss

        asyncio.run(bucket.get_max_capacity())

        assert bucket._rate_per_sec == pytest.approx(50.0)


class TestUpdateMaxCapacityFromResult:
    """Tests for update_max_capacity_from_result() — pipeline-based cache update."""

    def test_valid_payload_updates_cache_and_rate(self, bucket):
        """Valid anchored JSON payload updates cached max_capacity and rate."""
        payload = json.dumps(
            {"configured_max_capacity": 20.0, "override_max_capacity": 15.0}
        ).encode()
        bucket.update_max_capacity_from_result(payload)

        assert bucket._max_capacity_cached == 15.0
        assert bucket._rate_per_sec == pytest.approx(15.0)  # 15.0 / per_seconds=1
        assert bucket._max_capacity_cache_time > 0

    def test_none_falls_back_to_default(self, bucket, quota):
        """None input falls back to default max_capacity."""
        bucket.update_max_capacity_from_result(None)

        assert bucket._max_capacity_cached is None
        assert bucket.max_capacity == quota.limit
        assert bucket._rate_per_sec == pytest.approx(float(quota.limit))

    def test_invalid_bytes_raise(self, bucket):
        """Non-canonical bytes input surfaces to the caller."""
        with pytest.raises(MaxCapacityOverrideParseError, match="not valid JSON"):
            bucket.update_max_capacity_from_result(b"not-a-number")

    def test_nan_raises(self, bucket):
        """NaN bytes input surfaces to the caller."""
        with pytest.raises(MaxCapacityOverrideParseError, match="not valid JSON"):
            bucket.update_max_capacity_from_result(b"nan")

    def test_inf_raises(self, bucket):
        """Inf bytes input surfaces to the caller."""
        with pytest.raises(MaxCapacityOverrideParseError, match="not valid JSON"):
            bucket.update_max_capacity_from_result(b"inf")

    def test_negative_raises(self, bucket):
        """Negative bare value surfaces to the caller."""
        with pytest.raises(MaxCapacityOverrideParseError, match="object"):
            bucket.update_max_capacity_from_result(b"-5.0")

    def test_zero_raises(self, bucket):
        """Zero bare value surfaces to the caller."""
        with pytest.raises(MaxCapacityOverrideParseError, match="object"):
            bucket.update_max_capacity_from_result(b"0")

    def test_stale_override_metadata_for_old_config_is_ignored(self, bucket, quota):
        """Fresh processes should ignore overrides written against an old static limit."""
        payload = json.dumps(
            {
                "configured_max_capacity": 10.0,
                "override_max_capacity": 5.0,
            }
        ).encode()

        bucket.update_max_capacity_from_result(payload)

        assert bucket._max_capacity_cached is None
        assert bucket.max_capacity == quota.limit

    def test_rate_recalculation(self, bucket):
        """Rate is recalculated as new_value / per_seconds."""
        payload = json.dumps(
            {"configured_max_capacity": 20.0, "override_max_capacity": 50.0}
        ).encode()
        bucket.update_max_capacity_from_result(payload)

        # per_seconds=1, so rate = 50.0 / 1 = 50.0
        assert bucket._rate_per_sec == pytest.approx(50.0)

    def test_cache_time_always_updated(self, bucket):
        """Cache time is updated even when value doesn't change."""
        bucket._max_capacity_cache_time = 0.0
        bucket.update_max_capacity_from_result(None)  # Will use default=20

        assert bucket._max_capacity_cache_time > 0

    def test_bare_numeric_string_rejected(self, bucket):
        """Bare numeric strings are invalid unanchored overrides."""
        with pytest.raises(MaxCapacityOverrideParseError, match="object"):
            bucket.update_max_capacity_from_result(b"15.0")

    def test_dict_missing_configured_max_capacity_rejected(self, bucket):
        """Dict without configured_max_capacity surfaces to the caller."""
        payload = json.dumps({"override_max_capacity": 5.0}).encode()

        with pytest.raises(MaxCapacityOverrideParseError, match="configured"):
            bucket.update_max_capacity_from_result(payload)

    def test_no_redis_call(self, bucket, mock_redis):
        """update_max_capacity_from_result does not call Redis."""
        payload = json.dumps(
            {"configured_max_capacity": 20.0, "override_max_capacity": 15.0}
        ).encode()
        bucket.update_max_capacity_from_result(payload)

        mock_redis.get.assert_not_called()


class TestRedisKeyFormat:
    """Tests for Redis key format consistency."""

    def test_max_capacity_key_format(self, bucket):
        """Runtime override key follows the expected format."""
        expected = (
            "test:rate_limiting:bucket:test/model:requests:1:max_capacity_override"
        )
        assert bucket._max_capacity_key == expected

    def test_key_format_matches_token_throttle_convention(self, mock_redis):
        """Key format keeps the deployment prefix outermost for Redis ACLs."""
        quota = Quota(metric="requests", limit=20, per_seconds=1)
        config = PerModelConfig(
            model_family="anthropic",
            quotas=UsageQuotas([quota]),
        )
        bucket = RedisBucket(quota, config, mock_redis, key_prefix="prod")

        expected = (
            "prod:rate_limiting:bucket:anthropic:requests:1:max_capacity_override"
        )
        assert bucket._max_capacity_key == expected
