"""Tests for dynamic max_capacity in RedisBucket.

These tests verify that max_capacity can be updated at runtime via Redis,
enabling adaptive rate limiting scenarios where limits change dynamically.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._limiter_backends._redis._bucket import RedisBucket


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
        mock_redis.get.return_value = b"15.0"

        result = asyncio.get_event_loop().run_until_complete(bucket.get_max_capacity())

        assert result == 15.0
        mock_redis.get.assert_called_once_with(bucket._max_capacity_key)

    def test_returns_cached_value_when_fresh(self, bucket, mock_redis):
        """get_max_capacity() returns cached value without Redis call when fresh."""
        bucket._max_capacity_cached = 10.0
        bucket._max_capacity_cache_time = time.time()  # Fresh cache

        result = asyncio.get_event_loop().run_until_complete(bucket.get_max_capacity())

        assert result == 10.0
        mock_redis.get.assert_not_called()

    def test_refetches_when_cache_stale(self, bucket, mock_redis):
        """get_max_capacity() refetches from Redis when cache is stale."""
        bucket._max_capacity_cached = 10.0
        bucket._max_capacity_cache_time = time.time() - 2.0  # Stale (>1s TTL)
        mock_redis.get.return_value = b"8.0"

        result = asyncio.get_event_loop().run_until_complete(bucket.get_max_capacity())

        assert result == 8.0
        mock_redis.get.assert_called_once()

    def test_returns_default_when_redis_key_missing(self, bucket, mock_redis, quota):
        """get_max_capacity() returns default when Redis key doesn't exist."""
        mock_redis.get.return_value = None

        result = asyncio.get_event_loop().run_until_complete(bucket.get_max_capacity())

        assert result == quota.limit
        assert result == 20.0

    def test_handles_invalid_redis_value(self, bucket, mock_redis, quota):
        """get_max_capacity() returns default for invalid Redis values."""
        mock_redis.get.return_value = b"not-a-number"

        result = asyncio.get_event_loop().run_until_complete(bucket.get_max_capacity())

        assert result == quota.limit


class TestSetMaxCapacity:
    """Tests for set_max_capacity() async method."""

    def test_stores_value_in_redis(self, bucket, mock_redis):
        """set_max_capacity() stores the value in Redis."""
        asyncio.get_event_loop().run_until_complete(bucket.set_max_capacity(5.0))

        mock_redis.set.assert_called_once_with(bucket._max_capacity_key, 5.0)

    def test_updates_cache_immediately(self, bucket, mock_redis):
        """set_max_capacity() updates local cache immediately."""
        asyncio.get_event_loop().run_until_complete(bucket.set_max_capacity(5.0))

        assert bucket._max_capacity_cached == 5.0
        assert bucket._max_capacity_cache_time > 0

    def test_rejects_zero_value(self, bucket):
        """set_max_capacity() raises for zero value."""
        with pytest.raises(ValueError, match="must be greater than 0"):
            asyncio.get_event_loop().run_until_complete(bucket.set_max_capacity(0))

    def test_rejects_negative_value(self, bucket):
        """set_max_capacity() raises for negative value."""
        with pytest.raises(ValueError, match="must be greater than 0"):
            asyncio.get_event_loop().run_until_complete(bucket.set_max_capacity(-5.0))


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


class TestRedisKeyFormat:
    """Tests for Redis key format consistency."""

    def test_max_capacity_key_format(self, bucket):
        """max_capacity key follows the expected format."""
        expected = "rate_limiting:test/model:requests:1:max_capacity"
        assert bucket._max_capacity_key == expected

    def test_key_format_matches_token_throttle_convention(self, mock_redis):
        """Key format matches the convention for external rate limit controllers."""
        # External controllers set: rate_limiting:{client}:requests:1:max_capacity
        # This test verifies our key matches that format
        quota = Quota(metric="requests", limit=20, per_seconds=1)
        config = PerModelConfig(
            model_family="anthropic",
            quotas=UsageQuotas([quota]),
        )
        bucket = RedisBucket(quota, config, mock_redis)

        expected = "rate_limiting:anthropic:requests:1:max_capacity"
        assert bucket._max_capacity_key == expected
