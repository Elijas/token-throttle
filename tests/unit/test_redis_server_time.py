"""Tests for Redis server-time usage in the Redis backends.

Verifies that:
1. The server-time helpers correctly convert Redis TIME responses to floats
2. Redis bucket standalone paths use server time instead of local time.time()
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._bucket import RedisBucket
from token_throttle._limiter_backends._redis._server_time import (
    async_server_time,
    sync_server_time,
)
from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket

# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestAsyncServerTime:
    async def test_converts_redis_time_to_float(self):
        client = AsyncMock()
        client.time.return_value = (1700000000, 500000)

        result = await async_server_time(client)

        assert result == 1700000000.5
        client.time.assert_called_once()

    async def test_zero_microseconds(self):
        client = AsyncMock()
        client.time.return_value = (1700000000, 0)

        result = await async_server_time(client)

        assert result == 1700000000.0

    async def test_microsecond_precision(self):
        client = AsyncMock()
        client.time.return_value = (1700000000, 999999)

        result = await async_server_time(client)

        assert result == pytest.approx(1700000000.999999)


class TestSyncServerTime:
    def test_converts_redis_time_to_float(self):
        client = MagicMock()
        client.time.return_value = (1700000000, 500000)

        result = sync_server_time(client)

        assert result == 1700000000.5
        client.time.assert_called_once()

    def test_zero_microseconds(self):
        client = MagicMock()
        client.time.return_value = (1700000000, 0)

        result = sync_server_time(client)

        assert result == 1700000000.0

    def test_microsecond_precision(self):
        client = MagicMock()
        client.time.return_value = (1700000000, 999999)

        result = sync_server_time(client)

        assert result == pytest.approx(1700000000.999999)


# ---------------------------------------------------------------------------
# Async bucket integration — standalone paths use Redis server time
# ---------------------------------------------------------------------------


class TestAsyncBucketUsesServerTime:
    @pytest.fixture
    def mock_redis(self):
        mock = AsyncMock()
        mock.time.return_value = (1700000000, 123456)
        mock.get.return_value = None  # get_max_capacity fallback
        pipeline = MagicMock()
        pipeline.execute = AsyncMock(return_value=[None, None])
        # pipeline() is sync in redis-py even for async client
        mock.pipeline = MagicMock(return_value=pipeline)
        return mock

    @pytest.fixture
    def bucket(self, mock_redis):
        quota = Quota(metric="requests", limit=10, per_seconds=1)
        config = PerModelConfig(model_family="test/model", quotas=UsageQuotas([quota]))
        return RedisBucket(quota=quota, limit_config=config, redis_client=mock_redis)

    async def test_get_capacity_standalone_calls_redis_time(self, bucket, mock_redis):
        """get_capacity() without current_time should call Redis TIME, not time.time()."""
        await bucket.get_capacity()

        mock_redis.time.assert_called_once()

    async def test_set_capacity_standalone_writes_server_timestamp(
        self, bucket, mock_redis
    ):
        """set_capacity() without current_time should write the Redis server timestamp."""
        pipeline = mock_redis.pipeline.return_value

        await bucket.set_capacity(5.0)

        mock_redis.time.assert_called_once()
        # The last_checked key should be set to the Redis server time
        pipeline.set.assert_any_call(
            bucket._last_checked_key,
            1700000000.123456,
        )

    async def test_get_capacity_with_explicit_time_skips_redis_time(
        self, bucket, mock_redis
    ):
        """When current_time is provided, Redis TIME should not be called."""
        pipeline = MagicMock()
        pipeline.execute = AsyncMock(return_value=[None, None])

        await bucket.get_capacity(pipeline=pipeline, current_time=999.0)

        mock_redis.time.assert_not_called()


# ---------------------------------------------------------------------------
# Sync bucket integration — standalone paths use Redis server time
# ---------------------------------------------------------------------------


class TestSyncBucketUsesServerTime:
    @pytest.fixture
    def mock_redis(self):
        mock = MagicMock()
        mock.time.return_value = (1700000000, 123456)
        mock.get.return_value = None  # get_max_capacity fallback
        pipeline = MagicMock()
        pipeline.execute.return_value = [None, None]
        mock.pipeline.return_value = pipeline
        return mock

    @pytest.fixture
    def bucket(self, mock_redis):
        quota = Quota(metric="requests", limit=10, per_seconds=1)
        config = PerModelConfig(model_family="test/model", quotas=UsageQuotas([quota]))
        return SyncRedisBucket(
            quota=quota, limit_config=config, redis_client=mock_redis
        )

    def test_get_capacity_standalone_calls_redis_time(self, bucket, mock_redis):
        """get_capacity() without current_time should call Redis TIME."""
        bucket.get_capacity()

        mock_redis.time.assert_called_once()

    def test_set_capacity_standalone_writes_server_timestamp(self, bucket, mock_redis):
        """set_capacity() without current_time should write the Redis server timestamp."""
        pipeline = mock_redis.pipeline.return_value

        bucket.set_capacity(5.0)

        mock_redis.time.assert_called_once()
        pipeline.set.assert_any_call(
            bucket._last_checked_key,
            1700000000.123456,
        )

    def test_get_capacity_with_explicit_time_skips_redis_time(self, bucket, mock_redis):
        """When current_time is provided, Redis TIME should not be called."""
        pipeline = MagicMock()

        bucket.get_capacity(pipeline=pipeline, current_time=999.0)

        mock_redis.time.assert_not_called()
