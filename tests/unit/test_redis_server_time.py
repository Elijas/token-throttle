"""Tests for Redis server-time usage in the Redis backends.

Verifies that:
1. The server-time helpers correctly convert Redis TIME responses to floats
2. Redis bucket standalone paths use server time instead of local time.time()
3. R4 L21 hardening: forward-jump rejection (T01), shape validation (T02),
   Pipeline rejection (T03)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("redis", reason="redis package not installed")

import redis.asyncio.client
import redis.client

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis import _server_time
from token_throttle._limiter_backends._redis._bucket import RedisBucket
from token_throttle._limiter_backends._redis._server_time import (
    MAX_FORWARD_JUMP_SECONDS,
    async_server_time,
    sync_server_time,
)
from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket


@pytest.fixture
def freeze_local_time(monkeypatch):
    """Freeze ``time.time()`` inside the helper module for deterministic jump tests."""
    fixed = 1_700_000_000.0
    monkeypatch.setattr(_server_time.time, "time", lambda: fixed)
    return fixed


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
        return RedisBucket(
            quota=quota,
            limit_config=config,
            redis_client=mock_redis,
            key_prefix="test",
        )

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

    async def test_set_capacity_execute_false_requires_pipeline(
        self, bucket, mock_redis
    ):
        """Standalone execute=False would otherwise discard queued writes."""
        with pytest.raises(
            ValueError, match="execute=False requires an explicit pipeline"
        ):
            await bucket.set_capacity(5.0, execute=False)

        mock_redis.time.assert_not_called()
        mock_redis.pipeline.assert_not_called()

    async def test_set_capacity_with_explicit_pipeline_executes_by_default(
        self, bucket, mock_redis
    ):
        """Explicit pipelines should still execute unless execute=False is passed."""
        pipeline = MagicMock()
        pipeline.execute = AsyncMock(return_value=[None, None])

        await bucket.set_capacity(5.0, pipeline=pipeline, current_time=999.0)

        pipeline.execute.assert_awaited_once()
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
            quota=quota,
            limit_config=config,
            redis_client=mock_redis,
            key_prefix="test",
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

    def test_set_capacity_execute_false_requires_pipeline(self, bucket, mock_redis):
        """Standalone execute=False would otherwise discard queued writes."""
        with pytest.raises(
            ValueError, match="execute=False requires an explicit pipeline"
        ):
            bucket.set_capacity(5.0, execute=False)

        mock_redis.time.assert_not_called()
        mock_redis.pipeline.assert_not_called()

    def test_set_capacity_with_explicit_pipeline_executes_by_default(
        self, bucket, mock_redis
    ):
        """Explicit pipelines should still execute unless execute=False is passed."""
        pipeline = MagicMock()

        bucket.set_capacity(5.0, pipeline=pipeline, current_time=999.0)

        pipeline.execute.assert_called_once()
        mock_redis.time.assert_not_called()


# ---------------------------------------------------------------------------
# R4 L21 T01 — forward-jump detection (cluster default-node failover guard)
# ---------------------------------------------------------------------------


class TestForwardJumpDetection:
    """Reject Redis TIME values that are implausibly far ahead of local clock.

    Models a RedisCluster default-node failover where the new primary's clock
    is grossly skewed forward. Without this guard, ``calculate_capacity`` sees
    a giant ``time_passed`` and silently over-grants bucket capacity.
    """

    async def test_async_rejects_jump_just_over_threshold(self, freeze_local_time):
        client = AsyncMock()
        # 11 seconds ahead of local — past the 10s default
        client.time.return_value = (int(freeze_local_time) + 11, 0)
        with pytest.raises(RuntimeError, match="ahead of local wall clock"):
            await async_server_time(client)

    def test_sync_rejects_jump_just_over_threshold(self, freeze_local_time):
        client = MagicMock()
        client.time.return_value = (int(freeze_local_time) + 11, 0)
        with pytest.raises(RuntimeError, match="ahead of local wall clock"):
            sync_server_time(client)

    async def test_async_accepts_small_forward_skew(self, freeze_local_time):
        """A few seconds of forward skew is normal NTP slew — must NOT raise."""
        client = AsyncMock()
        client.time.return_value = (int(freeze_local_time) + 5, 0)
        result = await async_server_time(client)
        assert result == pytest.approx(freeze_local_time + 5)

    def test_sync_accepts_small_forward_skew(self, freeze_local_time):
        client = MagicMock()
        client.time.return_value = (int(freeze_local_time) + 5, 0)
        result = sync_server_time(client)
        assert result == pytest.approx(freeze_local_time + 5)

    async def test_async_accepts_backward_jump(self, freeze_local_time):
        """Backward direction is already clamped in calculate_capacity — must NOT raise here."""
        client = AsyncMock()
        client.time.return_value = (int(freeze_local_time) - 1_000_000, 0)
        result = await async_server_time(client)
        assert result == pytest.approx(freeze_local_time - 1_000_000)

    def test_sync_accepts_backward_jump(self, freeze_local_time):
        client = MagicMock()
        client.time.return_value = (int(freeze_local_time) - 1_000_000, 0)
        result = sync_server_time(client)
        assert result == pytest.approx(freeze_local_time - 1_000_000)

    async def test_async_jump_at_threshold_passes(self, freeze_local_time):
        """At exactly MAX_FORWARD_JUMP_SECONDS the check is inclusive (not strict-greater)."""
        client = AsyncMock()
        client.time.return_value = (
            int(freeze_local_time) + int(MAX_FORWARD_JUMP_SECONDS),
            0,
        )
        # Equal to threshold → not "more than" — must not raise.
        await async_server_time(client)


# ---------------------------------------------------------------------------
# R4 L21 T02 — runtime shape / range validation
# ---------------------------------------------------------------------------


class TestResponseShapeValidation:
    """Reject off-shape TIME responses instead of producing wrong-but-not-erroring math."""

    async def test_async_rejects_non_sequence(self, freeze_local_time):
        client = AsyncMock()
        client.time.return_value = 1_700_000_000  # bare int, not a tuple
        with pytest.raises(TypeError, match="unexpected shape"):
            await async_server_time(client)

    async def test_async_rejects_wrong_length(self, freeze_local_time):
        client = AsyncMock()
        client.time.return_value = (1_700_000_000, 0, 0)  # 3-tuple
        with pytest.raises(TypeError, match="unexpected shape"):
            await async_server_time(client)

    async def test_async_rejects_nan_seconds(self, freeze_local_time):
        client = AsyncMock()
        client.time.return_value = (float("nan"), 0)
        with pytest.raises(TypeError, match="integer-coercible"):
            await async_server_time(client)

    async def test_async_rejects_inf_seconds(self, freeze_local_time):
        client = AsyncMock()
        client.time.return_value = (float("inf"), 0)
        with pytest.raises(TypeError, match="integer-coercible"):
            await async_server_time(client)

    async def test_async_rejects_negative_seconds(self, freeze_local_time):
        client = AsyncMock()
        client.time.return_value = (-1, 0)
        with pytest.raises(ValueError, match="out of range"):
            await async_server_time(client)

    async def test_async_rejects_microseconds_at_million(self, freeze_local_time):
        client = AsyncMock()
        client.time.return_value = (int(freeze_local_time), 1_000_000)
        with pytest.raises(ValueError, match="out of range"):
            await async_server_time(client)

    async def test_async_rejects_negative_microseconds(self, freeze_local_time):
        client = AsyncMock()
        client.time.return_value = (int(freeze_local_time), -1)
        with pytest.raises(ValueError, match="out of range"):
            await async_server_time(client)

    async def test_async_accepts_list_form(self, freeze_local_time):
        """Real redis-py may return list rather than tuple — both are valid."""
        client = AsyncMock()
        client.time.return_value = [int(freeze_local_time), 500_000]
        result = await async_server_time(client)
        assert result == pytest.approx(freeze_local_time + 0.5)

    def test_sync_rejects_non_sequence(self, freeze_local_time):
        client = MagicMock()
        client.time.return_value = 1_700_000_000
        with pytest.raises(TypeError, match="unexpected shape"):
            sync_server_time(client)

    def test_sync_rejects_wrong_length(self, freeze_local_time):
        client = MagicMock()
        client.time.return_value = (1_700_000_000,)
        with pytest.raises(TypeError, match="unexpected shape"):
            sync_server_time(client)

    def test_sync_rejects_nan_seconds(self, freeze_local_time):
        client = MagicMock()
        client.time.return_value = (float("nan"), 0)
        with pytest.raises(TypeError, match="integer-coercible"):
            sync_server_time(client)

    def test_sync_rejects_negative_seconds(self, freeze_local_time):
        client = MagicMock()
        client.time.return_value = (-1, 0)
        with pytest.raises(ValueError, match="out of range"):
            sync_server_time(client)


# ---------------------------------------------------------------------------
# R4 L21 T03 — Pipeline-as-client rejection
# ---------------------------------------------------------------------------


class TestPipelineRejection:
    """``Pipeline`` is a ``Redis`` subclass; the static type accepts it.

    Without runtime rejection, ``client.time()`` queues TIME on the pipeline
    and returns the pipeline object; the unpack then fails with a confusing
    ``ValueError: not enough values to unpack (expected 2, got 0)``.
    """

    async def test_async_rejects_async_pipeline(self):
        pipeline = MagicMock(spec=redis.asyncio.client.Pipeline)
        with pytest.raises(TypeError, match="bare Redis client, not a Pipeline"):
            await async_server_time(pipeline)
        # And TIME should never have been issued.
        pipeline.time.assert_not_called()

    def test_sync_rejects_sync_pipeline(self):
        pipeline = MagicMock(spec=redis.client.Pipeline)
        with pytest.raises(TypeError, match="bare Redis client, not a Pipeline"):
            sync_server_time(pipeline)
        pipeline.time.assert_not_called()

    async def test_async_rejects_sync_pipeline_too(self):
        """Even a sync Pipeline passed to the async helper is rejected (defensive)."""
        pipeline = MagicMock(spec=redis.client.Pipeline)
        with pytest.raises(TypeError, match="bare Redis client, not a Pipeline"):
            await async_server_time(pipeline)

    def test_sync_rejects_async_pipeline_too(self):
        pipeline = MagicMock(spec=redis.asyncio.client.Pipeline)
        with pytest.raises(TypeError, match="bare Redis client, not a Pipeline"):
            sync_server_time(pipeline)
