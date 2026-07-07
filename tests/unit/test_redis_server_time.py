"""Tests for Redis server-time usage in the Redis backends.

Verifies that:
1. The server-time helpers correctly convert Redis TIME responses to floats
2. Redis bucket standalone paths use server time instead of local time.time()
3. Hardening: consecutive-reading forward-jump detection, TIME shape
   validation, and Pipeline-as-client rejection
"""

import logging
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


def test_module_docstring_references_real_entry_points():
    """The TIME shape contract must be documented against the real helpers,
    not a nonexistent gateway abstraction a reader cannot find.
    """
    doc = _server_time.__doc__
    assert doc is not None
    assert "AbstractRedisGateway" not in doc
    assert "async_server_time" in doc
    assert "sync_server_time" in doc


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
        pipeline.execute = AsyncMock(return_value=[None, None, False, False])
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
        pipeline.execute.return_value = [True, True]

        await bucket.set_capacity(5.0)

        mock_redis.time.assert_called_once()
        # The last_checked key should be set to the Redis server time
        pipeline.set.assert_any_call(
            bucket._last_checked_key,
            1700000000.123456,
            ex=bucket._bucket_ttl_seconds,
        )

    async def test_get_capacity_with_explicit_time_skips_redis_time(
        self, bucket, mock_redis
    ):
        """When current_time is provided, Redis TIME should not be called."""
        pipeline = MagicMock()
        pipeline.execute = AsyncMock(return_value=[None, None, False, False])

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
        pipeline.execute = AsyncMock(return_value=[True, True])

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
        pipeline.execute.return_value = [None, None, False, False]
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
        pipeline.execute.return_value = [True, True]

        bucket.set_capacity(5.0)

        mock_redis.time.assert_called_once()
        pipeline.set.assert_any_call(
            bucket._last_checked_key,
            1700000000.123456,
            ex=bucket._bucket_ttl_seconds,
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
        pipeline.execute.return_value = [True, True]

        bucket.set_capacity(5.0, pipeline=pipeline, current_time=999.0)

        pipeline.execute.assert_called_once()
        mock_redis.time.assert_not_called()


# ---------------------------------------------------------------------------
# Forward-jump detection — consecutive-reading guard against a jumped primary
# ---------------------------------------------------------------------------


class TestForwardJumpDetection:
    """Detect a genuine server-side forward jump between consecutive TIME reads.

    The rail compares consecutive Redis TIME values against locally-elapsed
    *monotonic* time (per client) with each reading's round-trip bounded into
    the tolerance, so it fires only on a real server clock jump (e.g. a
    Sentinel/managed failover to a clock-skewed primary) - never on a lagging
    local wall clock, a slow TIME reply, or a suspended/resumed host (where
    the local *wall* clock corroborates the server's advance and the rail
    re-anchors with a warning instead). The first reading only establishes
    the baseline, and a detected jump re-anchors exactly once so stale
    out-of-order readings cannot re-arm it.
    """

    @pytest.fixture(autouse=True)
    def _reset_state(self):
        """Isolate the per-client clock state between tests."""
        _server_time._client_states.clear()
        yield
        _server_time._client_states.clear()

    @pytest.fixture
    def monotonic(self, monkeypatch):
        """Controllable ``time.monotonic()`` for deterministic elapsed intervals."""
        clock = {"now": 1000.0}
        monkeypatch.setattr(_server_time.time, "monotonic", lambda: clock["now"])
        return clock

    @pytest.fixture
    def wall(self, monkeypatch):
        """Controllable ``time.time()`` for suspend-vs-jump discrimination tests."""
        clock = {"now": 1_700_000_000.0}
        monkeypatch.setattr(_server_time.time, "time", lambda: clock["now"])
        return clock

    # (a) A lagging local wall clock must NOT hard-fail operations.
    async def test_async_lagging_local_clock_does_not_raise(
        self, freeze_local_time, monotonic
    ):
        client = AsyncMock()
        # Server is a steady 30s ahead of the frozen (lagging) local wall clock.
        monotonic["now"] = 1000.0
        client.time.return_value = (int(freeze_local_time) + 30, 0)
        assert await async_server_time(client) == pytest.approx(freeze_local_time + 30)
        # Second reading: 5s later on BOTH the server and the monotonic clock.
        monotonic["now"] = 1005.0
        client.time.return_value = (int(freeze_local_time) + 35, 0)
        assert await async_server_time(client) == pytest.approx(freeze_local_time + 35)

    def test_sync_lagging_local_clock_does_not_raise(
        self, freeze_local_time, monotonic
    ):
        client = MagicMock()
        monotonic["now"] = 1000.0
        client.time.return_value = (int(freeze_local_time) + 30, 0)
        assert sync_server_time(client) == pytest.approx(freeze_local_time + 30)
        monotonic["now"] = 1005.0
        client.time.return_value = (int(freeze_local_time) + 35, 0)
        assert sync_server_time(client) == pytest.approx(freeze_local_time + 35)

    # (b) A genuine server jump between consecutive readings MUST raise.
    async def test_async_genuine_jump_raises(self, freeze_local_time, monotonic):
        client = AsyncMock()
        monotonic["now"] = 1000.0
        client.time.return_value = (1_700_000_000, 0)  # baseline
        await async_server_time(client)
        # 1s of monotonic elapsed, but the server clock leapt 12s → excess 11s.
        monotonic["now"] = 1001.0
        client.time.return_value = (1_700_000_012, 0)
        with pytest.raises(RuntimeError, match="jumped forward"):
            await async_server_time(client)

    def test_sync_genuine_jump_raises(self, freeze_local_time, monotonic):
        client = MagicMock()
        monotonic["now"] = 1000.0
        client.time.return_value = (1_700_000_000, 0)
        sync_server_time(client)
        monotonic["now"] = 1001.0
        client.time.return_value = (1_700_000_012, 0)
        with pytest.raises(RuntimeError, match="jumped forward"):
            sync_server_time(client)

    # (c) The very first reading only establishes the baseline — never raises,
    #     even when it is implausibly far ahead of the local wall clock.
    async def test_async_first_reading_never_raises(self, freeze_local_time, monotonic):
        client = AsyncMock()
        monotonic["now"] = 1000.0
        client.time.return_value = (int(freeze_local_time) + 1_000_000, 0)
        assert await async_server_time(client) == pytest.approx(
            freeze_local_time + 1_000_000
        )

    def test_sync_first_reading_never_raises(self, freeze_local_time, monotonic):
        client = MagicMock()
        monotonic["now"] = 1000.0
        client.time.return_value = (int(freeze_local_time) + 1_000_000, 0)
        assert sync_server_time(client) == pytest.approx(freeze_local_time + 1_000_000)

    # (d) Backward jumps and small forward skews are fine.
    async def test_async_backward_jump_does_not_raise(
        self, freeze_local_time, monotonic
    ):
        client = AsyncMock()
        monotonic["now"] = 1000.0
        client.time.return_value = (1_700_000_100, 0)
        await async_server_time(client)
        monotonic["now"] = 1001.0
        client.time.return_value = (1_700_000_050, 0)  # server went backward 50s
        assert await async_server_time(client) == pytest.approx(1_700_000_050.0)

    def test_sync_backward_jump_does_not_raise(self, freeze_local_time, monotonic):
        client = MagicMock()
        monotonic["now"] = 1000.0
        client.time.return_value = (1_700_000_100, 0)
        sync_server_time(client)
        monotonic["now"] = 1001.0
        client.time.return_value = (1_700_000_050, 0)
        assert sync_server_time(client) == pytest.approx(1_700_000_050.0)

    async def test_async_small_forward_skew_does_not_raise(
        self, freeze_local_time, monotonic
    ):
        client = AsyncMock()
        monotonic["now"] = 1000.0
        client.time.return_value = (1_700_000_000, 0)
        await async_server_time(client)
        # Server advanced 9s while 1s of monotonic elapsed → excess 8s < 10s.
        monotonic["now"] = 1001.0
        client.time.return_value = (1_700_000_009, 0)
        assert await async_server_time(client) == pytest.approx(1_700_000_009.0)

    def test_sync_small_forward_skew_does_not_raise(self, freeze_local_time, monotonic):
        client = MagicMock()
        monotonic["now"] = 1000.0
        client.time.return_value = (1_700_000_000, 0)
        sync_server_time(client)
        monotonic["now"] = 1001.0
        client.time.return_value = (1_700_000_009, 0)
        assert sync_server_time(client) == pytest.approx(1_700_000_009.0)

    async def test_async_jump_at_threshold_passes(self, freeze_local_time, monotonic):
        """At exactly MAX_FORWARD_JUMP_SECONDS the check is inclusive (not strict-greater)."""
        client = AsyncMock()
        monotonic["now"] = 1000.0
        client.time.return_value = (1_700_000_000, 0)
        await async_server_time(client)
        # 1s monotonic elapsed, server advanced 1s + threshold → excess == threshold.
        monotonic["now"] = 1001.0
        client.time.return_value = (
            1_700_000_000 + 1 + int(MAX_FORWARD_JUMP_SECONDS),
            0,
        )
        # Equal to threshold → not "more than" — must not raise.
        await async_server_time(client)

    # (e) Host suspend/VM pause: the server and wall clocks advance while the
    #     monotonic clock stalls. The wall clock corroborates the server, so
    #     the rail re-anchors with a warning instead of hard-failing.
    async def test_async_host_suspend_reanchors_instead_of_raising(
        self, monotonic, wall, caplog
    ):
        client = AsyncMock()
        monotonic["now"] = 1000.0
        wall["now"] = 1_700_000_000.0
        client.time.return_value = (1_700_000_000, 0)
        await async_server_time(client)
        # Host suspends for 300s: server and wall advance, monotonic stalls.
        monotonic["now"] = 1000.1
        wall["now"] = 1_700_000_300.0
        client.time.return_value = (1_700_000_300, 0)
        with caplog.at_level(logging.WARNING, logger="token_throttle"):
            result = await async_server_time(client)
        assert result == pytest.approx(1_700_000_300.0)
        assert "suspended/resumed host" in caplog.text
        # The rail keeps working from the re-anchored baseline.
        monotonic["now"] = 1001.1
        wall["now"] = 1_700_000_301.0
        client.time.return_value = (1_700_000_301, 0)
        assert await async_server_time(client) == pytest.approx(1_700_000_301.0)

    def test_sync_host_suspend_reanchors_instead_of_raising(
        self, monotonic, wall, caplog
    ):
        client = MagicMock()
        monotonic["now"] = 1000.0
        wall["now"] = 1_700_000_000.0
        client.time.return_value = (1_700_000_000, 0)
        sync_server_time(client)
        monotonic["now"] = 1000.1
        wall["now"] = 1_700_000_300.0
        client.time.return_value = (1_700_000_300, 0)
        with caplog.at_level(logging.WARNING, logger="token_throttle"):
            result = sync_server_time(client)
        assert result == pytest.approx(1_700_000_300.0)
        assert "suspended/resumed host" in caplog.text
        monotonic["now"] = 1001.1
        wall["now"] = 1_700_000_301.0
        client.time.return_value = (1_700_000_301, 0)
        assert sync_server_time(client) == pytest.approx(1_700_000_301.0)

    # (f) A genuine server jump: wall and monotonic agree that little time
    #     passed while the server leapt - the raise must survive the
    #     wall-clock discriminator.
    async def test_async_genuine_jump_with_agreeing_wall_raises(self, monotonic, wall):
        client = AsyncMock()
        monotonic["now"] = 1000.0
        wall["now"] = 1_700_000_000.0
        client.time.return_value = (1_700_000_000, 0)
        await async_server_time(client)
        # Wall and monotonic both advance 1s; the server leaps 20s.
        monotonic["now"] = 1001.0
        wall["now"] = 1_700_000_001.0
        client.time.return_value = (1_700_000_020, 0)
        with pytest.raises(RuntimeError, match="jumped forward"):
            await async_server_time(client)

    def test_sync_genuine_jump_with_agreeing_wall_raises(self, monotonic, wall):
        client = MagicMock()
        monotonic["now"] = 1000.0
        wall["now"] = 1_700_000_000.0
        client.time.return_value = (1_700_000_000, 0)
        sync_server_time(client)
        monotonic["now"] = 1001.0
        wall["now"] = 1_700_000_001.0
        client.time.return_value = (1_700_000_020, 0)
        with pytest.raises(RuntimeError, match="jumped forward"):
            sync_server_time(client)

    # (g) One slow TIME reply must not trip the rail: the reading's own
    #     round-trip is bounded into the tolerance because the anchor keeps
    #     the pre-call monotonic reading.
    async def test_async_slow_time_reply_does_not_raise(self, monotonic, wall):
        client = AsyncMock()
        monotonic["now"] = 1000.0
        wall["now"] = 1_700_000_000.0

        def slow_then_prompt_reply():
            if client.time.call_count == 1:
                # The server samples TIME at send time, but the reply takes
                # 12s to arrive; local clocks tick on during the transit.
                monotonic["now"] = 1012.0
                wall["now"] = 1_700_000_012.0
                return (1_700_000_000, 0)
            return (1_700_000_012, 500_000)

        client.time.side_effect = slow_then_prompt_reply
        await async_server_time(client)
        # A prompt reading 0.5s later: the server clock kept ticking during
        # the previous reply's 12s transit.
        monotonic["now"] = 1012.5
        wall["now"] = 1_700_000_012.5
        assert await async_server_time(client) == pytest.approx(1_700_000_012.5)

    def test_sync_slow_time_reply_does_not_raise(self, monotonic, wall):
        client = MagicMock()
        monotonic["now"] = 1000.0
        wall["now"] = 1_700_000_000.0

        def slow_then_prompt_reply():
            if client.time.call_count == 1:
                monotonic["now"] = 1012.0
                wall["now"] = 1_700_000_012.0
                return (1_700_000_000, 0)
            return (1_700_000_012, 500_000)

        client.time.side_effect = slow_then_prompt_reply
        sync_server_time(client)
        monotonic["now"] = 1012.5
        wall["now"] = 1_700_000_012.5
        assert sync_server_time(client) == pytest.approx(1_700_000_012.5)

    # (h) A detected jump re-anchors exactly once: a stale pre-jump reading
    #     arriving out of order (a concurrent caller) must neither raise nor
    #     regress the anchor, so the same jump is never reported twice.
    async def test_async_out_of_order_reading_does_not_rearm_the_rail(
        self, freeze_local_time, monotonic
    ):
        client = AsyncMock()
        monotonic["now"] = 1000.0
        client.time.return_value = (1_700_000_000, 0)
        await async_server_time(client)
        # A genuine 50s jump raises once and re-anchors to the jumped reading.
        monotonic["now"] = 1001.0
        client.time.return_value = (1_700_000_050, 0)
        with pytest.raises(RuntimeError, match="jumped forward"):
            await async_server_time(client)
        # A stale pre-jump reading arrives out of order: no raise, no anchor
        # regression.
        monotonic["now"] = 1002.0
        client.time.return_value = (1_700_000_001, 0)
        assert await async_server_time(client) == pytest.approx(1_700_000_001.0)
        # A later post-jump reading compares against the jumped baseline -
        # the same event must not raise a second time.
        monotonic["now"] = 1003.0
        client.time.return_value = (1_700_000_052, 0)
        assert await async_server_time(client) == pytest.approx(1_700_000_052.0)

    def test_sync_out_of_order_reading_does_not_rearm_the_rail(
        self, freeze_local_time, monotonic
    ):
        client = MagicMock()
        monotonic["now"] = 1000.0
        client.time.return_value = (1_700_000_000, 0)
        sync_server_time(client)
        monotonic["now"] = 1001.0
        client.time.return_value = (1_700_000_050, 0)
        with pytest.raises(RuntimeError, match="jumped forward"):
            sync_server_time(client)
        monotonic["now"] = 1002.0
        client.time.return_value = (1_700_000_001, 0)
        assert sync_server_time(client) == pytest.approx(1_700_000_001.0)
        monotonic["now"] = 1003.0
        client.time.return_value = (1_700_000_052, 0)
        assert sync_server_time(client) == pytest.approx(1_700_000_052.0)

    # The one-time wall-skew warning: a large server-vs-wall divergence is logged
    # (once per client) but never raises, because refill uses server time only.
    async def test_async_lagging_clock_warns_once(
        self, freeze_local_time, monotonic, caplog
    ):
        client = AsyncMock()
        monotonic["now"] = 1000.0
        client.time.return_value = (int(freeze_local_time) + 30, 0)
        with caplog.at_level(logging.WARNING, logger="token_throttle"):
            await async_server_time(client)
            monotonic["now"] = 1001.0
            client.time.return_value = (int(freeze_local_time) + 31, 0)
            await async_server_time(client)
        assert caplog.text.count("diverges from the local wall clock") == 1

    def test_sync_lagging_clock_warns_once(self, freeze_local_time, monotonic, caplog):
        client = MagicMock()
        monotonic["now"] = 1000.0
        client.time.return_value = (int(freeze_local_time) + 30, 0)
        with caplog.at_level(logging.WARNING, logger="token_throttle"):
            sync_server_time(client)
            monotonic["now"] = 1001.0
            client.time.return_value = (int(freeze_local_time) + 31, 0)
            sync_server_time(client)
        assert caplog.text.count("diverges from the local wall clock") == 1


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
