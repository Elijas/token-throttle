"""FHA5-N02 Redis parse-before-local-mutate regressions."""

# ruff: noqa: E402

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

redis = pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis import _backend, _sync_backend
from token_throttle._limiter_backends._redis._backend import RedisBackend
from token_throttle._limiter_backends._redis._bucket import (
    RedisBucket,
    RedisPipelineResultError,
)
from token_throttle._limiter_backends._redis._sync_backend import SyncRedisBackend
from token_throttle._limiter_backends._redis._sync_bucket import (
    RedisPipelineResultError as SyncRedisPipelineResultError,
)
from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket


class _AsyncPipeline:
    def __init__(self, result: object) -> None:
        self.result = result

    def get(self, *_args: object, **_kwargs: object) -> None:
        return None

    def expire(self, *_args: object, **_kwargs: object) -> None:
        return None

    def set(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def execute(self) -> object:
        return self.result


class _SyncPipeline(_AsyncPipeline):
    def execute(self) -> object:
        return self.result


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="test/model",
        quotas=UsageQuotas([Quota(metric="requests", limit=20, per_seconds=60)]),
    )


def _override_payload(bucket: RedisBucket | SyncRedisBucket, value: float) -> bytes:
    return json.dumps(
        {
            "configured_max_capacity": bucket.configured_max_capacity,
            "override_max_capacity": value,
        }
    ).encode()


def _assert_override_cache_unmodified(bucket: RedisBucket | SyncRedisBucket) -> None:
    assert bucket.max_capacity == pytest.approx(bucket.configured_max_capacity)
    assert bucket._max_capacity_cached is None
    assert bucket._max_capacity_cache_populated is False


def _assert_missing_data_context_unmodified(
    bucket: RedisBucket | SyncRedisBucket,
) -> None:
    assert bucket._missing_consumption_data_reason is None
    assert bucket._missing_consumption_data_missing_keys == ()
    assert bucket._missing_consumption_data_present_keys == ()


async def test_async_partial_state_repair_error_does_not_mutate_local_context() -> None:
    config = _config()
    redis_client = MagicMock()
    redis_client.get = AsyncMock(return_value=None)
    redis_client.expire = AsyncMock(return_value=True)
    redis_client.pipeline.return_value = _AsyncPipeline(
        [True, redis.exceptions.ResponseError("repair SET failed")]
    )
    bucket = RedisBucket(
        next(iter(config.quotas)), config, redis_client, key_prefix="test"
    )
    backend = RedisBackend([bucket], redis_client, config, key_prefix="test")
    initial = _AsyncPipeline(
        [b"1000.0", None, True, True, _override_payload(bucket, 7.0)]
    )

    with pytest.raises(RedisPipelineResultError, match="partial-state repair"):
        await backend._get_capacities_unsafe(pipeline=initial, current_time=1001.0)

    _assert_override_cache_unmodified(bucket)
    _assert_missing_data_context_unmodified(bucket)


def test_sync_partial_state_repair_error_does_not_mutate_local_context() -> None:
    config = _config()
    redis_client = MagicMock()
    redis_client.get.return_value = None
    redis_client.expire.return_value = True
    redis_client.pipeline.return_value = _SyncPipeline(
        [True, redis.exceptions.ResponseError("repair SET failed")]
    )
    bucket = SyncRedisBucket(
        next(iter(config.quotas)), config, redis_client, key_prefix="test"
    )
    backend = SyncRedisBackend([bucket], redis_client, config, key_prefix="test")
    initial = _SyncPipeline(
        [b"1000.0", None, True, True, _override_payload(bucket, 7.0)]
    )

    with pytest.raises(SyncRedisPipelineResultError, match="partial-state repair"):
        backend._get_capacities_unsafe(pipeline=initial, current_time=1001.0)

    _assert_override_cache_unmodified(bucket)
    _assert_missing_data_context_unmodified(bucket)


async def test_async_direct_bucket_hostile_read_does_not_mutate_override_cache() -> (
    None
):
    config = _config()
    redis_client = MagicMock()
    bucket = RedisBucket(
        next(iter(config.quotas)), config, redis_client, key_prefix="test"
    )
    redis_client.get = AsyncMock(return_value=_override_payload(bucket, 7.0))
    redis_client.expire = AsyncMock(return_value=True)
    redis_client.pipeline.return_value = _AsyncPipeline(
        [
            b"1000.0",
            redis.exceptions.ResponseError("capacity GET failed"),
            True,
            True,
        ]
    )

    with pytest.raises(RedisPipelineResultError):
        await bucket.get_capacity(current_time=1001.0)

    _assert_override_cache_unmodified(bucket)


def test_sync_direct_bucket_hostile_read_does_not_mutate_override_cache() -> None:
    config = _config()
    redis_client = MagicMock()
    bucket = SyncRedisBucket(
        next(iter(config.quotas)), config, redis_client, key_prefix="test"
    )
    redis_client.get.return_value = _override_payload(bucket, 7.0)
    redis_client.expire.return_value = True
    redis_client.pipeline.return_value = _SyncPipeline(
        [
            b"1000.0",
            redis.exceptions.ResponseError("capacity GET failed"),
            True,
            True,
        ]
    )

    with pytest.raises(SyncRedisPipelineResultError):
        bucket.get_capacity(current_time=1001.0)

    _assert_override_cache_unmodified(bucket)


async def test_async_snapshot_hostile_read_does_not_mutate_override_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_server_time(_redis: object) -> float:
        return 1001.0

    config = _config()
    redis_client = MagicMock()
    bucket = RedisBucket(
        next(iter(config.quotas)), config, redis_client, key_prefix="test"
    )
    redis_client.get = AsyncMock(return_value=_override_payload(bucket, 7.0))
    redis_client.expire = AsyncMock(return_value=True)
    redis_client.pipeline.return_value = _AsyncPipeline(
        [
            b"1000.0",
            redis.exceptions.ResponseError("capacity GET failed"),
            True,
            True,
        ]
    )
    backend = RedisBackend([bucket], redis_client, config, key_prefix="test")
    monkeypatch.setattr(_backend, "async_server_time", fake_server_time)

    with pytest.raises(RedisPipelineResultError):
        await backend._snapshot_bucket_state(bucket)

    _assert_override_cache_unmodified(bucket)


def test_sync_snapshot_hostile_read_does_not_mutate_override_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    redis_client = MagicMock()
    bucket = SyncRedisBucket(
        next(iter(config.quotas)), config, redis_client, key_prefix="test"
    )
    redis_client.get.return_value = _override_payload(bucket, 7.0)
    redis_client.expire.return_value = True
    redis_client.pipeline.return_value = _SyncPipeline(
        [
            b"1000.0",
            redis.exceptions.ResponseError("capacity GET failed"),
            True,
            True,
        ]
    )
    backend = SyncRedisBackend([bucket], redis_client, config, key_prefix="test")
    monkeypatch.setattr(_sync_backend, "sync_server_time", lambda _redis: 1001.0)

    with pytest.raises(SyncRedisPipelineResultError):
        backend._snapshot_bucket_state(bucket)

    _assert_override_cache_unmodified(bucket)
