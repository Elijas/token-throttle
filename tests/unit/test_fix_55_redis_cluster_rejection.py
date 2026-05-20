import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter


def _config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="requests", limit=100, per_seconds=60)])
    )


def _cluster_client(module: str) -> object:
    redis_cluster_cls = type("RedisCluster", (), {"__module__": module})
    return redis_cluster_cls()


def _async_builtin_cluster_client() -> object:
    redis_async = pytest.importorskip("redis.asyncio")
    redis_cluster_cls = type(
        "RedisCluster",
        (redis_async.Redis,),
        {"__module__": "redis.asyncio.cluster"},
    )
    return redis_cluster_cls()


def _sync_builtin_cluster_client() -> object:
    redis = pytest.importorskip("redis")
    redis_cluster_cls = type(
        "RedisCluster",
        (redis.Redis,),
        {"__module__": "redis.cluster"},
    )
    return redis_cluster_cls()


class _CustomAsyncBuilder:
    def __init__(self, redis_client: object) -> None:
        self._redis = redis_client
        self._delegate = MemoryBackendBuilder()

    def build(self, cfg: PerModelConfig, *, callbacks=None):
        return self._delegate.build(cfg, callbacks=callbacks)


class _CustomSyncBuilder:
    def __init__(self, redis_client: object) -> None:
        self._redis = redis_client
        self._delegate = SyncMemoryBackendBuilder()

    def build(self, cfg: PerModelConfig, *, callbacks=None):
        return self._delegate.build(cfg, callbacks=callbacks)


def test_async_rate_limiter_rejects_builtin_redis_cluster_builder_at_init():
    from token_throttle._limiter_backends._redis._backend import (  # noqa: PLC0415
        RedisBackendBuilder,
    )

    builder = RedisBackendBuilder(_async_builtin_cluster_client(), key_prefix="test")

    with pytest.raises(
        ValueError,
        match=r"does not support Redis Cluster.*Redis topology support",
    ):
        RateLimiter(_config(), backend=builder)


def test_async_rate_limiter_accepts_custom_builder_with_private_cluster_client():
    builder = _CustomAsyncBuilder(_cluster_client("redis.asyncio.cluster"))

    limiter = RateLimiter(_config(), backend=builder)

    assert limiter._backend is builder


def test_sync_rate_limiter_rejects_builtin_redis_cluster_builder_at_init():
    from token_throttle._limiter_backends._redis._sync_backend import (  # noqa: PLC0415
        SyncRedisBackendBuilder,
    )

    builder = SyncRedisBackendBuilder(_sync_builtin_cluster_client(), key_prefix="test")

    with pytest.raises(
        ValueError,
        match=r"does not support Redis Cluster.*Redis topology support",
    ):
        SyncRateLimiter(_config(), backend=builder)


def test_sync_rate_limiter_accepts_custom_builder_with_private_cluster_client():
    builder = _CustomSyncBuilder(_cluster_client("redis.cluster"))

    sync_limiter = SyncRateLimiter(_config(), backend=builder)

    assert sync_limiter._backend is builder


def test_custom_builders_with_non_cluster_private_client_still_initialize():
    async_builder = _CustomAsyncBuilder(object())
    sync_builder = _CustomSyncBuilder(object())

    limiter = RateLimiter(_config(), backend=async_builder)
    sync_limiter = SyncRateLimiter(_config(), backend=sync_builder)

    assert limiter._backend is async_builder
    assert sync_limiter._backend is sync_builder
