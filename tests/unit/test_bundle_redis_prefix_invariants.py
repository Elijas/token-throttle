"""Redis backend/bucket namespace invariant coverage."""

import pytest

pytest.importorskip("redis", reason="redis package not installed")

import redis as _sync_redis
import redis.asyncio as _async_redis

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import (
    RedisBackend,
    RedisBackendBuilder,
)
from token_throttle._limiter_backends._redis._bucket import RedisBucket
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackend,
    SyncRedisBackendBuilder,
)
from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket


class _RedisClient:
    def __init__(self, connection_pool: object | None = None) -> None:
        self.connection_pool = connection_pool


class _AsyncRedisClient(_RedisClient, _async_redis.Redis):
    pass


class _SyncRedisClient(_RedisClient, _sync_redis.Redis):
    pass


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="gpt-4",
        quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=60)]),
    )


def _quota(cfg: PerModelConfig) -> Quota:
    return next(iter(cfg.quotas))


def _async_bucket(
    cfg: PerModelConfig,
    redis_client: _RedisClient,
    *,
    key_prefix: str,
) -> RedisBucket:
    return RedisBucket(
        quota=_quota(cfg),
        limit_config=cfg,
        redis_client=redis_client,
        key_prefix=key_prefix,
    )


def _sync_bucket(
    cfg: PerModelConfig,
    redis_client: _RedisClient,
    *,
    key_prefix: str,
) -> SyncRedisBucket:
    return SyncRedisBucket(
        quota=_quota(cfg),
        limit_config=cfg,
        redis_client=redis_client,
        key_prefix=key_prefix,
    )


def test_async_backend_rejects_mixed_prefix_buckets() -> None:
    cfg = _config()
    redis_client = _RedisClient()
    bucket = _async_bucket(cfg, redis_client, key_prefix="tenant-b")

    with pytest.raises(ValueError, match="key_prefix must match"):
        RedisBackend(
            buckets=[bucket],
            redis=redis_client,
            limit_config=cfg,
            key_prefix="tenant-a",
        )


def test_sync_backend_rejects_mixed_prefix_buckets() -> None:
    cfg = _config()
    redis_client = _RedisClient()
    bucket = _sync_bucket(cfg, redis_client, key_prefix="tenant-b")

    with pytest.raises(ValueError, match="key_prefix must match"):
        SyncRedisBackend(
            buckets=[bucket],
            redis=redis_client,
            limit_config=cfg,
            key_prefix="tenant-a",
        )


def test_async_stock_builder_satisfies_prefix_invariant() -> None:
    cfg = _config()
    redis_client = _AsyncRedisClient()

    backend = RedisBackendBuilder(redis_client, key_prefix="tenant-a").build(cfg)

    assert [bucket.key_prefix for bucket in backend.sorted_buckets] == ["tenant-a"]


def test_sync_stock_builder_satisfies_prefix_invariant() -> None:
    cfg = _config()
    redis_client = _SyncRedisClient()

    backend = SyncRedisBackendBuilder(redis_client, key_prefix="tenant-a").build(cfg)

    assert [bucket.key_prefix for bucket in backend.sorted_buckets] == ["tenant-a"]


def test_async_add_bucket_enforces_prefix_invariant() -> None:
    cfg = _config()
    redis_client = _RedisClient()
    backend = RedisBackend(
        buckets=[],
        redis=redis_client,
        limit_config=cfg,
        key_prefix="tenant-a",
    )
    matching = _async_bucket(cfg, redis_client, key_prefix="tenant-a")
    mismatching = _async_bucket(cfg, redis_client, key_prefix="tenant-b")

    backend.add_bucket(matching)

    assert backend.sorted_buckets == [matching]
    with pytest.raises(ValueError, match="key_prefix must match"):
        backend.add_bucket(mismatching)
    assert backend.sorted_buckets == [matching]


def test_sync_add_bucket_enforces_prefix_invariant() -> None:
    cfg = _config()
    redis_client = _RedisClient()
    backend = SyncRedisBackend(
        buckets=[],
        redis=redis_client,
        limit_config=cfg,
        key_prefix="tenant-a",
    )
    matching = _sync_bucket(cfg, redis_client, key_prefix="tenant-a")
    mismatching = _sync_bucket(cfg, redis_client, key_prefix="tenant-b")

    backend.add_bucket(matching)

    assert backend.sorted_buckets == [matching]
    with pytest.raises(ValueError, match="key_prefix must match"):
        backend.add_bucket(mismatching)
    assert backend.sorted_buckets == [matching]


def test_async_backend_accepts_distinct_clients_sharing_connection_pool() -> None:
    cfg = _config()
    pool = object()
    bucket_client = _RedisClient(pool)
    backend_client = _RedisClient(pool)
    bucket = _async_bucket(cfg, bucket_client, key_prefix="tenant-a")

    backend = RedisBackend(
        buckets=[bucket],
        redis=backend_client,
        limit_config=cfg,
        key_prefix="tenant-a",
    )

    assert backend.sorted_buckets == [bucket]


def test_async_backend_rejects_distinct_clients_with_distinct_connection_pools() -> (
    None
):
    cfg = _config()
    bucket_client = _RedisClient(object())
    backend_client = _RedisClient(object())
    bucket = _async_bucket(cfg, bucket_client, key_prefix="tenant-a")

    with pytest.raises(ValueError, match="redis client"):
        RedisBackend(
            buckets=[bucket],
            redis=backend_client,
            limit_config=cfg,
            key_prefix="tenant-a",
        )


def test_sync_backend_accepts_distinct_clients_sharing_connection_pool() -> None:
    cfg = _config()
    pool = object()
    bucket_client = _RedisClient(pool)
    backend_client = _RedisClient(pool)
    bucket = _sync_bucket(cfg, bucket_client, key_prefix="tenant-a")

    backend = SyncRedisBackend(
        buckets=[bucket],
        redis=backend_client,
        limit_config=cfg,
        key_prefix="tenant-a",
    )

    assert backend.sorted_buckets == [bucket]


def test_sync_backend_rejects_distinct_clients_with_distinct_connection_pools() -> None:
    cfg = _config()
    bucket_client = _RedisClient(object())
    backend_client = _RedisClient(object())
    bucket = _sync_bucket(cfg, bucket_client, key_prefix="tenant-a")

    with pytest.raises(ValueError, match="redis client"):
        SyncRedisBackend(
            buckets=[bucket],
            redis=backend_client,
            limit_config=cfg,
            key_prefix="tenant-a",
        )
