from unittest.mock import MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter


def _config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="requests", limit=100, per_seconds=60)])
    )


def _cluster_client(module: str) -> object:
    redis_cluster_cls = type("RedisCluster", (), {"__module__": module})
    return redis_cluster_cls()


def _builder(redis_client: object) -> MagicMock:
    builder = MagicMock()
    builder._redis = redis_client
    return builder


def test_async_rate_limiter_rejects_redis_cluster_backend_at_init():
    with pytest.raises(
        ValueError,
        match=r"does not support Redis Cluster.*Redis topology support",
    ):
        RateLimiter(_config(), backend=_builder(_cluster_client("redis.cluster")))


def test_async_rate_limiter_rejects_async_redis_cluster_backend_at_init():
    with pytest.raises(
        ValueError,
        match=r"does not support Redis Cluster.*Redis topology support",
    ):
        RateLimiter(
            _config(),
            backend=_builder(_cluster_client("redis.asyncio.cluster")),
        )


def test_sync_rate_limiter_rejects_redis_cluster_backend_at_init():
    with pytest.raises(
        ValueError,
        match=r"does not support Redis Cluster.*Redis topology support",
    ):
        SyncRateLimiter(_config(), backend=_builder(_cluster_client("redis.cluster")))


def test_sync_rate_limiter_rejects_async_redis_cluster_backend_at_init():
    with pytest.raises(
        ValueError,
        match=r"does not support Redis Cluster.*Redis topology support",
    ):
        SyncRateLimiter(
            _config(),
            backend=_builder(_cluster_client("redis.asyncio.cluster")),
        )


def test_non_cluster_backend_still_initializes():
    builder = _builder(object())

    limiter = RateLimiter(_config(), backend=builder)
    sync_limiter = SyncRateLimiter(_config(), backend=builder)

    assert limiter._backend is builder
    assert sync_limiter._backend is builder
