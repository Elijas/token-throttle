"""Integration tests for apply_configured_max_capacity on Redis backends (F18.04).

Exercises the full lock → snapshot → clear-override → set-configured-max
pipeline on both async and sync Redis backends. The unit tests in
test_callable_config_refresh.py cover the flow against a mocked backend;
these tests verify the Redis pipeline/lock/snapshot path end-to-end.
"""

import pytest
from frozendict import frozendict

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)


def _make_config(
    *,
    limit: float = 100,
    per_seconds: int = 3600,
    metric: str = "requests",
    model_family: str = "test-apply-max",
) -> PerModelConfig:
    return PerModelConfig(
        model_family=model_family,
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
    )


@pytest.mark.redis
class TestAsyncApplyConfiguredMaxCapacity:
    async def test_updates_bucket_max(self, redis_client):
        """apply_configured_max_capacity updates the bucket's configured max."""
        config = _make_config()
        backend = RedisBackendBuilder(redis_client).build(config)

        assert backend.sorted_buckets[0].max_capacity == 100.0

        await backend.apply_configured_max_capacity("requests", 3600, 200.0)

        assert backend.sorted_buckets[0].max_capacity == 200.0

    async def test_clears_runtime_override(self, redis_client):
        """apply_configured_max_capacity clears any existing runtime override."""
        config = _make_config(model_family="test-apply-clear")
        backend = RedisBackendBuilder(redis_client).build(config)

        await backend.set_max_capacity("requests", 3600, 50.0)
        assert backend.sorted_buckets[0].max_capacity == 50.0

        await backend.apply_configured_max_capacity("requests", 3600, 200.0)

        assert backend.sorted_buckets[0].max_capacity == 200.0
        override_key = backend.sorted_buckets[0]._max_capacity_key
        assert await redis_client.get(override_key) is None

    async def test_capacity_reflects_new_max(self, redis_client):
        """After apply_configured_max_capacity, acquirable capacity reflects the new max."""
        config = _make_config(model_family="test-apply-cap")
        backend = RedisBackendBuilder(redis_client).build(config)

        await backend.apply_configured_max_capacity("requests", 3600, 500.0)

        await backend.await_for_capacity(frozendict({"requests": 400.0}))

    async def test_nonexistent_bucket_raises(self, redis_client):
        """apply_configured_max_capacity raises ValueError for unknown metric."""
        config = _make_config(model_family="test-apply-missing")
        backend = RedisBackendBuilder(redis_client).build(config)

        with pytest.raises(ValueError, match="not found"):
            await backend.apply_configured_max_capacity("nonexistent", 3600, 100.0)


@pytest.mark.redis
class TestSyncApplyConfiguredMaxCapacity:
    def test_updates_bucket_max(self, sync_redis_client):
        """Sync apply_configured_max_capacity updates the bucket's configured max."""
        config = _make_config(model_family="test-apply-max-sync")
        backend = SyncRedisBackendBuilder(sync_redis_client).build(config)

        assert backend.sorted_buckets[0].max_capacity == 100.0

        backend.apply_configured_max_capacity("requests", 3600, 200.0)

        assert backend.sorted_buckets[0].max_capacity == 200.0

    def test_clears_runtime_override(self, sync_redis_client):
        """Sync apply_configured_max_capacity clears runtime overrides."""
        config = _make_config(model_family="test-apply-clear-sync")
        backend = SyncRedisBackendBuilder(sync_redis_client).build(config)

        backend.set_max_capacity("requests", 3600, 50.0)
        assert backend.sorted_buckets[0].max_capacity == 50.0

        backend.apply_configured_max_capacity("requests", 3600, 200.0)

        assert backend.sorted_buckets[0].max_capacity == 200.0
        override_key = backend.sorted_buckets[0]._max_capacity_key
        assert sync_redis_client.get(override_key) is None

    def test_capacity_reflects_new_max(self, sync_redis_client):
        """Sync: acquirable capacity reflects the new max after apply."""
        config = _make_config(model_family="test-apply-cap-sync")
        backend = SyncRedisBackendBuilder(sync_redis_client).build(config)

        backend.apply_configured_max_capacity("requests", 3600, 500.0)

        backend.wait_for_capacity(frozendict({"requests": 400.0}))

    def test_nonexistent_bucket_raises(self, sync_redis_client):
        """Sync apply_configured_max_capacity raises ValueError for unknown metric."""
        config = _make_config(model_family="test-apply-missing-sync")
        backend = SyncRedisBackendBuilder(sync_redis_client).build(config)

        with pytest.raises(ValueError, match="not found"):
            backend.apply_configured_max_capacity("nonexistent", 3600, 100.0)
