"""Integration tests for Redis-specific internals (RedisBucket, RedisBackend locking)."""

import asyncio
import json
import re
import time
import warnings

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import (
    RedisBackend,
    RedisBackendBuilder,
)
from token_throttle._limiter_backends._redis._bucket import (
    CalculatedCapacity,
    RedisBucket,
)
from token_throttle._rate_limiter import RateLimiter


def make_bucket(
    redis_client,
    limit=100,
    per_seconds=60,
    metric="requests",
    model_family="test-model",
):
    quota = Quota(metric=metric, limit=limit, per_seconds=per_seconds)
    config = PerModelConfig(model_family=model_family, quotas=UsageQuotas([quota]))
    return RedisBucket(quota=quota, limit_config=config, redis_client=redis_client)


@pytest.mark.redis
class TestRedisBucketCapacity:
    async def test_get_capacity_on_empty_redis_returns_max_capacity_fresh_start(
        self, redis_client
    ):
        """get_capacity on empty Redis returns max_capacity with is_fresh_start=True."""
        bucket = make_bucket(redis_client, limit=100, per_seconds=60)

        result = await bucket.get_capacity()

        assert result == CalculatedCapacity(amount=100.0, is_fresh_start=True)

    async def test_set_capacity_then_get_capacity_roundtrip(self, redis_client):
        """set_capacity followed by get_capacity returns the stored value."""
        bucket = make_bucket(redis_client, limit=100, per_seconds=60)

        # Pin current_time for both calls to avoid time-based refill drift
        now = time.time()
        await bucket.set_capacity(42.5, current_time=now)
        result = await bucket.get_capacity(current_time=now)

        assert result is not None
        assert result.amount == pytest.approx(42.5)
        assert result.is_fresh_start is False


@pytest.mark.redis
class TestRedisBucketMaxCapacity:
    async def test_set_max_capacity_get_max_capacity_roundtrip(self, redis_client):
        """set_max_capacity then get_max_capacity returns the stored value."""
        bucket = make_bucket(redis_client, limit=100, per_seconds=60)

        await bucket.set_max_capacity(200.0)
        result = await bucket.get_max_capacity()

        assert result == pytest.approx(200.0)

    async def test_get_max_capacity_cache_ttl(self, redis_client):
        """Cached max_capacity is stale after TTL expires, then refreshes from Redis."""
        bucket = make_bucket(redis_client, limit=100, per_seconds=60)

        # Set initial value and populate cache
        await bucket.set_max_capacity(200.0)
        assert await bucket.get_max_capacity() == pytest.approx(200.0)

        # Manually update the Redis key directly, bypassing the cache
        await redis_client.set(
            bucket._max_capacity_key,
            json.dumps(
                {"configured_max_capacity": 100, "override_max_capacity": 999.0}
            ),
        )

        # Cache is still fresh, so we should still see the old value
        assert await bucket.get_max_capacity() == pytest.approx(200.0)

        # Wait for cache TTL to expire (TTL is 1.0 second)
        await asyncio.sleep(1.1)

        # Now the cache should be stale and fetch the new value from Redis
        assert await bucket.get_max_capacity() == pytest.approx(999.0)


@pytest.mark.redis
class TestRedisBucketLocking:
    async def test_lock_acquisition_and_release(self, redis_client):
        """Acquiring a lock blocks a second attempt; releasing allows it."""
        bucket = make_bucket(redis_client, limit=100, per_seconds=60)

        # Acquire the first lock (timeout is the lock TTL / auto-release)
        lock1 = bucket.lock(timeout=10)
        acquired = await lock1.acquire()
        assert acquired is True

        # A second lock attempt should fail (blocking_timeout controls wait)
        lock2 = bucket.lock(timeout=10)
        acquired2 = await lock2.acquire(blocking_timeout=0.1)
        assert acquired2 is False

        # Release the first lock
        await lock1.release()

        # Now a new lock should succeed
        lock3 = bucket.lock(timeout=10)
        acquired3 = await lock3.acquire(blocking_timeout=1)
        assert acquired3 is True
        await lock3.release()


@pytest.mark.redis
class TestRedisBucketKeyFormat:
    async def test_redis_key_format_matches_convention(self, redis_client):
        """All Redis keys match rate_limiting:{family}:{metric}:{per_seconds}:* pattern."""
        family = "test-model"
        metric = "requests"
        per_seconds = 60
        bucket = make_bucket(
            redis_client,
            limit=100,
            per_seconds=per_seconds,
            metric=metric,
            model_family=family,
        )

        expected_prefix = f"rate_limiting:{family}:{metric}:{per_seconds}"
        key_pattern = re.compile(
            rf"^rate_limiting:{re.escape(family)}:{re.escape(metric)}:{per_seconds}:\w+$"
        )

        all_keys = [
            bucket._last_checked_key,
            bucket._capacity_key,
            bucket._lock_key,
            bucket._max_capacity_key,
        ]

        for key in all_keys:
            assert key.startswith(expected_prefix), (
                f"Key {key!r} does not start with {expected_prefix!r}"
            )
            assert key_pattern.match(key), (
                f"Key {key!r} does not match pattern {key_pattern.pattern!r}"
            )


@pytest.mark.redis
class TestRedisBackendSortedLocking:
    async def test_sorted_lock_ordering_for_deadlock_prevention(self, redis_client):
        """RedisBackend sorts buckets by key to prevent deadlocks."""
        # Create quotas that would produce unsorted keys alphabetically
        quota_z = Quota(metric="z_tokens", limit=500, per_seconds=60)
        quota_a = Quota(metric="a_requests", limit=100, per_seconds=60)
        quotas = UsageQuotas([quota_z, quota_a])
        config = PerModelConfig(model_family="test-model", quotas=quotas)

        bucket_z = RedisBucket(
            quota=quota_z,
            limit_config=config,
            redis_client=redis_client,
            key_prefix="test",
        )
        bucket_a = RedisBucket(
            quota=quota_a,
            limit_config=config,
            redis_client=redis_client,
            key_prefix="test",
        )

        backend = RedisBackend(
            buckets=[bucket_z, bucket_a],
            redis=redis_client,
            limit_config=config,
            key_prefix="test",
        )

        # Verify the backend sorted the buckets by full_redis_key
        keys = [b.full_redis_key for b in backend.sorted_buckets]
        assert keys == sorted(keys), f"Buckets are not sorted by key: {keys}"

        # Verify the "a_requests" bucket comes before "z_tokens"
        assert "a_requests" in keys[0]
        assert "z_tokens" in keys[1]


@pytest.mark.redis
class TestRedisCallableConfigRefresh:
    async def test_metric_set_reconfigure_rewrites_surviving_max_capacity(
        self,
        redis_client,
    ):
        phase = 0

        def config_getter(_model_name: str) -> PerModelConfig:
            nonlocal phase
            if phase == 0:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=60)]
                )
            else:
                quotas = UsageQuotas(
                    [
                        Quota(metric="tokens", limit=50, per_seconds=60),
                        Quota(metric="requests", limit=10, per_seconds=60),
                    ]
                )
            return PerModelConfig(quotas=quotas, model_family="callable-refresh-redis")

        limiter = RateLimiter(config_getter, backend=RedisBackendBuilder(redis_client))

        reservation = await limiter.acquire_capacity({"tokens": 1}, "test-model")
        await limiter.refund_capacity({"tokens": 0}, reservation)

        await limiter.set_max_capacity("test-model", "tokens", 60, 80)

        phase = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=r"exceeds bucket max capacity"):
                await limiter.acquire_capacity(
                    {"tokens": 60, "requests": 1},
                    "test-model",
                    timeout=0,
                )

    async def test_metric_set_reconfigure_preserves_runtime_override_when_static_unchanged(
        self,
        redis_client,
    ):
        phase = 0

        def config_getter(_model_name: str) -> PerModelConfig:
            nonlocal phase
            quotas = [Quota(metric="tokens", limit=100, per_seconds=60)]
            if phase == 1:
                quotas.append(Quota(metric="requests", limit=10, per_seconds=60))
            return PerModelConfig(
                quotas=UsageQuotas(quotas),
                model_family="callable-refresh-redis-preserve",
            )

        limiter = RateLimiter(config_getter, backend=RedisBackendBuilder(redis_client))

        await limiter.acquire_capacity({"tokens": 0}, "test-model")
        await limiter.set_max_capacity("test-model", "tokens", 60, 20)

        with pytest.raises(ValueError, match=r"exceeds bucket max capacity"):
            await limiter.acquire_capacity({"tokens": 30}, "test-model", timeout=0)

        phase = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=r"exceeds bucket max capacity"):
                await limiter.acquire_capacity(
                    {"tokens": 30, "requests": 0},
                    "test-model",
                    timeout=0,
                )

    async def test_metric_remove_and_readd_drops_runtime_override(
        self,
        redis_client,
    ):
        phase = 0

        def config_getter(_model_name: str) -> PerModelConfig:
            nonlocal phase
            if phase in {0, 2}:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=3600)]
                )
            else:
                quotas = UsageQuotas(
                    [Quota(metric="requests", limit=10, per_seconds=3600)]
                )
            return PerModelConfig(
                quotas=quotas,
                model_family="callable-refresh-redis-readd",
            )

        limiter = RateLimiter(config_getter, backend=RedisBackendBuilder(redis_client))

        await limiter.acquire_capacity({"tokens": 0}, "test-model")
        await limiter.set_max_capacity("test-model", "tokens", 3600, 20)

        phase = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await limiter.acquire_capacity({"requests": 0}, "test-model")

        phase = 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reservation = await limiter.acquire_capacity(
                {"tokens": 30},
                "test-model",
                timeout=0,
            )

        assert reservation.usage["tokens"] == 30
