import pytest

pytest.importorskip("redis")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="redis-diag-family",
        quotas=UsageQuotas([Quota(metric="tokens", limit=20, per_seconds=60)]),
    )


@pytest.mark.redis
async def test_async_redis_diagnose_reads_live_bucket_state(redis_client):
    limiter = RateLimiter(
        _config(),
        backend=RedisBackendBuilder(redis_client, key_prefix="diag-async"),
    )
    reservation = await limiter.acquire_capacity({"tokens": 5}, model="redis-model")
    await limiter.set_max_capacity("redis-model", "tokens", 60, 12)

    diagnostic = await limiter.diagnose()

    assert diagnostic.backend_type == "redis"
    assert diagnostic.bucket_count == 1
    bucket = diagnostic.buckets[0]
    assert bucket.model_family == "redis-diag-family"
    assert bucket.metric == "tokens"
    assert bucket.current_capacity is not None
    assert 0 <= bucket.current_capacity <= 12
    assert bucket.configured_limit == pytest.approx(20.0)
    assert bucket.runtime_override == pytest.approx(12.0)
    assert bucket.override_source == "both"
    assert diagnostic.backend_health.redis is not None
    assert diagnostic.backend_health.redis.bucket_count == 1
    assert diagnostic.backend_health.redis.local_marker_count_estimate == 1
    assert diagnostic.reservations.in_flight_count == 1
    assert any(
        reservation.reservation_id in group.reservation_ids
        for group in diagnostic.reservations.groups
    )


@pytest.mark.redis
def test_sync_redis_diagnose_reads_live_bucket_state(sync_redis_client):
    limiter = SyncRateLimiter(
        _config(),
        backend=SyncRedisBackendBuilder(sync_redis_client, key_prefix="diag-sync"),
    )
    reservation = limiter.acquire_capacity({"tokens": 4}, model="redis-model")
    limiter.set_max_capacity("redis-model", "tokens", 60, 10)

    diagnostic = limiter.diagnose()

    assert diagnostic.backend_type == "redis"
    assert diagnostic.limiter_type == "sync"
    bucket = diagnostic.buckets[0]
    assert bucket.current_capacity is not None
    assert 0 <= bucket.current_capacity <= 10
    assert bucket.configured_limit == pytest.approx(20.0)
    assert bucket.runtime_override == pytest.approx(10.0)
    assert bucket.override_source == "both"
    assert diagnostic.backend_health.redis is not None
    assert diagnostic.backend_health.redis.local_marker_count_estimate == 1
    assert any(
        reservation.reservation_id in group.reservation_ids
        for group in diagnostic.reservations.groups
    )
