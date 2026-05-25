import time

import pytest

from token_throttle._diagnostic import (
    BackendHealthDiagnostic,
    BackendIntrospectionDiagnostic,
    BucketDiagnostic,
    RedisBackendHealthDiagnostic,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._rate_limiter import RateLimiter


class _MockRedisBackend:
    __module__ = "token_throttle._limiter_backends._redis._backend"

    def __init__(self, cfg: PerModelConfig) -> None:
        self.cfg = cfg
        self.override: float | None = None

    async def await_for_capacity(
        self,
        usage,
        *,
        timeout=None,
        reservation_id=None,
        reservation_lifetime_seconds=None,
    ):
        return time.time()

    async def consume_capacity(
        self,
        usage,
        *,
        reservation_id=None,
        reservation_lifetime_seconds=None,
    ):
        return time.time()

    async def refund_capacity(self, reserved_usage, actual_usage):
        return None

    async def set_max_capacity(self, metric, per_seconds, value):
        self.override = float(value)

    async def introspect(self) -> BackendIntrospectionDiagnostic:
        quota = next(iter(self.cfg.quotas))
        effective = self.override or float(quota.limit)
        return BackendIntrospectionDiagnostic(
            model_family=self.cfg.get_model_family(),
            backend_type="redis",
            as_of_monotonic=time.monotonic(),
            buckets=(
                BucketDiagnostic(
                    model_family=self.cfg.get_model_family(),
                    metric=quota.metric,
                    per_seconds=int(quota.per_seconds),
                    backend_type="redis",
                    current_capacity=3.0,
                    configured_limit=float(quota.limit),
                    runtime_override=self.override,
                    override_source="backend" if self.override else "none",
                    effective_max_capacity=effective,
                    configured_to_effective_gap=effective - float(quota.limit),
                    refill_rate_per_second=effective / int(quota.per_seconds),
                    status="ok",
                    as_of_monotonic=time.monotonic(),
                ),
            ),
            waits=(),
            memory_health=None,
            redis_health=RedisBackendHealthDiagnostic(
                model_family_count=1,
                bucket_count=1,
                connection_pool_class="MockPool",
                connection_pool_max_connections=10,
                connection_pool_in_use_connections=1,
                connection_pool_available_connections=9,
                pool_counts_observed_with_private_attrs=True,
                local_marker_count_estimate=0,
                local_refund_dedup_count_estimate=0,
            ),
            issues=(),
        )


class _MockRedisBuilder:
    __module__ = "token_throttle._limiter_backends._redis._backend"

    def __init__(self) -> None:
        self.backend: _MockRedisBackend | None = None

    def build(self, cfg: PerModelConfig, *, callbacks=None) -> _MockRedisBackend:
        self.backend = _MockRedisBackend(cfg)
        return self.backend


async def test_diagnose_redis_contract_without_live_redis():
    cfg = PerModelConfig(
        model_family="redis-family",
        quotas=UsageQuotas([Quota(metric="tokens", limit=10, per_seconds=60)]),
    )
    limiter = RateLimiter(cfg, backend=_MockRedisBuilder())
    await limiter.acquire_capacity({"tokens": 2}, model="redis-model")
    await limiter.set_max_capacity("redis-model", "tokens", 60, 6)

    diagnostic = await limiter.diagnose()

    assert diagnostic.backend_type == "redis"
    assert diagnostic.bucket_count == 1
    bucket = diagnostic.buckets[0]
    assert bucket.backend_type == "redis"
    assert bucket.current_capacity == pytest.approx(3.0)
    assert bucket.configured_limit == pytest.approx(10.0)
    assert bucket.runtime_override == pytest.approx(6.0)
    assert bucket.override_source == "both"
    assert diagnostic.runtime_overrides[0].source == "both"
    assert diagnostic.backend_health.redis is not None
    assert diagnostic.backend_health.redis.connection_pool_class == "MockPool"
    assert diagnostic.backend_health.redis.local_marker_count_estimate == 1
    assert isinstance(diagnostic.backend_health, BackendHealthDiagnostic)
