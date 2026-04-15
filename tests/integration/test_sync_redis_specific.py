"""
Sync Redis-specific integration tests.

Tests features that only apply to the Redis backend (e.g., dynamic max_capacity
via Redis keys).
"""

import time
import warnings

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)
from token_throttle._sync_rate_limiter import SyncRateLimiter


def test_dynamic_max_capacity_change(sync_redis_client):
    """
    Changing max_capacity via Redis should take effect after cache TTL.

    Mirrors the async test_dynamic_max_capacity_change in test_end_to_end.py.
    """
    config = PerModelConfig(
        model_family="dynamic_sync",
        quotas=UsageQuotas(
            [
                Quota(metric="requests", limit=10, per_seconds=1),
                Quota(metric="tokens", limit=1000, per_seconds=1),
            ],
        ),
    )
    builder = SyncRedisBackendBuilder(sync_redis_client)
    backend = builder.build(config)

    # Initial acquire to populate cache.
    backend.wait_for_capacity(frozen_usage({"requests": 1, "tokens": 10}))

    # Reduce max_capacity for requests from 10 to 3 via Redis directly.
    sync_redis_client.set(
        "rate_limiting:dynamic_sync:requests:1:max_capacity_override",
        3,
    )

    # Wait for cache TTL to expire (1 second cache + margin).
    time.sleep(1.2)

    # Request exactly 3 — should succeed since refill capped at new max (3).
    start = time.monotonic()
    backend.wait_for_capacity(frozen_usage({"requests": 3, "tokens": 10}))
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, "3 requests should succeed with new max_capacity=3"

    # Now requesting 1 more should block (just consumed all 3).
    start = time.monotonic()
    backend.wait_for_capacity(frozen_usage({"requests": 1, "tokens": 10}))
    elapsed = time.monotonic() - start
    assert elapsed >= 0.08, (
        f"Expected wait after exhausting dynamic capacity, got {elapsed:.3f} s"
    )


@pytest.mark.redis
def test_metric_set_reconfigure_rewrites_surviving_max_capacity(sync_redis_client):
    phase = 0

    def config_getter(_model_name: str) -> PerModelConfig:
        nonlocal phase
        if phase == 0:
            quotas = UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=60)])
        else:
            quotas = UsageQuotas(
                [
                    Quota(metric="tokens", limit=50, per_seconds=60),
                    Quota(metric="requests", limit=10, per_seconds=60),
                ]
            )
        return PerModelConfig(quotas=quotas, model_family="callable-refresh-sync-redis")

    limiter = SyncRateLimiter(
        config_getter,
        backend=SyncRedisBackendBuilder(sync_redis_client),
    )

    reservation = limiter.acquire_capacity({"tokens": 1}, "test-model")
    limiter.refund_capacity({"tokens": 0}, reservation)

    limiter.set_max_capacity("test-model", "tokens", 60, 80)

    phase = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=r"exceeds bucket max capacity"):
            limiter.acquire_capacity(
                {"tokens": 60, "requests": 1},
                "test-model",
                timeout=0,
            )


@pytest.mark.redis
def test_metric_set_reconfigure_preserves_runtime_override_when_static_unchanged(
    sync_redis_client,
):
    phase = 0

    def config_getter(_model_name: str) -> PerModelConfig:
        nonlocal phase
        quotas = [Quota(metric="tokens", limit=100, per_seconds=60)]
        if phase == 1:
            quotas.append(Quota(metric="requests", limit=10, per_seconds=60))
        return PerModelConfig(
            quotas=UsageQuotas(quotas),
            model_family="callable-refresh-sync-redis-preserve",
        )

    limiter = SyncRateLimiter(
        config_getter,
        backend=SyncRedisBackendBuilder(sync_redis_client),
    )

    limiter.acquire_capacity({"tokens": 0}, "test-model")
    limiter.set_max_capacity("test-model", "tokens", 60, 20)

    with pytest.raises(ValueError, match=r"exceeds bucket max capacity"):
        limiter.acquire_capacity({"tokens": 30}, "test-model", timeout=0)

    phase = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=r"exceeds bucket max capacity"):
            limiter.acquire_capacity(
                {"tokens": 30, "requests": 0},
                "test-model",
                timeout=0,
            )


@pytest.mark.redis
def test_metric_remove_and_readd_drops_runtime_override(sync_redis_client):
    phase = 0

    def config_getter(_model_name: str) -> PerModelConfig:
        nonlocal phase
        if phase in {0, 2}:
            quotas = UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=3600)])
        else:
            quotas = UsageQuotas([Quota(metric="requests", limit=10, per_seconds=3600)])
        return PerModelConfig(
            quotas=quotas,
            model_family="callable-refresh-sync-redis-readd",
        )

    limiter = SyncRateLimiter(
        config_getter,
        backend=SyncRedisBackendBuilder(sync_redis_client),
    )

    limiter.acquire_capacity({"tokens": 0}, "test-model")
    limiter.set_max_capacity("test-model", "tokens", 3600, 20)

    phase = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        limiter.acquire_capacity({"requests": 0}, "test-model")

    phase = 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reservation = limiter.acquire_capacity(
            {"tokens": 30},
            "test-model",
            timeout=0,
        )

    assert reservation.usage["tokens"] == 30
