"""
Sync Redis-specific integration tests.

Tests features that only apply to the Redis backend (e.g., dynamic max_capacity
via Redis keys).
"""

import time

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)


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
        "rate_limiting:dynamic_sync:requests:1:max_capacity",
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
