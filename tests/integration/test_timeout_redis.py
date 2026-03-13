"""Integration tests for the timeout parameter on await_for_capacity / wait_for_capacity.

Parameterized across all backends (memory + redis) via the backend_builder
and sync_backend_builder fixtures.

Tests cover timeout=0 (try-acquire), timeout=N (bounded wait), and
timeout=None (default).  These tests will fail until timeout is implemented
(TDD red phase).
"""

import contextlib
import time

import pytest
import redis as sync_redis
import redis.asyncio as redis

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder
from token_throttle._limiter_backends._redis._bucket import RedisBucket
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)
from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket


def _make_config(
    *, limit: float = 100, per_seconds: int = 3600, metric: str = "requests"
) -> PerModelConfig:
    return PerModelConfig(
        model_family="test",
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
    )


# ---------------------------------------------------------------------------
# Async -- timeout=0 (try-acquire)
# ---------------------------------------------------------------------------


async def test_timeout_zero_fails_when_no_capacity(backend_builder):
    """timeout=0 should raise TimeoutError immediately when capacity is exhausted."""
    config = _make_config(limit=10, per_seconds=3600)
    backend = backend_builder.build(config)

    # Exhaust all capacity
    await backend.await_for_capacity(frozen_usage({"requests": 10}))

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        await backend.await_for_capacity(frozen_usage({"requests": 1}), timeout=0)
    elapsed = time.monotonic() - start

    # Should fail nearly instantly
    assert elapsed < 0.5, f"timeout=0 should fail immediately, took {elapsed:.2f}s"


async def test_timeout_zero_succeeds_with_capacity(backend_builder):
    """timeout=0 should succeed immediately when capacity is available."""
    config = _make_config(limit=100, per_seconds=3600)
    backend = backend_builder.build(config)

    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1}), timeout=0)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5


# ---------------------------------------------------------------------------
# Async -- timeout=N (bounded wait)
# ---------------------------------------------------------------------------


async def test_timeout_n_raises_after_bounded_wait(backend_builder):
    """timeout=N should raise TimeoutError after ~N seconds."""
    # Very slow refill: 100 tokens over 3600s = 0.028/s
    config = _make_config(limit=100, per_seconds=3600)
    backend = backend_builder.build(config)

    # Exhaust all capacity
    await backend.await_for_capacity(frozen_usage({"requests": 100}))

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        await backend.await_for_capacity(frozen_usage({"requests": 50}), timeout=0.5)
    elapsed = time.monotonic() - start

    # Should have waited approximately 0.5s (bounded, not indefinite)
    assert elapsed >= 0.4, f"Expected wait of ~0.5s, got {elapsed:.2f}s"
    assert elapsed < 2.0, (
        f"Should not wait much longer than timeout, got {elapsed:.2f}s"
    )


# ---------------------------------------------------------------------------
# Async -- timeout=None (default behavior unchanged)
# ---------------------------------------------------------------------------


async def test_timeout_none_blocks_until_available(backend_builder):
    """timeout=None should block until capacity is available (same as default)."""
    # Fast refill: 100/s
    config = _make_config(limit=100, per_seconds=1)
    backend = backend_builder.build(config)

    # Exhaust all capacity
    await backend.consume_capacity(frozen_usage({"requests": 100}))

    # timeout=None should block until refill, then succeed
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1}), timeout=None)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"timeout=None should eventually succeed, took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Sync -- timeout=0 (try-acquire)
# ---------------------------------------------------------------------------


def test_sync_timeout_zero_fails_when_no_capacity(sync_backend_builder):
    """timeout=0 should raise TimeoutError immediately (sync)."""
    config = _make_config(limit=10, per_seconds=3600)
    backend = sync_backend_builder.build(config)

    # Exhaust all capacity
    backend.wait_for_capacity(frozen_usage({"requests": 10}))

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        backend.wait_for_capacity(frozen_usage({"requests": 1}), timeout=0)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, f"timeout=0 should fail immediately, took {elapsed:.2f}s"


def test_sync_timeout_zero_succeeds_with_capacity(sync_backend_builder):
    """timeout=0 should succeed immediately when capacity is available (sync)."""
    config = _make_config(limit=100, per_seconds=3600)
    backend = sync_backend_builder.build(config)

    start = time.monotonic()
    backend.wait_for_capacity(frozen_usage({"requests": 1}), timeout=0)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5


# ---------------------------------------------------------------------------
# Sync -- timeout=N (bounded wait)
# ---------------------------------------------------------------------------


def test_sync_timeout_n_raises_after_bounded_wait(sync_backend_builder):
    """timeout=N should raise TimeoutError after ~N seconds (sync)."""
    config = _make_config(limit=100, per_seconds=3600)
    backend = sync_backend_builder.build(config)

    # Exhaust all capacity
    backend.wait_for_capacity(frozen_usage({"requests": 100}))

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        backend.wait_for_capacity(frozen_usage({"requests": 50}), timeout=0.5)
    elapsed = time.monotonic() - start

    assert elapsed >= 0.4, f"Expected wait of ~0.5s, got {elapsed:.2f}s"
    assert elapsed < 2.0, (
        f"Should not wait much longer than timeout, got {elapsed:.2f}s"
    )


# ---------------------------------------------------------------------------
# Sync -- timeout=None (default behavior unchanged)
# ---------------------------------------------------------------------------


def test_sync_timeout_none_blocks_until_available(sync_backend_builder):
    """timeout=None should block until capacity is available (sync)."""
    # Fast refill: 100/s
    config = _make_config(limit=100, per_seconds=1)
    backend = sync_backend_builder.build(config)

    # Exhaust all capacity
    backend.consume_capacity(frozen_usage({"requests": 100}))

    # timeout=None should block until refill, then succeed
    start = time.monotonic()
    backend.wait_for_capacity(frozen_usage({"requests": 1}), timeout=None)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"timeout=None should eventually succeed, took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Redis lock acquisition must also respect caller timeout
# ---------------------------------------------------------------------------


async def test_async_timeout_zero_includes_redis_lock_wait(
    redis_url: str,
    redis_client,
):
    """timeout=0 must fail fast even if another worker holds the Redis lock."""
    config = _make_config(limit=10, per_seconds=60)
    backend = RedisBackendBuilder(redis_client).build(config)

    lock_client = redis.from_url(redis_url)
    lock = None
    try:
        bucket = RedisBucket(
            quota=Quota(metric="requests", limit=10, per_seconds=60),
            limit_config=config,
            redis_client=lock_client,
        )
        lock = bucket.lock(timeout=1.2)
        assert await lock.acquire() is True

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await backend.await_for_capacity(frozen_usage({"requests": 1}), timeout=0)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, (
            f"timeout=0 should not wait on Redis lock contention, took {elapsed:.2f}s"
        )
    finally:
        if lock is not None:
            with contextlib.suppress(Exception):
                await lock.release()
        await lock_client.aclose()


def test_sync_timeout_zero_includes_redis_lock_wait(
    redis_url: str,
    sync_redis_client,
):
    """Sync timeout=0 must fail fast even if another worker holds the Redis lock."""
    config = _make_config(limit=10, per_seconds=60)
    backend = SyncRedisBackendBuilder(sync_redis_client).build(config)

    lock_client = sync_redis.from_url(redis_url)
    lock = None
    try:
        bucket = SyncRedisBucket(
            quota=Quota(metric="requests", limit=10, per_seconds=60),
            limit_config=config,
            redis_client=lock_client,
        )
        lock = bucket.lock(timeout=1.2)
        assert lock.acquire() is True

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            backend.wait_for_capacity(frozen_usage({"requests": 1}), timeout=0)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, (
            f"timeout=0 should not wait on Redis lock contention, took {elapsed:.2f}s"
        )
    finally:
        if lock is not None:
            with contextlib.suppress(Exception):
                lock.release()
        lock_client.close()
