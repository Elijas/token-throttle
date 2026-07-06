"""The caller deadline — not lock_blocking_timeout_seconds — bounds the wait.

A waiter with a caller ``timeout`` that hits per-bucket lock contention must
keep retrying the acquire until *its own* deadline expires; the per-attempt
``lock_blocking_timeout_seconds`` cap must not short-circuit it into an early
``TimeoutError``. ``timeout=0`` must still fail fast under contention.

These are unit-style tests that talk to a real local Redis (they skip when one
is unavailable, e.g. the Redis-less unit CI lanes). Each test uses a unique
``key_prefix`` and deletes only its own keys — never ``FLUSHDB``. The lock is
held externally by ``SET``-ing the per-bucket lock key with a ``PX`` expiry, so
no cross-thread lock ownership is involved. ``lock_blocking_timeout_seconds`` is
set well below the hold so a single acquire attempt cannot span the hold: only
retry-until-deadline can succeed, which is exactly the regressed behavior.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("redis", reason="redis package not installed")

import time

import redis as _sync_redis
import redis.asyncio as _async_redis
import redis.exceptions

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)

# Per-attempt lock cap kept far below the external hold so no single acquire
# attempt can outlast the hold — only retry-until-deadline reaches capacity.
_LOCK_BLOCKING_TIMEOUT = 0.25
_HOLD_SECONDS = 1.5


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="deadline-fam",
        quotas=UsageQuotas([Quota(metric="requests", limit=100, per_seconds=3600)]),
    )


@pytest.fixture
def redis_url(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--redis-url")


def _unique_prefix() -> str:
    return f"fixlane-redis-deadline-{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@pytest.mark.redis
async def test_async_await_retries_until_deadline_under_held_lock(
    redis_url: str,
) -> None:
    client = _async_redis.from_url(redis_url)
    try:
        await client.ping()
    except redis.exceptions.RedisError as exc:  # pragma: no cover - env dependent
        await client.aclose()
        pytest.skip(f"Redis unavailable at {redis_url}: {exc}")

    prefix = _unique_prefix()
    try:
        backend = RedisBackendBuilder(
            client,
            key_prefix=prefix,
            lock_blocking_timeout_seconds=_LOCK_BLOCKING_TIMEOUT,
        ).build(_config())
        lock_key = backend.sorted_buckets[0]._lock_key
        # Hold the per-bucket lock externally for longer than one acquire attempt.
        await client.set(lock_key, b"held", px=int(_HOLD_SECONDS * 1000))

        start = time.monotonic()
        result = await backend.await_for_capacity(
            frozen_usage({"requests": 1}), timeout=8
        )
        elapsed = time.monotonic() - start

        assert result is not None
        # Waited out the held lock instead of giving up at the per-attempt cap.
        assert elapsed >= _HOLD_SECONDS - 0.7, (
            f"succeeded too early ({elapsed:.2f}s) — per-attempt cap leaked through"
        )
        assert elapsed < 8.0, f"should succeed before the deadline ({elapsed:.2f}s)"
    finally:
        await _cleanup_async(client, prefix)
        await client.aclose()


@pytest.mark.redis
async def test_async_await_timeout_zero_fails_fast_under_held_lock(
    redis_url: str,
) -> None:
    client = _async_redis.from_url(redis_url)
    try:
        await client.ping()
    except redis.exceptions.RedisError as exc:  # pragma: no cover - env dependent
        await client.aclose()
        pytest.skip(f"Redis unavailable at {redis_url}: {exc}")

    prefix = _unique_prefix()
    try:
        backend = RedisBackendBuilder(
            client,
            key_prefix=prefix,
            lock_blocking_timeout_seconds=_LOCK_BLOCKING_TIMEOUT,
        ).build(_config())
        lock_key = backend.sorted_buckets[0]._lock_key
        await client.set(lock_key, b"held", px=int(_HOLD_SECONDS * 1000))

        start = time.monotonic()
        with pytest.raises(TimeoutError) as excinfo:
            await backend.await_for_capacity(frozen_usage({"requests": 1}), timeout=0)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, f"timeout=0 should fail fast ({elapsed:.2f}s)"
        # Error names lock contention as the cause, not a bottleneck=None capacity.
        assert "lock" in str(excinfo.value).lower()
        assert isinstance(excinfo.value.__cause__, redis.exceptions.LockError)
    finally:
        await _cleanup_async(client, prefix)
        await client.aclose()


async def _cleanup_async(client: _async_redis.Redis, prefix: str) -> None:
    keys = [key async for key in client.scan_iter(match=f"{prefix}*")]
    if keys:
        await client.delete(*keys)


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


@pytest.mark.redis
def test_sync_wait_retries_until_deadline_under_held_lock(redis_url: str) -> None:
    client = _sync_redis.from_url(redis_url)
    try:
        client.ping()
    except redis.exceptions.RedisError as exc:  # pragma: no cover - env dependent
        client.close()
        pytest.skip(f"Redis unavailable at {redis_url}: {exc}")

    prefix = _unique_prefix()
    try:
        backend = SyncRedisBackendBuilder(
            client,
            key_prefix=prefix,
            lock_blocking_timeout_seconds=_LOCK_BLOCKING_TIMEOUT,
        ).build(_config())
        lock_key = backend.sorted_buckets[0]._lock_key
        client.set(lock_key, b"held", px=int(_HOLD_SECONDS * 1000))

        start = time.monotonic()
        result = backend.wait_for_capacity(frozen_usage({"requests": 1}), timeout=8)
        elapsed = time.monotonic() - start

        assert result is not None
        assert elapsed >= _HOLD_SECONDS - 0.7, (
            f"succeeded too early ({elapsed:.2f}s) — per-attempt cap leaked through"
        )
        assert elapsed < 8.0, f"should succeed before the deadline ({elapsed:.2f}s)"
    finally:
        _cleanup_sync(client, prefix)
        client.close()


@pytest.mark.redis
def test_sync_wait_timeout_zero_fails_fast_under_held_lock(redis_url: str) -> None:
    client = _sync_redis.from_url(redis_url)
    try:
        client.ping()
    except redis.exceptions.RedisError as exc:  # pragma: no cover - env dependent
        client.close()
        pytest.skip(f"Redis unavailable at {redis_url}: {exc}")

    prefix = _unique_prefix()
    try:
        backend = SyncRedisBackendBuilder(
            client,
            key_prefix=prefix,
            lock_blocking_timeout_seconds=_LOCK_BLOCKING_TIMEOUT,
        ).build(_config())
        lock_key = backend.sorted_buckets[0]._lock_key
        client.set(lock_key, b"held", px=int(_HOLD_SECONDS * 1000))

        start = time.monotonic()
        with pytest.raises(TimeoutError) as excinfo:
            backend.wait_for_capacity(frozen_usage({"requests": 1}), timeout=0)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, f"timeout=0 should fail fast ({elapsed:.2f}s)"
        assert "lock" in str(excinfo.value).lower()
        assert isinstance(excinfo.value.__cause__, redis.exceptions.LockError)
    finally:
        _cleanup_sync(client, prefix)
        client.close()


def _cleanup_sync(client: _sync_redis.Redis, prefix: str) -> None:
    keys = list(client.scan_iter(match=f"{prefix}*"))
    if keys:
        client.delete(*keys)
