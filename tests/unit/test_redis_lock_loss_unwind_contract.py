"""A lock lost mid-operation must surface as ``BackendLockContentionError``.

The v9 contract (see ``docs/operations.md`` and the v9.0.0 ``CHANGELOG`` entry)
promises that non-waiting ops (``consume_capacity`` etc.) raise
``BackendLockContentionError`` when the per-bucket lock is lost mid-operation.

Mechanism guarded here: ``_extend_locks`` detects the loss (its ``reacquire``
raises redis ``LockNotOwnedError``) and re-raises ``BackendLockContentionError``.
As that error unwinds the lock ``ExitStack``, the release callback runs
``lock.release()`` on the *already-lost* lock, which also raises
``LockNotOwnedError``. Without suppression that release error replaces the
in-flight ``BackendLockContentionError``, so callers saw the raw redis error
instead of the documented one.

These are unit-style tests that talk to a real local Redis (they skip when one
is unavailable, e.g. the Redis-less unit CI lanes). Each test uses a unique
``key_prefix`` and deletes only its own keys — never ``FLUSHDB``. The lock loss
is forced deterministically by deleting the per-bucket lock key between
acquisition and ``_extend_locks`` (hooked via ``_get_capacities_unsafe``), which
is exactly the state a real GC pause / TTL lapse / stolen lock produces.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("redis", reason="redis package not installed")

import redis as _sync_redis
import redis.asyncio as _async_redis
import redis.exceptions

from token_throttle._exceptions import BackendLockContentionError
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="lock-loss-fam",
        quotas=UsageQuotas([Quota(metric="requests", limit=100, per_seconds=3600)]),
    )


@pytest.fixture
def redis_url(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--redis-url")


def _unique_prefix() -> str:
    return f"fixlane2-redis-lock-unwind-{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@pytest.mark.redis
async def test_async_mid_operation_lock_loss_raises_contention(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _async_redis.from_url(redis_url)
    try:
        await client.ping()
    except redis.exceptions.RedisError as exc:  # pragma: no cover - env dependent
        await client.aclose()
        pytest.skip(f"Redis unavailable at {redis_url}: {exc}")

    prefix = _unique_prefix()
    try:
        backend = RedisBackendBuilder(client, key_prefix=prefix).build(_config())
        lock_keys = [bucket._lock_key for bucket in backend.sorted_buckets]

        original_get = backend._get_capacities_unsafe

        async def _drop_lock_then_get(*args: object, **kwargs: object) -> object:
            # After the lock is acquired but before _extend_locks runs, delete
            # the per-bucket lock key so reacquire() raises LockNotOwnedError.
            await client.delete(*lock_keys)
            return await original_get(*args, **kwargs)

        monkeypatch.setattr(backend, "_get_capacities_unsafe", _drop_lock_then_get)

        with pytest.raises(BackendLockContentionError) as excinfo:
            await backend.consume_capacity(frozen_usage({"requests": 1}))

        # The mid-operation loss is the trigger: the chained cause is the redis
        # LockNotOwnedError from _extend_locks' reacquire(), not a release error.
        assert isinstance(excinfo.value.__cause__, redis.exceptions.LockNotOwnedError)
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
def test_sync_mid_operation_lock_loss_raises_contention(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _sync_redis.from_url(redis_url)
    try:
        client.ping()
    except redis.exceptions.RedisError as exc:  # pragma: no cover - env dependent
        client.close()
        pytest.skip(f"Redis unavailable at {redis_url}: {exc}")

    prefix = _unique_prefix()
    try:
        backend = SyncRedisBackendBuilder(client, key_prefix=prefix).build(_config())
        lock_keys = [bucket._lock_key for bucket in backend.sorted_buckets]

        original_get = backend._get_capacities_unsafe

        def _drop_lock_then_get(*args: object, **kwargs: object) -> object:
            client.delete(*lock_keys)
            return original_get(*args, **kwargs)

        monkeypatch.setattr(backend, "_get_capacities_unsafe", _drop_lock_then_get)

        with pytest.raises(BackendLockContentionError) as excinfo:
            backend.consume_capacity(frozen_usage({"requests": 1}))

        assert isinstance(excinfo.value.__cause__, redis.exceptions.LockNotOwnedError)
    finally:
        _cleanup_sync(client, prefix)
        client.close()


def _cleanup_sync(client: _sync_redis.Redis, prefix: str) -> None:
    keys = list(client.scan_iter(match=f"{prefix}*"))
    if keys:
        client.delete(*keys)
