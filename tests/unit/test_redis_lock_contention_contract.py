"""Unit coverage for the per-bucket lock-contention contract.

These tests simulate lock-acquisition starvation without a real Redis by
monkeypatching the lock primitive (so ``_lock`` fails to acquire) or the
``_check_and_consume_capacity`` hot path (to exercise the wait-loop retry).
They assert the public contract:

* no-timeout ``await_for_capacity``/``wait_for_capacity`` retry on contention
  and eventually succeed (never raise on lock starvation);
* ``consume_capacity``/``refund_capacity``/``set_max_capacity`` raise the
  library ``BackendLockContentionError`` chained from the underlying
  ``redis.exceptions.LockError`` instead of leaking the raw redis error;
* a caller timeout still surfaces the library ``TimeoutError``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("redis", reason="redis package not installed")

import redis as _sync_redis
import redis.asyncio as _async_redis
import redis.exceptions
from frozendict import frozendict

from token_throttle._exceptions import BackendLockContentionError
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder
from token_throttle._limiter_backends._redis._bucket import RedisBucket
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)
from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket


class _Pool:
    def __init__(self, max_connections: int | None) -> None:
        self.max_connections = max_connections


class _FakeAsyncRedis(_async_redis.Redis):
    def __init__(self, *, max_connections: int | None = 50) -> None:
        self.connection_pool = _Pool(max_connections)

    def pipeline(self):
        raise AssertionError("pipeline should not be reached once the lock starves")


class _FakeSyncRedis(_sync_redis.Redis):
    def __init__(self, *, max_connections: int | None = 50) -> None:
        self.connection_pool = _Pool(max_connections)

    def pipeline(self):
        raise AssertionError("pipeline should not be reached once the lock starves")


class _StarvingAsyncLock:
    """Async lock that never succeeds — simulates acquisition starvation."""

    def __init__(self) -> None:
        self.name = "lock"

    async def acquire(self, *, blocking_timeout=None, token=None) -> bool:
        return False

    async def release(self) -> None:  # pragma: no cover - never acquired
        pass

    async def lua_release(self, *, keys, args, client) -> None:
        pass


class _StarvingSyncLock:
    """Sync lock that never succeeds — simulates acquisition starvation."""

    def __init__(self) -> None:
        self.name = "lock"

    def acquire(self, *, sleep=None, blocking_timeout=None, token=None) -> bool:
        return False

    def release(self) -> None:  # pragma: no cover - never acquired
        pass

    def lua_release(self, *, keys, args, client) -> None:
        pass


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="lock-family",
        quotas=UsageQuotas([Quota(metric="tokens", limit=1000.0, per_seconds=60)]),
    )


def _starving_async_lock(self, **kwargs):
    return _StarvingAsyncLock()


def _starving_sync_lock(self, **kwargs):
    return _StarvingSyncLock()


def _async_backend(monkeypatch):
    monkeypatch.setattr(RedisBucket, "lock", _starving_async_lock)
    return RedisBackendBuilder(_FakeAsyncRedis(), key_prefix="test").build(_config())


def _sync_backend(monkeypatch):
    monkeypatch.setattr(SyncRedisBucket, "lock", _starving_sync_lock)
    return SyncRedisBackendBuilder(_FakeSyncRedis(), key_prefix="test").build(_config())


# ---------------------------------------------------------------------------
# (a) no-timeout acquire retries and eventually succeeds after transient
#     starvation, for sync and async.
# ---------------------------------------------------------------------------


async def test_async_await_retries_then_succeeds_on_transient_contention(
    monkeypatch,
) -> None:
    backend = RedisBackendBuilder(_FakeAsyncRedis(), key_prefix="test").build(_config())
    calls = {"n": 0}
    success = (
        True,
        frozendict(),
        frozendict(),
        123.0,
        123.0,
        (),
    )

    async def fake_check(self, usage_, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise redis.exceptions.LockError("Unable to acquire lock")
        return success

    monkeypatch.setattr(
        type(backend), "_check_and_consume_capacity", fake_check, raising=True
    )
    monkeypatch.setattr(backend, "_lock_sleep_seconds", 0.0)

    result = await backend.await_for_capacity(frozendict({"tokens": 1.0}))

    assert result == 123.0
    assert calls["n"] == 4  # retried through 3 starvations, then succeeded


def test_sync_wait_retries_then_succeeds_on_transient_contention(monkeypatch) -> None:
    backend = SyncRedisBackendBuilder(_FakeSyncRedis(), key_prefix="test").build(
        _config()
    )
    calls = {"n": 0}
    success = (
        True,
        frozendict(),
        frozendict(),
        123.0,
        123.0,
        (),
    )

    def fake_check(self, usage_, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise redis.exceptions.LockError("Unable to acquire lock")
        return success

    monkeypatch.setattr(
        type(backend), "_check_and_consume_capacity", fake_check, raising=True
    )
    monkeypatch.setattr(backend, "_lock_sleep_seconds", 0.0)

    result = backend.wait_for_capacity(frozendict({"tokens": 1.0}))

    assert result == 123.0
    assert calls["n"] == 4


async def test_async_await_retries_on_mid_operation_lock_loss(monkeypatch) -> None:
    """Mid-operation lock loss (BackendLockContentionError) also retries."""
    backend = RedisBackendBuilder(_FakeAsyncRedis(), key_prefix="test").build(_config())
    calls = {"n": 0}
    success = (True, frozendict(), frozendict(), 7.0, 7.0, ())

    async def fake_check(self, usage_, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise BackendLockContentionError(
                BackendLockContentionError.LOCK_LOST_MESSAGE
            )
        return success

    monkeypatch.setattr(
        type(backend), "_check_and_consume_capacity", fake_check, raising=True
    )
    monkeypatch.setattr(backend, "_lock_sleep_seconds", 0.0)

    assert await backend.await_for_capacity(frozendict({"tokens": 1.0})) == 7.0
    assert calls["n"] == 3


# ---------------------------------------------------------------------------
# (b) consume/refund/set_max raise BackendLockContentionError chained from
#     redis.exceptions.LockError.
# ---------------------------------------------------------------------------


async def test_async_consume_raises_contention_chained_from_lock_error(
    monkeypatch,
) -> None:
    backend = _async_backend(monkeypatch)
    with pytest.raises(BackendLockContentionError) as excinfo:
        await backend.consume_capacity(frozendict({"tokens": 1.0}))
    assert isinstance(excinfo.value.__cause__, redis.exceptions.LockError)


async def test_async_refund_raises_contention_chained_from_lock_error(
    monkeypatch,
) -> None:
    backend = _async_backend(monkeypatch)
    with pytest.raises(BackendLockContentionError) as excinfo:
        await backend.refund_capacity(
            reserved_usage=frozendict({"tokens": 10.0}),
            actual_usage=frozendict({"tokens": 5.0}),
        )
    assert isinstance(excinfo.value.__cause__, redis.exceptions.LockError)


async def test_async_set_max_raises_contention_chained_from_lock_error(
    monkeypatch,
) -> None:
    backend = _async_backend(monkeypatch)
    with pytest.raises(BackendLockContentionError) as excinfo:
        await backend.set_max_capacity("tokens", 60, 500.0)
    assert isinstance(excinfo.value.__cause__, redis.exceptions.LockError)


def test_sync_consume_raises_contention_chained_from_lock_error(monkeypatch) -> None:
    backend = _sync_backend(monkeypatch)
    with pytest.raises(BackendLockContentionError) as excinfo:
        backend.consume_capacity(frozendict({"tokens": 1.0}))
    assert isinstance(excinfo.value.__cause__, redis.exceptions.LockError)


def test_sync_refund_raises_contention_chained_from_lock_error(monkeypatch) -> None:
    backend = _sync_backend(monkeypatch)
    with pytest.raises(BackendLockContentionError) as excinfo:
        backend.refund_capacity(
            reserved_usage=frozendict({"tokens": 10.0}),
            actual_usage=frozendict({"tokens": 5.0}),
        )
    assert isinstance(excinfo.value.__cause__, redis.exceptions.LockError)


def test_sync_set_max_raises_contention_chained_from_lock_error(monkeypatch) -> None:
    backend = _sync_backend(monkeypatch)
    with pytest.raises(BackendLockContentionError) as excinfo:
        backend.set_max_capacity("tokens", 60, 500.0)
    assert isinstance(excinfo.value.__cause__, redis.exceptions.LockError)


# ---------------------------------------------------------------------------
# (c) acquire with a caller timeout still raises the library TimeoutError.
# ---------------------------------------------------------------------------


async def test_async_await_with_timeout_raises_library_timeout_error(
    monkeypatch,
) -> None:
    backend = RedisBackendBuilder(_FakeAsyncRedis(), key_prefix="test").build(_config())

    async def always_starves(self, usage_, **kwargs):
        raise redis.exceptions.LockError("Unable to acquire lock")

    monkeypatch.setattr(
        type(backend), "_check_and_consume_capacity", always_starves, raising=True
    )

    with pytest.raises(TimeoutError) as excinfo:
        await backend.await_for_capacity(frozendict({"tokens": 1.0}), timeout=0.05)
    assert isinstance(excinfo.value.__cause__, redis.exceptions.LockError)


def test_sync_wait_with_timeout_raises_library_timeout_error(monkeypatch) -> None:
    backend = SyncRedisBackendBuilder(_FakeSyncRedis(), key_prefix="test").build(
        _config()
    )

    def always_starves(self, usage_, **kwargs):
        raise redis.exceptions.LockError("Unable to acquire lock")

    monkeypatch.setattr(
        type(backend), "_check_and_consume_capacity", always_starves, raising=True
    )

    with pytest.raises(TimeoutError) as excinfo:
        backend.wait_for_capacity(frozendict({"tokens": 1.0}), timeout=0.05)
    assert isinstance(excinfo.value.__cause__, redis.exceptions.LockError)
