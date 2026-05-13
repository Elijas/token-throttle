"""Regression coverage for FIX-40 Redis lock hot-path knobs."""

from __future__ import annotations

import asyncio
import math
import time
import warnings

import pytest

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import (
    DEFAULT_LOCK_SLEEP_SECONDS,
    RedisBackendBuilder,
)
from token_throttle._limiter_backends._redis._bucket import RedisBucket
from token_throttle._limiter_backends._redis._sync_backend import (
    DEFAULT_LOCK_BLOCKING_THREAD_SLEEP_SECONDS,
    SyncRedisBackendBuilder,
)
from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket


class _Pool:
    def __init__(self, max_connections: int | None) -> None:
        self.max_connections = max_connections


class _FakeAsyncRedis:
    def __init__(self, *, max_connections: int | None = 50) -> None:
        self.connection_pool = _Pool(max_connections)

    def pipeline(self):
        raise AssertionError("pipeline should not be used by these tests")


class _FakeSyncRedis:
    def __init__(self, *, max_connections: int | None = 50) -> None:
        self.connection_pool = _Pool(max_connections)

    def pipeline(self):
        raise AssertionError("pipeline should not be used by these tests")


class _AsyncLock:
    def __init__(self) -> None:
        self.name = "lock"
        self.acquire_blocking_timeout: float | None = None
        self.acquire_tokens: list[bytes] = []
        self.released = False

    async def acquire(
        self,
        *,
        blocking_timeout: float | None = None,
        token: bytes | None = None,
    ) -> bool:
        self.acquire_blocking_timeout = blocking_timeout
        if token is not None:
            self.acquire_tokens.append(token)
        await asyncio.sleep(0)
        return True

    async def release(self) -> None:
        self.released = True

    async def lua_release(self, *, keys, args, client) -> None:
        self.released = True


class _SyncLock:
    def __init__(self) -> None:
        self.name = "lock"
        self.acquire_sleep: float | None = None
        self.acquire_blocking_timeout: float | None = None
        self.acquire_tokens: list[bytes] = []
        self.released = False

    def acquire(
        self,
        *,
        sleep: float | None = None,
        blocking_timeout: float | None = None,
        token: bytes | None = None,
    ) -> bool:
        self.acquire_sleep = sleep
        self.acquire_blocking_timeout = blocking_timeout
        if token is not None:
            self.acquire_tokens.append(token)
        return True

    def release(self) -> None:
        self.released = True

    def lua_release(self, *, keys, args, client) -> None:
        self.released = True


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="hot-family",
        quotas=UsageQuotas([Quota(metric="tokens", limit=1000.0, per_seconds=60)]),
    )


async def test_async_builder_passes_lock_knobs_to_bucket_lock(monkeypatch) -> None:
    lock_kwargs: list[dict[str, object]] = []
    locks: list[_AsyncLock] = []

    def fake_lock(self, **kwargs):
        lock_kwargs.append(kwargs)
        lock = _AsyncLock()
        locks.append(lock)
        return lock

    monkeypatch.setattr(RedisBucket, "lock", fake_lock)
    backend = RedisBackendBuilder(
        _FakeAsyncRedis(),
        key_prefix="test",
        lock_blocking_timeout_seconds=1.25,
        lock_sleep_seconds=0.02,
    ).build(_config())

    async with await backend._lock(timeout=30):
        pass

    assert lock_kwargs == [{"timeout": 30, "sleep": 0.02}]
    assert locks[0].acquire_blocking_timeout == pytest.approx(1.25, abs=0.01)
    assert locks[0].acquire_tokens
    assert locks[0].released


def test_sync_builder_passes_lock_knobs_to_bucket_lock(monkeypatch) -> None:
    lock_kwargs: list[dict[str, object]] = []
    locks: list[_SyncLock] = []

    def fake_lock(self, **kwargs):
        lock_kwargs.append(kwargs)
        lock = _SyncLock()
        locks.append(lock)
        return lock

    monkeypatch.setattr(SyncRedisBucket, "lock", fake_lock)
    backend = SyncRedisBackendBuilder(
        _FakeSyncRedis(),
        key_prefix="test",
        lock_blocking_timeout_seconds=1.5,
        lock_sleep_seconds=0.03,
        lock_blocking_thread_sleep_seconds=0.04,
    ).build(_config())

    with backend._lock(timeout=30):
        pass

    assert lock_kwargs == [{"timeout": 30, "sleep": 0.03}]
    assert locks[0].acquire_blocking_timeout == pytest.approx(1.5, abs=0.01)
    assert locks[0].acquire_sleep == pytest.approx(0.04)
    assert locks[0].acquire_tokens
    assert locks[0].released


@pytest.mark.parametrize(
    ("builder_cls", "redis_client"),
    [
        (RedisBackendBuilder, _FakeAsyncRedis(max_connections=9)),
        (SyncRedisBackendBuilder, _FakeSyncRedis(max_connections=9)),
    ],
)
def test_builder_warns_when_connection_pool_is_likely_too_small(
    builder_cls,
    redis_client,
) -> None:
    builder = builder_cls(redis_client, key_prefix="test")

    with pytest.warns(RuntimeWarning, match="max_connections is less than 10"):
        builder.build(_config())


@pytest.mark.parametrize(
    ("builder_cls", "redis_client"),
    [
        (RedisBackendBuilder, _FakeAsyncRedis(max_connections=10)),
        (SyncRedisBackendBuilder, _FakeSyncRedis(max_connections=None)),
    ],
)
def test_builder_does_not_warn_for_sufficient_or_unknown_pool_size(
    builder_cls,
    redis_client,
) -> None:
    builder = builder_cls(redis_client, key_prefix="test")

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        builder.build(_config())

    assert warnings_record == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lock_blocking_timeout_seconds": 0},
        {"lock_sleep_seconds": 0},
    ],
)
def test_async_builder_rejects_non_positive_lock_knobs(kwargs) -> None:
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        RedisBackendBuilder(_FakeAsyncRedis(), key_prefix="test", **kwargs)


def test_sync_builder_rejects_non_positive_thread_sleep() -> None:
    with pytest.raises(ValueError, match="lock_blocking_thread_sleep_seconds"):
        SyncRedisBackendBuilder(
            _FakeSyncRedis(),
            key_prefix="test",
            lock_blocking_thread_sleep_seconds=0,
        )


async def test_hot_family_mock_acquires_use_lower_default_lock_poll_sleep(
    monkeypatch,
) -> None:
    lock_kwargs: list[dict[str, object]] = []

    def fake_lock(self, **kwargs):
        lock_kwargs.append(kwargs)
        return _AsyncLock()

    monkeypatch.setattr(RedisBucket, "lock", fake_lock)
    backend = RedisBackendBuilder(_FakeAsyncRedis(), key_prefix="test").build(_config())

    async def acquire_once() -> float:
        started = time.perf_counter()
        async with await backend._lock(timeout=30):
            await asyncio.sleep(0)
        return time.perf_counter() - started

    durations = sorted(await asyncio.gather(*(acquire_once() for _ in range(100))))
    p99 = durations[math.ceil(len(durations) * 0.99) - 1]

    assert p99 < 1.0
    assert len(lock_kwargs) == 100
    assert {kwargs["sleep"] for kwargs in lock_kwargs} == {DEFAULT_LOCK_SLEEP_SECONDS}


def test_sync_default_thread_sleep_matches_public_constant(monkeypatch) -> None:
    locks: list[_SyncLock] = []

    def fake_lock(self, **kwargs):
        lock = _SyncLock()
        locks.append(lock)
        return lock

    monkeypatch.setattr(SyncRedisBucket, "lock", fake_lock)
    backend = SyncRedisBackendBuilder(_FakeSyncRedis(), key_prefix="test").build(
        _config()
    )

    with backend._lock(timeout=30):
        pass

    assert locks[0].acquire_sleep == DEFAULT_LOCK_BLOCKING_THREAD_SLEEP_SECONDS
