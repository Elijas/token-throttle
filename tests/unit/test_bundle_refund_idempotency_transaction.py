"""Regression coverage for FIX-42 REFUND-IDEMPOTENCY-TRANSACTION."""

from __future__ import annotations

import asyncio
import inspect
import time
from contextlib import AbstractAsyncContextManager, AbstractContextManager

import pytest

from token_throttle._exceptions import DuplicateRefundError
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

MODEL = "model"
MODEL_FAMILY = "fam"
BUCKET_ID = ("tokens", 60)
RESERVED = {"tokens": 30.0}
ACTUAL = {"tokens": 10.0}


def _redis_get_value(value: object) -> object:
    if value is None or isinstance(value, (bytes, str)):
        return value
    return str(value)


def _config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=60)]),
        model_family=MODEL_FAMILY,
    )


def _redis_modules():
    pytest.importorskip("redis", reason="redis package not installed")
    from token_throttle._limiter_backends._redis._backend import (  # noqa: PLC0415
        RedisBackend,
    )
    from token_throttle._limiter_backends._redis._bucket import (  # noqa: PLC0415
        RedisBucket,
    )
    from token_throttle._limiter_backends._redis._keys import (  # noqa: PLC0415
        redis_acquired_marker_key,
        redis_acquired_marker_value,
        redis_refund_dedup_key,
    )
    from token_throttle._limiter_backends._redis._sync_backend import (  # noqa: PLC0415
        SyncRedisBackend,
    )
    from token_throttle._limiter_backends._redis._sync_bucket import (  # noqa: PLC0415
        SyncRedisBucket,
    )

    return {
        "redis_backend": RedisBackend,
        "redis_bucket": RedisBucket,
        "redis_acquired_marker_key": redis_acquired_marker_key,
        "redis_acquired_marker_value": redis_acquired_marker_value,
        "redis_refund_dedup_key": redis_refund_dedup_key,
        "sync_redis_backend": SyncRedisBackend,
        "sync_redis_bucket": SyncRedisBucket,
    }


def _acquire_marker_value(reservation_id: str) -> str:
    return _redis_modules()["redis_acquired_marker_value"](
        reservation_id=reservation_id,
        model_family=MODEL_FAMILY,
        bucket_ids=frozenset({BUCKET_ID}),
        usage=RESERVED,
    )


def _acquire_marker_key(reservation_id: str) -> str:
    return _redis_modules()["redis_acquired_marker_key"]("tenant", reservation_id)


def _marker_refund_kwargs(reservation_id: str) -> dict[str, object]:
    return {
        "bucket_ids": frozenset({BUCKET_ID}),
        "reservation_id": reservation_id,
        "reservation_model_family": MODEL_FAMILY,
        "reservation_bucket_ids": frozenset({BUCKET_ID}),
        "reservation_reserved_usage": RESERVED,
    }


def _seed_acquire_marker(redis_client: object, reservation_id: str) -> None:
    redis_client.store[_acquire_marker_key(reservation_id)] = _acquire_marker_value(
        reservation_id
    )


async def test_async_cancel_before_backend_write_leaves_local_guard_retryable() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 30}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    original_refund = backend.refund_capacity_for_buckets
    entered = asyncio.Event()
    keep_open = asyncio.Event()

    async def stall_before_write(*args, **kwargs):
        entered.set()
        await keep_open.wait()

    backend.refund_capacity_for_buckets = stall_before_write

    task = asyncio.create_task(limiter.refund_capacity({"tokens": 10}, reservation))
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert limiter._refunded_reservation_ids[reservation.reservation_id] == "pending"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert limiter._refunded_reservation_ids[reservation.reservation_id] == "failed"

    backend.refund_capacity_for_buckets = original_refund
    await limiter.refund_capacity({"tokens": 10}, reservation)

    assert limiter._refunded_reservation_ids[reservation.reservation_id] == "committed"
    with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
        await limiter.refund_capacity({"tokens": 10}, reservation)


async def test_async_failed_backend_write_leaves_local_guard_retryable() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 30}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    original_refund = backend.refund_capacity_for_buckets
    calls = 0

    async def fail_before_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("simulated refund write failure")

    backend.refund_capacity_for_buckets = fail_before_write

    with pytest.raises(RuntimeError, match="simulated refund write failure"):
        await limiter.refund_capacity({"tokens": 10}, reservation)

    assert calls == 1
    assert limiter._refunded_reservation_ids[reservation.reservation_id] == "failed"

    backend.refund_capacity_for_buckets = original_refund
    await limiter.refund_capacity({"tokens": 10}, reservation)

    with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
        await limiter.refund_capacity({"tokens": 10}, reservation)


def test_sync_failed_backend_write_leaves_local_guard_retryable() -> None:
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    reservation = limiter.acquire_capacity({"tokens": 30}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    original_refund = backend.refund_capacity_for_buckets
    calls = 0

    def fail_before_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("simulated refund write failure")

    backend.refund_capacity_for_buckets = fail_before_write

    with pytest.raises(RuntimeError, match="simulated refund write failure"):
        limiter.refund_capacity({"tokens": 10}, reservation)

    assert calls == 1
    assert limiter._refunded_reservation_ids[reservation.reservation_id] == "failed"

    backend.refund_capacity_for_buckets = original_refund
    limiter.refund_capacity({"tokens": 10}, reservation)

    with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
        limiter.refund_capacity({"tokens": 10}, reservation)


def test_sync_base_exception_before_backend_write_leaves_local_guard_retryable() -> (
    None
):
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    reservation = limiter.acquire_capacity({"tokens": 30}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    original_refund = backend.refund_capacity_for_buckets

    def interrupt_before_write(*args, **kwargs):
        raise SystemExit("simulated shutdown")

    backend.refund_capacity_for_buckets = interrupt_before_write

    with pytest.raises(SystemExit, match="simulated shutdown"):
        limiter.refund_capacity({"tokens": 10}, reservation)

    assert limiter._refunded_reservation_ids[reservation.reservation_id] == "failed"

    backend.refund_capacity_for_buckets = original_refund
    limiter.refund_capacity({"tokens": 10}, reservation)

    with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
        limiter.refund_capacity({"tokens": 10}, reservation)


class _AsyncPipeline:
    def __init__(self, redis_client: _AsyncRedis) -> None:
        self._redis = redis_client
        self._commands: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def get(self, key: str) -> None:
        self._commands.append(("get", (key,), {}))

    def set(self, key: str, value: object, **kwargs: object) -> None:
        self._commands.append(("set", (key, value), kwargs))

    def expire(self, key: str, seconds: int) -> None:
        self._commands.append(("expire", (key, seconds), {}))

    async def execute(self) -> list[object]:
        results: list[object] = []
        for name, args, kwargs in self._commands:
            result = getattr(self._redis, name)(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            results.append(result)
        self._commands.clear()
        return results


class _AsyncRedis:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.now = float(int(time.time()))
        self.pause_first_dedup_set: asyncio.Event | None = None
        self.first_dedup_set_entered = asyncio.Event()
        self._paused_first_dedup_set = False

    async def get(self, key: str) -> object:
        return _redis_get_value(self.store.get(key))

    async def exists(self, key: str) -> int:
        return int(key in self.store)

    async def set(
        self,
        key: str,
        value: object,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        if (
            ":refund_dedup:" in key
            and self.pause_first_dedup_set is not None
            and not self._paused_first_dedup_set
        ):
            self._paused_first_dedup_set = True
            self.first_dedup_set_entered.set()
            await self.pause_first_dedup_set.wait()
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self.store

    async def delete(self, key: str) -> int:
        existed = key in self.store
        self.store.pop(key, None)
        return int(existed)

    async def time(self) -> tuple[int, int]:
        return int(self.now), 0

    async def eval(self, _script: str, numkeys: int, *keys_and_args: object) -> str:
        keys = [str(key) for key in keys_and_args[:numkeys]]
        argv = list(keys_and_args[numkeys:])
        marker_key, dedup_key = keys[0], keys[1]
        marker = await self.get(marker_key)
        if marker is None:
            return (
                "duplicate_refund"
                if await self.exists(dedup_key)
                else "unknown_reservation"
            )
        if marker != argv[0]:
            return "marker_mismatch"
        claimed = await self.set(dedup_key, "1", ex=int(argv[1]), nx=True)
        if not claimed:
            return "duplicate_refund"
        arg_index = 2
        for key_index in range(2, len(keys), 2):
            await self.set(
                keys[key_index], argv[arg_index], ex=int(argv[arg_index + 2])
            )
            await self.set(
                keys[key_index + 1],
                argv[arg_index + 1],
                ex=int(argv[arg_index + 2]),
            )
            arg_index += 3
        await self.delete(marker_key)
        return "ok"

    def pipeline(self) -> _AsyncPipeline:
        return _AsyncPipeline(self)


class _SyncPipeline:
    def __init__(self, redis_client: _SyncRedis) -> None:
        self._redis = redis_client
        self._commands: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def get(self, key: str) -> None:
        self._commands.append(("get", (key,), {}))

    def set(self, key: str, value: object, **kwargs: object) -> None:
        self._commands.append(("set", (key, value), kwargs))

    def expire(self, key: str, seconds: int) -> None:
        self._commands.append(("expire", (key, seconds), {}))

    def execute(self) -> list[object]:
        results = []
        for name, args, kwargs in self._commands:
            results.append(getattr(self._redis, name)(*args, **kwargs))
        self._commands.clear()
        return results


class _SyncRedis:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.now = float(int(time.time()))

    def get(self, key: str) -> object:
        return _redis_get_value(self.store.get(key))

    def exists(self, key: str) -> int:
        return int(key in self.store)

    def set(
        self,
        key: str,
        value: object,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    def expire(self, key: str, seconds: int) -> bool:
        return key in self.store

    def delete(self, key: str) -> int:
        existed = key in self.store
        self.store.pop(key, None)
        return int(existed)

    def time(self) -> tuple[int, int]:
        return int(self.now), 0

    def eval(self, _script: str, numkeys: int, *keys_and_args: object) -> str:
        keys = [str(key) for key in keys_and_args[:numkeys]]
        argv = list(keys_and_args[numkeys:])
        marker_key, dedup_key = keys[0], keys[1]
        marker = self.get(marker_key)
        if marker is None:
            return (
                "duplicate_refund" if self.exists(dedup_key) else "unknown_reservation"
            )
        if marker != argv[0]:
            return "marker_mismatch"
        claimed = self.set(dedup_key, "1", ex=int(argv[1]), nx=True)
        if not claimed:
            return "duplicate_refund"
        arg_index = 2
        for key_index in range(2, len(keys), 2):
            self.set(keys[key_index], argv[arg_index], ex=int(argv[arg_index + 2]))
            self.set(
                keys[key_index + 1],
                argv[arg_index + 1],
                ex=int(argv[arg_index + 2]),
            )
            arg_index += 3
        self.delete(marker_key)
        return "ok"

    def pipeline(self) -> _SyncPipeline:
        return _SyncPipeline(self)


class _AsyncLockStack(AbstractAsyncContextManager):
    def __init__(self, lock: asyncio.Lock | None = None) -> None:
        self._lock = lock
        self.locks = []

    async def __aenter__(self):
        if self._lock is not None:
            await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if self._lock is not None:
            self._lock.release()
        return False


class _SyncLockStack(AbstractContextManager):
    def __init__(self, lock: object | None = None) -> None:
        self._lock = lock
        self.locks = []

    def __enter__(self):
        if self._lock is not None:
            self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._lock is not None:
            self._lock.release()
        return False


def _async_redis_backend(
    redis_client: _AsyncRedis,
    *,
    test_lock: asyncio.Lock | None = None,
) -> tuple[object, object]:
    redis_modules = _redis_modules()

    class TestRedisBackend(redis_modules["redis_backend"]):
        def __init__(
            self, *args, test_lock: asyncio.Lock | None = None, **kwargs
        ) -> None:
            super().__init__(*args, **kwargs)
            self._test_lock = test_lock

        async def _lock(self, **kwargs):
            return _AsyncLockStack(self._test_lock)

        @staticmethod
        async def _extend_locks(_stack, *, reservation_id=None) -> None:
            _ = reservation_id

    cfg = _config()
    bucket = redis_modules["redis_bucket"](
        next(iter(cfg.quotas)),
        cfg,
        redis_client,
        key_prefix="tenant",
    )
    redis_client.store[bucket._last_checked_key] = redis_client.now
    redis_client.store[bucket._capacity_key] = 70.0
    backend = TestRedisBackend(
        [bucket],
        redis_client,
        cfg,
        key_prefix="tenant",
        test_lock=test_lock,
    )
    return backend, bucket


def _sync_redis_backend(
    redis_client: _SyncRedis,
) -> tuple[object, object]:
    redis_modules = _redis_modules()

    class TestSyncRedisBackend(redis_modules["sync_redis_backend"]):
        def __init__(self, *args, test_lock: object | None = None, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._test_lock = test_lock

        def _lock(self, **kwargs):
            return _SyncLockStack(self._test_lock)

        @staticmethod
        def _extend_locks(_stack, *, reservation_id=None) -> None:
            _ = reservation_id

    cfg = _config()
    bucket = redis_modules["sync_redis_bucket"](
        next(iter(cfg.quotas)),
        cfg,
        redis_client,
        key_prefix="tenant",
    )
    redis_client.store[bucket._last_checked_key] = redis_client.now
    redis_client.store[bucket._capacity_key] = 70.0
    backend = TestSyncRedisBackend([bucket], redis_client, cfg, key_prefix="tenant")
    return backend, bucket


async def test_async_redis_failed_bucket_write_does_not_claim_tombstone() -> None:
    redis_client = _AsyncRedis()
    backend, bucket = _async_redis_backend(redis_client)
    _seed_acquire_marker(redis_client, "redis-r1")
    original_commit = backend._commit_refund_with_acquire_marker_unsafe
    calls = 0

    async def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated redis write failure")
        return await original_commit(*args, **kwargs)

    backend._commit_refund_with_acquire_marker_unsafe = fail_once
    dedup_key = _redis_modules()["redis_refund_dedup_key"]("tenant", "redis-r1")

    with pytest.raises(RuntimeError, match="simulated redis write failure"):
        await backend.refund_capacity_for_buckets(
            RESERVED,
            ACTUAL,
            **_marker_refund_kwargs("redis-r1"),
        )

    assert dedup_key not in redis_client.store
    assert redis_client.store[bucket._capacity_key] == 70.0

    assert await backend.refund_capacity_for_buckets(
        RESERVED,
        ACTUAL,
        **_marker_refund_kwargs("redis-r1"),
    )
    assert redis_client.store[bucket._capacity_key] == 90.0
    assert redis_client.store[dedup_key] == "1"

    with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
        await backend.refund_capacity_for_buckets(
            RESERVED,
            ACTUAL,
            **_marker_refund_kwargs("redis-r1"),
        )
    assert redis_client.store[bucket._capacity_key] == 90.0


def test_sync_redis_failed_bucket_write_does_not_claim_tombstone() -> None:
    redis_client = _SyncRedis()
    backend, bucket = _sync_redis_backend(redis_client)
    _seed_acquire_marker(redis_client, "redis-r2")
    original_commit = backend._commit_refund_with_acquire_marker_unsafe
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated redis write failure")
        return original_commit(*args, **kwargs)

    backend._commit_refund_with_acquire_marker_unsafe = fail_once
    dedup_key = _redis_modules()["redis_refund_dedup_key"]("tenant", "redis-r2")

    with pytest.raises(RuntimeError, match="simulated redis write failure"):
        backend.refund_capacity_for_buckets(
            RESERVED,
            ACTUAL,
            **_marker_refund_kwargs("redis-r2"),
        )

    assert dedup_key not in redis_client.store
    assert redis_client.store[bucket._capacity_key] == 70.0

    assert backend.refund_capacity_for_buckets(
        RESERVED,
        ACTUAL,
        **_marker_refund_kwargs("redis-r2"),
    )
    assert redis_client.store[bucket._capacity_key] == 90.0
    assert redis_client.store[dedup_key] == "1"

    with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
        backend.refund_capacity_for_buckets(
            RESERVED,
            ACTUAL,
            **_marker_refund_kwargs("redis-r2"),
        )
    assert redis_client.store[bucket._capacity_key] == 90.0


async def test_async_redis_deferred_tombstone_serializes_concurrent_retries() -> None:
    redis_client = _AsyncRedis()
    release_first_dedup_set = asyncio.Event()
    redis_client.pause_first_dedup_set = release_first_dedup_set
    backend, bucket = _async_redis_backend(redis_client, test_lock=asyncio.Lock())
    _seed_acquire_marker(redis_client, "redis-r3")
    dedup_key = _redis_modules()["redis_refund_dedup_key"]("tenant", "redis-r3")

    first = asyncio.create_task(
        backend.refund_capacity_for_buckets(
            RESERVED,
            ACTUAL,
            **_marker_refund_kwargs("redis-r3"),
        )
    )
    await asyncio.wait_for(redis_client.first_dedup_set_entered.wait(), timeout=1)
    second = asyncio.create_task(
        backend.refund_capacity_for_buckets(
            RESERVED,
            ACTUAL,
            **_marker_refund_kwargs("redis-r3"),
        )
    )

    release_first_dedup_set.set()
    results = await asyncio.gather(first, second, return_exceptions=True)

    assert results.count(True) == 1
    assert sum(isinstance(result, DuplicateRefundError) for result in results) == 1
    assert redis_client.store[dedup_key] == "1"
    assert redis_client.store[bucket._capacity_key] == 90.0
