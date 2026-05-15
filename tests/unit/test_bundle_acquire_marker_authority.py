"""Regression coverage for FIX-44b ACQUIRE-MARKER-AUTHORITY."""

from __future__ import annotations

import math
import time
from contextlib import AbstractAsyncContextManager, AbstractContextManager

import pytest

from token_throttle._exceptions import DuplicateRefundError, UnknownReservationError
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

try:
    from token_throttle._limiter_backends._redis._backend import RedisBackend
    from token_throttle._limiter_backends._redis._bucket import RedisBucket
    from token_throttle._limiter_backends._redis._keys import (
        redis_acquired_marker_key,
        redis_refund_dedup_key,
    )
    from token_throttle._limiter_backends._redis._sync_backend import SyncRedisBackend
    from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket
    from token_throttle._limiter_backends._redis._ttl import (
        resolve_max_reservation_lifetime_seconds_from_ttls,
        validate_reservation_lifetime_ttl_invariant,
    )
except ImportError:
    RedisBackend = object
    RedisBucket = None
    SyncRedisBackend = object
    SyncRedisBucket = None
    redis_acquired_marker_key = None
    redis_refund_dedup_key = None
    resolve_max_reservation_lifetime_seconds_from_ttls = None
    validate_reservation_lifetime_ttl_invariant = None
    _HAS_REDIS = False
else:
    _HAS_REDIS = True

_REDIS_SKIP = pytest.mark.skipif(not _HAS_REDIS, reason="redis package not installed")

MODEL = "model"
FAMILY = "fam"
PREFIX = "tenant"
BUCKET_ID = ("tokens", 60)


def _config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=60)]),
        model_family=FAMILY,
    )


def _forged_reservation(reservation_id: str = "forged") -> CapacityReservation:
    return CapacityReservation(
        reservation_id=reservation_id,
        usage={"tokens": 30.0},
        model_family=FAMILY,
        bucket_ids={BUCKET_ID},
        model=MODEL,
        limiter_instance_id="manual-modern",
        created_at_seconds=time.time(),
    )


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

    def delete(self, key: str) -> None:
        self._commands.append(("delete", (key,), {}))

    async def execute(self) -> list[object]:
        results = []
        for name, args, kwargs in self._commands:
            result = getattr(self._redis, name)(*args, **kwargs)
            results.append(await result)
        self._commands.clear()
        return results


class _AsyncRedis:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.deadlines: dict[str, float | None] = {}
        self.now = 1000.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def _purge_if_expired(self, key: str) -> None:
        deadline = self.deadlines.get(key)
        if deadline is not None and deadline <= self.now:
            self.store.pop(key, None)
            self.deadlines.pop(key, None)

    async def get(self, key: str) -> object:
        self._purge_if_expired(key)
        return self.store.get(key)

    async def exists(self, key: str) -> int:
        self._purge_if_expired(key)
        return int(key in self.store)

    async def set(
        self,
        key: str,
        value: object,
        *,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        self._purge_if_expired(key)
        if nx and key in self.store:
            return None
        self.store[key] = value
        if px is not None:
            self.deadlines[key] = self.now + (float(px) / 1000.0)
        else:
            self.deadlines[key] = None if ex is None else self.now + float(ex)
        return True

    async def expire(self, key: str, seconds: int) -> bool:
        self._purge_if_expired(key)
        if key not in self.store:
            return False
        self.deadlines[key] = self.now + float(seconds)
        return True

    async def delete(self, key: str) -> int:
        self._purge_if_expired(key)
        existed = key in self.store
        self.store.pop(key, None)
        self.deadlines.pop(key, None)
        return int(existed)

    async def ttl(self, key: str) -> int:
        self._purge_if_expired(key)
        if key not in self.store:
            return -2
        deadline = self.deadlines.get(key)
        if deadline is None:
            return -1
        return max(0, math.ceil(deadline - self.now))

    async def time(self) -> tuple[int, int]:
        return int(self.now), int((self.now % 1) * 1_000_000)

    async def eval(self, _script: str, numkeys: int, *keys_and_args: object) -> str:
        keys = [str(key) for key in keys_and_args[:numkeys]]
        argv = list(keys_and_args[numkeys:])
        if len(keys) >= 2 and ":refund_dedup:" in keys[1]:
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

        marker_key = keys[0]
        existing = await self.get(marker_key)
        if existing is not None:
            if existing == argv[1]:
                return "ok"
            return "duplicate_acquire"
        arg_index = 2
        for key_index in range(1, len(keys), 2):
            await self.set(
                keys[key_index], argv[arg_index], ex=int(argv[arg_index + 2])
            )
            await self.set(
                keys[key_index + 1],
                argv[arg_index + 1],
                ex=int(argv[arg_index + 2]),
            )
            arg_index += 3
        claimed = await self.set(marker_key, argv[1], px=int(argv[0]), nx=True)
        if not claimed:
            existing = await self.get(marker_key)
            if existing == argv[1]:
                return "ok"
            return "duplicate_acquire"
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

    def delete(self, key: str) -> None:
        self._commands.append(("delete", (key,), {}))

    def execute(self) -> list[object]:
        results = []
        for name, args, kwargs in self._commands:
            results.append(getattr(self._redis, name)(*args, **kwargs))
        self._commands.clear()
        return results


class _SyncRedis:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.deadlines: dict[str, float | None] = {}
        self.now = 1000.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def _purge_if_expired(self, key: str) -> None:
        deadline = self.deadlines.get(key)
        if deadline is not None and deadline <= self.now:
            self.store.pop(key, None)
            self.deadlines.pop(key, None)

    def get(self, key: str) -> object:
        self._purge_if_expired(key)
        return self.store.get(key)

    def exists(self, key: str) -> int:
        self._purge_if_expired(key)
        return int(key in self.store)

    def set(
        self,
        key: str,
        value: object,
        *,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        self._purge_if_expired(key)
        if nx and key in self.store:
            return None
        self.store[key] = value
        if px is not None:
            self.deadlines[key] = self.now + (float(px) / 1000.0)
        else:
            self.deadlines[key] = None if ex is None else self.now + float(ex)
        return True

    def expire(self, key: str, seconds: int) -> bool:
        self._purge_if_expired(key)
        if key not in self.store:
            return False
        self.deadlines[key] = self.now + float(seconds)
        return True

    def delete(self, key: str) -> int:
        self._purge_if_expired(key)
        existed = key in self.store
        self.store.pop(key, None)
        self.deadlines.pop(key, None)
        return int(existed)

    def ttl(self, key: str) -> int:
        self._purge_if_expired(key)
        if key not in self.store:
            return -2
        deadline = self.deadlines.get(key)
        if deadline is None:
            return -1
        return max(0, math.ceil(deadline - self.now))

    def time(self) -> tuple[int, int]:
        return int(self.now), int((self.now % 1) * 1_000_000)

    def eval(self, _script: str, numkeys: int, *keys_and_args: object) -> str:
        keys = [str(key) for key in keys_and_args[:numkeys]]
        argv = list(keys_and_args[numkeys:])
        if len(keys) >= 2 and ":refund_dedup:" in keys[1]:
            marker_key, dedup_key = keys[0], keys[1]
            marker = self.get(marker_key)
            if marker is None:
                return (
                    "duplicate_refund"
                    if self.exists(dedup_key)
                    else "unknown_reservation"
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

        marker_key = keys[0]
        existing = self.get(marker_key)
        if existing is not None:
            if existing == argv[1]:
                return "ok"
            return "duplicate_acquire"
        arg_index = 2
        for key_index in range(1, len(keys), 2):
            self.set(keys[key_index], argv[arg_index], ex=int(argv[arg_index + 2]))
            self.set(
                keys[key_index + 1],
                argv[arg_index + 1],
                ex=int(argv[arg_index + 2]),
            )
            arg_index += 3
        claimed = self.set(marker_key, argv[1], px=int(argv[0]), nx=True)
        if not claimed:
            existing = self.get(marker_key)
            if existing == argv[1]:
                return "ok"
            return "duplicate_acquire"
        return "ok"

    def pipeline(self) -> _SyncPipeline:
        return _SyncPipeline(self)


class _AsyncLockStack(AbstractAsyncContextManager):
    def __init__(self) -> None:
        self.locks = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _SyncLockStack(AbstractContextManager):
    def __init__(self) -> None:
        self.locks = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _TestAsyncRedisBackend(RedisBackend):
    async def _lock(self, **kwargs):
        return _AsyncLockStack()

    @staticmethod
    async def _extend_locks(_stack) -> None:
        return None


class _TestSyncRedisBackend(SyncRedisBackend):
    def _lock(self, **kwargs):
        return _SyncLockStack()

    @staticmethod
    def _extend_locks(_stack) -> None:
        return None


class _AsyncRedisBuilder:
    def __init__(
        self,
        redis_client: _AsyncRedis,
        *,
        bucket_ttl: int = 10,
        refund_dedup_ttl: int = 10,
    ) -> None:
        self.redis = redis_client
        self.bucket_ttl = bucket_ttl
        self.refund_dedup_ttl = refund_dedup_ttl

    def resolve_max_reservation_lifetime_seconds(
        self,
        max_reservation_lifetime_seconds: float | None,
    ) -> float | None:
        return resolve_max_reservation_lifetime_seconds_from_ttls(
            max_reservation_lifetime_seconds=max_reservation_lifetime_seconds,
            bucket_ttl_seconds=self.bucket_ttl,
            refund_dedup_ttl_seconds=self.refund_dedup_ttl,
        )

    def validate_reservation_lifetime_seconds(
        self,
        max_reservation_lifetime_seconds: float | None,
    ) -> None:
        validate_reservation_lifetime_ttl_invariant(
            max_reservation_lifetime_seconds=max_reservation_lifetime_seconds,
            bucket_ttl_seconds=self.bucket_ttl,
            refund_dedup_ttl_seconds=self.refund_dedup_ttl,
        )

    def build(self, cfg: PerModelConfig, *, callbacks=None) -> RedisBackend:
        bucket = RedisBucket(
            next(iter(cfg.quotas)),
            cfg,
            self.redis,
            key_prefix=PREFIX,
            bucket_ttl_seconds=self.bucket_ttl,
        )
        return _TestAsyncRedisBackend(
            [bucket],
            self.redis,
            cfg,
            key_prefix=PREFIX,
            refund_dedup_ttl_seconds=self.refund_dedup_ttl,
            callbacks=callbacks,
        )


class _SyncRedisBuilder(_AsyncRedisBuilder):
    redis: _SyncRedis

    def build(self, cfg: PerModelConfig, *, callbacks=None) -> SyncRedisBackend:
        bucket = SyncRedisBucket(
            next(iter(cfg.quotas)),
            cfg,
            self.redis,
            key_prefix=PREFIX,
            bucket_ttl_seconds=self.bucket_ttl,
        )
        return _TestSyncRedisBackend(
            [bucket],
            self.redis,
            cfg,
            key_prefix=PREFIX,
            refund_dedup_ttl_seconds=self.refund_dedup_ttl,
            callbacks=callbacks,
        )


@_REDIS_SKIP
async def test_async_redis_happy_path_deletes_marker_and_writes_tombstone() -> None:
    redis_client = _AsyncRedis()
    limiter = RateLimiter(_config(), backend=_AsyncRedisBuilder(redis_client))

    reservation = await limiter.acquire_capacity({"tokens": 30}, MODEL)
    marker_key = redis_acquired_marker_key(PREFIX, reservation.reservation_id)
    tombstone_key = redis_refund_dedup_key(PREFIX, reservation.reservation_id)
    assert marker_key in redis_client.store

    await limiter.refund_capacity({"tokens": 10}, reservation)

    assert marker_key not in redis_client.store
    assert redis_client.store[tombstone_key] == "1"
    assert (
        redis_client.store[f"{PREFIX}:rate_limiting:bucket:{FAMILY}:tokens:60:capacity"]
        == 90.0
    )


@_REDIS_SKIP
def test_sync_redis_happy_path_deletes_marker_and_writes_tombstone() -> None:
    redis_client = _SyncRedis()
    limiter = SyncRateLimiter(_config(), backend=_SyncRedisBuilder(redis_client))

    reservation = limiter.acquire_capacity({"tokens": 30}, MODEL)
    marker_key = redis_acquired_marker_key(PREFIX, reservation.reservation_id)
    tombstone_key = redis_refund_dedup_key(PREFIX, reservation.reservation_id)
    assert marker_key in redis_client.store

    limiter.refund_capacity({"tokens": 10}, reservation)

    assert marker_key not in redis_client.store
    assert redis_client.store[tombstone_key] == "1"
    assert (
        redis_client.store[f"{PREFIX}:rate_limiting:bucket:{FAMILY}:tokens:60:capacity"]
        == 90.0
    )


@_REDIS_SKIP
async def test_async_redis_forged_reservation_is_unknown() -> None:
    limiter = RateLimiter(_config(), backend=_AsyncRedisBuilder(_AsyncRedis()))

    with pytest.raises(
        UnknownReservationError,
        match="reservation was never acquired by this backend",
    ):
        await limiter.refund_capacity({"tokens": 0}, _forged_reservation())


@_REDIS_SKIP
def test_sync_redis_forged_reservation_is_unknown() -> None:
    limiter = SyncRateLimiter(_config(), backend=_SyncRedisBuilder(_SyncRedis()))

    with pytest.raises(
        UnknownReservationError,
        match="reservation was never acquired by this backend",
    ):
        limiter.refund_capacity({"tokens": 0}, _forged_reservation())


@_REDIS_SKIP
async def test_async_redis_duplicate_refund_raises_duplicate() -> None:
    limiter = RateLimiter(_config(), backend=_AsyncRedisBuilder(_AsyncRedis()))
    reservation = await limiter.acquire_capacity({"tokens": 30}, MODEL)

    await limiter.refund_capacity({"tokens": 0}, reservation)

    with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
        await limiter.refund_capacity({"tokens": 0}, reservation)


@_REDIS_SKIP
def test_sync_redis_duplicate_refund_raises_duplicate() -> None:
    limiter = SyncRateLimiter(_config(), backend=_SyncRedisBuilder(_SyncRedis()))
    reservation = limiter.acquire_capacity({"tokens": 30}, MODEL)

    limiter.refund_capacity({"tokens": 0}, reservation)

    with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
        limiter.refund_capacity({"tokens": 0}, reservation)


@_REDIS_SKIP
async def test_async_redis_cross_process_refund_uses_shared_marker() -> None:
    redis_client = _AsyncRedis()
    builder = _AsyncRedisBuilder(redis_client)
    limiter_a = RateLimiter(_config(), backend=builder)
    limiter_b = RateLimiter(_config(), backend=builder)
    reservation = await limiter_a.acquire_capacity({"tokens": 30}, MODEL)

    await limiter_b.refund_capacity({"tokens": 10}, reservation)

    assert (
        redis_acquired_marker_key(PREFIX, reservation.reservation_id)
        not in redis_client.store
    )
    assert (
        redis_client.store[redis_refund_dedup_key(PREFIX, reservation.reservation_id)]
        == "1"
    )


@_REDIS_SKIP
def test_sync_redis_cross_process_refund_uses_shared_marker() -> None:
    redis_client = _SyncRedis()
    builder = _SyncRedisBuilder(redis_client)
    limiter_a = SyncRateLimiter(_config(), backend=builder)
    limiter_b = SyncRateLimiter(_config(), backend=builder)
    reservation = limiter_a.acquire_capacity({"tokens": 30}, MODEL)

    limiter_b.refund_capacity({"tokens": 10}, reservation)

    assert (
        redis_acquired_marker_key(PREFIX, reservation.reservation_id)
        not in redis_client.store
    )
    assert (
        redis_client.store[redis_refund_dedup_key(PREFIX, reservation.reservation_id)]
        == "1"
    )


@_REDIS_SKIP
async def test_async_redis_marker_ttl_expiry_is_unknown() -> None:
    redis_client = _AsyncRedis()
    limiter = RateLimiter(
        _config(),
        backend=_AsyncRedisBuilder(redis_client, bucket_ttl=4, refund_dedup_ttl=4),
    )
    reservation = await limiter.acquire_capacity({"tokens": 30}, MODEL)

    redis_client.advance(3.0)

    with pytest.raises(UnknownReservationError):
        await limiter.refund_capacity({"tokens": 0}, reservation)


@_REDIS_SKIP
def test_sync_redis_marker_ttl_expiry_is_unknown() -> None:
    redis_client = _SyncRedis()
    limiter = SyncRateLimiter(
        _config(),
        backend=_SyncRedisBuilder(redis_client, bucket_ttl=4, refund_dedup_ttl=4),
    )
    reservation = limiter.acquire_capacity({"tokens": 30}, MODEL)

    redis_client.advance(3.0)

    with pytest.raises(UnknownReservationError):
        limiter.refund_capacity({"tokens": 0}, reservation)


async def test_async_memory_backend_error_parity() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 30}, MODEL)

    await limiter.refund_capacity({"tokens": 0}, reservation)
    with pytest.raises(DuplicateRefundError):
        await limiter.refund_capacity({"tokens": 0}, reservation)

    with pytest.raises(UnknownReservationError):
        await limiter.refund_capacity({"tokens": 0}, _forged_reservation())


def test_sync_memory_backend_error_parity() -> None:
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    reservation = limiter.acquire_capacity({"tokens": 30}, MODEL)

    limiter.refund_capacity({"tokens": 0}, reservation)
    with pytest.raises(DuplicateRefundError):
        limiter.refund_capacity({"tokens": 0}, reservation)

    with pytest.raises(UnknownReservationError):
        limiter.refund_capacity({"tokens": 0}, _forged_reservation())
