"""Regression coverage for FIX-51 TTL/reservation lifetime invariants."""

from __future__ import annotations

import fnmatch
import math
import warnings

import pytest

import token_throttle._rate_limiter as async_limiter_module
from token_throttle._interfaces._interfaces import PerModelConfig, RateLimiterBackend
from token_throttle._interfaces._models import (
    BucketId,
    FrozenUsage,
    Quota,
    UsageQuotas,
)
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._redis._keys import redis_refund_dedup_key
from token_throttle._limiter_backends._redis._ttl import (
    resolve_max_reservation_lifetime_seconds_from_ttls,
    validate_reservation_lifetime_ttl_invariant,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter
from token_throttle.migration import cleanup_legacy_buckets


def _config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=60)]),
        model_family="fam",
    )


class _ExpiringRedis:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.deadlines: dict[str, float | None] = {}
        self.now = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def _purge_if_expired(self, key: str) -> None:
        deadline = self.deadlines.get(key)
        if deadline is not None and deadline <= self.now:
            self.store.pop(key, None)
            self.deadlines.pop(key, None)

    async def set(
        self,
        key: str,
        value: object,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        self._purge_if_expired(key)
        if nx and key in self.store:
            return None
        self.store[key] = value
        self.deadlines[key] = None if ex is None else self.now + ex
        return True

    async def exists(self, key: str) -> int:
        self._purge_if_expired(key)
        return int(key in self.store)

    async def ttl(self, key: str) -> int:
        self._purge_if_expired(key)
        if key not in self.store:
            return -2
        deadline = self.deadlines.get(key)
        if deadline is None:
            return -1
        return max(0, int(deadline - self.now))


class _AsyncDedupBackend(RateLimiterBackend):
    def __init__(self, redis_client: _ExpiringRedis, *, refund_dedup_ttl: int) -> None:
        self.redis = redis_client
        self.refund_dedup_ttl = refund_dedup_ttl
        self.refund_calls = 0

    async def await_for_capacity(
        self,
        usage: FrozenUsage,
        *,
        timeout: float | None = None,
    ) -> None:
        _ = usage, timeout

    async def consume_capacity(self, usage: FrozenUsage) -> None:
        _ = usage

    async def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        await self.refund_capacity_for_buckets(reserved_usage, actual_usage)

    async def refund_capacity_for_buckets(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        *,
        bucket_ids: set[BucketId] | frozenset[BucketId] | None = None,
        reservation_id: str | None = None,
    ) -> bool:
        _ = reserved_usage, actual_usage, bucket_ids
        if reservation_id is not None:
            key = redis_refund_dedup_key("tenant", reservation_id)
            claimed = await self.redis.set(
                key,
                "1",
                ex=self.refund_dedup_ttl,
                nx=True,
            )
            if not claimed:
                warnings.warn("already been refunded according to Redis", UserWarning)
                return False
        self.refund_calls += 1
        return True

    def supports_durable_refund_dedup(self) -> bool:
        return True

    async def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        _ = metric, per_seconds, value


class _AsyncDedupBuilder:
    def __init__(
        self,
        redis_client: _ExpiringRedis,
        *,
        bucket_ttl: int = 3,
        refund_dedup_ttl: int = 3,
    ) -> None:
        self.redis = redis_client
        self.bucket_ttl = bucket_ttl
        self.refund_dedup_ttl = refund_dedup_ttl
        self.backends: list[_AsyncDedupBackend] = []

    def validate_reservation_lifetime_seconds(
        self,
        max_reservation_lifetime_seconds: float | None,
    ) -> None:
        validate_reservation_lifetime_ttl_invariant(
            max_reservation_lifetime_seconds=max_reservation_lifetime_seconds,
            bucket_ttl_seconds=self.bucket_ttl,
            refund_dedup_ttl_seconds=self.refund_dedup_ttl,
        )

    def resolve_max_reservation_lifetime_seconds(
        self,
        max_reservation_lifetime_seconds: float | None,
    ) -> float | None:
        return resolve_max_reservation_lifetime_seconds_from_ttls(
            max_reservation_lifetime_seconds=max_reservation_lifetime_seconds,
            bucket_ttl_seconds=self.bucket_ttl,
            refund_dedup_ttl_seconds=self.refund_dedup_ttl,
        )

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: object | None = None,
    ) -> _AsyncDedupBackend:
        _ = cfg, callbacks
        backend = _AsyncDedupBackend(
            self.redis,
            refund_dedup_ttl=self.refund_dedup_ttl,
        )
        self.backends.append(backend)
        return backend

    async def aclose(self) -> None:
        return None


def test_default_lifetime_is_derived_from_redis_ttls() -> None:
    builder = _AsyncDedupBuilder(
        _ExpiringRedis(),
        bucket_ttl=600,
        refund_dedup_ttl=300,
    )

    limiter = RateLimiter(_config(), backend=builder)

    max_lifetime = limiter._max_reservation_lifetime_seconds
    assert max_lifetime is not None
    assert max_lifetime <= 150.0
    assert math.isclose(max_lifetime, 150.0)


def test_sync_default_lifetime_is_derived_from_redis_ttls() -> None:
    builder = _AsyncDedupBuilder(
        _ExpiringRedis(),
        bucket_ttl=600,
        refund_dedup_ttl=300,
    )

    limiter = SyncRateLimiter(_config(), backend=builder)

    max_lifetime = limiter._max_reservation_lifetime_seconds
    assert max_lifetime is not None
    assert max_lifetime <= 150.0
    assert math.isclose(max_lifetime, 150.0)


def test_default_lifetime_stays_unbounded_when_backend_has_no_ttls() -> None:
    assert (
        resolve_max_reservation_lifetime_seconds_from_ttls(
            max_reservation_lifetime_seconds=None,
            bucket_ttl_seconds=None,
            refund_dedup_ttl_seconds=None,
        )
        is None
    )


class _SyncMigrationRedis:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.ttls: dict[str, int] = {}

    def scan_iter(self, *, match: str, count: int):
        _ = count
        for key in list(self.store):
            if fnmatch.fnmatch(key, match):
                yield key

    def ttl(self, key: str) -> int:
        return self.ttls.get(key, -2)

    def delete(self, key: str) -> int:
        existed = key in self.store
        self.store.pop(key, None)
        self.ttls.pop(key, None)
        return int(existed)


async def test_cm03_default_lifetime_rejects_after_bucket_ttl_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_client = _ExpiringRedis()
    builder = _AsyncDedupBuilder(redis_client, bucket_ttl=3, refund_dedup_ttl=3)
    now = 0.0
    monkeypatch.setattr(async_limiter_module.time, "time", lambda: now)
    limiter = RateLimiter(_config(), backend=builder)

    max_lifetime = limiter._max_reservation_lifetime_seconds
    assert max_lifetime is not None
    assert max_lifetime < builder.bucket_ttl

    reservation = await limiter.acquire_capacity({"tokens": 30}, "model")
    await redis_client.set(
        "tenant:rate_limiting:bucket:fam:tokens:60:last_checked",
        0.0,
        ex=3,
    )
    await redis_client.set(
        "tenant:rate_limiting:bucket:fam:tokens:60:capacity",
        70.0,
        ex=3,
    )

    now = 4.0
    redis_client.advance(4.0)

    assert (
        await redis_client.ttl("tenant:rate_limiting:bucket:fam:tokens:60:capacity")
        == -2
    )
    with pytest.raises(ValueError, match="Reservation lifetime exceeded"):
        await limiter.refund_capacity({"tokens": 0}, reservation)
    assert builder.backends[0].refund_calls == 0


async def test_cm03_expired_reservation_rejected_after_bucket_ttl_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_client = _ExpiringRedis()
    builder = _AsyncDedupBuilder(redis_client, bucket_ttl=3, refund_dedup_ttl=3)
    now = 0.0
    monkeypatch.setattr(async_limiter_module.time, "time", lambda: now)
    limiter = RateLimiter(
        _config(),
        backend=builder,
        max_reservation_lifetime_seconds=1,
    )

    reservation = await limiter.acquire_capacity({"tokens": 30}, "model")
    await redis_client.set(
        "tenant:rate_limiting:bucket:fam:tokens:60:last_checked",
        0.0,
        ex=3,
    )
    await redis_client.set(
        "tenant:rate_limiting:bucket:fam:tokens:60:capacity",
        70.0,
        ex=3,
    )

    now = 4.0
    redis_client.advance(4.0)

    assert (
        await redis_client.ttl("tenant:rate_limiting:bucket:fam:tokens:60:capacity")
        == -2
    )
    with pytest.raises(ValueError, match="Reservation lifetime exceeded"):
        await limiter.refund_capacity({"tokens": 0}, reservation)
    assert builder.backends[0].refund_calls == 0


def test_cm04_cleanup_removes_pre_fix38_idle_bucket_state_keys() -> None:
    redis_client = _SyncMigrationRedis()
    legacy_capacity = "tenant:rate_limiting:bucket:fam:tokens:60:capacity"
    legacy_last_checked = "tenant:rate_limiting:bucket:fam:tokens:60:last_checked"
    live_capacity = "tenant:rate_limiting:bucket:fam:requests:60:capacity"
    override = "tenant:rate_limiting:bucket:fam:tokens:60:max_capacity_override"
    foreign = "other:rate_limiting:bucket:fam:tokens:60:capacity"
    redis_client.store.update(
        {
            legacy_capacity: "70",
            legacy_last_checked: "0",
            live_capacity: "99",
            override: "50",
            foreign: "1",
        }
    )
    redis_client.ttls.update(
        {
            legacy_capacity: -1,
            legacy_last_checked: -1,
            live_capacity: 30,
            override: -1,
            foreign: -1,
        }
    )

    assert cleanup_legacy_buckets(redis_client, "tenant") == 2

    assert legacy_capacity not in redis_client.store
    assert legacy_last_checked not in redis_client.store
    assert live_capacity in redis_client.store
    assert override in redis_client.store
    assert foreign in redis_client.store


async def test_lr04_default_lifetime_rejects_after_dedup_ttl_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_client = _ExpiringRedis()
    now = 0.0
    monkeypatch.setattr(async_limiter_module.time, "time", lambda: now)
    first_builder = _AsyncDedupBuilder(redis_client, bucket_ttl=3, refund_dedup_ttl=3)
    first = RateLimiter(_config(), backend=first_builder)
    max_lifetime = first._max_reservation_lifetime_seconds
    assert max_lifetime is not None
    assert max_lifetime < first_builder.refund_dedup_ttl

    reservation = await first.acquire_capacity({"tokens": 30}, "model")

    now = 0.5
    redis_client.advance(0.5)
    await first.refund_capacity({"tokens": 0}, reservation)
    assert first_builder.backends[0].refund_calls == 1

    now = 3.6
    redis_client.advance(3.1)
    dedup_key = redis_refund_dedup_key("tenant", reservation.reservation_id)
    assert await redis_client.exists(dedup_key) == 0
    second_builder = _AsyncDedupBuilder(redis_client, bucket_ttl=3, refund_dedup_ttl=3)
    second = RateLimiter(_config(), backend=second_builder)
    second._limiter_instance_id = reservation.limiter_instance_id

    with pytest.raises(ValueError, match="Reservation lifetime exceeded"):
        await second.refund_capacity({"tokens": 0}, reservation)
    assert second_builder.backends == []


async def test_lr04_second_cross_process_refund_rejected_after_dedup_ttl_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_client = _ExpiringRedis()
    now = 0.0
    monkeypatch.setattr(async_limiter_module.time, "time", lambda: now)
    first_builder = _AsyncDedupBuilder(redis_client, bucket_ttl=3, refund_dedup_ttl=3)
    first = RateLimiter(
        _config(),
        backend=first_builder,
        max_reservation_lifetime_seconds=1,
    )
    reservation = await first.acquire_capacity({"tokens": 30}, "model")

    now = 0.5
    redis_client.advance(0.5)
    await first.refund_capacity({"tokens": 0}, reservation)
    assert first_builder.backends[0].refund_calls == 1

    now = 3.6
    redis_client.advance(3.1)
    dedup_key = redis_refund_dedup_key("tenant", reservation.reservation_id)
    assert await redis_client.exists(dedup_key) == 0
    second_builder = _AsyncDedupBuilder(redis_client, bucket_ttl=3, refund_dedup_ttl=3)
    second = RateLimiter(
        _config(),
        backend=second_builder,
        max_reservation_lifetime_seconds=1,
    )
    second._limiter_instance_id = reservation.limiter_instance_id

    with pytest.raises(ValueError, match="Reservation lifetime exceeded"):
        await second.refund_capacity({"tokens": 0}, reservation)
    assert second_builder.backends == []


def test_max_reservation_lifetime_longer_than_redis_ttls_raises() -> None:
    redis_client = _ExpiringRedis()
    builder = _AsyncDedupBuilder(redis_client, bucket_ttl=10, refund_dedup_ttl=30)

    with pytest.raises(ValueError, match="Redis TTLs must exceed"):
        RateLimiter(
            _config(),
            backend=builder,
            max_reservation_lifetime_seconds=6,
        )


def test_explicit_lifetime_is_respected_with_finite_redis_ttls() -> None:
    redis_client = _ExpiringRedis()
    builder = _AsyncDedupBuilder(redis_client, bucket_ttl=10, refund_dedup_ttl=30)

    limiter = RateLimiter(
        _config(),
        backend=builder,
        max_reservation_lifetime_seconds=1,
    )

    assert limiter._max_reservation_lifetime_seconds == 1.0


async def test_default_unbounded_lifetime_preserves_missing_timestamp_refund() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    assert limiter._max_reservation_lifetime_seconds is None
    reservation = await limiter.acquire_capacity({"tokens": 30}, "model")
    object.__setattr__(reservation, "created_at_seconds", None)

    await limiter.refund_capacity({"tokens": 0}, reservation)

    assert reservation.reservation_id in limiter._refunded_reservation_ids
