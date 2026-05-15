"""Regression coverage for FIX-29 UPGRADE-LEGACY-RESERVATION."""

from __future__ import annotations

import collections
import types

import pytest
from pydantic import ValidationError

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter


def _config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=60)]),
        model_family="fam",
    )


def _reservation(
    *,
    reservation_id: str = "reservation",
    limiter_instance_id: str = "limiter",
) -> CapacityReservation:
    return CapacityReservation(
        reservation_id=reservation_id,
        usage={"tokens": 30.0},
        model_family="fam",
        bucket_ids={("tokens", 60)},
        model="model",
        limiter_instance_id=limiter_instance_id,
    )


async def test_capacity_reservation_requires_limiter_instance_id() -> None:
    with pytest.raises(ValidationError, match="limiter_instance_id"):
        CapacityReservation(
            usage={"tokens": 1.0},
            model_family="fam",
            bucket_ids={("tokens", 60)},
            model="model",
        )


async def test_cross_limiter_refund_is_rejected() -> None:
    limiter_a = RateLimiter(_config(), backend=MemoryBackendBuilder())
    limiter_b = RateLimiter(_config(), backend=MemoryBackendBuilder())
    reservation = await limiter_a.acquire_capacity({"tokens": 30}, "model")

    with pytest.raises(ValueError, match="different limiter"):
        await limiter_b.refund_capacity({"tokens": 0}, reservation)


async def test_duplicate_refund_is_local_noop() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 30}, "model")

    await limiter.refund_capacity({"tokens": 0}, reservation)

    with pytest.warns(UserWarning, match="already been refunded"):
        await limiter.refund_capacity({"tokens": 0}, reservation)


async def test_memory_backend_rejects_cold_restart_refund() -> None:
    issuing_limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    reservation = await issuing_limiter.acquire_capacity({"tokens": 30}, "model")

    restarted_limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    # Keep the owner id aligned so this isolates the local in-flight check.
    restarted_limiter._limiter_instance_id = reservation.limiter_instance_id

    with pytest.raises(ValueError, match="cold-restart refunds require"):
        await restarted_limiter.refund_capacity({"tokens": 0}, reservation)


async def test_legacy_reservation_rejection_has_sync_async_parity() -> None:
    async_limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    sync_limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    async_legacy = _reservation()
    sync_legacy = _reservation()
    object.__setattr__(async_legacy, "limiter_instance_id", None)
    object.__setattr__(sync_legacy, "limiter_instance_id", None)

    match = "legacy v1.4.x reservations no longer supported"
    with pytest.raises(ValueError, match=match) as async_excinfo:
        await async_limiter.refund_capacity({"tokens": 0}, async_legacy)
    async_outcome = (type(async_excinfo.value), str(async_excinfo.value))

    with pytest.raises(ValueError, match=match) as sync_excinfo:
        sync_limiter.refund_capacity({"tokens": 0}, sync_legacy)
    sync_outcome = (type(sync_excinfo.value), str(sync_excinfo.value))

    assert async_outcome == sync_outcome
    assert "legacy v1.4.x reservations no longer supported" in async_outcome[1]


class _AsyncDedupRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.set_calls: list[tuple[str, str, int | None, bool]] = []

    async def set(
        self,
        key: str,
        value: str,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        self.set_calls.append((key, value, ex, nx))
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True


class _SyncDedupRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.set_calls: list[tuple[str, str, int | None, bool]] = []

    def set(
        self,
        key: str,
        value: str,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        self.set_calls.append((key, value, ex, nx))
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True


def _redis_modules():
    pytest.importorskip("redis", reason="redis package not installed")
    from token_throttle._limiter_backends._redis._backend import (  # noqa: PLC0415
        RedisBackend,
    )
    from token_throttle._limiter_backends._redis._bucket import (  # noqa: PLC0415
        RedisBucket,
    )
    from token_throttle._limiter_backends._redis._keys import (  # noqa: PLC0415
        DEFAULT_REFUND_DEDUP_TTL_SECONDS,
        redis_refund_dedup_key,
    )
    from token_throttle._limiter_backends._redis._sync_backend import (  # noqa: PLC0415
        SyncRedisBackendBuilder,
    )

    return types.SimpleNamespace(
        redis_backend=RedisBackend,
        redis_bucket=RedisBucket,
        sync_redis_backend_builder=SyncRedisBackendBuilder,
        default_refund_dedup_ttl_seconds=DEFAULT_REFUND_DEDUP_TTL_SECONDS,
        redis_refund_dedup_key=redis_refund_dedup_key,
    )


async def test_redis_refund_dedup_ttl_default_and_configurable() -> None:
    redis_modules = _redis_modules()
    cfg = _config()
    redis_client = _AsyncDedupRedis()
    bucket = redis_modules.redis_bucket(
        next(iter(cfg.quotas)),
        cfg,
        redis_client,
        key_prefix="tenant",
    )
    backend = redis_modules.redis_backend(
        [bucket], redis_client, cfg, key_prefix="tenant"
    )

    assert (
        backend._refund_dedup_ttl_seconds
        == redis_modules.default_refund_dedup_ttl_seconds
    )

    configured = redis_modules.redis_backend(
        [bucket],
        redis_client,
        cfg,
        key_prefix="tenant",
        refund_dedup_ttl_seconds=123,
    )
    await configured.refund_capacity_for_buckets(
        reserved_usage={},
        actual_usage={},
        bucket_ids=frozenset(),
        reservation_id="ttl-check",
    )

    assert redis_client.set_calls[-1] == (
        "tenant:rate_limiting:refund_dedup:ttl-check",
        "1",
        123,
        True,
    )

    sync_builder = redis_modules.sync_redis_backend_builder(
        _SyncDedupRedis(),
        key_prefix="tenant",
        refund_dedup_ttl_seconds=456,
    )
    assert sync_builder._refund_dedup_ttl_seconds == 456


async def test_redis_cold_restart_refund_uses_durable_dedup() -> None:
    redis_modules = _redis_modules()

    class StubRedisBackend(redis_modules.redis_backend):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.applied_refunds = 0

        async def refund_capacity_for_buckets(
            self,
            reserved_usage,
            actual_usage,
            *,
            bucket_ids=None,
            reservation_id=None,
        ) -> bool:
            if not await self._claim_refund_dedup(reservation_id):
                return False
            self.applied_refunds += 1
            return True

    class StubRedisBuilder:
        def __init__(self, redis_client: _AsyncDedupRedis) -> None:
            self.redis_client = redis_client

        def build(self, cfg, *, callbacks=None):
            bucket = redis_modules.redis_bucket(
                next(iter(cfg.quotas)),
                cfg,
                self.redis_client,
                key_prefix="tenant",
            )
            return StubRedisBackend(
                [bucket],
                self.redis_client,
                cfg,
                key_prefix="tenant",
                callbacks=callbacks,
            )

    issuing_limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    reservation = await issuing_limiter.acquire_capacity({"tokens": 30}, "model")
    redis_client = _AsyncDedupRedis()

    restarted_limiter = RateLimiter(_config(), backend=StubRedisBuilder(redis_client))
    # v2.0.0 still rejects cross-limiter owner mismatches; this simulates a
    # restarted worker that preserved the reservation owner id but lost local
    # in-flight memory.
    restarted_limiter._limiter_instance_id = reservation.limiter_instance_id
    await restarted_limiter.refund_capacity({"tokens": 0}, reservation)
    backend = restarted_limiter._model_family_to_backend["fam"]
    assert backend.applied_refunds == 1

    second_restarted_limiter = RateLimiter(
        _config(),
        backend=StubRedisBuilder(redis_client),
    )
    second_restarted_limiter._limiter_instance_id = reservation.limiter_instance_id
    with pytest.warns(UserWarning, match="already been refunded according to Redis"):
        await second_restarted_limiter.refund_capacity({"tokens": 0}, reservation)
    second_backend = second_restarted_limiter._model_family_to_backend["fam"]
    assert second_backend.applied_refunds == 0


async def test_refund_storm_fifo_eviction_does_not_reopen_redis_duplicate() -> None:
    redis_modules = _redis_modules()

    class StubRedisBackend(redis_modules.redis_backend):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.applied_refunds = 0

        async def refund_capacity_for_buckets(
            self,
            reserved_usage,
            actual_usage,
            *,
            bucket_ids=None,
            reservation_id=None,
        ) -> bool:
            if not await self._claim_refund_dedup(reservation_id):
                return False
            self.applied_refunds += 1
            return True

    class StubRedisBuilder:
        def __init__(self, redis_client: _AsyncDedupRedis) -> None:
            self.redis_client = redis_client

        def build(self, cfg, *, callbacks=None):
            bucket = redis_modules.redis_bucket(
                next(iter(cfg.quotas)),
                cfg,
                self.redis_client,
                key_prefix="tenant",
            )
            return StubRedisBackend(
                [bucket],
                self.redis_client,
                cfg,
                key_prefix="tenant",
                callbacks=callbacks,
            )

    redis_client = _AsyncDedupRedis()
    local_fifo = collections.OrderedDict()
    for index in range(200_000):
        rid = f"r{index}"
        redis_client.store[redis_modules.redis_refund_dedup_key("tenant", rid)] = "1"
        local_fifo[rid] = None
        if len(local_fifo) > 131_072:
            local_fifo.popitem(last=False)

    assert "r0" not in local_fifo

    limiter = RateLimiter(_config(), backend=StubRedisBuilder(redis_client))
    limiter._limiter_instance_id = "limiter"
    limiter._refunded_reservation_ids = local_fifo
    evicted = _reservation(reservation_id="r0", limiter_instance_id="limiter")

    with pytest.warns(UserWarning, match="already been refunded according to Redis"):
        await limiter.refund_capacity({"tokens": 0}, evicted)

    backend = limiter._model_family_to_backend["fam"]
    assert backend.applied_refunds == 0
