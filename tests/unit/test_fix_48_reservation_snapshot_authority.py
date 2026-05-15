from __future__ import annotations

import asyncio
import threading
import time

import pytest

import token_throttle._rate_limiter as async_limiter_module
import token_throttle._sync_rate_limiter as sync_limiter_module
from token_throttle._exceptions import UnknownReservationError
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    RateLimiterBackend,
    SyncRateLimiterBackend,
)
from token_throttle._interfaces._models import (
    CapacityReservation,
    FrozenUsage,
    Quota,
    UsageQuotas,
)
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

MODEL = "model"
FAMILY = "fam"


def _two_metric_config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric="tokens", limit=10, per_seconds=60),
                Quota(metric="requests", limit=10, per_seconds=60),
            ]
        ),
        model_family=FAMILY,
    )


def _one_metric_config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=1, per_seconds=60)]),
        model_family=FAMILY,
    )


async def test_async_model_copy_usage_rewrite_does_not_refund_unreserved_metric() -> (
    None
):
    limiter = RateLimiter(_two_metric_config(), backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 1, "requests": 0}, MODEL)
    await limiter.acquire_capacity({"tokens": 0, "requests": 10}, MODEL)

    rewritten = reservation.model_copy(update={"usage": {"tokens": 0, "requests": 10}})
    await limiter.refund_capacity({"tokens": 0, "requests": 0}, rewritten)

    with pytest.raises(TimeoutError):
        await limiter.acquire_capacity({"tokens": 0, "requests": 1}, MODEL, timeout=0)


def test_sync_model_copy_usage_rewrite_does_not_refund_unreserved_metric() -> None:
    limiter = SyncRateLimiter(_two_metric_config(), backend=SyncMemoryBackendBuilder())
    reservation = limiter.acquire_capacity({"tokens": 1, "requests": 0}, MODEL)
    limiter.acquire_capacity({"tokens": 0, "requests": 10}, MODEL)

    rewritten = reservation.model_copy(update={"usage": {"tokens": 0, "requests": 10}})
    limiter.refund_capacity({"tokens": 0, "requests": 0}, rewritten)

    with pytest.raises(TimeoutError):
        limiter.acquire_capacity({"tokens": 0, "requests": 1}, MODEL, timeout=0)


async def test_async_created_at_model_copy_refresh_does_not_bypass_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(async_limiter_module.time, "time", lambda: now)
    limiter = RateLimiter(
        _one_metric_config(),
        backend=MemoryBackendBuilder(),
        max_reservation_lifetime_seconds=1,
    )
    reservation = await limiter.acquire_capacity({"tokens": 1}, MODEL)

    now = 102.0
    refreshed = reservation.model_copy(update={"created_at_seconds": now})
    with pytest.raises(ValueError, match="Reservation lifetime exceeded"):
        await limiter.refund_capacity({"tokens": 0}, refreshed)


def test_sync_created_at_model_copy_refresh_does_not_bypass_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(sync_limiter_module.time, "time", lambda: now)
    limiter = SyncRateLimiter(
        _one_metric_config(),
        backend=SyncMemoryBackendBuilder(),
        max_reservation_lifetime_seconds=1,
    )
    reservation = limiter.acquire_capacity({"tokens": 1}, MODEL)

    now = 102.0
    refreshed = reservation.model_copy(update={"created_at_seconds": now})
    with pytest.raises(ValueError, match="Reservation lifetime exceeded"):
        limiter.refund_capacity({"tokens": 0}, refreshed)


async def test_async_waited_acquire_issue_time_starts_after_capacity_commit() -> None:
    limiter = RateLimiter(
        _one_metric_config(),
        backend=MemoryBackendBuilder(sleep_interval=0.01),
        max_reservation_lifetime_seconds=0.2,
    )
    first = await limiter.acquire_capacity({"tokens": 1}, MODEL)
    waiter = asyncio.create_task(
        limiter.acquire_capacity({"tokens": 1}, MODEL, timeout=1)
    )

    await asyncio.sleep(0.25)
    assert not waiter.done()
    backend = limiter._model_family_to_backend[FAMILY]
    await backend.refund_capacity_for_buckets(
        {"tokens": 1},
        {"tokens": 0},
        bucket_ids=frozenset({("tokens", 60)}),
        reservation_id=first.reservation_id,
    )

    waited = await asyncio.wait_for(waiter, timeout=1)
    assert waited.created_at_seconds is not None
    assert time.time() - waited.created_at_seconds < 0.2
    await limiter.refund_capacity({"tokens": 0}, waited)


def test_sync_waited_acquire_issue_time_starts_after_capacity_commit() -> None:
    limiter = SyncRateLimiter(
        _one_metric_config(),
        backend=SyncMemoryBackendBuilder(sleep_interval=0.01),
        max_reservation_lifetime_seconds=0.2,
    )
    first = limiter.acquire_capacity({"tokens": 1}, MODEL)
    result: dict[str, CapacityReservation | BaseException] = {}

    def wait_for_reservation() -> None:
        try:
            result["reservation"] = limiter.acquire_capacity(
                {"tokens": 1}, MODEL, timeout=1
            )
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=wait_for_reservation)
    thread.start()
    time.sleep(0.25)
    assert thread.is_alive()
    backend = limiter._model_family_to_backend[FAMILY]
    backend.refund_capacity_for_buckets(
        {"tokens": 1},
        {"tokens": 0},
        bucket_ids=frozenset({("tokens", 60)}),
        reservation_id=first.reservation_id,
    )
    thread.join(timeout=1)

    assert "error" not in result
    waited = result["reservation"]
    assert isinstance(waited, CapacityReservation)
    assert waited.created_at_seconds is not None
    assert time.time() - waited.created_at_seconds < 0.2
    limiter.refund_capacity({"tokens": 0}, waited)


class _LyingAsyncBackend(RateLimiterBackend):
    def __init__(self) -> None:
        self.refunded = False

    async def await_for_capacity(
        self,
        usage: FrozenUsage,
        *,
        timeout: float | None = None,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        _ = usage, timeout, reservation_id, reservation_lifetime_seconds
        return time.time()

    async def consume_capacity(
        self,
        usage: FrozenUsage,
        *,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        _ = usage, reservation_id, reservation_lifetime_seconds
        return time.time()

    async def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        _ = reserved_usage, actual_usage
        self.refunded = True

    def supports_acquire_marker_authority(self) -> bool:
        return True

    async def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        _ = metric, per_seconds, value


class _LyingAsyncBuilder:
    def __init__(self) -> None:
        self.backend = _LyingAsyncBackend()

    def build(self, cfg: PerModelConfig, *, callbacks=None) -> _LyingAsyncBackend:
        _ = cfg, callbacks
        return self.backend


class _LyingSyncBackend(SyncRateLimiterBackend):
    def __init__(self) -> None:
        self.refunded = False

    def wait_for_capacity(
        self,
        usage: FrozenUsage,
        *,
        timeout: float | None = None,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        _ = usage, timeout, reservation_id, reservation_lifetime_seconds
        return time.time()

    def consume_capacity(
        self,
        usage: FrozenUsage,
        *,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        _ = usage, reservation_id, reservation_lifetime_seconds
        return time.time()

    def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        _ = reserved_usage, actual_usage
        self.refunded = True

    def supports_acquire_marker_authority(self) -> bool:
        return True

    def set_max_capacity(self, metric: str, per_seconds: int, value: float) -> None:
        _ = metric, per_seconds, value


class _LyingSyncBuilder:
    def __init__(self) -> None:
        self.backend = _LyingSyncBackend()

    def build(self, cfg: PerModelConfig, *, callbacks=None) -> _LyingSyncBackend:
        _ = cfg, callbacks
        return self.backend


def _forged_reservation() -> CapacityReservation:
    return CapacityReservation(
        reservation_id="forged",
        usage={"tokens": 1},
        model_family=FAMILY,
        bucket_ids={("tokens", 60)},
        model=MODEL,
        limiter_instance_id="manual-modern",
        created_at_seconds=time.time(),
    )


async def test_async_custom_backend_marker_authority_lie_is_rejected() -> None:
    builder = _LyingAsyncBuilder()
    limiter = RateLimiter(_one_metric_config(), backend=builder)

    with pytest.raises(RuntimeError, match="supports_acquire_marker_authority=True"):
        await limiter.refund_capacity({"tokens": 0}, _forged_reservation())

    assert builder.backend.refunded is False


def test_sync_custom_backend_marker_authority_lie_is_rejected() -> None:
    builder = _LyingSyncBuilder()
    limiter = SyncRateLimiter(_one_metric_config(), backend=builder)

    with pytest.raises(RuntimeError, match="supports_acquire_marker_authority=True"):
        limiter.refund_capacity({"tokens": 0}, _forged_reservation())

    assert builder.backend.refunded is False


try:
    from tests.unit.test_bundle_acquire_marker_authority import (
        _HAS_REDIS,
        _AsyncRedis,
        _AsyncRedisBuilder,
        _SyncRedis,
        _SyncRedisBuilder,
    )
except ImportError:
    _HAS_REDIS_HELPERS = False
else:
    _HAS_REDIS_HELPERS = _HAS_REDIS


@pytest.mark.skipif(not _HAS_REDIS_HELPERS, reason="redis package not installed")
async def test_async_redis_cross_process_usage_rewrite_is_rejected_by_marker() -> None:
    redis_client = _AsyncRedis()
    builder = _AsyncRedisBuilder(redis_client)
    limiter_a = RateLimiter(_one_metric_config(), backend=builder)
    limiter_b = RateLimiter(_one_metric_config(), backend=builder)
    reservation = await limiter_a.acquire_capacity({"tokens": 1}, MODEL)
    rewritten = reservation.model_copy(update={"usage": {"tokens": 10}})

    with pytest.raises(UnknownReservationError):
        await limiter_b.refund_capacity({"tokens": 0}, rewritten)

    await limiter_b.refund_capacity({"tokens": 0}, reservation)


@pytest.mark.skipif(not _HAS_REDIS_HELPERS, reason="redis package not installed")
def test_sync_redis_cross_process_usage_rewrite_is_rejected_by_marker() -> None:
    redis_client = _SyncRedis()
    builder = _SyncRedisBuilder(redis_client)
    limiter_a = SyncRateLimiter(_one_metric_config(), backend=builder)
    limiter_b = SyncRateLimiter(_one_metric_config(), backend=builder)
    reservation = limiter_a.acquire_capacity({"tokens": 1}, MODEL)
    rewritten = reservation.model_copy(update={"usage": {"tokens": 10}})

    with pytest.raises(UnknownReservationError):
        limiter_b.refund_capacity({"tokens": 0}, rewritten)

    limiter_b.refund_capacity({"tokens": 0}, reservation)
