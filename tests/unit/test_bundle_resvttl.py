import logging
import time

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


def _limited_config(
    *,
    family: str = "fam",
    metric: str = "tokens",
    limit: float = 100.0,
) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric=metric, limit=limit, per_seconds=60)]),
        model_family=family,
    )


async def _async_capacity(limiter: RateLimiter, family: str, metric: str) -> float:
    backend = limiter._model_family_to_backend[family]
    async with backend._condition:
        capacities, _ = backend._get_capacities(time.time())
    return capacities[(metric, 60)]


def _sync_capacity(limiter: SyncRateLimiter, family: str, metric: str) -> float:
    backend = limiter._model_family_to_backend[family]
    with backend._condition:
        capacities, _ = backend._get_capacities(time.time())
    return capacities[(metric, 60)]


async def test_n01_async_cross_limiter_memory_refund_is_unknown():
    limiter_a = RateLimiter(_limited_config(), backend=MemoryBackendBuilder())
    limiter_b = RateLimiter(_limited_config(), backend=MemoryBackendBuilder())

    reservation = await limiter_a.acquire_capacity({"tokens": 30}, "model")
    await limiter_b.acquire_capacity({"tokens": 1}, "model")
    before = await _async_capacity(limiter_b, "fam", "tokens")

    with pytest.raises(UnknownReservationError):
        await limiter_b.refund_capacity({"tokens": 0}, reservation)

    assert reservation.limiter_instance_id == limiter_a._limiter_instance_id
    assert reservation.reservation_id not in limiter_b._refunded_reservation_ids
    assert await _async_capacity(limiter_b, "fam", "tokens") - before < 1


async def test_legacy_reservation_without_limiter_instance_id_is_rejected(
    caplog: pytest.LogCaptureFixture,
):
    limiter = RateLimiter(_limited_config(), backend=MemoryBackendBuilder())
    await limiter.acquire_capacity({"tokens": 30}, "model")
    legacy = CapacityReservation(
        usage={"tokens": 30},
        model_family="fam",
        bucket_ids={("tokens", 60)},
        model="model",
        limiter_instance_id=limiter._limiter_instance_id,
    )
    object.__setattr__(legacy, "limiter_instance_id", None)
    before = await _async_capacity(limiter, "fam", "tokens")

    with (
        caplog.at_level(logging.WARNING, logger="token_throttle"),
        pytest.raises(ValueError, match=r"legacy v1\.4\.x reservations"),
    ):
        await limiter.refund_capacity({"tokens": 0}, legacy)

    assert await _async_capacity(limiter, "fam", "tokens") - before < 1
    assert "legacy v1.4.x reservations are rejected" in caplog.text


async def test_n02_empty_projection_commits_dedup_before_return():
    state = "tokens"

    def config_getter(_model: str) -> PerModelConfig:
        metric = "tokens" if state == "tokens" else "requests"
        return _limited_config(metric=metric)

    limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 30}, "model")

    state = "requests"
    with pytest.warns(RuntimeWarning, match="Refund dropped"):
        await limiter.refund_capacity({"tokens": 0}, reservation)

    assert reservation.reservation_id in limiter._refunded_reservation_ids

    state = "tokens"
    await limiter.record_usage({"tokens": 0}, "model")
    with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
        await limiter.refund_capacity({"tokens": 0}, reservation)

    assert await _async_capacity(limiter, "fam", "tokens") == 100


async def test_n03_limited_to_unlimited_flip_rejected_on_refund():
    use_unlimited = False

    def config_getter(_model: str) -> PerModelConfig:
        if use_unlimited:
            return PerModelConfig(
                quotas=UsageQuotas.unlimited(),
                model_family="fam",
            )
        return _limited_config()

    limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 30}, "model")

    use_unlimited = True
    with pytest.raises(ValueError, match=r"limited-to-unlimited.*L13 N03"):
        await limiter.refund_capacity({"tokens": 0}, reservation)

    assert reservation.reservation_id not in limiter._refunded_reservation_ids


async def test_n05_model_family_reroute_rejected_on_refund():
    family = "fam-old"

    def config_getter(_model: str) -> PerModelConfig:
        return _limited_config(family=family)

    limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 30}, "model")

    family = "fam-new"
    with pytest.raises(ValueError, match=r"model_family rerouting.*L13 N05"):
        await limiter.refund_capacity({"tokens": 0}, reservation)

    assert reservation.reservation_id not in limiter._refunded_reservation_ids


async def test_n09_async_close_logs_outstanding_reservations_and_blocks_use(
    caplog: pytest.LogCaptureFixture,
):
    limiter = RateLimiter(_limited_config(), backend=MemoryBackendBuilder())
    first = await limiter.acquire_capacity({"tokens": 10}, "model")
    await limiter.acquire_capacity({"tokens": 10}, "model")
    await limiter.refund_capacity({"tokens": 0}, first)

    with caplog.at_level(logging.WARNING, logger="token_throttle"):
        await limiter.aclose()

    assert "limiter closed; 1 reservations still in flight may not be refundable" in (
        caplog.text
    )
    with pytest.raises(RuntimeError, match="closed"):
        await limiter.acquire_capacity({"tokens": 1}, "model")


def test_n01_sync_cross_limiter_memory_refund_is_unknown():
    limiter_a = SyncRateLimiter(_limited_config(), backend=SyncMemoryBackendBuilder())
    limiter_b = SyncRateLimiter(_limited_config(), backend=SyncMemoryBackendBuilder())

    reservation = limiter_a.acquire_capacity({"tokens": 30}, "model")
    limiter_b.acquire_capacity({"tokens": 1}, "model")
    before = _sync_capacity(limiter_b, "fam", "tokens")

    with pytest.raises(UnknownReservationError):
        limiter_b.refund_capacity({"tokens": 0}, reservation)

    assert reservation.reservation_id not in limiter_b._refunded_reservation_ids
    assert _sync_capacity(limiter_b, "fam", "tokens") - before < 1


def test_n04_sync_empty_projection_warns_and_commits_dedup():
    state = "tokens"

    def config_getter(_model: str) -> PerModelConfig:
        metric = "tokens" if state == "tokens" else "requests"
        return _limited_config(metric=metric)

    limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())
    reservation = limiter.acquire_capacity({"tokens": 30}, "model")

    state = "requests"
    with pytest.warns(RuntimeWarning, match="Refund dropped"):
        limiter.refund_capacity({"tokens": 0}, reservation)

    assert reservation.reservation_id in limiter._refunded_reservation_ids


def test_n09_sync_close_logs_outstanding_reservations_and_blocks_use(
    caplog: pytest.LogCaptureFixture,
):
    limiter = SyncRateLimiter(_limited_config(), backend=SyncMemoryBackendBuilder())
    reservation = limiter.acquire_capacity({"tokens": 10}, "model")
    limiter.acquire_capacity({"tokens": 10}, "model")
    limiter.refund_capacity({"tokens": 0}, reservation)

    with caplog.at_level(logging.WARNING, logger="token_throttle"):
        limiter.close()

    assert "limiter closed; 1 reservations still in flight may not be refundable" in (
        caplog.text
    )
    with pytest.raises(RuntimeError, match="closed"):
        limiter.acquire_capacity({"tokens": 1}, "model")


async def test_async_close_no_warning_on_clean_shutdown(
    caplog: pytest.LogCaptureFixture,
):
    limiter = RateLimiter(_limited_config(), backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 10}, "model")
    await limiter.refund_capacity({"tokens": 0}, reservation)

    with caplog.at_level(logging.WARNING, logger="token_throttle"):
        await limiter.aclose()

    assert "limiter closed" not in caplog.text


def test_sync_close_no_warning_on_clean_shutdown(
    caplog: pytest.LogCaptureFixture,
):
    limiter = SyncRateLimiter(_limited_config(), backend=SyncMemoryBackendBuilder())
    reservation = limiter.acquire_capacity({"tokens": 10}, "model")
    limiter.refund_capacity({"tokens": 0}, reservation)

    with caplog.at_level(logging.WARNING, logger="token_throttle"):
        limiter.close()

    assert "limiter closed" not in caplog.text
