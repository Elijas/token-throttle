"""Regression tests for FIX-31 LK acquire transactionality."""

import asyncio

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

MODEL = "test-model"
MODEL_FAMILY = "test-family"


def _config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=60)]),
        model_family=MODEL_FAMILY,
    )


async def _assert_async_cancel_after_backend_success_is_refundable(
    method_name: str,
) -> None:
    cfg = _config()
    limiter = RateLimiter(cfg, backend=MemoryBackendBuilder())
    finalize_entered = asyncio.Event()
    release_finalize = asyncio.Event()
    captured: list[CapacityReservation] = []
    original_finalize = limiter._finalize_pending_acquire

    async def controlled_finalize(
        reservation: CapacityReservation,
        model: str,
    ) -> None:
        captured.append(reservation)
        finalize_entered.set()
        await release_finalize.wait()
        await original_finalize(reservation, model)

    limiter._finalize_pending_acquire = controlled_finalize

    method = getattr(limiter, method_name)
    task = asyncio.create_task(method({"tokens": 10}, MODEL))
    await asyncio.wait_for(finalize_entered.wait(), timeout=1.0)
    assert len(limiter._pending_acquire_reservations) == 1

    task.cancel()
    release_finalize.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    reservation = captured[0]
    if method_name == "acquire_capacity":
        assert reservation.reservation_id not in limiter._in_flight_reservation_ids
        assert reservation.reservation_id not in limiter._pending_acquire_reservations
        next_reservation = await limiter.acquire_capacity(
            {"tokens": 100},
            MODEL,
            timeout=0,
        )
        await limiter.refund_capacity({"tokens": 0}, next_reservation)
        return

    assert reservation.reservation_id not in limiter._in_flight_reservation_ids
    assert reservation.reservation_id not in limiter._pending_acquire_reservations
    with pytest.raises(TimeoutError):
        await limiter.acquire_capacity({"tokens": 91}, MODEL, timeout=0)
    next_reservation = await limiter.acquire_capacity({"tokens": 90}, MODEL, timeout=0)

    await limiter.refund_capacity({"tokens": 0}, next_reservation)
    assert next_reservation.reservation_id not in limiter._in_flight_reservation_ids


class TestAsyncAcquireTransaction:
    async def test_lk01_cancel_after_acquire_backend_success_keeps_reservation_refundable(
        self,
    ):
        for _ in range(100):
            await _assert_async_cancel_after_backend_success_is_refundable(
                "acquire_capacity"
            )

    async def test_record_usage_cancel_after_backend_success_keeps_reservation_refundable(
        self,
    ):
        await _assert_async_cancel_after_backend_success_is_refundable("record_usage")

    async def test_backend_failure_cleans_pending_acquire(self):
        cfg = _config()
        limiter = RateLimiter(cfg, backend=MemoryBackendBuilder())
        backend = await limiter._get_backend(cfg)

        async def fail_after_no_consume(*_args, **_kwargs) -> None:
            raise RuntimeError("simulated backend failure")

        backend.await_for_capacity = fail_after_no_consume

        with pytest.raises(RuntimeError, match="simulated backend failure"):
            await limiter.acquire_capacity({"tokens": 10}, MODEL)

        assert limiter._pending_acquire_reservations == set()
        assert limiter._in_flight_reservation_ids == set()

    async def test_backend_cancel_during_acquire_cleans_pending(self):
        cfg = _config()
        limiter = RateLimiter(cfg, backend=MemoryBackendBuilder())
        backend = await limiter._get_backend(cfg)

        async def cancel_before_consume(*_args, **_kwargs) -> None:
            raise asyncio.CancelledError

        backend.await_for_capacity = cancel_before_consume

        with pytest.raises(asyncio.CancelledError):
            await limiter.acquire_capacity({"tokens": 10}, MODEL)

        assert limiter._pending_acquire_reservations == set()
        assert limiter._in_flight_reservation_ids == set()

    async def test_record_usage_backend_failure_cleans_pending(self):
        cfg = _config()
        limiter = RateLimiter(cfg, backend=MemoryBackendBuilder())
        backend = await limiter._get_backend(cfg)

        async def fail_after_no_consume(*_args, **_kwargs) -> None:
            raise RuntimeError("simulated consume failure")

        backend.consume_capacity = fail_after_no_consume

        with pytest.raises(RuntimeError, match="simulated consume failure"):
            await limiter.record_usage({"tokens": 10}, MODEL)

        assert limiter._pending_acquire_reservations == set()
        assert limiter._in_flight_reservation_ids == set()

    async def test_record_usage_backend_cancel_cleans_pending(self):
        cfg = _config()
        limiter = RateLimiter(cfg, backend=MemoryBackendBuilder())
        backend = await limiter._get_backend(cfg)

        async def cancel_before_consume(*_args, **_kwargs) -> None:
            raise asyncio.CancelledError

        backend.consume_capacity = cancel_before_consume

        with pytest.raises(asyncio.CancelledError):
            await limiter.record_usage({"tokens": 10}, MODEL)

        assert limiter._pending_acquire_reservations == set()
        assert limiter._in_flight_reservation_ids == set()


class TestSyncAcquireTransaction:
    @pytest.mark.parametrize(
        ("method_name", "backend_method_name"),
        [
            ("acquire_capacity", "wait_for_capacity"),
            ("record_usage", "consume_capacity"),
        ],
    )
    def test_success_moves_pending_to_in_flight(
        self,
        method_name: str,
        backend_method_name: str,
    ):
        cfg = _config()
        limiter = SyncRateLimiter(cfg, backend=SyncMemoryBackendBuilder())
        backend = limiter._get_backend(cfg)
        original_backend_method = getattr(backend, backend_method_name)
        observed_pending: set[str] = set()

        def wrapped_backend_method(*args, **kwargs) -> None:
            observed_pending.update(limiter._pending_acquire_reservations)
            original_backend_method(*args, **kwargs)

        setattr(backend, backend_method_name, wrapped_backend_method)

        reservation = getattr(limiter, method_name)({"tokens": 10}, MODEL)

        assert observed_pending == {reservation.reservation_id}
        assert limiter._pending_acquire_reservations == set()
        assert reservation.reservation_id in limiter._in_flight_reservation_ids

    @pytest.mark.parametrize(
        ("method_name", "backend_method_name"),
        [
            ("acquire_capacity", "wait_for_capacity"),
            ("record_usage", "consume_capacity"),
        ],
    )
    def test_backend_failure_cleans_pending(
        self,
        method_name: str,
        backend_method_name: str,
    ):
        cfg = _config()
        limiter = SyncRateLimiter(cfg, backend=SyncMemoryBackendBuilder())
        backend = limiter._get_backend(cfg)

        def fail_after_no_consume(*_args, **_kwargs) -> None:
            raise RuntimeError("simulated sync backend failure")

        setattr(backend, backend_method_name, fail_after_no_consume)

        with pytest.raises(RuntimeError, match="simulated sync backend failure"):
            getattr(limiter, method_name)({"tokens": 10}, MODEL)

        assert limiter._pending_acquire_reservations == set()
        assert limiter._in_flight_reservation_ids == set()
