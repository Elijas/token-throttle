"""Regression tests for FIX-43 acquire delivery transactionality."""

import asyncio

import pytest
from frozendict import frozendict

from token_throttle._exceptions import AcquireRefundFailedError
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
        quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=3600)]),
        model_family=MODEL_FAMILY,
    )


async def test_async_cancel_after_backend_consume_refunds_undelivered_reservation():
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    finalize_entered = asyncio.Event()
    release_finalize = asyncio.Event()
    original_finalize = limiter._finalize_pending_acquire

    async def controlled_finalize(
        reservation: CapacityReservation,
        model: str,
    ) -> None:
        finalize_entered.set()
        await release_finalize.wait()
        await original_finalize(reservation, model)

    limiter._finalize_pending_acquire = controlled_finalize

    task = asyncio.create_task(limiter.acquire_capacity({"tokens": 60}, MODEL))
    await asyncio.wait_for(finalize_entered.wait(), timeout=1.0)

    task.cancel()
    release_finalize.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert limiter._pending_acquire_reservations == set()
    assert limiter._in_flight_reservation_ids == set()

    reservation = await limiter.acquire_capacity({"tokens": 100}, MODEL, timeout=0)
    await limiter.refund_capacity({"tokens": 0}, reservation)


async def test_async_cancel_after_backend_consume_refund_failure_delivers_reservation():
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    finalize_entered = asyncio.Event()
    release_finalize = asyncio.Event()
    original_finalize = limiter._finalize_pending_acquire
    original_refund = limiter.refund_capacity

    async def controlled_finalize(
        reservation: CapacityReservation,
        model: str,
    ) -> None:
        finalize_entered.set()
        await release_finalize.wait()
        await original_finalize(reservation, model)

    async def failing_refund(*_args, **_kwargs) -> None:
        raise RuntimeError("simulated refund failure")

    limiter._finalize_pending_acquire = controlled_finalize
    limiter.refund_capacity = failing_refund

    task = asyncio.create_task(limiter.acquire_capacity({"tokens": 60}, MODEL))
    await asyncio.wait_for(finalize_entered.wait(), timeout=1.0)

    task.cancel()
    release_finalize.set()
    with pytest.raises(AcquireRefundFailedError) as exc_info:
        await asyncio.wait_for(task, timeout=1.0)

    error = exc_info.value
    assert isinstance(error, asyncio.CancelledError)
    assert isinstance(error.refund_error, RuntimeError)
    assert error.reservation.reservation_id in limiter._in_flight_reservation_ids
    assert limiter._pending_acquire_reservations == set()

    limiter.refund_capacity = original_refund
    await limiter.refund_capacity({"tokens": 0}, error.reservation)
    reservation = await limiter.acquire_capacity({"tokens": 100}, MODEL, timeout=0)
    await limiter.refund_capacity({"tokens": 0}, reservation)


def test_acquire_refund_failed_error_is_caught_as_cancelled_error():
    reservation = CapacityReservation(
        usage=frozendict({"tokens": 1.0}),
        model_family=MODEL_FAMILY,
        bucket_ids=frozenset({("tokens", 3600)}),
        model=MODEL,
        limiter_instance_id="test-limiter",
    )

    with pytest.raises(asyncio.CancelledError) as exc_info:
        raise AcquireRefundFailedError(
            reservation=reservation,
            refund_error=RuntimeError("simulated refund failure"),
        )

    assert isinstance(exc_info.value, AcquireRefundFailedError)
    assert exc_info.value.reservation is reservation


async def test_async_cancel_before_backend_consume_does_not_consume_capacity():
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    first = await limiter.acquire_capacity({"tokens": 100}, MODEL)
    backend = await limiter._get_backend(_config())
    original_await_for_capacity = backend.await_for_capacity
    wait_entered = asyncio.Event()

    async def observed_await_for_capacity(*args, **kwargs) -> None:
        wait_entered.set()
        await original_await_for_capacity(*args, **kwargs)

    backend.await_for_capacity = observed_await_for_capacity

    task = asyncio.create_task(limiter.acquire_capacity({"tokens": 10}, MODEL))
    await asyncio.wait_for(wait_entered.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    await limiter.refund_capacity({"tokens": 0}, first)
    reservation = await limiter.acquire_capacity({"tokens": 100}, MODEL, timeout=0)
    await limiter.refund_capacity({"tokens": 0}, reservation)


def test_sync_baseexception_after_backend_consume_refunds_undelivered_reservation():
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    original_finalize = limiter._finalize_pending_acquire
    interrupted = False

    def interrupt_once_before_finalize(
        reservation: CapacityReservation,
        model: str,
    ) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise SystemExit("simulated post-consume interrupt")
        original_finalize(reservation, model)

    limiter._finalize_pending_acquire = interrupt_once_before_finalize

    with pytest.raises(SystemExit, match="simulated post-consume interrupt"):
        limiter.acquire_capacity({"tokens": 60}, MODEL)

    limiter._finalize_pending_acquire = original_finalize

    assert interrupted
    assert limiter._pending_acquire_reservations == set()
    assert limiter._in_flight_reservation_ids == set()

    reservation = limiter.acquire_capacity({"tokens": 100}, MODEL, timeout=0)
    limiter.refund_capacity({"tokens": 0}, reservation)


def test_sync_baseexception_after_backend_consume_refund_failure_delivers_reservation():
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    original_finalize = limiter._finalize_pending_acquire
    original_refund = limiter.refund_capacity
    interrupted = False

    def interrupt_once_before_finalize(
        reservation: CapacityReservation,
        model: str,
    ) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise SystemExit("simulated post-consume interrupt")
        original_finalize(reservation, model)

    def failing_refund(*_args, **_kwargs) -> None:
        raise RuntimeError("simulated refund failure")

    limiter._finalize_pending_acquire = interrupt_once_before_finalize
    limiter.refund_capacity = failing_refund

    with pytest.raises(AcquireRefundFailedError) as exc_info:
        limiter.acquire_capacity({"tokens": 60}, MODEL)

    error = exc_info.value
    assert interrupted
    assert isinstance(error.interrupted_by, SystemExit)
    assert isinstance(error.refund_error, RuntimeError)
    assert error.reservation.reservation_id in limiter._in_flight_reservation_ids
    assert limiter._pending_acquire_reservations == set()

    limiter._finalize_pending_acquire = original_finalize
    limiter.refund_capacity = original_refund
    limiter.refund_capacity({"tokens": 0}, error.reservation)
    reservation = limiter.acquire_capacity({"tokens": 100}, MODEL, timeout=0)
    limiter.refund_capacity({"tokens": 0}, reservation)
