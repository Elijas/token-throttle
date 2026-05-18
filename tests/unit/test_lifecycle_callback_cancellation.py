"""Regression tests for lifecycle callback cancellation propagation."""

import asyncio
import concurrent.futures

import pytest

from token_throttle import (
    CapacityReservation,
    PerModelConfig,
    Quota,
    RateLimiter,
    RateLimiterCallbacks,
    SyncRateLimiter,
    SyncRateLimiterCallbacks,
    UsageQuotas,
)
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)

MODEL = "test-model"
MODEL_FAMILY = "test-family"


def _config(*, limit: float = 10.0) -> PerModelConfig:
    return PerModelConfig(
        model_family=MODEL_FAMILY,
        quotas=UsageQuotas([Quota(metric="tokens", limit=limit, per_seconds=3600)]),
    )


def _flatten_group(exc: BaseException) -> list[BaseException]:
    if not isinstance(exc, BaseExceptionGroup):
        return [exc]
    leaves: list[BaseException] = []
    for nested in exc.exceptions:
        leaves.extend(_flatten_group(nested))
    return leaves


async def test_async_lifecycle_callback_cancellation_propagates_and_refunds() -> None:
    callback_entered = asyncio.Event()
    block_next_consumed = True

    async def on_lifecycle_event(*, event) -> None:
        nonlocal block_next_consumed
        if event.event_type != "capacity_consumed" or not block_next_consumed:
            return
        block_next_consumed = False
        callback_entered.set()
        await asyncio.sleep(60)

    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        callbacks=RateLimiterCallbacks(on_lifecycle_event=on_lifecycle_event),
    )

    task = asyncio.create_task(limiter.acquire_capacity({"tokens": 10}, MODEL))
    await asyncio.wait_for(callback_entered.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert limiter.snapshot_state()["in_flight_reservations"] == 0
    reservation = await limiter.acquire_capacity({"tokens": 10}, MODEL, timeout=0)
    await limiter.refund_capacity({"tokens": 0}, reservation)


async def test_async_lifecycle_callback_grouped_cancellation_propagates() -> None:
    async def on_lifecycle_event(*, event) -> None:
        if event.event_type == "capacity_consumed":
            raise BaseExceptionGroup(
                "grouped lifecycle cancellation",
                [asyncio.CancelledError("cancelled"), ValueError("ordinary")],
            )

    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        callbacks=RateLimiterCallbacks(on_lifecycle_event=on_lifecycle_event),
    )

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await limiter.acquire_capacity({"tokens": 10}, MODEL)

    leaves = _flatten_group(exc_info.value)
    assert any(isinstance(leaf, asyncio.CancelledError) for leaf in leaves)
    assert limiter.snapshot_state()["in_flight_reservations"] == 0


def test_sync_lifecycle_callback_asyncio_cancelled_error_propagates_and_refunds() -> (
    None
):
    raise_next_consumed = True

    def on_lifecycle_event(*, event) -> None:
        nonlocal raise_next_consumed
        if event.event_type != "capacity_consumed" or not raise_next_consumed:
            return
        raise_next_consumed = False
        raise asyncio.CancelledError("sync callback cancellation")

    limiter = SyncRateLimiter(
        _config(),
        backend=SyncMemoryBackendBuilder(),
        callbacks=SyncRateLimiterCallbacks(on_lifecycle_event=on_lifecycle_event),
    )

    with pytest.raises(asyncio.CancelledError):
        limiter.acquire_capacity({"tokens": 10}, MODEL)

    assert limiter.snapshot_state()["in_flight_reservations"] == 0
    reservation = limiter.acquire_capacity({"tokens": 10}, MODEL, timeout=0)
    limiter.refund_capacity({"tokens": 0}, reservation)


def test_sync_lifecycle_callback_futures_cancelled_error_propagates_and_refunds() -> (
    None
):
    raise_next_consumed = True

    def on_lifecycle_event(*, event) -> None:
        nonlocal raise_next_consumed
        if event.event_type != "capacity_consumed" or not raise_next_consumed:
            return
        raise_next_consumed = False
        raise concurrent.futures.CancelledError("sync future cancellation")

    limiter = SyncRateLimiter(
        _config(),
        backend=SyncMemoryBackendBuilder(),
        callbacks=SyncRateLimiterCallbacks(on_lifecycle_event=on_lifecycle_event),
    )

    with pytest.raises(concurrent.futures.CancelledError):
        limiter.acquire_capacity({"tokens": 10}, MODEL)

    assert limiter.snapshot_state()["in_flight_reservations"] == 0
    reservation = limiter.acquire_capacity({"tokens": 10}, MODEL, timeout=0)
    limiter.refund_capacity({"tokens": 0}, reservation)


async def test_async_record_usage_lifecycle_cancellation_forgets_reservation_only() -> (
    None
):
    raise_next_consumed = True

    async def on_lifecycle_event(*, event) -> None:
        nonlocal raise_next_consumed
        if event.event_type != "capacity_consumed" or not raise_next_consumed:
            return
        raise_next_consumed = False
        raise asyncio.CancelledError("record lifecycle cancellation")

    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        callbacks=RateLimiterCallbacks(on_lifecycle_event=on_lifecycle_event),
    )

    with pytest.raises(asyncio.CancelledError):
        await limiter.record_usage({"tokens": 10}, MODEL)

    assert limiter.snapshot_state()["in_flight_reservations"] == 0
    with pytest.raises(TimeoutError):
        await limiter.acquire_capacity({"tokens": 10}, MODEL, timeout=0)


async def test_async_record_usage_finalize_cancellation_forgets_reservation_only() -> (
    None
):
    limiter = RateLimiter(_config(limit=100.0), backend=MemoryBackendBuilder())
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

    task = asyncio.create_task(limiter.record_usage({"tokens": 10}, MODEL))
    await asyncio.wait_for(finalize_entered.wait(), timeout=1.0)

    task.cancel()
    release_finalize.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert limiter.snapshot_state()["in_flight_reservations"] == 0
    with pytest.raises(TimeoutError):
        await limiter.acquire_capacity({"tokens": 91}, MODEL, timeout=0)
    reservation = await limiter.acquire_capacity({"tokens": 90}, MODEL, timeout=0)
    await limiter.refund_capacity({"tokens": 0}, reservation)


def test_sync_record_usage_lifecycle_cancellation_forgets_reservation_only() -> None:
    raise_next_consumed = True

    def on_lifecycle_event(*, event) -> None:
        nonlocal raise_next_consumed
        if event.event_type != "capacity_consumed" or not raise_next_consumed:
            return
        raise_next_consumed = False
        raise asyncio.CancelledError("record lifecycle cancellation")

    limiter = SyncRateLimiter(
        _config(),
        backend=SyncMemoryBackendBuilder(),
        callbacks=SyncRateLimiterCallbacks(on_lifecycle_event=on_lifecycle_event),
    )

    with pytest.raises(asyncio.CancelledError):
        limiter.record_usage({"tokens": 10}, MODEL)

    assert limiter.snapshot_state()["in_flight_reservations"] == 0
    with pytest.raises(TimeoutError):
        limiter.acquire_capacity({"tokens": 10}, MODEL, timeout=0)


async def test_async_fallback_refund_lifecycle_cancellation_is_not_refund_failure() -> (
    None
):
    consumed_callback_entered = asyncio.Event()
    block_next_consumed = True
    cancel_next_refunded = True

    async def on_lifecycle_event(*, event) -> None:
        nonlocal block_next_consumed, cancel_next_refunded
        if event.event_type == "capacity_consumed" and block_next_consumed:
            block_next_consumed = False
            consumed_callback_entered.set()
            await asyncio.sleep(60)
        if event.event_type == "capacity_refunded" and cancel_next_refunded:
            cancel_next_refunded = False
            raise asyncio.CancelledError("refund lifecycle cancellation")

    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        callbacks=RateLimiterCallbacks(on_lifecycle_event=on_lifecycle_event),
    )

    task = asyncio.create_task(limiter.acquire_capacity({"tokens": 10}, MODEL))
    await asyncio.wait_for(consumed_callback_entered.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert limiter.snapshot_state()["in_flight_reservations"] == 0
    reservation = await limiter.acquire_capacity({"tokens": 10}, MODEL, timeout=0)
    await limiter.refund_capacity({"tokens": 0}, reservation)


def test_sync_fallback_refund_lifecycle_cancellation_is_not_refund_failure() -> None:
    cancel_next_consumed = True
    cancel_next_refunded = True

    def on_lifecycle_event(*, event) -> None:
        nonlocal cancel_next_consumed, cancel_next_refunded
        if event.event_type == "capacity_consumed" and cancel_next_consumed:
            cancel_next_consumed = False
            raise asyncio.CancelledError("consume lifecycle cancellation")
        if event.event_type == "capacity_refunded" and cancel_next_refunded:
            cancel_next_refunded = False
            raise asyncio.CancelledError("refund lifecycle cancellation")

    limiter = SyncRateLimiter(
        _config(),
        backend=SyncMemoryBackendBuilder(),
        callbacks=SyncRateLimiterCallbacks(on_lifecycle_event=on_lifecycle_event),
    )

    with pytest.raises(asyncio.CancelledError):
        limiter.acquire_capacity({"tokens": 10}, MODEL)

    assert limiter.snapshot_state()["in_flight_reservations"] == 0
    reservation = limiter.acquire_capacity({"tokens": 10}, MODEL, timeout=0)
    limiter.refund_capacity({"tokens": 0}, reservation)
