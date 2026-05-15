"""Regression tests for FIX-50 exception contract v4."""

import asyncio
import multiprocessing
import pickle

import pytest
from frozendict import frozendict

from token_throttle._exceptions import (
    AcquireRefundFailedError,
    DuplicateRefundError,
)
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


def _reservation() -> CapacityReservation:
    return CapacityReservation(
        reservation_id="test-reservation",
        usage=frozendict({"tokens": 1.0}),
        model_family=MODEL_FAMILY,
        bucket_ids=frozenset({("tokens", 60)}),
        model=MODEL,
        limiter_instance_id="test-limiter",
    )


def _acquire_refund_failed_error() -> tuple[
    CapacityReservation,
    AcquireRefundFailedError,
]:
    reservation = _reservation()
    return reservation, AcquireRefundFailedError(
        reservation,
        interrupted_by=asyncio.CancelledError("client cancelled"),
        refund_error=RuntimeError("refund failed"),
    )


async def _raise_acquire_refund_failed_error(
    error: AcquireRefundFailedError,
) -> None:
    raise error


async def _raise_acquire_refund_failed_error_in_task_group(
    error: AcquireRefundFailedError,
) -> None:
    async with asyncio.TaskGroup() as task_group:
        task_group.create_task(_raise_acquire_refund_failed_error(error))


def _send_acquire_refund_failed_error(conn) -> None:
    _reservation, error = _acquire_refund_failed_error()
    conn.send(error)
    conn.close()


def _assert_acquire_refund_failed_error_round_trip(
    error: AcquireRefundFailedError,
    reservation: CapacityReservation,
) -> None:
    assert error.reservation == reservation
    assert isinstance(error.interrupted_by, asyncio.CancelledError)
    assert error.interrupted_by.args == ("client cancelled",)
    assert isinstance(error.refund_error, RuntimeError)
    assert error.refund_error.args == ("refund failed",)


async def test_wait_for_preserves_acquire_refund_failed_payload() -> None:
    reservation, error = _acquire_refund_failed_error()

    with pytest.raises(AcquireRefundFailedError) as exc_info:
        await asyncio.wait_for(_raise_acquire_refund_failed_error(error), timeout=1.0)

    assert exc_info.value is error
    _assert_acquire_refund_failed_error_round_trip(exc_info.value, reservation)


async def test_shield_preserves_acquire_refund_failed_payload() -> None:
    reservation, error = _acquire_refund_failed_error()

    with pytest.raises(AcquireRefundFailedError) as exc_info:
        await asyncio.shield(_raise_acquire_refund_failed_error(error))

    assert exc_info.value is error
    _assert_acquire_refund_failed_error_round_trip(exc_info.value, reservation)


async def test_gather_return_exceptions_preserves_acquire_refund_failed_payload() -> (
    None
):
    reservation, error = _acquire_refund_failed_error()

    result = await asyncio.gather(
        _raise_acquire_refund_failed_error(error),
        return_exceptions=True,
    )

    assert result == [error]
    _assert_acquire_refund_failed_error_round_trip(result[0], reservation)


async def test_task_group_preserves_acquire_refund_failed_payload() -> None:
    reservation, error = _acquire_refund_failed_error()

    with pytest.raises(ExceptionGroup) as exc_info:
        await _raise_acquire_refund_failed_error_in_task_group(error)

    matching, rest = exc_info.value.split(AcquireRefundFailedError)
    assert rest is None
    assert matching is not None
    assert matching.exceptions == (error,)
    _assert_acquire_refund_failed_error_round_trip(error, reservation)


def test_acquire_refund_failed_error_pickle_round_trip_preserves_payload() -> None:
    reservation, error = _acquire_refund_failed_error()

    round_tripped = pickle.loads(pickle.dumps(error))  # noqa: S301

    assert isinstance(round_tripped, AcquireRefundFailedError)
    _assert_acquire_refund_failed_error_round_trip(round_tripped, reservation)


def test_acquire_refund_failed_error_cross_process_round_trip_preserves_payload() -> (
    None
):
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_send_acquire_refund_failed_error,
        args=(child_conn,),
    )
    process.start()
    child_conn.close()
    try:
        assert parent_conn.poll(10), "child did not send exception"
        error = parent_conn.recv()
    finally:
        parent_conn.close()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert process.exitcode == 0
    assert isinstance(error, AcquireRefundFailedError)
    _assert_acquire_refund_failed_error_round_trip(error, _reservation())


async def test_async_duplicate_refund_error_reason_already_refunded() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 1}, MODEL)

    await limiter.refund_capacity({"tokens": 0}, reservation)
    with pytest.raises(DuplicateRefundError) as exc_info:
        await limiter.refund_capacity({"tokens": 0}, reservation)

    assert exc_info.value.reason == "already_refunded"


def test_sync_duplicate_refund_error_reason_already_refunded() -> None:
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    reservation = limiter.acquire_capacity({"tokens": 1}, MODEL)

    limiter.refund_capacity({"tokens": 0}, reservation)
    with pytest.raises(DuplicateRefundError) as exc_info:
        limiter.refund_capacity({"tokens": 0}, reservation)

    assert exc_info.value.reason == "already_refunded"


async def test_async_duplicate_refund_error_reason_in_progress() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 1}, MODEL)

    limiter._refund_in_progress.add(reservation.reservation_id)
    try:
        with pytest.raises(DuplicateRefundError) as exc_info:
            await limiter.refund_capacity({"tokens": 0}, reservation)
        assert exc_info.value.reason == "in_progress"
    finally:
        limiter._refund_in_progress.discard(reservation.reservation_id)
        await limiter.refund_capacity({"tokens": 0}, reservation)


def test_sync_duplicate_refund_error_reason_in_progress() -> None:
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    reservation = limiter.acquire_capacity({"tokens": 1}, MODEL)

    limiter._refund_in_progress.add(reservation.reservation_id)
    try:
        with pytest.raises(DuplicateRefundError) as exc_info:
            limiter.refund_capacity({"tokens": 0}, reservation)
        assert exc_info.value.reason == "in_progress"
    finally:
        limiter._refund_in_progress.discard(reservation.reservation_id)
        limiter.refund_capacity({"tokens": 0}, reservation)


async def test_async_duplicate_refund_error_reason_duplicate_acquire() -> None:
    backend = MemoryBackendBuilder().build(_config())

    await backend.consume_capacity(frozendict({"tokens": 1.0}), reservation_id="same")
    with pytest.raises(DuplicateRefundError) as exc_info:
        await backend.consume_capacity(
            frozendict({"tokens": 1.0}),
            reservation_id="same",
        )

    assert exc_info.value.reason == "duplicate_acquire"


def test_sync_duplicate_refund_error_reason_duplicate_acquire() -> None:
    backend = SyncMemoryBackendBuilder().build(_config())

    backend.consume_capacity(frozendict({"tokens": 1.0}), reservation_id="same")
    with pytest.raises(DuplicateRefundError) as exc_info:
        backend.consume_capacity(frozendict({"tokens": 1.0}), reservation_id="same")

    assert exc_info.value.reason == "duplicate_acquire"


async def test_async_callback_propagates_acquire_refund_failed_error() -> None:
    backend = MemoryBackendBuilder().build(_config())
    _reservation, error = _acquire_refund_failed_error()

    async def callback(**_kwargs) -> None:
        raise error

    with pytest.raises(AcquireRefundFailedError) as exc_info:
        await backend._invoke_callback_safe(callback)

    assert exc_info.value is error


def test_sync_callback_propagates_acquire_refund_failed_error() -> None:
    backend = SyncMemoryBackendBuilder().build(_config())
    _reservation, error = _acquire_refund_failed_error()

    def callback(**_kwargs) -> None:
        raise error

    with pytest.raises(AcquireRefundFailedError) as exc_info:
        backend._invoke_callback_safe(callback)

    assert exc_info.value is error
