"""
Unit tests for the ``_refund_or_forget_reservation_on_raise`` context-manager
helpers on ``RateLimiter`` and ``SyncRateLimiter``.

These helpers wrap reservation lifecycle emission such that any exception
escaping the body triggers either ``_forget_in_flight_reservation`` (forget
branch, used for unlimited reservations and non-blocking acquire) or
``_refund_undelivered_acquire_or_deliver`` / ``_finalize_and_refund_undelivered_acquire``
(refund branch, used for blocking-acquire delivery cleanup).

Sync helper signature is asymmetric with async: it accepts ``model`` because
``_finalize_and_refund_undelivered_acquire`` does its own finalize and needs
the model name. The plan KNOWN UNKNOWN #2 documents the asymmetry; this test
suite also exercises the fail-fast path.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from token_throttle import RateLimiter, SyncRateLimiter
from token_throttle._exceptions import AcquireRefundFailedError
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import (
    CapacityReservation,
    Quota,
    UsageQuotas,
)
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)


def _cfg() -> PerModelConfig:
    return PerModelConfig(
        model_family="test/model",
        quotas=UsageQuotas([Quota(metric="requests", limit=20, per_seconds=60)]),
    )


def _async_limiter() -> RateLimiter:
    return RateLimiter(_cfg(), backend=MemoryBackendBuilder())


def _sync_limiter() -> SyncRateLimiter:
    return SyncRateLimiter(_cfg(), backend=SyncMemoryBackendBuilder())


def _reservation(limiter_id: str) -> CapacityReservation:
    return CapacityReservation(
        usage={"requests": 1},
        model_family="test/model",
        limiter_instance_id=limiter_id,
    )


# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------


async def test_async_helper_no_raise_does_nothing() -> None:
    limiter = _async_limiter()
    reservation = _reservation(limiter._limiter_instance_id)
    forget = MagicMock()
    refund = AsyncMock()
    limiter._forget_in_flight_reservation = forget  # type: ignore[method-assign]
    limiter._refund_undelivered_acquire_or_deliver = refund  # type: ignore[method-assign]

    async with limiter._refund_or_forget_reservation_on_raise(
        reservation, refund_undelivered=False
    ):
        pass

    forget.assert_not_called()
    refund.assert_not_called()


async def test_async_helper_forget_branch_on_cancelled_error() -> None:
    limiter = _async_limiter()
    reservation = _reservation(limiter._limiter_instance_id)
    forget = MagicMock()
    refund = AsyncMock()
    limiter._forget_in_flight_reservation = forget  # type: ignore[method-assign]
    limiter._refund_undelivered_acquire_or_deliver = refund  # type: ignore[method-assign]

    exc = asyncio.CancelledError("forced")
    with pytest.raises(asyncio.CancelledError):
        async with limiter._refund_or_forget_reservation_on_raise(
            reservation, refund_undelivered=False
        ):
            raise exc

    forget.assert_called_once_with(reservation.reservation_id)
    refund.assert_not_called()


async def test_async_helper_refund_branch_on_cancelled_error_passes_interrupted_by() -> (
    None
):
    limiter = _async_limiter()
    reservation = _reservation(limiter._limiter_instance_id)
    forget = MagicMock()
    refund = AsyncMock()
    limiter._forget_in_flight_reservation = forget  # type: ignore[method-assign]
    limiter._refund_undelivered_acquire_or_deliver = refund  # type: ignore[method-assign]

    exc = asyncio.CancelledError("forced")
    with pytest.raises(asyncio.CancelledError):
        async with limiter._refund_or_forget_reservation_on_raise(
            reservation, refund_undelivered=True
        ):
            raise exc

    forget.assert_not_called()
    refund.assert_called_once()
    args, kwargs = refund.call_args
    assert args[0] is reservation
    assert kwargs["interrupted_by"] is exc


async def test_async_helper_refund_failure_propagates_acquire_refund_failed_error() -> (
    None
):
    limiter = _async_limiter()
    reservation = _reservation(limiter._limiter_instance_id)

    async def _refund_raises_acquire_refund_failed(
        _reservation: CapacityReservation, *, interrupted_by: BaseException
    ) -> None:
        raise AcquireRefundFailedError(
            reservation=_reservation,
            refund_error=RuntimeError("refund failed"),
            interrupted_by=interrupted_by,
        )

    limiter._refund_undelivered_acquire_or_deliver = (  # type: ignore[method-assign]
        _refund_raises_acquire_refund_failed
    )

    with pytest.raises(AcquireRefundFailedError):
        async with limiter._refund_or_forget_reservation_on_raise(
            reservation, refund_undelivered=True
        ):
            raise asyncio.CancelledError("forced")


# ---------------------------------------------------------------------------
# Sync helper
# ---------------------------------------------------------------------------


def test_sync_helper_no_raise_does_nothing() -> None:
    limiter = _sync_limiter()
    reservation = _reservation(limiter._limiter_instance_id)
    forget = MagicMock()
    finalize_refund = MagicMock()
    limiter._forget_in_flight_reservation = forget  # type: ignore[method-assign]
    limiter._finalize_and_refund_undelivered_acquire = finalize_refund  # type: ignore[method-assign]

    with limiter._refund_or_forget_reservation_on_raise(
        reservation, refund_undelivered=False
    ):
        pass

    forget.assert_not_called()
    finalize_refund.assert_not_called()


def test_sync_helper_forget_branch_on_cancelled_error() -> None:
    limiter = _sync_limiter()
    reservation = _reservation(limiter._limiter_instance_id)
    forget = MagicMock()
    finalize_refund = MagicMock()
    limiter._forget_in_flight_reservation = forget  # type: ignore[method-assign]
    limiter._finalize_and_refund_undelivered_acquire = finalize_refund  # type: ignore[method-assign]

    exc = asyncio.CancelledError("forced")
    with (
        pytest.raises(asyncio.CancelledError),
        limiter._refund_or_forget_reservation_on_raise(
            reservation, refund_undelivered=False
        ),
    ):
        raise exc

    forget.assert_called_once_with(reservation.reservation_id)
    finalize_refund.assert_not_called()


def test_sync_helper_refund_branch_on_cancelled_error_passes_interrupted_by() -> None:
    limiter = _sync_limiter()
    reservation = _reservation(limiter._limiter_instance_id)
    forget = MagicMock()
    finalize_refund = MagicMock()
    limiter._forget_in_flight_reservation = forget  # type: ignore[method-assign]
    limiter._finalize_and_refund_undelivered_acquire = finalize_refund  # type: ignore[method-assign]

    exc = asyncio.CancelledError("forced")
    with (
        pytest.raises(asyncio.CancelledError),
        limiter._refund_or_forget_reservation_on_raise(
            reservation, "test-model", refund_undelivered=True
        ),
    ):
        raise exc

    forget.assert_not_called()
    finalize_refund.assert_called_once()
    args, kwargs = finalize_refund.call_args
    assert args[0] is reservation
    assert args[1] == "test-model"
    assert kwargs["interrupted_by"] is exc


def test_sync_helper_refund_failure_propagates_acquire_refund_failed_error() -> None:
    limiter = _sync_limiter()
    reservation = _reservation(limiter._limiter_instance_id)

    def _finalize_refund_raises(
        _reservation: CapacityReservation,
        _model: str,
        *,
        interrupted_by: BaseException,
    ) -> None:
        raise AcquireRefundFailedError(
            reservation=_reservation,
            refund_error=RuntimeError("refund failed"),
            interrupted_by=interrupted_by,
        )

    limiter._finalize_and_refund_undelivered_acquire = (  # type: ignore[method-assign]
        _finalize_refund_raises
    )

    with (
        pytest.raises(AcquireRefundFailedError),
        limiter._refund_or_forget_reservation_on_raise(
            reservation, "test-model", refund_undelivered=True
        ),
    ):
        raise asyncio.CancelledError("forced")


def test_sync_helper_fails_fast_when_refund_requested_without_model() -> None:
    """KNOWN UNKNOWN #2: sync helper's ``model`` is optional but required on the
    refund branch. If the caller asks for refund without supplying a model,
    raise ``ValueError`` immediately rather than crashing inside finalize.
    """
    limiter = _sync_limiter()
    reservation = _reservation(limiter._limiter_instance_id)

    with (
        pytest.raises(ValueError, match="model is required"),
        limiter._refund_or_forget_reservation_on_raise(
            reservation, refund_undelivered=True
        ),
    ):
        pass


def test_sync_helper_forget_branch_accepts_missing_model() -> None:
    """Forget branch has no use for ``model`` — caller may omit it."""
    limiter = _sync_limiter()
    reservation = _reservation(limiter._limiter_instance_id)
    forget = MagicMock()
    limiter._forget_in_flight_reservation = forget  # type: ignore[method-assign]

    with (
        pytest.raises(RuntimeError),
        limiter._refund_or_forget_reservation_on_raise(
            reservation, refund_undelivered=False
        ),
    ):
        raise RuntimeError("boom")

    forget.assert_called_once_with(reservation.reservation_id)
