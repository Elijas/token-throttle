from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from token_throttle._interfaces._models import CapacityReservation

DuplicateRefundReason = Literal[
    "already_refunded",
    "in_progress",
    "duplicate_acquire",
]

_DUPLICATE_REFUND_REASONS = frozenset(
    {
        "already_refunded",
        "in_progress",
        "duplicate_acquire",
    }
)

_DUPLICATE_REFUND_MESSAGES: dict[str, str] = {
    "already_refunded": "reservation already refunded",
    "in_progress": "reservation refund already in progress",
    "duplicate_acquire": "reservation already acquired",
}


def _infer_duplicate_refund_reason(message: str | None) -> DuplicateRefundReason:
    if message == _DUPLICATE_REFUND_MESSAGES["in_progress"]:
        return "in_progress"
    if message == _DUPLICATE_REFUND_MESSAGES["duplicate_acquire"]:
        return "duplicate_acquire"
    return "already_refunded"


def _rebuild_acquire_refund_failed_error(
    reservation: CapacityReservation,
    interrupted_by: BaseException | None,
    refund_error: BaseException | None,
) -> AcquireRefundFailedError:
    return AcquireRefundFailedError(
        reservation,
        interrupted_by=interrupted_by,
        refund_error=refund_error,
    )


class AcquireRefundFailedError(Exception):
    """
    Raised when interrupted acquire delivery cannot refund reserved capacity.

    This exception is a regular ``Exception``. It is not an
    ``asyncio.CancelledError`` subclass; catch this class directly to recover
    the delivered ``.reservation`` and inspect ``.interrupted_by`` or
    ``.refund_error``.
    """

    _MESSAGE = (
        "acquire was interrupted after capacity was reserved, and the fallback "
        "refund failed; inspect .reservation to refund or use the reservation "
        "explicitly"
    )
    reason = "acquire_refund_failed"

    def __init__(
        self,
        reservation: CapacityReservation,
        interrupted_by: BaseException | None = None,
        refund_error: BaseException | None = None,
    ) -> None:
        super().__init__(self._MESSAGE)
        self.reservation = reservation
        self.refund_error = refund_error
        self.interrupted_by = interrupted_by

    def __reduce__(self):
        return (
            _rebuild_acquire_refund_failed_error,
            (self.reservation, self.interrupted_by, self.refund_error),
        )


class CardinalityLimitExceededError(ValueError):
    """Raised when a mandatory limiter cardinality or length cap is exceeded."""

    reason = "cardinality_limit_exceeded"


class BackendConformanceError(Exception):
    """Raised when a backend fails the public conformance contract."""

    reason = "backend_conformance_error"


class DuplicateRefundError(ValueError):
    """Raised when a refund is duplicate, already in progress, or already acquired."""

    def __init__(
        self,
        message: str | None = None,
        *,
        reason: DuplicateRefundReason | None = None,
    ) -> None:
        if reason is None:
            reason = _infer_duplicate_refund_reason(message)
        if reason not in _DUPLICATE_REFUND_REASONS:
            raise ValueError(f"unknown duplicate refund reason: {reason!r}")
        if message is None:
            message = _DUPLICATE_REFUND_MESSAGES[reason]
        super().__init__(message)
        self.reason = reason


class UnknownReservationError(ValueError):
    """Raised when a backend has no record that a reservation was acquired."""

    reason = "unknown_reservation"


_UNKNOWN_RESERVATION_FORGET_IN_FLIGHT_ATTR = (
    "_token_throttle_forget_in_flight_on_unknown"
)


def _mark_unknown_reservation_forget_in_flight(
    exc: UnknownReservationError,
) -> UnknownReservationError:
    setattr(exc, _UNKNOWN_RESERVATION_FORGET_IN_FLIGHT_ATTR, True)
    return exc


def _unknown_reservation_should_forget_in_flight(
    exc: UnknownReservationError,
) -> bool:
    return bool(getattr(exc, _UNKNOWN_RESERVATION_FORGET_IN_FLIGHT_ATTR, False))
