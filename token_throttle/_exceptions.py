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


class BackendLockContentionError(Exception):
    """
    Raised when a backend cannot acquire (or loses) its internal per-bucket lock.

    The Redis backend serializes every bucket mutation through a short-lived
    per-bucket lock. Two situations surface as this exception:

    * **Acquisition starvation** — the lock could not be acquired within the
      configured ``lock_blocking_timeout_seconds`` because other workers held
      it for the whole window. Retry, raise ``lock_blocking_timeout_seconds``,
      or reduce contention (fewer concurrent callers, smaller bucket fan-out).
    * **Mid-operation loss** — the lock expired or was stolen by another worker
      between the read and the write, so the in-progress write was aborted to
      avoid clobbering another worker's state. The operation made no change and
      is safe to retry.

    ``wait_for_capacity`` / ``await_for_capacity`` with no caller timeout absorb
    this internally and keep waiting, so callers normally see it only from
    ``consume_capacity``, ``refund_capacity``, ``set_max_capacity``, and
    reconfiguration. It always chains the underlying cause via ``__cause__``.
    """

    reason = "backend_lock_contention"

    ACQUISITION_MESSAGE = (
        "could not acquire the internal per-bucket Redis lock within the "
        "configured lock_blocking_timeout_seconds; the bucket is under heavy "
        "contention. Retry, raise lock_blocking_timeout_seconds, or reduce "
        "the number of concurrent callers."
    )
    LOCK_LOST_MESSAGE = (
        "lost the internal per-bucket Redis lock mid-operation (it expired or "
        "was stolen by another worker); the write was aborted and made no "
        "change. Retry the operation."
    )

    def __init__(self, message: str | None = None) -> None:
        super().__init__(self.ACQUISITION_MESSAGE if message is None else message)


class DuplicateRefundError(ValueError):
    """Raised when a refund is duplicate, already in progress, or already acquired."""

    def __init__(
        self,
        message: str | None = None,
        *,
        reason: DuplicateRefundReason | None = None,
        reservation_id: str | None = None,
        model_family: str | None = None,
    ) -> None:
        if reason is None:
            reason = _infer_duplicate_refund_reason(message)
        if reason not in _DUPLICATE_REFUND_REASONS:
            raise ValueError(f"unknown duplicate refund reason: {reason!r}")
        if message is None:
            message = _DUPLICATE_REFUND_MESSAGES[reason]
        self.reservation_id = reservation_id
        self.model_family = model_family
        context = []
        if reservation_id is not None:
            context.append(f"reservation_id={reservation_id!r}")
        if model_family is not None:
            context.append(f"model_family={model_family!r}")
        if context:
            message = f"{message} ({', '.join(context)})"
        super().__init__(message)
        self.reason = reason


class UnknownReservationError(ValueError):
    """Raised when a backend has no record that a reservation was acquired."""

    reason = "unknown_reservation"

    def __init__(
        self,
        message: str = "reservation was never acquired by this backend",
        *,
        reservation_id: str | None = None,
        model_family: str | None = None,
    ) -> None:
        self.reservation_id = reservation_id
        self.model_family = model_family
        context = []
        if reservation_id is not None:
            context.append(f"reservation_id={reservation_id!r}")
        if model_family is not None:
            context.append(f"model_family={model_family!r}")
        if context:
            message = f"{message} ({', '.join(context)})"
        super().__init__(message)


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
