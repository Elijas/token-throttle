from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from token_throttle._interfaces._models import CapacityReservation


class AcquireRefundFailedError(asyncio.CancelledError):
    """Raised when cancellation fallback cannot refund an acquired reservation."""

    def __init__(
        self,
        *,
        reservation: CapacityReservation,
        refund_error: BaseException,
        interrupted_by: BaseException | None = None,
    ) -> None:
        super().__init__(
            "acquire was interrupted after capacity was reserved, and the "
            "fallback refund failed; inspect .reservation to refund or use "
            "the reservation explicitly"
        )
        self.reservation = reservation
        self.refund_error = refund_error
        self.interrupted_by = interrupted_by


class CardinalityLimitExceededError(ValueError):
    """Raised when a mandatory limiter cardinality or length cap is exceeded."""


class DuplicateRefundError(ValueError):
    """Raised when a reservation has already been refunded."""


class UnknownReservationError(ValueError):
    """Raised when a backend has no record that a reservation was acquired."""


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
