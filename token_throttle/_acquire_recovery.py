"""Private, opt-in handoff of confirmed built-in acquisition cleanup failures."""

from __future__ import annotations

import contextvars
import math
from typing import TYPE_CHECKING

from token_throttle._exceptions import AcquireRefundFailedError
from token_throttle._interfaces._callbacks import (
    LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
    _exception_group_contains_critical,
)

if TYPE_CHECKING:
    from token_throttle._interfaces._models import CapacityReservation


class _AcquireRecovery:
    def __init__(self, reservation_id: str) -> None:
        self.reservation_id = reservation_id
        self.failure: tuple[float, BaseException, BaseException] | None = None

    def error(
        self, reservation: CapacityReservation
    ) -> AcquireRefundFailedError | None:
        if self.failure is None:
            return None
        if reservation.reservation_id != self.reservation_id:
            raise RuntimeError("Acquisition cleanup recovery identity mismatch")
        issued_at, interruption, refund_error = self.failure
        return AcquireRefundFailedError(
            reservation.model_copy(update={"created_at_seconds": issued_at}),
            interrupted_by=interruption,
            refund_error=refund_error,
        )


_ACQUIRE_RECOVERY: contextvars.ContextVar[_AcquireRecovery | None] = (
    contextvars.ContextVar("token_throttle_acquire_recovery", default=None)
)


def _record_acquire_cleanup_failure(
    refund_error: BaseException,
    *,
    reservation_id: str | None,
    issued_at_seconds: float | None,
    interrupted_by: BaseException,
) -> None:
    if isinstance(
        refund_error, LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS
    ) or _exception_group_contains_critical(
        refund_error, LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS
    ):
        raise refund_error
    recovery = _ACQUIRE_RECOVERY.get()
    if recovery is None or recovery.reservation_id != reservation_id:
        return
    if (
        type(issued_at_seconds) not in (int, float)
        or issued_at_seconds is None
        or not math.isfinite(issued_at_seconds)
    ):
        raise RuntimeError("Acquisition cleanup recovery requires a known issue time")
    recovery.failure = (issued_at_seconds, interrupted_by, refund_error)
