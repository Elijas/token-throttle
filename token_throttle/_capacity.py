"""Shared token-bucket capacity math — used by all backends (Redis, in-memory, sync, async)."""

import math
import warnings

from pydantic import BaseModel

_backward_clock_warned: bool = False


class CalculatedCapacity(BaseModel):
    """
    Result of a token-bucket capacity calculation.

    ``is_fresh_start`` is True when no prior capacity data exists
    (``last_checked`` or ``outdated_capacity`` is None).  Backends use
    this to fire the ``on_missing_consumption_data`` callback, signalling
    that the bucket assumed full capacity because there was nothing to
    refill from.
    """

    amount: float
    is_fresh_start: bool


def calculate_capacity(  # noqa: PLR0913
    last_checked: float | None,
    outdated_capacity: float | None,
    current_time: float,
    max_capacity: float,
    rate_per_sec: float,
    bucket_id: str,
) -> CalculatedCapacity:
    """
    Calculate current bucket capacity based on time elapsed since last check.

    Pure function — no I/O, no locks. Shared by Redis and in-memory backends.

    Args:
        last_checked: Timestamp of last capacity update (None = fresh start).
        outdated_capacity: Capacity value at last_checked (None = fresh start).
        current_time: Current timestamp.
        max_capacity: Maximum capacity for the bucket.
        rate_per_sec: Refill rate in units per second.
        bucket_id: Identifier for the bucket (used in warning messages).

    """
    # Partial state (one None, one non-None) is treated as a fresh start:
    # anchoring with incomplete data would produce a wrong capacity value,
    # so we reset to max_capacity. Any negative debt is intentionally lost.
    if last_checked is None or outdated_capacity is None:
        return CalculatedCapacity(amount=max_capacity, is_fresh_start=True)

    try:
        raw_last_checked, raw_outdated_capacity = last_checked, outdated_capacity
        last_checked = float(last_checked)
        outdated_capacity = float(outdated_capacity)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid last_checked or capacity values: last_checked={raw_last_checked}, capacity={raw_outdated_capacity}",
        ) from e
    if not (math.isfinite(last_checked) and math.isfinite(outdated_capacity)):
        raise ValueError(
            f"Invalid last_checked or capacity values: last_checked={raw_last_checked}, capacity={raw_outdated_capacity}",
        )

    time_passed = current_time - last_checked
    if time_passed < 0:
        global _backward_clock_warned  # noqa: PLW0603
        if not _backward_clock_warned:
            _backward_clock_warned = True
            warnings.warn(
                f"Negative time_passed ({time_passed:.4f}s) detected in bucket "
                f"'{bucket_id}' — likely NTP clock correction. "
                f"Clamping to 0. Further backward-clock warnings suppressed.",
                RuntimeWarning,
                stacklevel=2,
            )
        time_passed = 0.0

    current_preconsumption_capacity = min(
        max_capacity,
        outdated_capacity + time_passed * rate_per_sec,
    )
    return CalculatedCapacity(
        amount=current_preconsumption_capacity,
        is_fresh_start=False,
    )
