"""Shared token-bucket capacity math — used by all backends (Redis, in-memory, sync, async)."""

import math
import warnings

from pydantic import BaseModel

_backward_clock_warned: bool = False
MIN_MAX_CAPACITY = 1e-9


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


def _validate_plain_number(value: object, *, name: str) -> float:
    if type(value) is bool:
        raise ValueError(f"{name} must not be a boolean")
    if type(value) is not float and type(value) is not int:
        raise ValueError(f"{name} must be an int or float (got {type(value).__name__})")
    try:
        return float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is too large to convert to float") from exc


def _validate_max_capacity_finite_positive(value: object) -> float:
    float_value = _validate_plain_number(value, name="max_capacity")
    if not math.isfinite(float_value) or float_value < MIN_MAX_CAPACITY:
        raise ValueError(
            "max_capacity must be finite and greater than 0 "
            f"(minimum supported value is {MIN_MAX_CAPACITY!r}; got {value!r})"
        )
    return float_value


def _validate_rate_per_sec_finite_positive(value: object) -> float:
    float_value = _validate_plain_number(value, name="rate_per_sec")
    if not math.isfinite(float_value) or float_value <= 0:
        raise ValueError(
            "Bucket rate is non-positive/non-finite — likely a misconfigured "
            f"max_capacity (got {value!r})"
        )
    return float_value


def _calculate_rate_per_sec(max_capacity: float, per_seconds: int) -> float:
    try:
        rate = max_capacity / float(per_seconds)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            "Bucket rate is non-positive/non-finite — likely a misconfigured "
            "max_capacity"
        ) from exc
    return _validate_rate_per_sec_finite_positive(rate)


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
    try:
        raw_current_time = current_time
        current_time = float(current_time)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(
            f"current_time must be finite and non-negative (got {raw_current_time!r})"
        ) from e
    if not math.isfinite(current_time) or current_time < 0:
        raise ValueError(
            f"current_time must be finite and non-negative (got {raw_current_time!r})"
        )

    max_capacity = _validate_max_capacity_finite_positive(max_capacity)
    rate_per_sec = _validate_rate_per_sec_finite_positive(rate_per_sec)

    # Partial state (one None, one non-None) is treated as a fresh start:
    # anchoring with incomplete data would produce a wrong capacity value,
    # so we reset to max_capacity. Any negative debt is intentionally lost.
    if last_checked is None or outdated_capacity is None:
        return CalculatedCapacity(amount=max_capacity, is_fresh_start=True)

    try:
        raw_last_checked, raw_outdated_capacity = last_checked, outdated_capacity
        last_checked = float(last_checked)
        outdated_capacity = float(outdated_capacity)
    except (TypeError, ValueError, OverflowError) as e:
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
