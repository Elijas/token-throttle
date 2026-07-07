"""Shared token-bucket capacity math — used by all backends (Redis, in-memory, sync, async)."""

import logging
import math
import time
import warnings

from pydantic import Field, field_validator

from token_throttle._dto import StrictDTO

_backward_clock_warned: bool = False
_backward_clock_last_warning_at: float | None = None
_BACKWARD_CLOCK_WARNING_INTERVAL_SECONDS = 300.0
MIN_MAX_CAPACITY = 1e-9
_logger = logging.getLogger("token_throttle")
_acquire_logger = logging.getLogger("token_throttle.acquire")
_MEMORY_BUCKET_ID_PARTS = 4
_REDIS_BUCKET_ID_PARTS = 6


def _bucket_context_from_bucket_id(bucket_id: str) -> tuple[str, str]:
    """Best-effort extraction for memory and Redis bucket key formats."""
    parts = bucket_id.split(":")
    if len(parts) >= _MEMORY_BUCKET_ID_PARTS and parts[0] == "memory":
        return parts[1], parts[2]
    if (
        len(parts) >= _REDIS_BUCKET_ID_PARTS
        and parts[1] == "rate_limiting"
        and parts[2] == "bucket"
    ):
        return parts[3], parts[4]
    return "<unknown>", "<unknown>"


class CalculatedCapacity(StrictDTO):
    """
    Result of a token-bucket capacity calculation.

    ``CalculatedCapacity`` is an exact-type immutable DTO, not a subclass
    extension point. Construction, assignment, copy, pickle restore,
    ``model_copy()``, and ``model_construct()`` all preserve finite capacity
    validation; ``model_construct()`` is disabled.

    ``is_fresh_start`` is True when no prior capacity data exists
    (``last_checked`` or ``outdated_capacity`` is None).  Backends use
    this to fire the ``on_missing_consumption_data`` callback, signalling
    that the bucket assumed full capacity because there was nothing to
    refill from.
    """

    amount: float = Field(allow_inf_nan=False)
    is_fresh_start: bool

    @field_validator("amount", mode="after")
    @classmethod
    def _require_finite_amount(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"amount must be finite (got {value!r})")
        return value


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
    last_checked: float | str | bytes | None,
    outdated_capacity: float | str | bytes | None,
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

    # Backends should only pass None for genuine new buckets where both state
    # fields are missing. Redis normalizes partial state before reaching this
    # shared math so a missing key cannot reset a drained bucket to full.
    if last_checked is None or outdated_capacity is None:
        return CalculatedCapacity(amount=max_capacity, is_fresh_start=True)

    try:
        raw_last_checked, raw_outdated_capacity = last_checked, outdated_capacity
        last_checked = float(last_checked)
        outdated_capacity = float(outdated_capacity)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(
            "Invalid last_checked or capacity values: "
            f"last_checked={raw_last_checked!r}, capacity={raw_outdated_capacity!r}",
        ) from e
    if not (math.isfinite(last_checked) and math.isfinite(outdated_capacity)):
        raise ValueError(
            "Invalid last_checked or capacity values: "
            f"last_checked={raw_last_checked!r}, capacity={raw_outdated_capacity!r}",
        )

    time_passed = current_time - last_checked
    if time_passed < 0:
        global _backward_clock_last_warning_at, _backward_clock_warned  # noqa: PLW0603
        now = time.monotonic()
        should_warn = (
            not _backward_clock_warned
            or _backward_clock_last_warning_at is None
            or now - _backward_clock_last_warning_at
            >= _BACKWARD_CLOCK_WARNING_INTERVAL_SECONDS
        )
        if should_warn:
            _backward_clock_warned = True
            _backward_clock_last_warning_at = now
            model_family, metric = _bucket_context_from_bucket_id(bucket_id)
            _logger.warning(
                "Negative time_passed detected; metric=%s model_family=%s "
                "value=%.4f bucket_id=%s. Likely NTP clock correction. "
                "Clamping to 0. Further backward-clock warnings suppressed for %.0fs.",
                metric,
                model_family,
                time_passed,
                bucket_id,
                _BACKWARD_CLOCK_WARNING_INTERVAL_SECONDS,
                extra={
                    "token_throttle_metric": metric,
                    "token_throttle_model_family": model_family,
                    "token_throttle_value": time_passed,
                    "token_throttle_bucket_id": bucket_id,
                },
            )
            message = (
                f"Negative time_passed ({time_passed:.4f}s) detected in bucket "
                f"'{bucket_id}' — likely NTP clock correction. "
                "Clamping to 0. Further backward-clock warnings suppressed "
                f"for {_BACKWARD_CLOCK_WARNING_INTERVAL_SECONDS:.0f}s."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            _acquire_logger.warning(message)
        time_passed = 0.0

    current_preconsumption_capacity = min(
        max_capacity,
        outdated_capacity + time_passed * rate_per_sec,
    )
    return CalculatedCapacity(
        amount=current_preconsumption_capacity,
        is_fresh_start=False,
    )
