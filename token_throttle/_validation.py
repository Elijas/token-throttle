"""Shared validation logic — used by both async RateLimiter and SyncRateLimiter."""

import math
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

from token_throttle._interfaces._interfaces import PerModelConfig, PerModelConfigGetter
from token_throttle._interfaces._models import (
    CapacityReservation,
    FrozenUsage,
    Usage,
    UsageQuotas,
    _coerce_usage_value,
    frozen_usage,
)

_UNLIMITED_FLAG = "__rate_limiting_disabled__"


def is_unlimited_reservation(reservation: CapacityReservation) -> bool:
    return reservation.model_family == _UNLIMITED_FLAG and not reservation.usage


def extract_usage_from_response(response: object) -> object:
    """Extract usage payload from a response object or raw response mapping."""
    if isinstance(response, Mapping):
        try:
            usage = response["usage"]
        except KeyError:
            raise ValueError(
                "response must include usage data — pass actual usage via "
                "refund_capacity() instead."
            ) from None
    else:
        usage = getattr(response, "usage", None)
        if usage is None:
            if hasattr(response, "usage"):
                raise ValueError(
                    "response.usage is None — cannot extract token counts. "
                    "Streaming responses may not include usage data; "
                    "pass actual usage via refund_capacity() instead."
                )
            raise ValueError(
                "response must include usage data — pass actual usage via "
                "refund_capacity() instead."
            )
    if usage is None:
        raise ValueError(
            "response.usage is None — cannot extract token counts. "
            "Streaming responses may not include usage data; "
            "pass actual usage via refund_capacity() instead."
        )
    return usage


def extract_total_tokens(usage: object) -> float:
    """Extract total_tokens from a usage object (attribute or mapping access)."""
    if hasattr(usage, "total_tokens"):
        total_tokens = usage.total_tokens
    elif isinstance(usage, Mapping):
        try:
            total_tokens = usage["total_tokens"]
        except KeyError:
            raise ValueError(
                "'total_tokens' key not found in usage data — "
                "pass actual usage via refund_capacity() instead."
            ) from None
    else:
        raise ValueError(
            "usage must be an object with total_tokens attribute or a mapping"
        )
    if total_tokens is None:
        raise ValueError(
            "total_tokens is None — cannot compute refund. "
            "Pass actual usage via refund_capacity() instead."
        )
    if isinstance(total_tokens, bool):
        raise ValueError(  # noqa: TRY004
            "total_tokens must not be a boolean"
        )
    try:
        value = float(total_tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"total_tokens must be a finite non-negative number (got {total_tokens!r})"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"total_tokens must be a finite non-negative number (got {total_tokens!r})"
        )
    if value < 0:
        raise ValueError(
            f"total_tokens must be a finite non-negative number (got {total_tokens!r})"
        )
    return value


def _validate_usage_mapping(
    usage: Usage,
    expected_keys: AbstractSet[str],
    *,
    mapping_label: str,
    expected_keys_label: str,
    value_label: str,
) -> None:
    if not isinstance(usage, Mapping):
        raise ValueError(  # noqa: TRY004
            f"{mapping_label} must be a mapping (got {type(usage).__name__})"
        )
    if set(usage) != expected_keys:
        raise ValueError(
            f"Usage keys {set(usage)} do not match {expected_keys_label} {expected_keys}",
        )
    for metric, amount_ in usage.items():
        amount = _coerce_usage_value(
            metric,
            amount_,
            label=value_label,
        )
        if amount < 0:
            raise ValueError(f"{value_label} for {metric} must be non-negative")


def validate_acquire_usage(usage: FrozenUsage, quotas: UsageQuotas) -> None:
    """
    Check usage keys match quota keys, values are finite and non-negative.

    Over-limit checks are performed by the backend against the live
    bucket max_capacity (which may differ from the static quota.limit
    after a set_max_capacity call).

    Raises:
        ValueError: If keys mismatch, or values are NaN/Inf/negative.

    """
    _validate_usage_mapping(
        usage,
        set(quotas.names),
        mapping_label="usage",
        expected_keys_label="quota keys",
        value_label="Usage value",
    )


def validate_refund_usage(actual_usage: Usage, reservation_keys: set[str]) -> None:
    """
    Check that actual usage keys match the reservation and values are finite/non-negative.

    Raises:
        ValueError: If keys don't match, or values are NaN/Inf/negative.

    """
    _validate_usage_mapping(
        actual_usage,
        reservation_keys,
        mapping_label="actual_usage",
        expected_keys_label="reservation usage keys",
        value_label="Actual usage value",
    )


def validate_backend_usage(
    usage: Usage,
    backend_metric_names: AbstractSet[str],
) -> None:
    """
    Validate direct backend usage against the backend's metric set.

    Exported backends are part of the public API, so they must reject
    impossible input even when callers bypass ``RateLimiter``.
    """
    _validate_usage_mapping(
        usage,
        backend_metric_names,
        mapping_label="usage",
        expected_keys_label="backend metric keys",
        value_label="Usage value",
    )


def validate_backend_refund_usage(
    reserved_usage: Usage,
    actual_usage: Usage,
    backend_metric_names: AbstractSet[str],
) -> None:
    """Validate direct backend refund inputs."""
    _validate_usage_mapping(
        reserved_usage,
        backend_metric_names,
        mapping_label="reserved_usage",
        expected_keys_label="backend metric keys",
        value_label="Reserved usage value",
    )
    validate_refund_usage(actual_usage, set(reserved_usage))


def validate_timeout(timeout: object) -> float | None:
    """Validate timeout values used by blocking acquire/wait operations."""
    if timeout is None:
        return None
    if isinstance(timeout, bool):
        raise ValueError("timeout must not be a boolean")  # noqa: TRY004
    try:
        timeout_value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"timeout must be finite or None (got {timeout!r})"
        ) from exc
    if not math.isfinite(timeout_value):
        raise ValueError(f"timeout must be finite or None (got {timeout!r})")
    if timeout_value < 0:
        raise ValueError(
            f"timeout must be non-negative or None (got {timeout!r})"
        )
    return timeout_value


def validate_max_capacity_value(value: object) -> float:
    """Validate the value parameter for set_max_capacity."""
    if isinstance(value, bool):
        raise ValueError("max_capacity must not be a boolean")  # noqa: TRY004
    try:
        float_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"max_capacity must be finite and greater than 0 (got {value!r})"
        ) from exc
    if not (math.isfinite(float_value) and float_value > 0):
        raise ValueError(
            f"max_capacity must be finite and greater than 0 (got {value!r})"
        )
    return float_value


def merge_extra_usage(
    usage: FrozenUsage,
    extra_usage: Mapping[str, object] | None,
) -> FrozenUsage:
    """Merge extra usage values into counted usage with consistent numeric checks."""
    if extra_usage is None:
        return usage
    if not isinstance(extra_usage, Mapping):
        raise ValueError(  # noqa: TRY004
            f"extra_usage must be a mapping or None (got {type(extra_usage).__name__})"
        )
    if not extra_usage:
        return usage

    merged_usage = dict(usage)
    for metric, raw_amount in extra_usage.items():
        if metric not in merged_usage:
            raise ValueError(
                f"Usage key '{metric}' not found in usage counter",
            )
        if isinstance(raw_amount, bool):
            raise ValueError(  # noqa: TRY004
                f"Usage value for {metric} must not be a boolean"
            )
        try:
            amount = float(raw_amount)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Usage value for {metric} must be a finite number (got {raw_amount!r})"
            ) from exc
        if not math.isfinite(amount):
            raise ValueError(
                f"Usage value for {metric} must be a finite number (got {raw_amount!r})"
            )
        if amount < 0:
            raise ValueError(f"Usage value for {metric} must be non-negative")
        merged_usage[metric] += amount
    return frozen_usage(merged_usage)


def validate_extra_usage(
    extra_usage: object,
) -> Mapping[str, object] | None:
    """Validate optional extra_usage payloads for request-based acquire helpers."""
    if extra_usage is None:
        return None
    if not isinstance(extra_usage, Mapping):
        raise ValueError(  # noqa: TRY004
            f"extra_usage must be a mapping or None (got {type(extra_usage).__name__})"
        )
    return extra_usage


def validate_metric(metric: object) -> str:
    """Validate the metric parameter for set_max_capacity."""
    if not isinstance(metric, str):
        raise ValueError(  # noqa: TRY004
            f"metric must be a non-empty string (got {type(metric).__name__})"
        )
    if not metric:
        raise ValueError("metric must be a non-empty string")
    return metric


def validate_per_seconds(per_seconds: object) -> int:
    """Validate the per_seconds parameter for set_max_capacity."""
    if isinstance(per_seconds, bool):
        raise ValueError("per_seconds must not be a boolean")  # noqa: TRY004
    if not isinstance(per_seconds, int | float):
        raise ValueError(  # noqa: TRY004
            f"per_seconds must be a positive integer (got {per_seconds!r})"
        )
    value = float(per_seconds)
    if not math.isfinite(value) or value <= 0 or not value.is_integer():
        raise ValueError(
            f"per_seconds must be a positive integer (got {per_seconds!r})"
        )
    return int(value)


def resolve_config(
    cfg: PerModelConfig | PerModelConfigGetter, model_name: str
) -> PerModelConfig:
    """
    Resolve a config (static or callable) and default model_family to model_name.

    Raises:
        ValueError: If model_name is not a non-empty string.

    """
    if not isinstance(model_name, str):
        raise ValueError(  # noqa: TRY004
            f"model_name must be a string (got {type(model_name).__name__})"
        )
    if not model_name:
        raise ValueError("model_name cannot be empty")
    r = cfg(model_name) if callable(cfg) else cfg
    return r if r.model_family else r.model_copy(update={"model_family": model_name})
