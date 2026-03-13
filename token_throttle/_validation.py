"""Shared validation logic — used by both async RateLimiter and SyncRateLimiter."""

import math
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

from token_throttle._interfaces._interfaces import PerModelConfig, PerModelConfigGetter
from token_throttle._interfaces._models import (
    FrozenUsage,
    Usage,
    UsageQuotas,
    _coerce_usage_value,
    frozen_usage,
)


def _validate_usage_mapping(
    usage: Usage,
    expected_keys: AbstractSet[str],
    *,
    expected_keys_label: str,
    value_label: str,
) -> None:
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
        if not math.isfinite(amount):
            raise ValueError(
                f"{value_label} for {metric} must be finite (got {amount_!r})"
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
        expected_keys_label="backend metric keys",
        value_label="Reserved usage value",
    )
    validate_refund_usage(actual_usage, set(reserved_usage))


def merge_extra_usage(
    usage: FrozenUsage,
    extra_usage: Mapping[str, object] | None,
) -> FrozenUsage:
    """Merge extra usage values into counted usage with consistent numeric checks."""
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


def resolve_config(
    cfg: PerModelConfig | PerModelConfigGetter, model_name: str
) -> PerModelConfig:
    """
    Resolve a config (static or callable) and default model_family to model_name.

    Raises:
        ValueError: If model_name is empty.

    """
    if not model_name:
        raise ValueError("model_name cannot be empty")
    r = cfg(model_name) if callable(cfg) else cfg
    return r if r.model_family else r.model_copy(update={"model_family": model_name})
