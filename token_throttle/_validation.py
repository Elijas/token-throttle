"""Shared validation logic — used by both async RateLimiter and SyncRateLimiter."""

import math
from collections.abc import Mapping

from token_throttle._interfaces._interfaces import PerModelConfig, PerModelConfigGetter
from token_throttle._interfaces._models import (
    FrozenUsage,
    Usage,
    UsageQuotas,
    _coerce_usage_value,
    frozen_usage,
)


def validate_acquire_usage(usage: FrozenUsage, quotas: UsageQuotas) -> None:
    """
    Check usage keys match quota keys, values are finite and non-negative.

    Over-limit checks are performed by the backend against the live
    bucket max_capacity (which may differ from the static quota.limit
    after a set_max_capacity call).

    Raises:
        ValueError: If keys mismatch, or values are NaN/Inf/negative.

    """
    if set(usage) != set(quotas.names):
        raise ValueError(
            f"Usage keys {set(usage)} do not match quota keys {set(quotas.names)}",
        )
    for metric, amount_ in usage.items():
        amount = _coerce_usage_value(metric, amount_)
        if not math.isfinite(amount):
            raise ValueError(
                f"Usage value for {metric} must be finite (got {amount_!r})"
            )
        if amount < 0:
            raise ValueError(f"Usage value for {metric} must be non-negative")


def validate_refund_usage(actual_usage: Usage, reservation_keys: set[str]) -> None:
    """
    Check that actual usage keys match the reservation and values are finite/non-negative.

    Raises:
        ValueError: If keys don't match, or values are NaN/Inf/negative.

    """
    if set(actual_usage) != reservation_keys:
        raise ValueError(
            f"Usage keys {set(actual_usage)} do not match reservation usage keys {reservation_keys}",
        )
    for metric, amount_ in actual_usage.items():
        amount = _coerce_usage_value(
            metric,
            amount_,
            label="Actual usage value",
        )
        if not math.isfinite(amount):
            raise ValueError(
                f"Actual usage value for {metric} must be finite (got {amount_!r})"
            )
        if amount < 0:
            raise ValueError(f"Actual usage value for {metric} must be non-negative")


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
