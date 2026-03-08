"""Shared validation logic — used by both async RateLimiter and SyncRateLimiter."""

import math

from token_throttle._interfaces._interfaces import PerModelConfig, PerModelConfigGetter
from token_throttle._interfaces._models import FrozenUsage, Usage, UsageQuotas


def validate_acquire_usage(usage: FrozenUsage, quotas: UsageQuotas) -> None:
    """
    Check usage keys match quota keys, no negatives, no over-limit.

    Raises:
        ValueError: If keys mismatch, negative values, or exceeds limit.

    """
    if set(usage) != set(quotas.names):
        raise ValueError(
            f"Usage keys {set(usage)} do not match quota keys {set(quotas.names)}",
        )
    for metric, amount_ in usage.items():
        amount = float(amount_)
        if not math.isfinite(amount):
            raise ValueError(
                f"Usage value for {metric} must be finite (got {amount_!r})"
            )
        if amount < 0:
            raise ValueError(f"Usage value for {metric} must be non-negative")


def validate_refund_usage(
    actual_usage: Usage, reservation_keys: set[str]
) -> None:
    """
    Check that refund usage keys match the reservation and values are finite/non-negative.

    Raises:
        ValueError: If keys don't match, or values are NaN/Inf/negative.

    """
    if set(actual_usage) != reservation_keys:
        raise ValueError(
            f"Usage keys {set(actual_usage)} do not match reservation usage keys {reservation_keys}",
        )
    for metric, amount_ in actual_usage.items():
        amount = float(amount_)
        if not math.isfinite(amount):
            raise ValueError(
                f"Refund value for {metric} must be finite (got {amount_!r})"
            )
        if amount < 0:
            raise ValueError(f"Refund value for {metric} must be non-negative")


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
