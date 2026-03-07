"""Shared validation logic — used by both async RateLimiter and SyncRateLimiter."""

from token_throttle._interfaces._interfaces import PerModelConfig, PerModelConfigGetter
from token_throttle._interfaces._models import FrozenUsage, UsageQuotas


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
        if amount < 0:
            raise ValueError(f"Usage value for {metric} must be non-negative")
        for quota in quotas.get_quotas(metric):
            if amount > float(quota.limit):
                raise ValueError(
                    f"Usage value for {metric} ({amount}) exceeds the limit ({quota.limit})",
                )


def validate_refund_keys(actual_keys: set[str], reservation_keys: set[str]) -> None:
    """
    Check that refund usage keys match the reservation keys.

    Raises:
        ValueError: If keys don't match.

    """
    if actual_keys != reservation_keys:
        raise ValueError(
            f"Usage keys {actual_keys} do not match reservation usage keys {reservation_keys}",
        )


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
