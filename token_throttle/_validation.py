"""Shared validation logic — used by both async RateLimiter and SyncRateLimiter."""

import inspect
import math
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

from token_throttle._interfaces._callable_utils import (
    close_awaitable_if_possible,
    is_async_callable,
)
from token_throttle._interfaces._interfaces import PerModelConfig, PerModelConfigGetter
from token_throttle._interfaces._models import (
    BucketId,
    CapacityReservation,
    FrozenUsage,
    Usage,
    UsageQuotas,
    _coerce_usage_value,
    _is_bool_like,
    frozen_usage,
)

_UNLIMITED_FLAG = "__rate_limiting_disabled__"


def is_unlimited_reservation(reservation: CapacityReservation) -> bool:
    """
    True when a reservation represents a disabled/unlimited rate limit.

    New code path: ``_unlimited_reservation`` sets ``is_unlimited=True`` on
    every reservation it creates, so that's the authoritative signal.

    Legacy path: reservations created before the ``is_unlimited`` field
    existed carry only the sentinel ``model_family`` and an empty
    ``usage`` / ``bucket_ids``. The extra-conservative fallback preserves
    back-compat for those, but it intentionally does NOT match reservations
    that have the sentinel family plus non-empty usage — those cannot be
    produced by the current code and are likely hand-constructed. Callers
    constructing ``CapacityReservation`` manually must set
    ``is_unlimited=True`` for unlimited reservations.
    """
    return bool(
        reservation.is_unlimited
        or (
            reservation.model_family == _UNLIMITED_FLAG
            and not reservation.usage
            and reservation.bucket_ids is None
        )
    )


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
    if _is_bool_like(total_tokens):
        raise ValueError("total_tokens must not be a boolean")
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


def validate_backend_refund_usage_for_bucket_ids(
    reserved_usage: Usage,
    actual_usage: Usage,
    bucket_ids: AbstractSet[BucketId],
    backend_bucket_ids: AbstractSet[BucketId],
) -> None:
    """Validate backend refund inputs for a specific subset of buckets."""
    if not bucket_ids:
        if reserved_usage or actual_usage:
            raise ValueError(
                "bucket_ids cannot be empty when refund usage is non-empty"
            )
        return

    missing_bucket_ids = set(bucket_ids) - set(backend_bucket_ids)
    if missing_bucket_ids:
        raise ValueError(
            f"Refund bucket ids {sorted(missing_bucket_ids)} not found in backend"
        )

    metric_names = {metric for metric, _ in bucket_ids}
    _validate_usage_mapping(
        reserved_usage,
        metric_names,
        mapping_label="reserved_usage",
        expected_keys_label="refund bucket metric keys",
        value_label="Reserved usage value",
    )
    validate_refund_usage(actual_usage, set(reserved_usage))


def validate_timeout(timeout: object) -> float | None:
    """Validate timeout values used by blocking acquire/wait operations."""
    if timeout is None:
        return None
    if _is_bool_like(timeout):
        raise ValueError("timeout must not be a boolean")
    try:
        timeout_value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"timeout must be finite or None (got {timeout!r})") from exc
    if not math.isfinite(timeout_value):
        raise ValueError(f"timeout must be finite or None (got {timeout!r})")
    if timeout_value < 0:
        raise ValueError(f"timeout must be non-negative or None (got {timeout!r})")
    return timeout_value


def validate_max_capacity_value(value: object) -> float:
    """Validate the value parameter for set_max_capacity."""
    if _is_bool_like(value):
        raise ValueError("max_capacity must not be a boolean")
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
    return _merge_extra_usage(
        usage,
        extra_usage,
        allow_new_keys=False,
    )


def merge_extra_usage_unrestricted(
    usage: FrozenUsage,
    extra_usage: Mapping[str, object] | None,
) -> FrozenUsage:
    """
    Merge extra usage values while allowing new metrics.

    Used by unlimited configs so disabling the limiter is a drop-in no-op even
    when callers pass metrics not produced by a usage_counter.
    """
    return _merge_extra_usage(
        usage,
        extra_usage,
        allow_new_keys=True,
    )


def _merge_extra_usage(
    usage: FrozenUsage,
    extra_usage: Mapping[str, object] | None,
    *,
    allow_new_keys: bool,
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
        if not allow_new_keys and metric not in merged_usage:
            raise ValueError(
                f"Usage key '{metric}' not found in usage counter",
            )
        if _is_bool_like(raw_amount):
            raise ValueError(f"Usage value for {metric} must not be a boolean")
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
        merged_usage[metric] = merged_usage.get(metric, 0.0) + amount
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
    if ":" in metric:
        raise ValueError("metric must not contain ':' (used as Redis key separator)")
    return metric


def validate_per_seconds(per_seconds: object) -> int:
    """
    Validate the per_seconds parameter for set_max_capacity.

    Accepts ``int`` directly.  Whole ``float`` values (e.g. ``60.0``) are
    coerced to ``int`` for parity with Pydantic's lax-mode coercion on
    ``Quota.per_seconds: int``.  All other types are rejected.
    """
    if _is_bool_like(per_seconds):
        raise ValueError("per_seconds must not be a boolean")
    if isinstance(per_seconds, int):
        if per_seconds <= 0:
            raise ValueError(
                f"per_seconds must be a positive integer (got {per_seconds!r})"
            )
        return per_seconds
    if isinstance(per_seconds, float):
        if (
            not math.isfinite(per_seconds)
            or per_seconds <= 0
            or not per_seconds.is_integer()
        ):
            raise ValueError(
                f"per_seconds must be a positive integer (got {per_seconds!r})"
            )
        return int(per_seconds)
    raise ValueError(f"per_seconds must be a positive integer (got {per_seconds!r})")


def resolve_config(
    cfg: PerModelConfig | PerModelConfigGetter, model_name: str
) -> PerModelConfig:
    """
    Resolve a config (static or callable) and default model_family to model_name.

    Raises:
        ValueError: If model_name is not a non-empty, non-whitespace string.

    """
    if not isinstance(model_name, str):
        raise ValueError(  # noqa: TRY004
            f"model_name must be a string (got {type(model_name).__name__})"
        )
    if not model_name:
        raise ValueError("model_name cannot be empty")
    # Whitespace-only names would silently default model_family to the same
    # whitespace, causing two callers using e.g. "  " vs "   " to route to
    # different backends without noticing.
    if not model_name.strip():
        raise ValueError(f"model_name cannot be whitespace-only (got {model_name!r})")
    if callable(cfg) and is_async_callable(cfg):
        raise ValueError("cfg must be a synchronous PerModelConfig getter")
    r = cfg(model_name) if callable(cfg) else cfg
    if inspect.isawaitable(r):
        close_awaitable_if_possible(r)
        raise ValueError("cfg must be a synchronous PerModelConfig getter")
    if not isinstance(r, PerModelConfig):
        raise ValueError(  # noqa: TRY004
            f"cfg must resolve to PerModelConfig (got {type(r).__name__})"
        )
    resolved = (
        r if r.model_family else r.model_copy(update={"model_family": model_name})
    )
    model_family = resolved.get_model_family()
    if ":" in model_family:
        raise ValueError(
            f"model_family must not contain ':' (used as Redis key separator); "
            f"got {model_family!r}"
        )
    return resolved


def resolve_usage_counter_result(usage_counter, /, **request) -> FrozenUsage:
    """Run a usage_counter and reject awaitable return values with a clear error."""
    result = _call_usage_counter(usage_counter, request)
    if inspect.isawaitable(result):
        close_awaitable_if_possible(result)
        raise ValueError(
            "usage_counter must be a synchronous callable returning a usage mapping"
        )
    return frozen_usage(result)


def _call_usage_counter(usage_counter, request: Mapping[str, object]) -> object:
    """
    Call a usage_counter with kwargs matched to its signature.

    Counters that accept ``**kwargs`` receive the full request payload. Fixed-
    signature counters receive only the named request fields they declare, so
    callers do not need to accept unrelated kwargs like ``model``.

    The counter's parameters must be keyword-addressable. Positional-only
    parameters (declared before ``/``) cannot receive request fields because
    request data arrives as kwargs, so a required positional-only parameter
    is rejected with a clear error rather than letting Python raise a
    cryptic ``TypeError: missing 1 required positional argument`` from
    inside the counter.
    """
    try:
        signature = inspect.signature(usage_counter)
    except (TypeError, ValueError):
        return usage_counter(**request)

    # Required positional-only check runs BEFORE the VAR_KEYWORD early-return:
    # a counter like ``def counter(model, /, **kwargs)`` otherwise slipped
    # through to ``usage_counter(**request)`` and raised Python's cryptic
    # ``TypeError: missing 1 required positional argument`` — exactly the
    # error this check was added to prevent.
    required_positional_only = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY
        and parameter.default is inspect.Parameter.empty
    ]
    if required_positional_only:
        raise ValueError(
            f"usage_counter has required positional-only parameter(s) "
            f"{required_positional_only}; request fields are passed as "
            "keyword arguments, so counter parameters must be "
            "POSITIONAL_OR_KEYWORD, KEYWORD_ONLY, or accept **kwargs."
        )

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return usage_counter(**request)

    accepted_kwargs = {
        name: request[name]
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and name in request
    }
    return usage_counter(**accepted_kwargs)
