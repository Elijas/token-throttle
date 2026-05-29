"""Shared validation logic — used by both async RateLimiter and SyncRateLimiter."""

import inspect
import math
import warnings
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

from token_throttle._capacity import _validate_max_capacity_finite_positive
from token_throttle._dto import StrictDTO
from token_throttle._exceptions import CardinalityLimitExceededError
from token_throttle._interfaces._callable_utils import (
    close_awaitable_if_possible,
    is_async_callable,
)
from token_throttle._interfaces._interfaces import PerModelConfig, PerModelConfigGetter
from token_throttle._interfaces._models import (
    _UNLIMITED_FLAG,
    MAX_ALIAS_LENGTH,
    MAX_KEY_PREFIX_LENGTH,
    MAX_METRIC_LENGTH,
    MAX_MODEL_FAMILY_LENGTH,
    MAX_PER_SECONDS,
    MAX_RESERVATION_ID_LENGTH,
    BucketId,
    CapacityReservation,
    FrozenUsage,
    Usage,
    UsageQuotas,
    _coerce_usage_value,
    _is_bool_like,
    _validate_key_segment,
    frozen_usage,
)

MAX_TOTAL_KEY_LENGTH = 8192

# Re-exported from ``_models`` so external callers that imported
# ``_UNLIMITED_FLAG`` from this module keep working.
__all__ = ["_UNLIMITED_FLAG"]


def _revalidate_dto[StrictDTO_T: StrictDTO](instance: StrictDTO_T) -> StrictDTO_T:
    """Force a fresh validation pass for an exact DTO instance."""
    return instance.revalidate()


def _validate_key_prefix(value: object) -> str:
    """Validate the deployment-scoped Redis key prefix."""
    return _validate_key_segment(
        value,
        field_name="key_prefix",
        max_length=MAX_KEY_PREFIX_LENGTH,
    )


def _validate_reservation_id(value: object) -> str:
    """Validate a reservation id before it is embedded in a Redis key."""
    return _validate_key_segment(
        value,
        field_name="reservation_id",
        max_length=MAX_RESERVATION_ID_LENGTH,
    )


def _validate_total_key_length(key: str) -> str:
    """Reject constructed Redis keys above token-throttle's bounded key cap."""
    if len(key) > MAX_TOTAL_KEY_LENGTH:
        raise CardinalityLimitExceededError(
            f"Redis key must be at most {MAX_TOTAL_KEY_LENGTH} characters "
            f"(got {len(key)})"
        )
    return key


def is_unlimited_reservation(reservation: object) -> bool:
    """
    True when a reservation represents a disabled/unlimited rate limit.

    The ``is_unlimited`` flag is the single source of truth. The
    ``CapacityReservation`` field validator now enforces that
    ``is_unlimited=True`` requires the canonical sentinel shape
    (``model_family == _UNLIMITED_FLAG``, empty ``usage``,
    ``bucket_ids is None``), so the flag is reliable end-to-end.

    Legacy fallback removed (L05 I10): a reservation with the sentinel
    ``model_family`` but ``is_unlimited=False`` is no longer treated as
    unlimited. Hand-constructing the sentinel string was the second
    bypass vector beyond V05; closing it requires this tightening AND
    the validator above.
    """
    if type(reservation) is not CapacityReservation:
        raise ValueError(
            "reservation must be a CapacityReservation "
            f"(got {type(reservation).__name__})"
        )
    reservation = _revalidate_dto(reservation)
    return reservation.is_unlimited is True


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
    sentinel = object()
    total_tokens = getattr(usage, "total_tokens", sentinel)
    if total_tokens is sentinel:
        if not isinstance(usage, Mapping):
            token_attrs = ("prompt_tokens", "completion_tokens")
            if any(hasattr(usage, attr) for attr in token_attrs):
                raise ValueError(
                    "usage object has prompt_tokens/completion_tokens but no "
                    "total_tokens; sum them manually and use refund_capacity()"
                )
            raise ValueError(
                "usage must be an object with total_tokens attribute or a mapping"
            )
        try:
            total_tokens = usage["total_tokens"]
        except KeyError:
            raise ValueError(
                "'total_tokens' key not found in usage data — "
                "pass actual usage via refund_capacity() instead."
            ) from None
    if total_tokens is None:
        raise ValueError(
            "total_tokens is None — cannot compute refund. "
            "Pass actual usage via refund_capacity() instead."
        )
    if _is_bool_like(total_tokens):
        raise ValueError("total_tokens must not be a boolean")
    if not isinstance(total_tokens, (int, float)):
        raise ValueError(  # noqa: TRY004 - public validators raise ValueError.
            f"total_tokens must be an int or float (got {type(total_tokens).__name__})"
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
    usage_keys = set(usage)
    if usage_keys != expected_keys:
        missing = sorted(expected_keys - usage_keys)
        extra = sorted(usage_keys - expected_keys)
        raise ValueError(
            f"Usage keys do not match {expected_keys_label}: "
            f"missing={missing}, extra={extra}. "
            f"Usage keys={sorted(usage_keys)}, {expected_keys_label}={sorted(expected_keys)}",
        )
    for metric, amount_ in usage.items():
        _validate_key_segment(metric, field_name="metric")
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


def validate_usage_values_non_negative(
    usage: FrozenUsage,
    *,
    value_label: str = "Usage value",
) -> None:
    """
    Validate usage values without requiring a configured quota key set.

    Unlimited configs discard usage before reserving backend capacity, but the
    public APIs still validate finite, non-negative usage for drop-in parity
    with limited configs.
    """
    for metric, amount in usage.items():
        if amount < 0:
            raise ValueError(f"{value_label} for {metric} must be non-negative")


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
    if not isinstance(timeout, (int, float)):
        raise ValueError(  # noqa: TRY004 - public validators raise ValueError.
            f"timeout must be finite int, float, or None (got {type(timeout).__name__})"
        )
    try:
        timeout_value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"timeout must be finite or None (got {timeout!r})") from exc
    if not math.isfinite(timeout_value):
        raise ValueError(f"timeout must be finite or None (got {timeout!r})")
    if timeout_value < 0:
        raise ValueError(f"timeout must be non-negative or None (got {timeout!r})")
    return timeout_value


def validate_sleep_interval(sleep_interval: object) -> float | None:
    """Validate backend poll sleep intervals."""
    if sleep_interval is None:
        return None
    if _is_bool_like(sleep_interval):
        raise ValueError("sleep_interval must not be a boolean")
    if not isinstance(sleep_interval, (int, float)):
        raise ValueError(  # noqa: TRY004 - public validators raise ValueError.
            f"sleep_interval must be finite and greater than 0 (got {sleep_interval!r})"
        )
    try:
        sleep_interval_value = float(sleep_interval)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"sleep_interval must be finite and greater than 0 (got {sleep_interval!r})"
        ) from exc
    if not math.isfinite(sleep_interval_value) or sleep_interval_value <= 0:
        raise ValueError(
            f"sleep_interval must be finite and greater than 0 (got {sleep_interval!r})"
        )
    return sleep_interval_value


def validate_max_capacity_value(value: object) -> float:
    """Validate the value parameter for set_max_capacity."""
    if _is_bool_like(value):
        raise ValueError("max_capacity must not be a boolean")
    if not isinstance(value, (int, float)):
        raise ValueError(  # noqa: TRY004 - public validators raise ValueError.
            f"max_capacity must be finite and greater than 0 (got {value!r})"
        )
    try:
        return _validate_max_capacity_finite_positive(value)
    except ValueError as exc:
        raise ValueError(
            f"max_capacity must be finite and greater than 0 (got {value!r})"
        ) from exc


def merge_extra_usage(
    usage: FrozenUsage,
    extra_usage: Mapping[str, object] | None,
) -> FrozenUsage:
    """
    Add extra usage values to counted usage with consistent numeric checks.

    Values are increments, not replacements. In limited configs, every
    extra_usage key must already be present in the usage_counter output.
    """
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
        normalized_metric = _validate_key_segment(metric, field_name="metric")
        if not allow_new_keys and normalized_metric not in merged_usage:
            raise ValueError(
                f"extra_usage key '{normalized_metric}' is not in counter output - "
                "to add custom metrics, ensure the counter emits the key first.",
            )
        amount = _coerce_extra_usage_value(normalized_metric, raw_amount)
        if amount < 0:
            raise ValueError(
                f"extra_usage value for {normalized_metric} must be non-negative"
            )
        try:
            merged_value = merged_usage.get(normalized_metric, 0.0) + amount
        except OverflowError as exc:
            raise ValueError(
                f"extra_usage value for {normalized_metric} "
                "too large to fit in IEEE 754 double"
            ) from exc
        if not math.isfinite(merged_value):
            raise ValueError(
                f"extra_usage value for {normalized_metric} "
                "too large to fit in IEEE 754 double"
            )
        merged_usage[normalized_metric] = merged_value
    return frozen_usage(merged_usage)


def validate_extra_usage(
    extra_usage: object,
) -> Mapping[str, object] | None:
    """
    Validate optional extra_usage payloads for request-based acquire helpers.

    This materializes custom Mapping instances into a plain dict, rejects
    duplicate keys from non-dict Mapping implementations, and validates value
    type/range before any usage_counter is invoked. Limited-config key checks
    still happen later because they depend on the counter output.
    """
    if extra_usage is None:
        return None
    if not isinstance(extra_usage, Mapping):
        raise ValueError(  # noqa: TRY004
            f"extra_usage must be a mapping or None (got {type(extra_usage).__name__})"
        )
    try:
        keys = list(extra_usage.keys())
    except Exception as exc:
        raise ValueError(
            "extra_usage must yield consistent key/value pairs "
            f"(got {type(exc).__name__}: {exc})"
        ) from exc
    try:
        if len(keys) != len(set(keys)):
            raise ValueError("extra_usage must not contain duplicate keys")
    except TypeError as exc:
        raise ValueError(
            f"extra_usage must yield hashable keys (got {type(exc).__name__}: {exc})"
        ) from exc
    converted: dict[str, float] = {}
    seen_metrics: set[str] = set()
    for metric, raw_amount in _materialize_extra_usage_items(extra_usage):
        normalized_metric = _validate_key_segment(metric, field_name="metric")
        if normalized_metric in seen_metrics:
            raise ValueError("extra_usage must not contain duplicate metric keys")
        seen_metrics.add(normalized_metric)
        amount = _coerce_extra_usage_value(normalized_metric, raw_amount)
        if amount < 0:
            raise ValueError(
                f"extra_usage value for {normalized_metric} must be non-negative"
            )
        converted[normalized_metric] = amount
    return converted


def _materialize_extra_usage_items(
    extra_usage: Mapping[object, object],
) -> list[tuple[object, object]]:
    try:
        raw_items = extra_usage.items()
    except Exception as exc:
        raise ValueError(
            "extra_usage must yield consistent key/value pairs "
            f"(got {type(exc).__name__}: {exc})"
        ) from exc

    materialized: list[tuple[object, object]] = []
    try:
        for raw_item in raw_items:
            try:
                metric, amount = raw_item
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"extra_usage must yield metric/value pairs (got {raw_item!r})"
                ) from exc
            materialized.append((metric, amount))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "extra_usage must yield consistent key/value pairs "
            f"(got {type(exc).__name__}: {exc})"
        ) from exc
    return materialized


def _coerce_extra_usage_value(metric: str, raw_amount: object) -> float:
    try:
        return _coerce_usage_value(metric, raw_amount, label="extra_usage value")
    except ValueError as exc:
        if "too large to fit in IEEE 754 double" in str(exc):
            raise ValueError(
                f"extra_usage value for {metric} too large to fit in IEEE 754 double"
            ) from exc
        raise


def validate_metric(metric: object, *, max_length: int = MAX_METRIC_LENGTH) -> str:
    """Validate the metric parameter for set_max_capacity."""
    return _validate_key_segment(metric, field_name="metric", max_length=max_length)


def validate_model_family(
    model_family: object,
    *,
    max_length: int = MAX_MODEL_FAMILY_LENGTH,
) -> str:
    """Validate a model_family key segment."""
    return _validate_key_segment(
        model_family,
        field_name="model_family",
        max_length=max_length,
    )


def validate_per_seconds(per_seconds: object) -> int:
    """
    Validate the per_seconds parameter for set_max_capacity.

    Accepts integer values directly.  Float and other numeric-looking
    objects are rejected so public validation matches strict model
    construction and cannot be reached through arbitrary ``__float__`` or
    ``__index__`` implementations.
    """
    if _is_bool_like(per_seconds):
        raise ValueError("per_seconds must not be a boolean")
    if type(per_seconds) is not int:
        raise ValueError(
            f"per_seconds must be a positive integer exact int (got {per_seconds!r}); "
            "use a plain int number of seconds such as 60"
        )
    if per_seconds <= 0:
        raise ValueError(
            f"per_seconds must be a positive integer exact int (got {per_seconds!r}); "
            "use a plain int number of seconds such as 60"
        )
    if per_seconds > MAX_PER_SECONDS:
        raise ValueError(
            f"per_seconds must be <= {MAX_PER_SECONDS} seconds "
            f"(got {per_seconds!r}); choose a smaller quota window"
        )
    return per_seconds


def _validate_model_name(model_name: object, *, max_alias_length: int) -> str:
    if type(model_name) is not str:
        raise ValueError(
            f"model_name must be a string (got {type(model_name).__name__}); "
            "set the 'model' parameter to a non-empty model name string"
        )
    if len(model_name) > max_alias_length:
        raise CardinalityLimitExceededError(
            f"max_alias_length exceeded: model_name must be at most "
            f"{max_alias_length} characters "
            f"(got {len(model_name)}); shorten the model alias or increase "
            "max_alias_length"
        )
    if not model_name:
        raise ValueError(
            "model_name cannot be empty; set the 'model' parameter to a "
            "non-empty model name string"
        )
    if not model_name.strip():
        raise ValueError(
            f"model_name cannot be whitespace-only (got {model_name!r}); "
            "set the 'model' parameter to a non-empty model name string"
        )
    return _validate_key_segment(
        model_name,
        field_name="model_name",
        max_length=max_alias_length,
    )


def resolve_config(
    cfg: PerModelConfig | PerModelConfigGetter,
    model_name: str,
    *,
    max_model_family_length: int = MAX_MODEL_FAMILY_LENGTH,
    max_alias_length: int = MAX_ALIAS_LENGTH,
) -> PerModelConfig:
    """
    Resolve a config (static or callable) and default model_family to model_name.

    Raises:
        ValueError: If model_name is not a non-empty, non-whitespace string.

    """
    model_name = _validate_model_name(model_name, max_alias_length=max_alias_length)
    if callable(cfg) and is_async_callable(cfg):
        raise ValueError("cfg must be a synchronous PerModelConfig getter")
    r = cfg(model_name) if callable(cfg) else cfg
    if inspect.isawaitable(r):
        close_awaitable_if_possible(r)
        raise ValueError("cfg must be a synchronous PerModelConfig getter")
    if type(r) is not PerModelConfig:
        raise ValueError(f"cfg must resolve to PerModelConfig (got {type(r).__name__})")
    r = _revalidate_dto(r)
    resolved = (
        r.model_copy()
        if r.model_family
        else r.model_copy(update={"model_family": model_name})
    )
    model_family = resolved.get_model_family()
    _validate_key_segment(
        model_family,
        field_name="model_family",
        max_length=max_model_family_length,
    )
    return resolved


def resolve_usage_counter_result(
    usage_counter,
    /,
    *,
    warn_if_sync_counter_blocks_event_loop: bool = False,
    **request,
) -> FrozenUsage:
    """Run a usage_counter and reject awaitable return values with a clear error."""
    if warn_if_sync_counter_blocks_event_loop:
        warnings.warn(
            "Synchronous usage_counter is being invoked inline on the asyncio "
            "event loop; use a fast counter or wrap blocking work explicitly "
            "with asyncio.to_thread.",
            UserWarning,
            stacklevel=2,
        )
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
        # Builtin C functions (e.g. len) and some extension types don't
        # expose an inspectable signature. Fall back to **kwargs
        # invocation, which will raise TypeError if the counter doesn't
        # accept keyword arguments.
        return _invoke_usage_counter(usage_counter, request)

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
        return _invoke_usage_counter(usage_counter, request)

    warnings.warn(
        "usage_counter without **kwargs uses deprecated signature-filtered "
        "dispatch; request fields not named in the counter signature are "
        "not passed to the counter. Add **kwargs to receive the full request.",
        UserWarning,
        stacklevel=2,
    )
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
    return _invoke_usage_counter(usage_counter, accepted_kwargs)


def _invoke_usage_counter(
    usage_counter,
    request: Mapping[str, object],
) -> object:
    try:
        return usage_counter(**request)
    except Exception as exc:
        raise ValueError(f"usage_counter raised: {type(exc).__name__}: {exc}") from exc
