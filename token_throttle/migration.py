"""
Pre-upgrade helpers for token-throttle v1.4.x to v2.0.0 migrations.

This module does not cover later v5, v6, v7, or v8 breaking changes; use
``MIGRATION.md`` for those release-specific upgrade notes.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import (
    CapacityReservation,
    Quota,
    UsageQuotas,
)
from token_throttle._limiter_backends._redis._keys import redis_namespace_key
from token_throttle._validation import _validate_key_prefix


@dataclass(frozen=True, slots=True)
class ConfigMigrationIssue:
    """One operator-actionable v1.4.x -> v2.0.0 migration issue."""

    field_path: str
    value: Any
    reason: str
    suggested_fix: str | None


def validate_config_for_v2_0(
    config: Mapping[str, Any]
    | PerModelConfig
    | Quota
    | UsageQuotas
    | CapacityReservation,
) -> list[ConfigMigrationIssue]:
    """
    Report v2.0.0 strictness issues in a config-like object without mutating it.

    The helper is intentionally read-only and non-coercive. It accepts common
    operator shapes: raw ``PerModelConfig`` dictionaries, mappings of model name
    to config dictionaries, strict DTO instances, Redis builder option
    dictionaries, and serialized ``CapacityReservation`` dictionaries.
    """
    issues: list[ConfigMigrationIssue] = []
    _scan_value(config, path="", issues=issues)
    return issues


def _scan_value(value: Any, *, path: str, issues: list[ConfigMigrationIssue]) -> None:
    if type(value) is Quota:
        _scan_quota_object(value, path=path or "quota", issues=issues)
        return
    if type(value) is UsageQuotas:
        for index, quota in enumerate(value):
            _scan_value(quota, path=_join_index(path or "quotas", index), issues=issues)
        return
    if type(value) is PerModelConfig:
        _scan_config_object(value, path=path or "config", issues=issues)
        return
    if type(value) is CapacityReservation:
        _scan_reservation_object(value, path=path or "reservation", issues=issues)
        return

    if isinstance(value, Mapping):
        _scan_mapping(value, path=path, issues=issues)
        return

    if path:
        _strict_construct_issue(path, value, issues)


def _scan_mapping(
    value: Mapping[str, Any],
    *,
    path: str,
    issues: list[ConfigMigrationIssue],
) -> None:
    # A "redis" key whose value is itself a mapping is a nested builder-options
    # section (its key_prefix lives inside that nested mapping); any other
    # redis indicator means this mapping itself is the flat builder config.
    nested_redis_config = value.get("redis")
    if isinstance(nested_redis_config, Mapping):
        _scan_redis_builder_config(
            nested_redis_config,
            path=_join_path(path, "redis"),
            issues=issues,
        )
    elif _looks_like_redis_builder_config(value):
        _scan_redis_builder_config(value, path=path or "redis", issues=issues)

    if _looks_like_quota(value):
        _scan_quota_mapping(value, path=path or "quota", issues=issues)
        return

    if _looks_like_reservation(value):
        _scan_reservation_mapping(value, path=path or "reservation", issues=issues)
        return

    if _looks_like_per_model_config(value):
        _scan_config_mapping(value, path=path or "config", issues=issues)
        return

    for key, item in value.items():
        child_path = _join_path(path, str(key))
        if isinstance(item, Mapping) or _is_supported_dto(item):
            _scan_value(item, path=child_path, issues=issues)
        elif isinstance(item, list):
            for index, element in enumerate(item):
                if isinstance(element, Mapping) or _is_supported_dto(element):
                    _scan_value(
                        element,
                        path=_join_index(child_path, index),
                        issues=issues,
                    )


def _scan_config_object(
    config: PerModelConfig,
    *,
    path: str,
    issues: list[ConfigMigrationIssue],
) -> None:
    _scan_segment(
        config.model_family,
        field_name="model_family",
        path=_join_path(path, "model_family"),
        issues=issues,
        allow_none=True,
    )
    _scan_value(config.quotas, path=_join_path(path, "quotas"), issues=issues)


def _scan_config_mapping(
    config: Mapping[str, Any],
    *,
    path: str,
    issues: list[ConfigMigrationIssue],
) -> None:
    if "model_family" in config:
        _scan_segment(
            config["model_family"],
            field_name="model_family",
            path=_join_path(path, "model_family"),
            issues=issues,
            allow_none=True,
        )

    quotas = config.get("quotas")
    quota_issues_before = len(issues)
    if isinstance(quotas, Iterable) and not isinstance(quotas, (str, bytes, Mapping)):
        quota_objects: list[Quota] = []
        for index, quota_value in enumerate(quotas):
            quota_path = _join_index(_join_path(path, "quotas"), index)
            if isinstance(quota_value, Mapping):
                _scan_quota_mapping(quota_value, path=quota_path, issues=issues)
                with contextlib.suppress(TypeError, ValidationError, ValueError):
                    quota_objects.append(Quota(**quota_value))
            elif type(quota_value) is Quota:
                _scan_value(quota_value, path=quota_path, issues=issues)
                quota_objects.append(quota_value)
            else:
                _strict_construct_issue(quota_path, quota_value, issues)
        if len(issues) == quota_issues_before:
            try:
                UsageQuotas(quota_objects)
            except ValueError as exc:
                issues.append(
                    ConfigMigrationIssue(
                        field_path=_join_path(path, "quotas"),
                        value=quotas,
                        reason=str(exc),
                        suggested_fix="Use valid, non-duplicate Quota entries or UsageQuotas.unlimited().",
                    )
                )
    elif type(quotas) is UsageQuotas:
        _scan_value(quotas, path=_join_path(path, "quotas"), issues=issues)
    elif quotas is not None:
        _strict_construct_issue(_join_path(path, "quotas"), quotas, issues)


def _scan_quota_object(
    quota: Quota,
    *,
    path: str,
    issues: list[ConfigMigrationIssue],
) -> None:
    _scan_segment(
        quota.metric,
        field_name="metric",
        path=_join_path(path, "metric"),
        issues=issues,
    )
    _scan_limit(quota.limit, path=_join_path(path, "limit"), issues=issues)
    _scan_per_seconds(
        quota.per_seconds,
        path=_join_path(path, "per_seconds"),
        issues=issues,
    )


def _scan_quota_mapping(
    quota: Mapping[str, Any],
    *,
    path: str,
    issues: list[ConfigMigrationIssue],
) -> None:
    if "metric" in quota:
        _scan_segment(
            quota["metric"],
            field_name="metric",
            path=_join_path(path, "metric"),
            issues=issues,
        )
    if "limit" in quota:
        _scan_limit(quota["limit"], path=_join_path(path, "limit"), issues=issues)
    if "per_seconds" in quota:
        _scan_per_seconds(
            quota["per_seconds"],
            path=_join_path(path, "per_seconds"),
            issues=issues,
        )

    _add_unreported_validation_errors(
        path=path,
        value=quota,
        issues=issues,
        construct=lambda: Quota(**quota),
    )


def _scan_reservation_object(
    reservation: CapacityReservation,
    *,
    path: str,
    issues: list[ConfigMigrationIssue],
) -> None:
    _scan_segment(
        reservation.model_family,
        field_name="model_family",
        path=_join_path(path, "model_family"),
        issues=issues,
    )
    _scan_limiter_instance_id(
        reservation.limiter_instance_id,
        path=_join_path(path, "limiter_instance_id"),
        issues=issues,
        present=True,
    )


def _scan_reservation_mapping(
    reservation: Mapping[str, Any],
    *,
    path: str,
    issues: list[ConfigMigrationIssue],
) -> None:
    if "model_family" in reservation:
        _scan_segment(
            reservation["model_family"],
            field_name="model_family",
            path=_join_path(path, "model_family"),
            issues=issues,
        )
    _scan_limiter_instance_id(
        reservation.get("limiter_instance_id"),
        path=_join_path(path, "limiter_instance_id"),
        issues=issues,
        present="limiter_instance_id" in reservation,
    )
    bucket_ids = reservation.get("bucket_ids")
    if bucket_ids is not None and isinstance(bucket_ids, Iterable):
        for index, item in enumerate(bucket_ids):
            if isinstance(item, (list, tuple)) and item:
                _scan_segment(
                    item[0],
                    field_name="bucket_id metric",
                    path=_join_path(
                        _join_index(_join_path(path, "bucket_ids"), index), "metric"
                    ),
                    issues=issues,
                )

    _add_unreported_validation_errors(
        path=path,
        value=reservation,
        issues=issues,
        construct=lambda: CapacityReservation(**reservation),
    )


def _scan_redis_builder_config(
    config: Mapping[str, Any],
    *,
    path: str,
    issues: list[ConfigMigrationIssue],
) -> None:
    key_path = _join_path(path, "key_prefix")
    if "key_prefix" not in config:
        issues.append(
            ConfigMigrationIssue(
                field_path=key_path,
                value=None,
                reason="Redis builders require key_prefix in v2.0.0",
                suggested_fix="Add a deployment-scoped key_prefix such as 'prod-api'.",
            )
        )
        return
    _scan_key_prefix(
        config["key_prefix"],
        path=key_path,
        issues=issues,
    )


def _scan_key_prefix(
    value: Any,
    *,
    path: str,
    issues: list[ConfigMigrationIssue],
) -> None:
    try:
        _validate_key_prefix(value)
    except ValueError as exc:
        issues.append(
            ConfigMigrationIssue(
                field_path=path,
                value=value,
                reason=str(exc),
                suggested_fix="Use a non-empty deployment-scoped key_prefix of at most 128 characters without whitespace, ':', '{', or '}'.",
            )
        )


def _scan_limit(value: Any, *, path: str, issues: list[ConfigMigrationIssue]) -> None:
    if type(value) is str:
        issues.append(
            ConfigMigrationIssue(
                field_path=path,
                value=value,
                reason="limit must be int or float, not str",
                suggested_fix="Replace the quoted number with an int or float literal.",
            )
        )
    elif type(value) is bytes:
        issues.append(
            ConfigMigrationIssue(
                field_path=path,
                value=value,
                reason="limit must be int or float, not bytes",
                suggested_fix="Decode the value and store it as an int or float literal.",
            )
        )


def _scan_per_seconds(
    value: Any,
    *,
    path: str,
    issues: list[ConfigMigrationIssue],
) -> None:
    if type(value) is float:
        suggested = (
            f"Use the integer {int(value)}."
            if value.is_integer()
            else "Use a positive integer number of seconds."
        )
        issues.append(
            ConfigMigrationIssue(
                field_path=path,
                value=value,
                reason="per_seconds must be int, not float",
                suggested_fix=suggested,
            )
        )
    elif type(value) is str:
        issues.append(
            ConfigMigrationIssue(
                field_path=path,
                value=value,
                reason="per_seconds must be int, not str",
                suggested_fix="Replace the quoted number with an int literal.",
            )
        )
    elif type(value) is bytes:
        issues.append(
            ConfigMigrationIssue(
                field_path=path,
                value=value,
                reason="per_seconds must be int, not bytes",
                suggested_fix="Decode the value and store it as an int literal.",
            )
        )


def _scan_limiter_instance_id(
    value: Any,
    *,
    path: str,
    issues: list[ConfigMigrationIssue],
    present: bool,
) -> None:
    if present and value is None:
        issues.append(
            ConfigMigrationIssue(
                field_path=path,
                value=value,
                reason="legacy reservation has limiter_instance_id=None",
                suggested_fix="Drain or refund in-flight reservations before upgrading.",
            )
        )
    else:
        _scan_segment(
            value,
            field_name="limiter_instance_id",
            path=path,
            issues=issues,
            allow_none=True,
        )


def _scan_segment(
    value: Any,
    *,
    field_name: str,
    path: str,
    issues: list[ConfigMigrationIssue],
    allow_none: bool = False,
) -> None:
    if value is None and allow_none:
        return
    if type(value) is bytes:
        issues.append(
            ConfigMigrationIssue(
                field_path=path,
                value=value,
                reason=f"{field_name} must be str, not bytes",
                suggested_fix="Decode the value to text before constructing the config.",
            )
        )
        return
    if type(value) is not str:
        issues.append(
            ConfigMigrationIssue(
                field_path=path,
                value=value,
                reason=f"{field_name} must be str",
                suggested_fix="Use a plain str value.",
            )
        )
        return
    if value != value.strip():
        issues.append(
            ConfigMigrationIssue(
                field_path=path,
                value=value,
                reason=f"{field_name} must not contain leading/trailing whitespace",
                suggested_fix=f"Use {value.strip()!r}.",
            )
        )
    elif any(char.isspace() for char in value):
        issues.append(
            ConfigMigrationIssue(
                field_path=path,
                value=value,
                reason=f"{field_name} must not contain whitespace",
                suggested_fix="Remove whitespace or replace it with '_' or '-'.",
            )
        )
    if ":" in value:
        issues.append(
            ConfigMigrationIssue(
                field_path=path,
                value=value,
                reason=f"{field_name} must not contain ':'",
                suggested_fix="Remove ':' because Redis keys use it as a separator.",
            )
        )
    if "{" in value or "}" in value:
        issues.append(
            ConfigMigrationIssue(
                field_path=path,
                value=value,
                reason=f"{field_name} must not contain '{{' or '}}'",
                suggested_fix="Remove braces because Redis Cluster uses them as hash tag delimiters.",
            )
        )


def _add_unreported_validation_errors(
    *,
    path: str,
    value: Any,
    issues: list[ConfigMigrationIssue],
    construct,
) -> None:
    seen_paths = {issue.field_path for issue in issues}
    try:
        construct()
    except ValidationError as exc:
        for error in exc.errors():
            field_path = _join_error_path(path, error.get("loc", ()))
            if field_path in seen_paths:
                continue
            issues.append(
                ConfigMigrationIssue(
                    field_path=field_path,
                    value=_lookup_path(value, error.get("loc", ())),
                    reason=str(error.get("msg", "strict validation failed")),
                    suggested_fix="Update this value to satisfy v2.0.0 strict validation.",
                )
            )
    except (TypeError, ValueError) as exc:
        if path not in seen_paths:
            issues.append(
                ConfigMigrationIssue(
                    field_path=path,
                    value=value,
                    reason=str(exc),
                    suggested_fix="Update this value to satisfy v2.0.0 strict validation.",
                )
            )


def _strict_construct_issue(
    path: str,
    value: Any,
    issues: list[ConfigMigrationIssue],
) -> None:
    issues.append(
        ConfigMigrationIssue(
            field_path=path,
            value=value,
            reason=f"unsupported config value type {type(value).__name__}",
            suggested_fix="Pass a PerModelConfig, Quota, UsageQuotas, CapacityReservation, or plain config dictionary.",
        )
    )


def _looks_like_quota(value: Mapping[str, Any]) -> bool:
    return "metric" in value and ("limit" in value or "per_seconds" in value)


def _looks_like_per_model_config(value: Mapping[str, Any]) -> bool:
    return "quotas" in value or "model_family" in value or "usage_counter" in value


def _looks_like_reservation(value: Mapping[str, Any]) -> bool:
    reservation_keys = {
        "reservation_id",
        "usage",
        "bucket_ids",
        "is_unlimited",
        "limiter_instance_id",
    }
    return "model_family" in value and bool(reservation_keys & set(value))


def _looks_like_redis_builder_config(value: Mapping[str, Any]) -> bool:
    indicators = (
        value.get("backend"),
        value.get("backend_type"),
        value.get("builder"),
        value.get("builder_type"),
        value.get("backend_builder"),
    )
    if any(isinstance(item, str) and "redis" in item.lower() for item in indicators):
        return True
    return any(key in value for key in ("redis", "redis_client", "redis_url"))


def _is_supported_dto(value: Any) -> bool:
    return type(value) in {Quota, UsageQuotas, PerModelConfig, CapacityReservation}


def _join_path(prefix: str, field: str) -> str:
    return f"{prefix}.{field}" if prefix else field


def _join_index(prefix: str, index: int) -> str:
    return f"{prefix}[{index}]"


def _join_error_path(prefix: str, loc: Iterable[Any]) -> str:
    path = prefix
    for item in loc:
        if isinstance(item, int):
            path = _join_index(path, item)
        else:
            path = _join_path(path, str(item))
    return path


def _lookup_path(value: Any, loc: Iterable[Any]) -> Any:
    current = value
    for item in loc:
        try:
            current = current[item]
        except (KeyError, IndexError, TypeError):
            return value
    return current


_LEGACY_BUCKET_STATE_SUFFIXES = (":last_checked", ":capacity")


def _validate_scan_count(value: object) -> int:
    if type(value) is not int:
        raise ValueError(f"count must be an int (got {type(value).__name__})")
    if value <= 0:
        raise ValueError("count must be greater than 0")
    return value


# Redis SCAN/KEYS use glob-style matching where `\`, `*`, `?`, `[`, `]` are
# metacharacters; a key_prefix containing any of them must be escaped so the
# MATCH pattern targets exactly this deployment's keys, not a wildcard shadow
# of unrelated ones.
_REDIS_GLOB_METACHARACTERS = frozenset("\\*?[]")


def _escape_redis_glob(value: str) -> str:
    return "".join(
        f"\\{char}" if char in _REDIS_GLOB_METACHARACTERS else char for char in value
    )


def _bucket_scan_match(key_prefix: str) -> str:
    fixed_prefix = redis_namespace_key(_validate_key_prefix(key_prefix), "bucket")
    return f"{_escape_redis_glob(fixed_prefix)}:*"


def _is_legacy_bucket_state_key(key: object) -> bool:
    if isinstance(key, bytes):
        try:
            key = key.decode()
        except UnicodeDecodeError:
            return False
    if not isinstance(key, str):
        key = str(key)
    return key.endswith(_LEGACY_BUCKET_STATE_SUFFIXES)


async def async_cleanup_legacy_buckets(
    redis_client: Any,
    key_prefix: str,
    *,
    count: int = 1000,
) -> int:
    """
    Delete Redis bucket state keys written before the v2.0.0 bucket-state TTL
    change that still have no expiry.

    Only ``:last_checked`` and ``:capacity`` keys under the configured
    token-throttle bucket namespace are considered, and only when Redis reports
    ``TTL == -1``. Keys with a positive TTL, missing keys, locks, runtime
    overrides, and schema-version keys are left untouched.
    """
    count = _validate_scan_count(count)
    match = _bucket_scan_match(key_prefix)
    deleted = 0
    iterator = redis_client.scan_iter(match=match, count=count)
    if inspect.isawaitable(iterator):
        iterator = await iterator
    if hasattr(iterator, "__aiter__"):
        async for key in iterator:
            deleted += await _async_cleanup_legacy_bucket_key(redis_client, key)
        return deleted
    for key in iterator:
        deleted += await _async_cleanup_legacy_bucket_key(redis_client, key)
    return deleted


async def _async_cleanup_legacy_bucket_key(redis_client: Any, key: object) -> int:
    if not _is_legacy_bucket_state_key(key):
        return 0
    ttl = redis_client.ttl(key)
    if inspect.isawaitable(ttl):
        ttl = await ttl
    if ttl != -1:
        return 0
    result = redis_client.delete(key)
    if inspect.isawaitable(result):
        result = await result
    return int(result or 0)


def cleanup_legacy_buckets(
    redis_client: Any,
    key_prefix: str,
    *,
    count: int = 1000,
) -> int:
    """
    Synchronous cleanup for Redis bucket state keys written before the
    v2.0.0 bucket-state TTL change that still have no expiry.

    Run during a maintenance window after draining in-flight reservations.
    Async Redis clients should use :func:`async_cleanup_legacy_buckets`.
    """
    count = _validate_scan_count(count)
    match = _bucket_scan_match(key_prefix)
    deleted = 0
    for key in redis_client.scan_iter(match=match, count=count):
        if not _is_legacy_bucket_state_key(key):
            continue
        if redis_client.ttl(key) != -1:
            continue
        deleted += int(redis_client.delete(key) or 0)
    return deleted


__all__ = [
    "ConfigMigrationIssue",
    "async_cleanup_legacy_buckets",
    "cleanup_legacy_buckets",
    "validate_config_for_v2_0",
]
