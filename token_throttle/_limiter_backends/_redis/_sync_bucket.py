import contextlib
import hashlib
import json
import logging
import math
import time
import typing
from collections.abc import Sequence

try:
    import redis
    import redis.client
    import redis.lock
except ImportError as exc:
    raise ImportError(
        'The "redis" package is required for the Redis backend. '
        'Install it with: pip install "token-throttle[redis]"'
    ) from exc

from token_throttle._capacity import (
    CalculatedCapacity,
    _calculate_rate_per_sec,
    _validate_max_capacity_finite_positive,
    _validate_rate_per_sec_finite_positive,
    calculate_capacity,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, _is_bool_like
from token_throttle._validation import _revalidate_dto, validate_per_seconds

from ._keys import redis_key_with_suffix, redis_namespace_key, validate_redis_key_prefix
from ._server_time import sync_server_time
from ._ttl import DEFAULT_BUCKET_TTL_SECONDS, validate_redis_ttl_seconds

__all__ = ["CalculatedCapacity", "SyncRedisBucket"]

_logger = logging.getLogger(__name__)
_MAX_REDIS_GET_VALUE_BYTES = 16 * 1024
_MAX_REDIS_DIAGNOSTIC_BYTES = 96


class RedisPipelineResultError(RuntimeError):
    """Redis pipeline returned an unusable result shape."""


def _raise_pipeline_response_error(
    context: str, exc: redis.exceptions.ResponseError
) -> typing.NoReturn:
    raise RedisPipelineResultError(
        f"Redis pipeline failed at {context}: {_safe_redis_value_repr(exc)}"
    ) from exc


def _safe_redis_value_repr(value: object) -> str:
    if isinstance(value, bytes):
        prefix = value[:_MAX_REDIS_DIAGNOSTIC_BYTES]
        suffix = "" if len(value) <= _MAX_REDIS_DIAGNOSTIC_BYTES else "..."
        digest = hashlib.sha256(value).hexdigest()[:12]
        return f"bytes(len={len(value)}, sha256={digest}, prefix={prefix!r}{suffix})"
    if isinstance(value, str):
        prefix = value[:_MAX_REDIS_DIAGNOSTIC_BYTES]
        suffix = "" if len(value) <= _MAX_REDIS_DIAGNOSTIC_BYTES else "..."
        return f"str(len={len(value)}, prefix={prefix!r}{suffix})"
    return f"{type(value).__name__}({value!r})"


def _validate_pipeline_results(
    results: object,
    *,
    context: str,
    expected_count: int,
) -> Sequence[object]:
    if results is None:
        raise RedisPipelineResultError(
            f"{context}: pipeline.execute() returned None, expected "
            f"{expected_count} results"
        )
    if not isinstance(results, Sequence) or isinstance(results, (bytes, str)):
        raise RedisPipelineResultError(
            f"{context}: pipeline.execute() returned {type(results).__name__}, "
            f"expected a result sequence"
        )
    if len(results) != expected_count:
        raise RedisPipelineResultError(
            f"{context}: pipeline returned {len(results)} results, "
            f"expected {expected_count}"
        )
    for index, value in enumerate(results):
        if isinstance(value, BaseException):
            raise RedisPipelineResultError(
                f"{context}: pipeline slot {index} returned an error response: "
                f"{_safe_redis_value_repr(value)}"
            )
    return results


def _validate_redis_get_result(value: object, *, context: str) -> bytes | str | None:
    if isinstance(value, BaseException):
        raise RedisPipelineResultError(
            f"{context}: Redis GET returned an error response: "
            f"{_safe_redis_value_repr(value)}"
        )
    if value is None:
        return None
    if not isinstance(value, (bytes, str)):
        raise RedisPipelineResultError(
            f"{context}: unexpected Redis GET result type {type(value).__name__}; "
            "expected bytes, str, or None"
        )
    if len(value) > _MAX_REDIS_GET_VALUE_BYTES:
        raise RedisPipelineResultError(
            f"{context}: Redis GET value is too large "
            f"({len(value)} bytes; max {_MAX_REDIS_GET_VALUE_BYTES}) "
            f"{_safe_redis_value_repr(value)}"
        )
    return value


def _validate_bucket_state_result(value: object, *, context: str) -> object:
    return _validate_redis_get_result(value, context=context)


def _validate_expire_result(value: object, *, context: str) -> bool:
    if isinstance(value, BaseException):
        raise RedisPipelineResultError(
            f"{context}: Redis EXPIRE returned an error response: "
            f"{_safe_redis_value_repr(value)}"
        )
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise RedisPipelineResultError(
        f"{context}: unexpected Redis EXPIRE result "
        f"{_safe_redis_value_repr(value)}; expected bool or 0/1"
    )


def _validate_set_result(value: object, *, context: str) -> bool:
    if isinstance(value, BaseException):
        raise RedisPipelineResultError(
            f"{context}: Redis SET returned an error response: "
            f"{_safe_redis_value_repr(value)}"
        )
    if value is True or value in ("OK", b"OK"):
        return True
    raise RedisPipelineResultError(
        f"{context}: unexpected Redis SET result {_safe_redis_value_repr(value)}; "
        "expected True or OK"
    )


def _validate_set_nx_result(value: object, *, context: str) -> bool:
    if isinstance(value, BaseException):
        raise RedisPipelineResultError(
            f"{context}: Redis SET NX returned an error response: "
            f"{_safe_redis_value_repr(value)}"
        )
    if value is True or value in ("OK", b"OK"):
        return True
    if value is None or value is False:
        return False
    raise RedisPipelineResultError(
        f"{context}: unexpected Redis SET NX result "
        f"{_safe_redis_value_repr(value)}; expected True/OK, False, or None"
    )


def _validate_delete_result(value: object, *, context: str) -> int:
    if isinstance(value, BaseException):
        raise RedisPipelineResultError(
            f"{context}: Redis DEL returned an error response: "
            f"{_safe_redis_value_repr(value)}"
        )
    if type(value) is bool:
        return int(value)
    if type(value) is int and value >= 0:
        return value
    raise RedisPipelineResultError(
        f"{context}: unexpected Redis DEL result {_safe_redis_value_repr(value)}; "
        "expected non-negative integer"
    )


def _normalize_bucket_state_pair(
    last_checked: object,
    capacity: object,
    *,
    context: str,
) -> tuple[object, object]:
    last_checked = _validate_bucket_state_result(
        last_checked, context=f"{context} last_checked"
    )
    capacity = _validate_bucket_state_result(capacity, context=f"{context} capacity")
    if last_checked is None or capacity is None:
        return None, None
    return last_checked, capacity


class MaxCapacityOverrideParseError(ValueError):
    """Redis max-capacity override exists but is not canonical."""


class SyncRedisBucket:
    """
    Token bucket implementation backed by Redis for distributed rate limiting (sync).

    Redis key format is
    ``{key_prefix}:rate_limiting:bucket:{model_family}:{metric}:{per_seconds}:...``.

    Uses the same key format as the async RedisBucket, so sync and async
    backends can share the same Redis override state. Static quota limits come
    from the current ``PerModelConfig`` in each process.
    """

    # Cache TTL for max_capacity reads (in seconds)
    MAX_CAPACITY_CACHE_TTL = 1.0

    # Pipeline command offsets within get_capacity(). The backend's
    # _get_capacities_unsafe indexes results using these offsets.
    PIPELINE_LAST_CHECKED_OFFSET = 0
    PIPELINE_CAPACITY_OFFSET = 1

    def __init__(  # noqa: PLR0913
        self,
        quota: Quota,
        limit_config: PerModelConfig,
        redis_client: redis.Redis,
        *,
        key_prefix: str,
        bucket_ttl_seconds: int = DEFAULT_BUCKET_TTL_SECONDS,
        override_ttl_seconds: int | None = None,
    ):
        quota = _revalidate_dto(quota)
        limit_config = _revalidate_dto(limit_config)
        self.usage_metric = quota.metric
        self.per_seconds = validate_per_seconds(quota.per_seconds)
        self.key_prefix = validate_redis_key_prefix(key_prefix)
        self.model_family = limit_config.get_model_family()
        self.full_redis_key = redis_namespace_key(
            self.key_prefix,
            "bucket",
            self.model_family,
            self.usage_metric,
            int(self.per_seconds),
        )

        # Default/configured max_capacity from quota (used when no runtime
        # override is present in Redis).
        self._max_capacity_default = _validate_max_capacity_finite_positive(quota.limit)
        # Cache for a runtime override fetched from Redis.
        self._max_capacity_cached: float | None = None
        self._max_capacity_cache_populated: bool = False
        self._max_capacity_cache_time: float = 0.0
        self._bucket_ttl_seconds = validate_redis_ttl_seconds(
            bucket_ttl_seconds, name="bucket_ttl_seconds"
        )
        self._override_ttl_seconds = validate_redis_ttl_seconds(
            (
                self._bucket_ttl_seconds
                if override_ttl_seconds is None
                else override_ttl_seconds
            ),
            name="override_ttl_seconds",
        )

        self._redis = redis_client
        # Initialised from quota.limit; corrected on the first override refresh.
        self._rate_per_sec = _calculate_rate_per_sec(
            self._max_capacity_default, self.per_seconds
        )
        # Keys for Redis
        self._last_checked_key = redis_key_with_suffix(
            self.full_redis_key, "last_checked"
        )
        self._capacity_key = redis_key_with_suffix(self.full_redis_key, "capacity")
        self._lock_key = redis_key_with_suffix(self.full_redis_key, "lock")
        self._max_capacity_key = redis_key_with_suffix(
            self.full_redis_key, "max_capacity_override"
        )
        self._schema_version_key = redis_namespace_key(
            self.key_prefix,
            "schema_version",
        )

    @property
    def configured_max_capacity(self) -> float:
        """
        Returns the configured max_capacity value from the current PerModelConfig.
        """
        return self._max_capacity_default

    @property
    def max_capacity(self) -> float:
        """Returns the current max_capacity value (cached or default)."""
        if self._max_capacity_cached is not None:
            return self._max_capacity_cached
        return self._max_capacity_default

    @property
    def _rate_per_sec(self) -> float:
        return self._rate_per_sec_value

    @_rate_per_sec.setter
    def _rate_per_sec(self, value: object) -> None:
        self._rate_per_sec_value = _validate_rate_per_sec_finite_positive(value)

    def refresh_max_capacity_from_redis(self) -> float:
        """
        Force-refresh the runtime override from Redis, bypassing the local TTL.

        Lock-held rate-change snapshots use this so another worker's recent
        override cannot be hidden by this process's 1-second convenience cache.
        """
        stored_value = self._redis.get(self._max_capacity_key)
        override_value = self._deserialize_max_capacity_override(stored_value)
        if override_value is not None:
            self._redis.expire(self._max_capacity_key, self._override_ttl_seconds)
        self._set_cached_max_capacity_override(override_value)
        self._max_capacity_cache_time = time.time()
        return self.max_capacity

    def get_max_capacity(self) -> float:
        """Fetch the runtime override from Redis (if cache is stale) and return the effective max capacity."""
        # time.time() is intentional: this TTL guards a per-process cache,
        # not shared state, so local wall-clock is the correct reference.
        current_time = time.time()
        cache_age = current_time - self._max_capacity_cache_time

        # Return cached override if fresh
        if (
            self._max_capacity_cached is not None or self._max_capacity_cache_populated
        ) and cache_age < self.MAX_CAPACITY_CACHE_TTL:
            return self.max_capacity

        return self.refresh_max_capacity_from_redis()

    def _set_cached_max_capacity_override(self, value: float | None) -> None:
        self._max_capacity_cached = (
            None if value is None else _validate_max_capacity_finite_positive(value)
        )
        self._max_capacity_cache_populated = True
        effective_limit = self.max_capacity
        self._rate_per_sec = _calculate_rate_per_sec(effective_limit, self.per_seconds)

    @staticmethod
    def _parse_positive_finite_value(raw_value: object) -> float | None:
        if type(raw_value) is bool or not isinstance(raw_value, (int, float)):
            if raw_value is not None:
                _logger.warning(
                    "Stale Redis bucket state at %s: expected positive finite value",
                    _safe_redis_value_repr(raw_value),
                )
            return None
        try:
            parsed = _validate_max_capacity_finite_positive(raw_value)
        except ValueError as parse_error:
            _logger.warning(
                "Stale Redis bucket state at %s: %r",
                _safe_redis_value_repr(raw_value),
                parse_error,
            )
            return None
        return parsed

    def _handle_corrupt_override(self, reason: str, *, raw_value: object) -> None:
        _logger.warning(
            "Ignoring corrupt Redis max_capacity override for bucket %s at key %s: "
            "%s (raw=%r). Treating it as missing override.",
            self.full_redis_key,
            self._max_capacity_key,
            reason,
            _safe_redis_value_repr(raw_value),
        )

    def _handle_legacy_override(
        self,
        era: str,
        reason: str,
        *,
        raw_value: object,
    ) -> None:
        _logger.warning(
            "Ignoring legacy Redis max_capacity override for bucket %s at key %s: "
            "%s format detected (%s, raw=%r). Re-set the override after upgrade "
            "to write the current anchored format.",
            self.full_redis_key,
            self._max_capacity_key,
            era,
            reason,
            _safe_redis_value_repr(raw_value),
        )

    def _deserialize_max_capacity_override(self, raw_value: object) -> float | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, BaseException):
            raise MaxCapacityOverrideParseError(
                "Redis max_capacity override command returned an error response"
            )
        if not isinstance(raw_value, (bytes, str)):
            raise MaxCapacityOverrideParseError(
                "Redis max_capacity override must be bytes, str, or None "
                f"(got {type(raw_value).__name__})"
            )
        if len(raw_value) > _MAX_REDIS_GET_VALUE_BYTES:
            raise MaxCapacityOverrideParseError(
                "Redis max_capacity override is too large "
                f"({len(raw_value)} bytes; max {_MAX_REDIS_GET_VALUE_BYTES}) "
                f"{_safe_redis_value_repr(raw_value)}"
            )

        decoded: object = raw_value
        if isinstance(raw_value, bytes):
            try:
                decoded = raw_value.decode()
            except UnicodeDecodeError:
                self._handle_corrupt_override("not valid UTF-8", raw_value=raw_value)
                return None

        if isinstance(decoded, str):
            try:
                decoded = json.loads(decoded)
            except json.JSONDecodeError:
                self._handle_corrupt_override("not valid JSON", raw_value=raw_value)
                return None

        if not isinstance(decoded, dict):
            if self._parse_positive_finite_value(decoded) is not None:
                self._handle_legacy_override(
                    "Era 2",
                    "bare numeric override without configured_max_capacity anchor",
                    raw_value=raw_value,
                )
                return None
            self._handle_corrupt_override(
                "JSON must decode to an object", raw_value=raw_value
            )
            return None

        override_value = self._parse_positive_finite_value(
            decoded.get(self._OVERRIDE_LIMIT_KEY)
        )
        if override_value is None:
            self._handle_corrupt_override(
                "invalid override_max_capacity", raw_value=raw_value
            )
            return None

        configured_limit = self._parse_positive_finite_value(
            decoded.get(self._CONFIGURED_LIMIT_KEY)
        )
        if configured_limit is None:
            if self._CONFIGURED_LIMIT_KEY not in decoded:
                self._handle_legacy_override(
                    "Era 2",
                    "JSON override without configured_max_capacity anchor",
                    raw_value=raw_value,
                )
                return None
            self._handle_corrupt_override(
                "invalid configured_max_capacity", raw_value=raw_value
            )
            return None
        if not math.isclose(
            configured_limit,
            self._max_capacity_default,
            rel_tol=1e-12,
        ):
            return None
        return override_value

    def update_max_capacity_from_result(self, raw_value: object) -> bool:
        """Update the runtime-override cache from a pre-fetched pipeline result."""
        new_value = self._deserialize_max_capacity_override(raw_value)
        self._set_cached_max_capacity_override(new_value)
        self._max_capacity_cache_time = time.time()
        return new_value is not None

    def set_max_capacity(self, value: float) -> None:
        """Persist a runtime max-capacity override in Redis."""
        if _is_bool_like(value):
            raise ValueError("max_capacity must not be a boolean")
        value = _validate_max_capacity_finite_positive(value)

        payload = json.dumps(
            {
                self._CONFIGURED_LIMIT_KEY: self._max_capacity_default,
                self._OVERRIDE_LIMIT_KEY: value,
            }
        )
        self._redis.set(self._schema_version_key, self._SCHEMA_VERSION, nx=True)
        self._redis.set(
            self._max_capacity_key,
            payload,
            ex=self._override_ttl_seconds,
        )
        # Update runtime override cache immediately
        try:
            self._set_cached_max_capacity_override(value)
            self._max_capacity_cache_time = time.time()
        except Exception as exc:  # noqa: BLE001 - Redis write already committed.
            _logger.warning(
                "Redis max_capacity override write for bucket %s at key %s "
                "succeeded, but local cache repair failed: %r. Treating the "
                "override as committed.",
                self.full_redis_key,
                self._max_capacity_key,
                exc,
            )
            with contextlib.suppress(Exception):
                self.refresh_max_capacity_from_redis()

    def set_configured_max_capacity(self, value: float) -> None:
        """
        Update the configured/static max capacity without persisting an override.

        Only updates ``_rate_per_sec`` when no runtime override is cached.
        When an override is active, the override's rate takes precedence;
        updating the rate here would cause capacity accrual at the wrong
        rate until the next override refresh.
        """
        if _is_bool_like(value):
            raise ValueError("max_capacity must not be a boolean")
        value = _validate_max_capacity_finite_positive(value)

        self._max_capacity_default = value
        if self._max_capacity_cached is None:
            self._rate_per_sec = _calculate_rate_per_sec(value, self.per_seconds)

    def clear_max_capacity_override(self) -> None:
        """Remove any persisted runtime override for this bucket."""
        self._redis.delete(self._max_capacity_key)
        self._set_cached_max_capacity_override(None)
        self._max_capacity_cache_time = time.time()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SyncRedisBucket):
            return False
        return (
            self.usage_metric == other.usage_metric
            and self.full_redis_key == other.full_redis_key
            and self._rate_per_sec == other._rate_per_sec
            and self._max_capacity_default == other._max_capacity_default
            and self.per_seconds == other.per_seconds
            and self._last_checked_key == other._last_checked_key
            and self._capacity_key == other._capacity_key
            and self._lock_key == other._lock_key
            and self._max_capacity_key == other._max_capacity_key
        )

    __hash__ = None  # Mutable — unhashable by design (capacity/rate change at runtime). Audited 2026-04.

    def lock(self, **kwargs) -> redis.lock.Lock:
        return redis.lock.Lock(self._redis, self._lock_key, **kwargs)

    def get_capacity(
        self,
        pipeline: redis.client.Pipeline | None = None,
        current_time: float | None = None,
    ) -> CalculatedCapacity | None:
        """Get the current capacity of the bucket."""
        if current_time is None:
            current_time = sync_server_time(self._redis)

        own_pipeline = pipeline is None
        if own_pipeline:
            pipeline = self._redis.pipeline()

        # Order must match PIPELINE_LAST_CHECKED_OFFSET / PIPELINE_CAPACITY_OFFSET
        pipeline.get(self._last_checked_key)
        pipeline.get(self._capacity_key)
        pipeline.expire(self._last_checked_key, self._bucket_ttl_seconds)
        pipeline.expire(self._capacity_key, self._bucket_ttl_seconds)

        if own_pipeline:
            # Refresh max_capacity cache before calculating
            self.get_max_capacity()
            try:
                results = pipeline.execute()
            except redis.exceptions.ResponseError as exc:
                _raise_pipeline_response_error("SyncRedisBucket.get_capacity", exc)
            results = _validate_pipeline_results(
                results,
                context=f"SyncRedisBucket.get_capacity({self.full_redis_key})",
                expected_count=4,
            )
            last_checked, capacity = _normalize_bucket_state_pair(
                results[self.PIPELINE_LAST_CHECKED_OFFSET],
                results[self.PIPELINE_CAPACITY_OFFSET],
                context=f"SyncRedisBucket.get_capacity({self.full_redis_key})",
            )
            _validate_expire_result(
                results[2],
                context=f"SyncRedisBucket.get_capacity({self.full_redis_key}) "
                "last_checked TTL",
            )
            _validate_expire_result(
                results[3],
                context=f"SyncRedisBucket.get_capacity({self.full_redis_key}) "
                "capacity TTL",
            )
            return self.calculate_capacity(last_checked, capacity, current_time)
        return None

    def set_capacity(
        self,
        new_capacity: float,
        pipeline: redis.client.Pipeline | None = None,
        current_time: float | None = None,
        *,
        execute: bool = True,
        allow_negative: bool = False,
    ) -> None:
        """
        Set bucket capacity in Redis and update the timestamp.

        allow_negative: False for acquire (blocking guarantees non-negative),
        True for consume_capacity (speedometer overshoot) and refund_capacity
        (must preserve negative debt for natural refill recovery).
        When execute=True (default), queued writes are executed immediately,
        including on a caller-supplied pipeline. Pass execute=False with an
        explicit pipeline to batch the writes yourself.
        """
        own_pipeline = pipeline is None
        if own_pipeline and not execute:
            raise ValueError("execute=False requires an explicit pipeline")
        if own_pipeline:
            pipeline = self._redis.pipeline()

        if current_time is None:
            current_time = sync_server_time(self._redis)

        if not math.isfinite(new_capacity):
            raise ValueError(f"capacity must be finite (got {new_capacity!r})")
        if new_capacity == 0.0:
            new_capacity = 0.0
        new_capacity = new_capacity if allow_negative else max(0, new_capacity)
        pipeline.set(self._last_checked_key, current_time, ex=self._bucket_ttl_seconds)
        pipeline.set(self._capacity_key, new_capacity, ex=self._bucket_ttl_seconds)

        if execute:
            try:
                results = pipeline.execute()
            except redis.exceptions.ResponseError as exc:
                _raise_pipeline_response_error("SyncRedisBucket.set_capacity", exc)
            results = _validate_pipeline_results(
                results,
                context=f"SyncRedisBucket.set_capacity({self.full_redis_key})",
                expected_count=2,
            )
            _validate_set_result(
                results[0],
                context=f"SyncRedisBucket.set_capacity({self.full_redis_key}) "
                "last_checked",
            )
            _validate_set_result(
                results[1],
                context=f"SyncRedisBucket.set_capacity({self.full_redis_key}) capacity",
            )

    def calculate_capacity(
        self,
        last_checked,
        outdated_capacity,
        current_time: float,
    ) -> CalculatedCapacity:
        return calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=current_time,
            max_capacity=self.max_capacity,
            rate_per_sec=self._rate_per_sec,
            bucket_id=self.full_redis_key,
        )

    _OVERRIDE_LIMIT_KEY = "override_max_capacity"
    _CONFIGURED_LIMIT_KEY = "configured_max_capacity"
    _SCHEMA_VERSION = "3"
