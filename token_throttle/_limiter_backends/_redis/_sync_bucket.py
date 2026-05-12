import json
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

from token_throttle._capacity import CalculatedCapacity, calculate_capacity
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, _is_bool_like

from ._server_time import sync_server_time

__all__ = ["CalculatedCapacity", "SyncRedisBucket"]


class RedisPipelineResultError(RuntimeError):
    """Redis pipeline returned an unusable result shape."""


def _raise_pipeline_response_error(
    context: str, exc: redis.exceptions.ResponseError
) -> typing.NoReturn:
    raise RedisPipelineResultError(
        f"Redis pipeline failed at {context}: {exc}"
    ) from exc


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
    return results


def _validate_bucket_state_result(value: object, *, context: str) -> object:
    if isinstance(value, BaseException):
        raise RedisPipelineResultError(
            f"{context}: Redis returned an error response for bucket state: {value}"
        )
    if value is None:
        return None
    if type(value) is bool or not isinstance(value, (bytes, str, int, float)):
        raise RedisPipelineResultError(
            f"{context}: unexpected bucket-state result type {type(value).__name__}"
        )
    return value


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

    Runtime override Redis Key Format:
        rate_limiting:{model_family}:{metric}:{per_seconds}:max_capacity_override

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

    def __init__(
        self,
        quota: Quota,
        limit_config: PerModelConfig,
        redis_client: redis.Redis,
    ):
        self.usage_metric = quota.metric
        self.per_seconds = quota.per_seconds
        self.full_redis_key = f"rate_limiting:{limit_config.model_family}:{self.usage_metric}:{int(self.per_seconds)}"
        self.model_family = limit_config.get_model_family()

        # Default/configured max_capacity from quota (used when no runtime
        # override is present in Redis).
        self._max_capacity_default = float(quota.limit)
        # Cache for a runtime override fetched from Redis.
        self._max_capacity_cached: float | None = None
        self._max_capacity_cache_populated: bool = False
        self._max_capacity_cache_time: float = 0.0

        self._redis = redis_client
        # Initialised from quota.limit; corrected on the first override refresh.
        self._rate_per_sec = float(quota.limit) / float(quota.per_seconds)
        # Keys for Redis
        self._last_checked_key = f"{self.full_redis_key}:last_checked"
        self._capacity_key = f"{self.full_redis_key}:capacity"
        self._lock_key = f"{self.full_redis_key}:lock"
        self._max_capacity_key = f"{self.full_redis_key}:max_capacity_override"

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

        # Fetch runtime override from Redis
        stored_value = self._redis.get(self._max_capacity_key)
        self.update_max_capacity_from_result(stored_value)
        self._max_capacity_cache_time = current_time
        return self.max_capacity

    def _set_cached_max_capacity_override(self, value: float | None) -> None:
        self._max_capacity_cached = value
        self._max_capacity_cache_populated = True
        effective_limit = self.max_capacity
        self._rate_per_sec = effective_limit / float(self.per_seconds)

    @staticmethod
    def _parse_positive_finite_value(raw_value: object) -> float | None:
        if type(raw_value) is bool or not isinstance(raw_value, (int, float)):
            return None
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(parsed) and parsed > 0):
            return None
        return parsed

    def _deserialize_max_capacity_override(self, raw_value: object) -> float | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, BaseException):
            raise MaxCapacityOverrideParseError(
                "Redis max_capacity override command returned an error response"
            )
        if not isinstance(raw_value, (bytes, str, dict)):
            raise MaxCapacityOverrideParseError(
                "Redis max_capacity override must be bytes, str, dict, or None "
                f"(got {type(raw_value).__name__})"
            )

        decoded: object = raw_value
        if isinstance(raw_value, bytes):
            try:
                decoded = raw_value.decode()
            except UnicodeDecodeError as exc:
                raise MaxCapacityOverrideParseError(
                    "Redis max_capacity override is not valid UTF-8"
                ) from exc

        if isinstance(decoded, str):
            try:
                decoded = json.loads(decoded)
            except json.JSONDecodeError as exc:
                raise MaxCapacityOverrideParseError(
                    "Redis max_capacity override is not valid JSON"
                ) from exc

        if not isinstance(decoded, dict):
            raise MaxCapacityOverrideParseError(
                "Redis max_capacity override JSON must decode to an object"
            )

        override_value = self._parse_positive_finite_value(
            decoded.get(self._OVERRIDE_LIMIT_KEY)
        )
        if override_value is None:
            raise MaxCapacityOverrideParseError(
                "Redis max_capacity override has invalid override_max_capacity"
            )

        configured_limit = self._parse_positive_finite_value(
            decoded.get(self._CONFIGURED_LIMIT_KEY)
        )
        if configured_limit is None:
            raise MaxCapacityOverrideParseError(
                "Redis max_capacity override has invalid configured_max_capacity"
            )
        if configured_limit != self._max_capacity_default:
            return None
        return override_value

    def update_max_capacity_from_result(self, raw_value: object) -> None:
        """Update the runtime-override cache from a pre-fetched pipeline result."""
        new_value = self._deserialize_max_capacity_override(raw_value)
        self._set_cached_max_capacity_override(new_value)
        self._max_capacity_cache_time = time.time()

    def set_max_capacity(self, value: float) -> None:
        """Persist a runtime max-capacity override in Redis."""
        if _is_bool_like(value):
            raise ValueError("max_capacity must not be a boolean")
        if not (math.isfinite(value) and value > 0):
            raise ValueError("max_capacity must be finite and greater than 0")

        payload = json.dumps(
            {
                self._CONFIGURED_LIMIT_KEY: self._max_capacity_default,
                self._OVERRIDE_LIMIT_KEY: value,
            }
        )
        self._redis.set(self._max_capacity_key, payload)
        # Update runtime override cache immediately
        self._set_cached_max_capacity_override(value)
        self._max_capacity_cache_time = time.time()

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
        if not (math.isfinite(value) and value > 0):
            raise ValueError("max_capacity must be finite and greater than 0")

        self._max_capacity_default = value
        if self._max_capacity_cached is None:
            self._rate_per_sec = value / float(self.per_seconds)

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
                expected_count=2,
            )
            last_checked, capacity = _normalize_bucket_state_pair(
                results[self.PIPELINE_LAST_CHECKED_OFFSET],
                results[self.PIPELINE_CAPACITY_OFFSET],
                context=f"SyncRedisBucket.get_capacity({self.full_redis_key})",
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
        new_capacity = new_capacity if allow_negative else max(0, new_capacity)
        pipeline.set(self._last_checked_key, current_time)
        pipeline.set(self._capacity_key, new_capacity)

        if execute:
            try:
                pipeline.execute()
            except redis.exceptions.ResponseError as exc:
                _raise_pipeline_response_error("SyncRedisBucket.set_capacity", exc)

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
