import json
import logging
import math
import time
import typing
from collections.abc import Sequence

try:
    import redis.asyncio
    import redis.asyncio.client
    import redis.asyncio.lock
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
from token_throttle._validation import validate_per_seconds

from ._server_time import async_server_time

# Re-export for backwards compatibility
__all__ = ["CalculatedCapacity", "RedisBucket"]

_logger = logging.getLogger(__name__)


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


class RedisBucket:
    """
    Token bucket implementation backed by Redis for distributed rate limiting.

    Runtime override Redis Key Format:
        rate_limiting:{model_family}:{metric}:{per_seconds}:max_capacity_override
        Example:
            rate_limiting:gemini/gemini-2.0-flash:requests:1:max_capacity_override

    Legacy override formats are intentionally not migrated because they were
    unanchored and cannot be safely applied after a config change. Era 1
    ``:max_capacity`` keys and Era 2 bare-numeric / unanchored JSON values are
    logged and ignored; operators should re-set runtime overrides after upgrade.

    Static quota limits come from the current ``PerModelConfig`` in each process.
    Only explicit runtime overrides from ``set_max_capacity()`` are persisted in
    Redis so stale static config from an old deployment cannot pin future
    processes to the wrong limit.
    """

    # Per-process convenience cache; correctness is guaranteed by lock-held re-read. 1s staleness is advisory. Audited 2026-04.
    MAX_CAPACITY_CACHE_TTL = 1.0

    # Pipeline command offsets within get_capacity(). The backend's
    # _get_capacities_unsafe indexes results using these offsets.
    PIPELINE_LAST_CHECKED_OFFSET = 0
    PIPELINE_CAPACITY_OFFSET = 1

    def __init__(
        self,
        quota: Quota,
        limit_config: PerModelConfig,
        redis_client: redis.asyncio.Redis,
        *,
        override_ttl_seconds: int | None = None,
    ):
        self.usage_metric = quota.metric
        self.per_seconds = validate_per_seconds(quota.per_seconds)
        self.full_redis_key = f"rate_limiting:{limit_config.model_family}:{self.usage_metric}:{int(self.per_seconds)}"
        self.model_family = limit_config.get_model_family()

        # Default/configured max_capacity from quota (used when no runtime
        # override is present in Redis).
        self._max_capacity_default = _validate_max_capacity_finite_positive(quota.limit)
        # Cache for a runtime override fetched from Redis.
        self._max_capacity_cached: float | None = None
        self._max_capacity_cache_populated: bool = False
        self._max_capacity_cache_time: float = 0.0
        self._override_ttl_seconds = self._validate_override_ttl_seconds(
            override_ttl_seconds
        )

        self._redis = redis_client
        # Initialised from quota.limit; corrected on the first override refresh.
        self._rate_per_sec = _calculate_rate_per_sec(
            self._max_capacity_default, self.per_seconds
        )
        # Keys for Redis
        self._last_checked_key = f"{self.full_redis_key}:last_checked"
        self._capacity_key = f"{self.full_redis_key}:capacity"
        self._lock_key = f"{self.full_redis_key}:lock"
        self._max_capacity_key = f"{self.full_redis_key}:max_capacity_override"
        self._legacy_max_capacity_key = f"{self.full_redis_key}:max_capacity"

    @property
    def configured_max_capacity(self) -> float:
        """
        Returns the configured max_capacity value from the current PerModelConfig.
        """
        return self._max_capacity_default

    @property
    def max_capacity(self) -> float:
        """
        Returns the current max_capacity value.

        Returns the cached value if it has been fetched, otherwise returns
        the default from quota.limit. The cached value persists until
        explicitly refreshed via get_max_capacity().
        """
        if self._max_capacity_cached is not None:
            return self._max_capacity_cached
        return self._max_capacity_default

    @property
    def _rate_per_sec(self) -> float:
        return self._rate_per_sec_value

    @_rate_per_sec.setter
    def _rate_per_sec(self, value: object) -> None:
        self._rate_per_sec_value = _validate_rate_per_sec_finite_positive(value)

    async def get_max_capacity(self) -> float:
        """
        Fetch the runtime override from Redis (if cache is stale) and return the
        effective max capacity.

        Uses a 1-second cache TTL to avoid excessive Redis calls. Returns
        the cached override immediately if it's fresh, otherwise fetches from
        Redis. If the override key doesn't exist, the configured quota limit is
        used.
        """
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
        stored_value = await self._redis.get(self._max_capacity_key)
        self.update_max_capacity_from_result(stored_value)
        self._max_capacity_cache_time = current_time
        return self.max_capacity

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
                    "Stale Redis bucket state at %r: expected positive finite value",
                    raw_value,
                )
            return None
        try:
            parsed = _validate_max_capacity_finite_positive(raw_value)
        except ValueError as parse_error:
            _logger.warning(
                "Stale Redis bucket state at %r: %r",
                raw_value,
                parse_error,
            )
            return None
        return parsed

    @staticmethod
    def _validate_override_ttl_seconds(value: object) -> int | None:
        if value is None:
            return None
        if type(value) is bool or not isinstance(value, int):
            raise TypeError("override_ttl_seconds must be an int number of seconds")
        if value <= 0:
            raise ValueError("override_ttl_seconds must be greater than 0")
        return value

    def _handle_corrupt_override(self, reason: str, *, raw_value: object) -> None:
        _logger.warning(
            "Ignoring corrupt Redis max_capacity override for bucket %s at key %s: "
            "%s (raw=%r). Treating it as missing override.",
            self.full_redis_key,
            self._max_capacity_key,
            reason,
            raw_value,
        )

    def _handle_legacy_override(
        self,
        era: str,
        reason: str,
        *,
        key: str | None = None,
        raw_value: object,
    ) -> None:
        _logger.warning(
            "Ignoring legacy Redis max_capacity override for bucket %s at key %s: "
            "%s format detected (%s, raw=%r). Re-set the override after upgrade "
            "to write the current anchored format.",
            self.full_redis_key,
            key if key is not None else self._max_capacity_key,
            era,
            reason,
            raw_value,
        )

    def handle_legacy_max_capacity_key(self, raw_value: object) -> None:
        if raw_value is not None:
            self._handle_legacy_override(
                "Era 1",
                "old :max_capacity key path",
                key=self._legacy_max_capacity_key,
                raw_value=raw_value,
            )

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
            _logger.warning(
                "Ignoring Redis max_capacity override for bucket %s at key %s: "
                "configured_max_capacity anchor %r does not match current "
                "configured limit %r.",
                self.full_redis_key,
                self._max_capacity_key,
                configured_limit,
                self._max_capacity_default,
            )
            return None
        return override_value

    def update_max_capacity_from_result(self, raw_value: object) -> None:
        """Update the runtime-override cache from a pre-fetched pipeline result."""
        new_value = self._deserialize_max_capacity_override(raw_value)
        self._set_cached_max_capacity_override(new_value)
        self._max_capacity_cache_time = time.time()

    async def set_max_capacity(self, value: float) -> None:
        """
        Persist a runtime max-capacity override in Redis.

        This allows explicit runtime modification of the bucket's max capacity
        without rebuilding backends, useful for adaptive rate limiting
        scenarios. Static quota updates should use
        ``set_configured_max_capacity()`` instead.

        Args:
            value: The new max capacity value (must be > 0).

        """
        if _is_bool_like(value):
            raise ValueError("max_capacity must not be a boolean")
        value = _validate_max_capacity_finite_positive(value)

        payload = json.dumps(
            {
                self._CONFIGURED_LIMIT_KEY: self._max_capacity_default,
                self._OVERRIDE_LIMIT_KEY: value,
            }
        )
        await self._redis.set(self._SCHEMA_VERSION_KEY, self._SCHEMA_VERSION, nx=True)
        if self._override_ttl_seconds is None:
            await self._redis.set(self._max_capacity_key, payload)
        else:
            await self._redis.set(
                self._max_capacity_key,
                payload,
                ex=self._override_ttl_seconds,
            )
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
        value = _validate_max_capacity_finite_positive(value)

        self._max_capacity_default = value
        if self._max_capacity_cached is None:
            self._rate_per_sec = _calculate_rate_per_sec(value, self.per_seconds)

    async def clear_max_capacity_override(self) -> None:
        """Remove any persisted runtime override for this bucket."""
        await self._redis.delete(self._max_capacity_key)
        self._set_cached_max_capacity_override(None)
        self._max_capacity_cache_time = time.time()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RedisBucket):
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

    def lock(self, **kwargs) -> redis.asyncio.lock.Lock:
        return redis.asyncio.lock.Lock(self._redis, self._lock_key, **kwargs)

    async def get_capacity(
        self,
        pipeline: redis.asyncio.client.Pipeline | None = None,
        current_time: float | None = None,
    ) -> CalculatedCapacity | None:
        """Get the current capacity of the bucket."""
        if current_time is None:
            current_time = await async_server_time(self._redis)

        own_pipeline = pipeline is None
        if own_pipeline:
            pipeline = self._redis.pipeline()
        assert pipeline is not None  # noqa: S101

        # Order must match PIPELINE_LAST_CHECKED_OFFSET / PIPELINE_CAPACITY_OFFSET
        pipeline.get(self._last_checked_key)
        pipeline.get(self._capacity_key)

        if own_pipeline:
            # Refresh max_capacity cache before calculating
            await self.get_max_capacity()
            try:
                results = await pipeline.execute()
            except redis.exceptions.ResponseError as exc:
                _raise_pipeline_response_error("RedisBucket.get_capacity", exc)
            results = _validate_pipeline_results(
                results,
                context=f"RedisBucket.get_capacity({self.full_redis_key})",
                expected_count=2,
            )
            last_checked, capacity = _normalize_bucket_state_pair(
                results[self.PIPELINE_LAST_CHECKED_OFFSET],
                results[self.PIPELINE_CAPACITY_OFFSET],
                context=f"RedisBucket.get_capacity({self.full_redis_key})",
            )
            return self.calculate_capacity(last_checked, capacity, current_time)
        return None

    async def set_capacity(
        self,
        new_capacity: float,
        pipeline: redis.asyncio.client.Pipeline | None = None,
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
        assert pipeline is not None  # noqa: S101

        if current_time is None:
            current_time = await async_server_time(self._redis)

        if not math.isfinite(new_capacity):
            raise ValueError(f"capacity must be finite (got {new_capacity!r})")
        if new_capacity == 0.0:
            new_capacity = 0.0
        new_capacity = new_capacity if allow_negative else max(0, new_capacity)
        pipeline.set(self._last_checked_key, current_time)
        pipeline.set(self._capacity_key, new_capacity)

        if execute:
            try:
                await pipeline.execute()
            except redis.exceptions.ResponseError as exc:
                _raise_pipeline_response_error("RedisBucket.set_capacity", exc)

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
    _SCHEMA_VERSION_KEY = "rate_limiting:schema_version"
