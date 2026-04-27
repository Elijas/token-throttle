import contextlib
import json
import math
import time

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
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(parsed) and parsed > 0):
            return None
        return parsed

    def _deserialize_max_capacity_override(
        self, raw_value: bytes | str | None
    ) -> float | None:
        if raw_value is None:
            return None

        decoded: object = raw_value
        if isinstance(raw_value, bytes):
            decoded = raw_value.decode()

        if isinstance(decoded, str):
            with contextlib.suppress(json.JSONDecodeError):
                decoded = json.loads(decoded)

        if not isinstance(decoded, dict):
            return None

        override_value = self._parse_positive_finite_value(
            decoded.get(self._OVERRIDE_LIMIT_KEY)
        )
        if override_value is None:
            return None

        configured_limit = self._parse_positive_finite_value(
            decoded.get(self._CONFIGURED_LIMIT_KEY)
        )
        if configured_limit is None:
            return None
        if configured_limit != self._max_capacity_default:
            return None
        return override_value

    def update_max_capacity_from_result(self, raw_value: bytes | None) -> None:
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
        """Update the configured/static max capacity without persisting an override."""
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

    __hash__ = None  # mutable; defining __eq__ without __hash__

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

        pipeline.get(self._last_checked_key)
        pipeline.get(self._capacity_key)

        if own_pipeline:
            # Refresh max_capacity cache before calculating
            self.get_max_capacity()
            results = pipeline.execute()
            last_checked, capacity = results
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

        new_capacity = new_capacity if allow_negative else max(0, new_capacity)
        pipeline.set(self._last_checked_key, current_time)
        pipeline.set(self._capacity_key, new_capacity)

        if execute:
            pipeline.execute()

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
