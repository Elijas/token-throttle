import math
import time

try:
    import redis.asyncio
    import redis.asyncio.client
    import redis.asyncio.lock
except ImportError as exc:
    raise ImportError(
        'The "redis" package is required for the Redis backend. '
        'Install it with: pip install "token-throttle[redis]"'
    ) from exc

from token_throttle._capacity import CalculatedCapacity, calculate_capacity
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota

# Re-export for backwards compatibility
__all__ = ["CalculatedCapacity", "RedisBucket"]


class RedisBucket:
    """
    Token bucket implementation backed by Redis for distributed rate limiting.

    Redis Key Format:
        rate_limiting:{model_family}:{metric}:{per_seconds}:max_capacity
        Example: rate_limiting:gemini/gemini-2.0-flash:requests:1:max_capacity

    The max_capacity can be updated at runtime via set_max_capacity() to support
    adaptive rate limiting scenarios where limits need to change dynamically.
    """

    # Cache TTL for max_capacity reads (in seconds)
    MAX_CAPACITY_CACHE_TTL = 1.0

    def __init__(
        self,
        quota: Quota,
        limit_config: PerModelConfig,
        redis_client: redis.asyncio.Redis,
    ):
        self.usage_metric = quota.metric
        self.per_seconds = quota.per_seconds
        self.full_redis_key = f"rate_limiting:{limit_config.model_family}:{self.usage_metric}:{int(self.per_seconds)}"
        self.model_family = limit_config.get_model_family()

        # Default max_capacity from quota (used as fallback if Redis key doesn't exist)
        self._max_capacity_default = float(quota.limit)
        # Cache for dynamic max_capacity from Redis
        self._max_capacity_cached: float | None = None
        self._max_capacity_cache_time: float = 0.0

        self._redis = redis_client
        self._rate_per_sec = float(quota.limit) / float(quota.per_seconds)
        # Keys for Redis
        self._last_checked_key = f"{self.full_redis_key}:last_checked"
        self._capacity_key = f"{self.full_redis_key}:capacity"
        self._lock_key = f"{self.full_redis_key}:lock"
        self._max_capacity_key = f"{self.full_redis_key}:max_capacity"

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

    async def get_max_capacity(self) -> float:
        """
        Fetch max_capacity from Redis (if cache is stale) and return it.

        Uses a 1-second cache TTL to avoid excessive Redis calls. Returns
        the cached value immediately if it's fresh, otherwise fetches from
        Redis. If the Redis key doesn't exist, caches and returns the
        default value from quota.limit.
        """
        current_time = time.time()
        cache_age = current_time - self._max_capacity_cache_time

        # Return cached value if fresh
        if (
            self._max_capacity_cached is not None
            and cache_age < self.MAX_CAPACITY_CACHE_TTL
        ):
            return self._max_capacity_cached

        # Fetch from Redis
        stored_value = await self._redis.get(self._max_capacity_key)

        if stored_value is not None:
            try:
                parsed = float(stored_value)
                new_value = (
                    parsed
                    if math.isfinite(parsed) and parsed > 0
                    else self._max_capacity_default
                )
            except (TypeError, ValueError):
                # Invalid value in Redis, fall back to default
                new_value = self._max_capacity_default
        else:
            # Key doesn't exist, use default
            new_value = self._max_capacity_default

        if new_value != self._max_capacity_cached:
            self._max_capacity_cached = new_value
            self._rate_per_sec = new_value / float(self.per_seconds)

        self._max_capacity_cache_time = current_time
        return self._max_capacity_cached

    async def set_max_capacity(self, value: float) -> None:
        """
        Set the max_capacity in Redis for dynamic rate limit adjustment.

        This allows runtime modification of the bucket's max capacity without
        rebuilding backends, useful for adaptive rate limiting scenarios.

        Args:
            value: The new max capacity value (must be > 0).

        """
        if not (math.isfinite(value) and value > 0):
            raise ValueError("max_capacity must be finite and greater than 0")

        await self._redis.set(self._max_capacity_key, value)
        # Update cache immediately
        self._max_capacity_cached = value
        self._max_capacity_cache_time = time.time()
        self._rate_per_sec = value / float(self.per_seconds)

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

    __hash__ = None  # mutable; defining __eq__ without __hash__

    def lock(self, **kwargs) -> redis.asyncio.lock.Lock:
        return redis.asyncio.lock.Lock(self._redis, self._lock_key, **kwargs)

    async def get_capacity(
        self,
        pipeline: redis.asyncio.client.Pipeline | None = None,
        current_time: float | None = None,
    ) -> CalculatedCapacity | None:
        """Get the current capacity of the bucket."""
        if current_time is None:
            current_time = time.time()

        own_pipeline = pipeline is None
        if own_pipeline:
            pipeline = self._redis.pipeline()

        pipeline.get(self._last_checked_key)
        pipeline.get(self._capacity_key)

        if own_pipeline:
            # Refresh max_capacity cache before calculating
            await self.get_max_capacity()
            results = await pipeline.execute()
            last_checked, capacity = results
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
        if current_time is None:
            current_time = time.time()

        own_pipeline = pipeline is None
        if own_pipeline:
            pipeline = self._redis.pipeline()

        new_capacity = new_capacity if allow_negative else max(0, new_capacity)
        pipeline.set(self._last_checked_key, current_time)
        pipeline.set(self._capacity_key, new_capacity)

        if execute and own_pipeline:
            await pipeline.execute()

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
