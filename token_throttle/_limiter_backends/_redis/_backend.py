import asyncio
import time
import typing
import warnings
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, ClassVar

try:
    import redis.asyncio
    import redis.asyncio.client
except ImportError as exc:
    raise ImportError(
        'The "redis" package is required for the Redis backend. '
        'Install it with: pip install "token-throttle[redis]"'
    ) from exc
from frozendict import frozendict

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    RateLimiterBackend,
    RateLimiterBackendBuilderInterface,
)
from token_throttle._interfaces._models import Capacities, FrozenUsage

from ._bucket import RedisBucket


class CapacitiesGetterResult(typing.NamedTuple):
    capacities: Capacities
    fresh_start_buckets: list[RedisBucket]


if TYPE_CHECKING:
    from collections.abc import Mapping

LOCK_TIMEOUT_SECONDS = 30


class RedisBackendBuilder(RateLimiterBackendBuilderInterface):
    def __init__(
        self,
        redis_client: redis.asyncio.Redis,
        *,
        sleep_interval: float | None = None,
    ) -> None:
        super().__init__()
        self._redis = redis_client
        self._sleep_interval = sleep_interval

    def build(
        self,
        limit_config: PerModelConfig,
        *,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> "RateLimiterBackend":
        redis_buckets = []
        for quota in limit_config.quotas:
            b = RedisBucket(
                quota=quota,
                limit_config=limit_config,
                redis_client=self._redis,
            )
            redis_buckets.append(b)
        return RedisBackend(
            buckets=redis_buckets,
            redis=self._redis,
            sleep_interval=self._sleep_interval,
            callbacks=callbacks,
            limit_config=limit_config,
        )


class RedisBackend(RateLimiterBackend):
    DEFAULT_SLEEP_INTERVAL: ClassVar[float] = 0.1

    def __init__(
        self,
        buckets: list[RedisBucket],
        redis: redis.asyncio.Redis,
        limit_config: PerModelConfig,
        *,
        sleep_interval: float | None = None,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> None:
        super().__init__()
        self.sorted_buckets = sorted(buckets, key=lambda b: b.full_redis_key)
        self._redis = redis
        self._sleep_interval: float = (
            self.DEFAULT_SLEEP_INTERVAL if sleep_interval is None else sleep_interval
        )
        self._callbacks = callbacks
        self._limit_config = limit_config

    async def _lock(self, **kwargs) -> AsyncExitStack:
        """Acquire locks for all buckets in a consistent order."""
        stack = AsyncExitStack()

        # Sorted buckets to ensure consistent locking order
        key_sorted_buckets = sorted(self.sorted_buckets, key=lambda b: b.full_redis_key)
        for bucket in key_sorted_buckets:
            await stack.enter_async_context(bucket.lock(**kwargs))

        return stack

    async def _get_capacities_unsafe(
        self,
        pipeline: redis.asyncio.client.Pipeline | None = None,
        current_time: float | None = None,
    ) -> CapacitiesGetterResult:
        """Get capacities for all buckets."""
        if pipeline is None:
            pipeline = self._redis.pipeline()

        if current_time is None:
            current_time = time.time()

        # Assert that buckets are already sorted by key
        if self.sorted_buckets != sorted(
            self.sorted_buckets,
            key=lambda b: b.full_redis_key,
        ):
            raise RuntimeError("Buckets must be sorted by key to prevent deadlocks")
        for bucket in self.sorted_buckets:
            await bucket.get_capacity(pipeline=pipeline, current_time=current_time)

        # Execute the pipeline to get all results
        results = await pipeline.execute()

        # Refresh max_capacity cache for all buckets (uses 1-second TTL caching)
        for bucket in self.sorted_buckets:
            await bucket.get_max_capacity()

        # We're using dict instead of Usage because two different application
        # versions might use the same Redis backend that's not cleaned up
        # between deployments, and the new version might have a different
        # Usage class.
        new_capacities: Mapping[tuple[str, int], float] = {}
        fresh_start_buckets: list[RedisBucket] = []
        for i, bucket in enumerate(self.sorted_buckets):
            idx = i * 2  # Each bucket adds 2 commands
            # Process results in pairs (last_checked, capacity) for each bucket
            last_checked = results[idx]
            capacity = results[idx + 1]
            result = bucket.calculate_capacity(
                last_checked,
                capacity,
                current_time,
            )
            if result.is_fresh_start:
                fresh_start_buckets.append(bucket)
            new_capacities[(bucket.usage_metric, int(bucket.per_seconds))] = (
                result.amount
            )

        return CapacitiesGetterResult(
            capacities=frozendict(new_capacities),
            fresh_start_buckets=fresh_start_buckets,
        )

    async def _set_capacities_unsafe(
        self,
        new_capacities: Capacities,
        pipeline: redis.asyncio.client.Pipeline | None = None,
        current_time: float | None = None,
        *,
        allow_negative: bool = False,
    ) -> None:
        """Set capacities for all buckets."""
        if pipeline is None:
            pipeline = self._redis.pipeline()

        if current_time is None:
            current_time = time.time()

        for (usage_metric, per_seconds), amount in new_capacities.items():
            bucket = next(
                (
                    b
                    for b in self.sorted_buckets
                    if b.usage_metric == usage_metric and b.per_seconds == per_seconds
                ),
                None,
            )
            if bucket is None:
                raise ValueError(f"Bucket '{usage_metric}/{per_seconds}s' not found")
            await bucket.set_capacity(
                amount,
                pipeline=pipeline,
                current_time=current_time,
                execute=False,
                allow_negative=allow_negative,
            )
        await pipeline.execute()

    async def _check_and_consume_capacity(
        self,
        usage_: FrozenUsage,
    ) -> tuple[bool, Capacities, Capacities]:
        """Check if there's enough capacity and consume it if available."""
        usage: FrozenUsage = frozendict(
            {metric: float(amount) for metric, amount in usage_.items()},
        )
        completed = False
        preconsumption_capacities: Capacities = frozendict()
        postconsumption_capacities: Capacities = frozendict()
        current_time: float = 0.0
        fresh_start_buckets: list[RedisBucket] = []
        async with await self._lock(timeout=LOCK_TIMEOUT_SECONDS):
            pipeline = self._redis.pipeline()
            current_time = time.time()

            (
                preconsumption_capacities,
                fresh_start_buckets,
            ) = await self._get_capacities_unsafe(
                pipeline=pipeline,
                current_time=current_time,
            )

            for usage_metric_name, usage_amount in usage.items():
                for (
                    capacity_metric_name,
                    _,
                ), capacity_amount in preconsumption_capacities.items():
                    if usage_metric_name != capacity_metric_name:
                        continue
                    if usage_amount > capacity_amount:
                        return (
                            False,
                            preconsumption_capacities,
                            postconsumption_capacities,
                        )

            postconsumption_dict = {}
            for (
                capacity_metric_name,
                per_seconds,
            ), capacity_amount in preconsumption_capacities.items():
                for usage_metric_name, usage_amount in usage.items():
                    if capacity_metric_name != usage_metric_name:
                        continue
                    postconsumption_dict[(capacity_metric_name, per_seconds)] = (
                        capacity_amount - usage_amount
                    )
            postconsumption_capacities = frozendict(postconsumption_dict)
            await self._set_capacities_unsafe(
                postconsumption_capacities,
                pipeline=pipeline,
                current_time=current_time,
            )
            completed = True  # Release the lock before the callback
        if not completed:
            raise RuntimeError("Unexpected fallthrough in _check_and_consume_capacity")
        await self._fresh_start_buckets_callback(fresh_start_buckets)
        if self._callbacks and self._callbacks.on_capacity_consumed:
            if not all(
                [
                    preconsumption_capacities,
                    postconsumption_capacities,
                    usage,
                    current_time,
                ],
            ):
                raise ValueError("One or more arguments are empty")
            await self._callbacks.on_capacity_consumed(
                model_family=self._limit_config.get_model_family(),
                preconsumption_capacities=preconsumption_capacities,
                postconsumption_capacities=postconsumption_capacities,
                usage=usage,
                current_time=current_time,
            )
        return True, preconsumption_capacities, postconsumption_capacities

    async def consume_capacity(self, usage: FrozenUsage) -> None:
        """Consume capacity unconditionally. Capacity may go negative."""
        usage = frozendict(
            {metric: float(amount) for metric, amount in usage.items()},
        )
        preconsumption_capacities: Capacities = frozendict()
        postconsumption_capacities: Capacities = frozendict()
        current_time: float = 0.0
        fresh_start_buckets: list[RedisBucket] = []
        async with await self._lock(timeout=LOCK_TIMEOUT_SECONDS):
            pipeline = self._redis.pipeline()
            current_time = time.time()

            (
                preconsumption_capacities,
                fresh_start_buckets,
            ) = await self._get_capacities_unsafe(
                pipeline=pipeline,
                current_time=current_time,
            )

            postconsumption_dict = {}
            for (
                capacity_metric_name,
                per_seconds,
            ), capacity_amount in preconsumption_capacities.items():
                for usage_metric_name, usage_amount in usage.items():
                    if capacity_metric_name != usage_metric_name:
                        continue
                    postconsumption_dict[(capacity_metric_name, per_seconds)] = (
                        capacity_amount - usage_amount
                    )
            postconsumption_capacities = frozendict(postconsumption_dict)
            await self._set_capacities_unsafe(
                postconsumption_capacities,
                pipeline=pipeline,
                current_time=current_time,
                allow_negative=True,
            )
        await self._fresh_start_buckets_callback(fresh_start_buckets)
        if self._callbacks and self._callbacks.on_capacity_consumed:
            if not all(
                [
                    preconsumption_capacities,
                    postconsumption_capacities,
                    usage,
                    current_time,
                ],
            ):
                raise ValueError("One or more arguments are empty")
            await self._callbacks.on_capacity_consumed(
                model_family=self._limit_config.get_model_family(),
                preconsumption_capacities=preconsumption_capacities,
                postconsumption_capacities=postconsumption_capacities,
                usage=usage,
                current_time=current_time,
            )

    async def await_for_capacity(
        self,
        usage: FrozenUsage,
    ) -> None:
        """Wait until all buckets have the required capacity."""
        has_waited = False
        start_time = time.time()
        while True:
            (
                available,
                preconsumption,
                postconsumption,
            ) = await self._check_and_consume_capacity(usage)
            if available:
                if has_waited:
                    wait_time_s = time.time() - start_time
                    if self._callbacks and self._callbacks.after_wait_end_consumption:
                        await self._callbacks.after_wait_end_consumption(
                            model_family=self._limit_config.get_model_family(),
                            preconsumption_capacities=preconsumption,
                            postconsumption_capacities=postconsumption,
                            usage=frozendict(usage),
                            wait_time_s=wait_time_s,
                        )
                return

            if not has_waited:
                has_waited = True
                if self._callbacks and self._callbacks.on_wait_start:
                    await self._callbacks.on_wait_start(
                        model_family=self._limit_config.get_model_family(),
                        preconsumption_capacities=preconsumption,
                        usage=usage,
                    )

            # Wait before trying again
            await asyncio.sleep(self._sleep_interval)

    async def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        """
        Refund unused capacity back to the rate limiter based on actual usage.

        The refund mechanism handles two distinct adjustments:

        1. Token Usage Adjustment:
        - If fewer tokens were actually used than initially reserved
            (e.g., reserved 100 tokens but used only 80), the difference (20)
            is refunded.
        - If more tokens were used than initially reserved
            (e.g., reserved 100 tokens but used 120), the excess (-20)
            is treated as a negative refund, further reducing available capacity.

        2. Consumption Time Adjustment:
        - When capacity is initially acquired, we conservatively assume all
            consumption happens at the START time of the operation.
        - When refunding, we update to assume all consumption happened at the
            END time of the operation.
        - This adjustment occurs EVEN IF no token refund is needed, ensuring
            the system always records the actual end time of consumption.

        This approach provides tight adherence to rate limits without requiring
        knowledge of how resources were consumed between start and end times.
        We don't assume linear consumption or any specific pattern of usage
        during processing.

        Overuse Handling:
        If actual usage exceeds reserved usage for any metric (e.g., reserved 100
        tokens but used 120), this method will:
        1. Log a warning
        2. Apply a negative refund (-20 tokens), reducing available capacity further

        Args:
            reserved_usage: The usage that was originally reserved at the start
                            of the operation
            actual_usage: The actual usage consumed by the end of the operation
                        (may be more or less than reserved_usage)

        Example:
            TIME N=0: Reserve 100 tokens (assumes all consumed immediately)
            TIME N=10: Operation completes, but only used 80 tokens

            The refund will:
            1. Return 20 unused tokens (100-80)
            2. Update the timestamp to N=10, giving full credit for the elapsed time

            Alternative scenario:
            TIME N=0: Reserve 100 tokens
            TIME N=10: Operation completes, but used 120 tokens

            The refund will:
            1. Apply a negative refund of -20 tokens (100-120)
            2. Update the timestamp to N=10

        """
        # Calculate how much to refund for each metric
        refund_usage_: dict[str, float] = {}
        for metric, reserved_amount in reserved_usage.items():
            actual_amount = actual_usage.get(metric, 0)
            refund_amount = float(reserved_amount) - float(actual_amount)

            # Check for overuse and log a warning
            if refund_amount < 0:
                warnings.warn(
                    f"Actual usage ({actual_amount}) for {metric} exceeds "
                    f"reserved usage ({reserved_amount}). Applying negative refund.",
                    RuntimeWarning,
                )

            # Include both positive and negative refunds
            refund_usage_[metric] = refund_amount
        refund_usage: frozendict[str, float] = frozendict(refund_usage_)

        fresh_start_buckets: list[RedisBucket] = []
        completed = False
        async with await self._lock(timeout=LOCK_TIMEOUT_SECONDS):
            pipeline = self._redis.pipeline()
            current_time = time.time()

            # Get current capacities (which already account for time-based refill)
            (
                prerefund_capacities,
                fresh_start_buckets,
            ) = await self._get_capacities_unsafe(
                pipeline=pipeline,
                current_time=current_time,
            )

            # Apply refund amounts to current capacity
            updated_capacities_: dict[tuple[str, int], float] = dict(
                prerefund_capacities,
            )
            for (
                capability_usage_metric,
                per_seconds,
            ) in prerefund_capacities:
                for usage_metric, refund_amount in refund_usage.items():
                    if capability_usage_metric != usage_metric:
                        continue
                    bucket = next(
                        (
                            b
                            for b in self.sorted_buckets
                            if b.usage_metric == usage_metric
                            and b.per_seconds == per_seconds
                        ),
                        None,
                    )
                    if bucket is None:
                        raise ValueError(
                            f"Bucket '{usage_metric}/{per_seconds}s' not found",
                        )

                    # Apply refund (positive or negative) and ensure minimum of 0
                    updated_capacities_[(usage_metric, int(per_seconds))] = min(
                        max(
                            updated_capacities_[(usage_metric, int(per_seconds))]
                            + refund_amount,
                            0,
                        ),
                        bucket.max_capacity,
                    )
            updated_capacities = frozendict(updated_capacities_)

            # Always update capacities in Redis with the current time
            await self._set_capacities_unsafe(
                frozendict(updated_capacities),
                pipeline=pipeline,
                current_time=current_time,
            )
            completed = True
        if not completed:
            raise RuntimeError("Unexpected fallthrough in refund_capacity")
        await self._fresh_start_buckets_callback(fresh_start_buckets)
        if self._callbacks and self._callbacks.on_capacity_refunded:
            await self._callbacks.on_capacity_refunded(
                model_family=self._limit_config.get_model_family(),
                reserved_usage=reserved_usage,
                actual_usage=actual_usage,
                refunded_usage=refund_usage,
                prerefund_capacities=prerefund_capacities,
                postrefund_capacities=updated_capacities,
            )

    async def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        bucket = next(
            (
                b
                for b in self.sorted_buckets
                if b.usage_metric == metric and int(b.per_seconds) == per_seconds
            ),
            None,
        )
        if bucket is None:
            raise ValueError(f"Bucket '{metric}/{per_seconds}s' not found")
        await bucket.set_max_capacity(value)

    async def _fresh_start_buckets_callback(
        self,
        fresh_start_buckets: list[RedisBucket],
    ) -> None:
        if (
            fresh_start_buckets
            and self._callbacks
            and self._callbacks.on_missing_consumption_data
        ):
            for bucket in fresh_start_buckets:
                await self._callbacks.on_missing_consumption_data(
                    model_family=self._limit_config.get_model_family(),
                    usage_metric=bucket.usage_metric,
                    per_seconds=bucket.per_seconds,
                )
