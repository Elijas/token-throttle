import asyncio
import contextlib
import time
import typing
import warnings
from contextlib import AsyncExitStack
from typing import ClassVar

try:
    import redis.asyncio
    import redis.asyncio.client
    import redis.exceptions
except ImportError as exc:
    raise ImportError(
        'The "redis" package is required for the Redis backend. '
        'Install it with: pip install "token-throttle[redis]"'
    ) from exc
from frozendict import frozendict

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._callable_utils import (
    suppress_current_task_cancellation,
)
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    RateLimiterBackend,
    RateLimiterBackendBuilderInterface,
)
from token_throttle._interfaces._models import Capacities, FrozenUsage
from token_throttle._validation import (
    validate_backend_refund_usage_for_bucket_ids,
    validate_backend_usage,
    validate_timeout,
)

from ._bucket import RedisBucket
from ._server_time import async_server_time


class CapacitiesGetterResult(typing.NamedTuple):
    capacities: Capacities
    fresh_start_buckets: list[RedisBucket]


LOCK_TIMEOUT_SECONDS = 30

# Each bucket enqueues exactly 2 pipeline commands in get_capacity()
# (GET last_checked, GET capacity).  Used to index pipeline results.
_PIPELINE_CMDS_PER_BUCKET = 2


def _raise_lock_timeout_error() -> typing.NoReturn:
    raise redis.exceptions.LockError("Unable to acquire lock within the time specified")


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
        cfg: PerModelConfig,
        *,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> "RateLimiterBackend":
        redis_buckets = []
        for quota in cfg.quotas:
            b = RedisBucket(
                quota=quota,
                limit_config=cfg,
                redis_client=self._redis,
            )
            redis_buckets.append(b)
        return RedisBackend(
            buckets=redis_buckets,
            redis=self._redis,
            sleep_interval=self._sleep_interval,
            callbacks=callbacks,
            limit_config=cfg,
        )


class RedisBackend(RateLimiterBackend):
    DEFAULT_SLEEP_INTERVAL: ClassVar[float] = 0.1
    MAX_CROSS_WORKER_POLL: ClassVar[float] = 1.0

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
        self._usage_metric_names: set[str] = {bucket.usage_metric for bucket in buckets}
        self._local_condition = asyncio.Condition()

    def supports_metric_set_change(self) -> bool:
        # Surviving bucket state lives in Redis under stable keys, so a rebuilt
        # backend can safely point at the same buckets.
        return True

    def _snapshot_buckets(self) -> tuple[RedisBucket, ...]:
        return tuple(getattr(self, "sorted_buckets", ()))

    def _validation_metric_names(
        self,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
    ) -> set[str]:
        target_buckets = self._snapshot_buckets() if buckets is None else tuple(buckets)
        if target_buckets:
            return self._usage_metric_names_for(target_buckets)
        return set(getattr(self, "_usage_metric_names", set()))

    @staticmethod
    def _usage_metric_names_for(
        buckets: tuple[RedisBucket, ...] | list[RedisBucket],
    ) -> set[str]:
        return {bucket.usage_metric for bucket in buckets}

    @staticmethod
    def _ensure_usage_metrics_are_active(
        usage: FrozenUsage,
        active_metric_names: set[str],
    ) -> None:
        missing_metrics = sorted(set(usage) - active_metric_names)
        if missing_metrics:
            raise ValueError(
                "Usage metrics "
                f"{missing_metrics} are no longer active after backend reconfiguration. "
                f"Active metrics are {sorted(active_metric_names)}."
            )

    @staticmethod
    def _find_bucket(
        buckets: tuple[RedisBucket, ...] | list[RedisBucket],
        metric: str,
        per_seconds: int,
    ) -> RedisBucket | None:
        return next(
            (
                bucket
                for bucket in buckets
                if bucket.usage_metric == metric
                and int(bucket.per_seconds) == int(per_seconds)
            ),
            None,
        )

    @staticmethod
    def _combined_bucket_snapshot(
        *bucket_groups: tuple[RedisBucket, ...] | list[RedisBucket],
    ) -> tuple[RedisBucket, ...]:
        buckets_by_key: dict[str, RedisBucket] = {}
        for bucket_group in bucket_groups:
            for bucket in bucket_group:
                buckets_by_key[bucket.full_redis_key] = bucket
        return tuple(
            sorted(buckets_by_key.values(), key=lambda bucket: bucket.full_redis_key)
        )

    async def _lock(
        self,
        *,
        timeout: float,
        blocking_timeout: float | None = None,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
    ) -> AsyncExitStack:
        """
        Acquire distributed Redis locks for a fixed bucket snapshot.

        Returns an ``AsyncExitStack`` that holds the acquired locks.  Use as::

            async with await self._lock(timeout=LOCK_TIMEOUT_SECONDS):
                ...  # all bucket locks held here

        If acquiring lock N fails, locks 0..N-1 are released immediately
        (via ``stack.aclose()``) so we never leak partially-acquired locks.
        """
        stack = AsyncExitStack()
        target_buckets = self.sorted_buckets if buckets is None else buckets
        loop = asyncio.get_running_loop()
        stop_trying_at = (
            None if blocking_timeout is None else loop.time() + blocking_timeout
        )

        try:
            for bucket in target_buckets:
                remaining = (
                    None
                    if stop_trying_at is None
                    else max(0.0, stop_trying_at - loop.time())
                )
                lock = bucket.lock(timeout=timeout)
                acquired = await lock.acquire(blocking_timeout=remaining)
                if not acquired:
                    _raise_lock_timeout_error()
                stack.push_async_callback(lock.release)
        except BaseException:
            await stack.aclose()
            raise

        return stack

    async def _get_capacities_unsafe(
        self,
        pipeline: redis.asyncio.client.Pipeline | None = None,
        current_time: float | None = None,
        *,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
    ) -> CapacitiesGetterResult:
        """Get capacities for all buckets."""
        if pipeline is None:
            pipeline = self._redis.pipeline()
        target_buckets = self.sorted_buckets if buckets is None else buckets

        if current_time is None:
            current_time = await async_server_time(self._redis)

        for bucket in target_buckets:
            await bucket.get_capacity(pipeline=pipeline, current_time=current_time)

        # Include max_capacity in the pipeline to avoid extra round-trips
        for bucket in target_buckets:
            pipeline.get(bucket._max_capacity_key)  # noqa: SLF001

        # Execute the pipeline to get all results
        results = await pipeline.execute()

        num_buckets = len(target_buckets)
        expected_results = num_buckets * _PIPELINE_CMDS_PER_BUCKET + num_buckets
        if len(results) != expected_results:
            raise RuntimeError(
                f"Pipeline returned {len(results)} results, expected {expected_results} "
                f"({num_buckets} buckets x {_PIPELINE_CMDS_PER_BUCKET} cmds + "
                f"{num_buckets} max_capacity GETs)"
            )

        # We're using dict instead of Usage because two different application
        # versions might use the same Redis backend that's not cleaned up
        # between deployments, and the new version might have a different
        # Usage class.
        new_capacities: dict[tuple[str, int], float] = {}
        fresh_start_buckets: list[RedisBucket] = []
        for i, bucket in enumerate(target_buckets):
            idx = i * _PIPELINE_CMDS_PER_BUCKET
            last_checked = results[idx]
            capacity = results[idx + 1]
            # max_capacity GETs come after all the per-bucket command pairs
            bucket.update_max_capacity_from_result(
                results[num_buckets * _PIPELINE_CMDS_PER_BUCKET + i]
            )
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
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
    ) -> None:
        """
        Set capacities for all buckets. Caller must hold the distributed lock.

        allow_negative=True is required for consume_capacity (speedometer)
        and refund_capacity (preserves negative debt for natural refill).
        """
        if pipeline is None:
            pipeline = self._redis.pipeline()
        target_buckets = self.sorted_buckets if buckets is None else buckets

        if current_time is None:
            current_time = await async_server_time(self._redis)

        for (usage_metric, per_seconds), amount in new_capacities.items():
            bucket = self._find_bucket(
                target_buckets,
                usage_metric,
                per_seconds,
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

    def _bucket_ids(
        self,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
    ) -> frozenset[tuple[str, int]]:
        target_buckets = self.sorted_buckets if buckets is None else buckets
        return frozenset(
            (bucket.usage_metric, int(bucket.per_seconds)) for bucket in target_buckets
        )

    def _normalize_check_result(
        self,
        result: tuple[bool, Capacities, Capacities, float | None]
        | tuple[bool, Capacities, Capacities, float | None, tuple[RedisBucket, ...]],
    ) -> tuple[bool, Capacities, Capacities, float | None, tuple[RedisBucket, ...]]:
        if len(result) == 4:  # noqa: PLR2004
            available, preconsumption, postconsumption, consumed_monotonic = result
            return (
                available,
                preconsumption,
                postconsumption,
                consumed_monotonic,
                self._snapshot_buckets(),
            )
        if len(result) == 5:  # noqa: PLR2004
            available, preconsumption, postconsumption, consumed_monotonic, buckets = (
                result
            )
            return (
                available,
                preconsumption,
                postconsumption,
                consumed_monotonic,
                tuple(buckets),
            )
        raise RuntimeError(
            "_check_and_consume_capacity() must return 4 or 5 values",
        )

    def _compute_sleep_for_wait(
        self,
        usage: FrozenUsage,
        preconsumption: Capacities,
        *,
        buckets: tuple[RedisBucket, ...],
    ) -> float:
        if buckets:
            return self._compute_sleep(usage, preconsumption, buckets=buckets)
        return self._compute_sleep(usage, preconsumption)

    async def _wait_for_task_outcome_while_cancelled(
        self,
        task: asyncio.Task[None],
    ) -> bool:
        """
        Wait for a shielded write task to settle after outer-task cancellation.

        The caller has already received ``CancelledError``.  We still need the
        write task's final outcome to decide whether capacity was actually
        recorded (and therefore must be refunded or preserved) before we
        propagate cancellation.
        """
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.done():
                    break
            except BaseException:
                break
        return task.done() and not task.cancelled() and task.exception() is None

    async def _check_and_consume_capacity(
        self,
        usage_: FrozenUsage,
        *,
        lock_blocking_timeout: float | None = None,
    ) -> tuple[bool, Capacities, Capacities, float | None, tuple[RedisBucket, ...]]:
        """Check if there's enough capacity and consume it if available."""
        usage: FrozenUsage = frozendict(
            {metric: float(amount) for metric, amount in usage_.items()},
        )
        buckets = self._snapshot_buckets()
        preconsumption_capacities: Capacities = frozendict()
        # Empty on the failure path; callers only read postconsumption on success.
        postconsumption_capacities: Capacities = frozendict()
        consumed_monotonic: float | None = None
        current_time: float = 0.0
        fresh_start_buckets: list[RedisBucket] = []
        consumed = False
        try:
            async with await self._lock(
                timeout=LOCK_TIMEOUT_SECONDS,
                blocking_timeout=lock_blocking_timeout,
                buckets=buckets,
            ):
                # Pipeline is reused: _get_capacities_unsafe executes it (clearing
                # the command buffer), then _set_capacities_unsafe adds new commands
                # and executes again.  Safe because redis-py clears the buffer on execute().
                pipeline = self._redis.pipeline()
                current_time = await async_server_time(self._redis)

                (
                    preconsumption_capacities,
                    fresh_start_buckets,
                ) = await self._get_capacities_unsafe(
                    pipeline=pipeline,
                    current_time=current_time,
                    buckets=buckets,
                )
                active_metric_names = {
                    metric for metric, _ in preconsumption_capacities
                }
                self._ensure_usage_metrics_are_active(usage, active_metric_names)

                # Fail fast: if usage exceeds any bucket's max_capacity, it can
                # never be satisfied (capacity is capped at max_capacity).
                for usage_metric_name, usage_amount in usage.items():
                    for bucket in buckets:
                        if bucket.usage_metric != usage_metric_name:
                            continue
                        # Uses cached max_capacity — refreshed by get_max_capacity() in _refresh_capacity
                        if usage_amount > bucket.max_capacity:
                            raise ValueError(
                                f"Usage value for {usage_metric_name} ({usage_amount}) "
                                f"exceeds bucket max capacity ({bucket.max_capacity})",
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
                                consumed_monotonic,
                                buckets,
                            )

                postconsumption_dict = dict(preconsumption_capacities)
                for (
                    capacity_metric_name,
                    per_seconds,
                ), capacity_amount in preconsumption_capacities.items():
                    usage_amount = usage.get(capacity_metric_name)
                    if usage_amount is None:
                        continue
                    postconsumption_dict[(capacity_metric_name, per_seconds)] = (
                        capacity_amount - usage_amount
                    )
                postconsumption_capacities = frozendict(postconsumption_dict)
                write_task = asyncio.create_task(
                    self._set_capacities_unsafe(
                        postconsumption_capacities,
                        pipeline=pipeline,
                        current_time=current_time,
                        buckets=buckets,
                    )
                )
                try:
                    await asyncio.shield(write_task)
                except asyncio.CancelledError:
                    consumed = await self._wait_for_task_outcome_while_cancelled(
                        write_task
                    )
                    raise
                consumed = True
                consumed_monotonic = time.monotonic()
            await self._fresh_start_buckets_callback(fresh_start_buckets)
            if self._callbacks and self._callbacks.on_capacity_consumed:
                await self._invoke_callback_safe(
                    self._callbacks.on_capacity_consumed,
                    model_family=self._limit_config.get_model_family(),
                    preconsumption_capacities=preconsumption_capacities,
                    postconsumption_capacities=postconsumption_capacities,
                    usage=usage,
                    current_time=current_time,
                )
        except asyncio.CancelledError:
            if consumed:
                try:  # noqa: SIM105
                    await self._refund_cancelled_consumption(usage, buckets=buckets)
                except BaseException:  # noqa: BLE001, S110
                    # Best-effort: shield ensures background completion.
                    # Swallow so CancelledError always propagates for
                    # structured concurrency (TaskGroups).
                    pass
            raise
        return (
            True,
            preconsumption_capacities,
            postconsumption_capacities,
            consumed_monotonic,
            buckets,
        )

    async def consume_capacity(self, usage: FrozenUsage) -> None:
        """
        Consume capacity unconditionally.

        Capacity may go negative by design (speedometer pattern); this tracks
        overshoot rather than blocking.
        """
        buckets = self._snapshot_buckets()
        validate_backend_usage(usage, self._validation_metric_names(buckets))
        usage = frozendict(
            {metric: float(amount) for metric, amount in usage.items()},
        )
        preconsumption_capacities: Capacities = frozendict()
        postconsumption_capacities: Capacities = frozendict()
        current_time: float = 0.0
        fresh_start_buckets: list[RedisBucket] = []
        consumed = False
        try:
            async with await self._lock(timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets):
                pipeline = self._redis.pipeline()
                current_time = await async_server_time(self._redis)

                (
                    preconsumption_capacities,
                    fresh_start_buckets,
                ) = await self._get_capacities_unsafe(
                    pipeline=pipeline,
                    current_time=current_time,
                    buckets=buckets,
                )
                active_metric_names = {
                    metric for metric, _ in preconsumption_capacities
                }
                self._ensure_usage_metrics_are_active(usage, active_metric_names)

                for usage_metric_name, usage_amount in usage.items():
                    for bucket in buckets:
                        if bucket.usage_metric != usage_metric_name:
                            continue
                        if usage_amount > bucket.max_capacity:
                            warnings.warn(
                                f"record_usage value for {usage_metric_name} ({usage_amount}) exceeds "
                                f"bucket max capacity ({bucket.max_capacity}). "
                                f"Capacity will go deeply negative.",
                                RuntimeWarning,
                                stacklevel=2,
                            )

                postconsumption_dict = dict(preconsumption_capacities)
                for (
                    capacity_metric_name,
                    per_seconds,
                ), capacity_amount in preconsumption_capacities.items():
                    usage_amount = usage.get(capacity_metric_name)
                    if usage_amount is None:
                        continue
                    postconsumption_dict[(capacity_metric_name, per_seconds)] = (
                        capacity_amount - usage_amount
                    )
                postconsumption_capacities = frozendict(postconsumption_dict)
                write_task = asyncio.create_task(
                    self._set_capacities_unsafe(
                        postconsumption_capacities,
                        pipeline=pipeline,
                        current_time=current_time,
                        allow_negative=True,
                        buckets=buckets,
                    )
                )
                try:
                    await asyncio.shield(write_task)
                except asyncio.CancelledError:
                    consumed = await self._wait_for_task_outcome_while_cancelled(
                        write_task
                    )
                    if not consumed:
                        raise
                    suppress_current_task_cancellation()
                    return
                consumed = True
            await self._fresh_start_buckets_callback(fresh_start_buckets)
            if self._callbacks and self._callbacks.on_capacity_consumed:
                await self._invoke_callback_safe(
                    self._callbacks.on_capacity_consumed,
                    model_family=self._limit_config.get_model_family(),
                    preconsumption_capacities=preconsumption_capacities,
                    postconsumption_capacities=postconsumption_capacities,
                    usage=usage,
                    current_time=current_time,
                )
        except asyncio.CancelledError:
            if not consumed:
                raise
            suppress_current_task_cancellation()
            return

    async def await_for_capacity(
        self,
        usage: FrozenUsage,
        *,
        timeout: float | None = None,
    ) -> None:
        """Wait until all buckets have the required capacity."""
        validate_backend_usage(usage, self._validation_metric_names())
        timeout = validate_timeout(timeout)
        usage = frozendict({metric: float(amount) for metric, amount in usage.items()})
        deadline = None if timeout is None else time.monotonic() + timeout
        has_waited = False
        wait_started_at: float | None = None
        wait_start_callback_overhead = 0.0
        while True:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            try:
                (
                    available,
                    preconsumption,
                    postconsumption,
                    consumed_monotonic,
                    buckets,
                ) = self._normalize_check_result(
                    await self._check_and_consume_capacity(
                        usage,
                        lock_blocking_timeout=remaining,
                    )
                )
            except redis.exceptions.LockError as exc:
                if deadline is None:  # pragma: no cover
                    raise
                raise TimeoutError("Timed out waiting for capacity") from exc
            if available:
                try:
                    if has_waited:
                        wait_time_s = max(
                            0.0,
                            (consumed_monotonic or time.monotonic())
                            - (
                                wait_started_at
                                or (consumed_monotonic or time.monotonic())
                            )
                            - wait_start_callback_overhead,
                        )
                        if (
                            self._callbacks
                            and self._callbacks.after_wait_end_consumption
                        ):
                            await self._invoke_callback_safe(
                                self._callbacks.after_wait_end_consumption,
                                model_family=self._limit_config.get_model_family(),
                                preconsumption_capacities=preconsumption,
                                postconsumption_capacities=postconsumption,
                                usage=frozendict(usage),
                                wait_time_s=wait_time_s,
                            )
                except asyncio.CancelledError:
                    try:  # noqa: SIM105
                        await self._refund_cancelled_consumption(usage, buckets=buckets)
                    except BaseException:  # noqa: BLE001, S110
                        # Best-effort: shield ensures background completion.
                        # Swallow so CancelledError always propagates for
                        # structured concurrency (TaskGroups).
                        pass
                    raise
                return

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for capacity")

            if not has_waited:
                has_waited = True
                wait_started_at = time.monotonic()
                if self._callbacks and self._callbacks.on_wait_start:
                    callback_started = time.monotonic()
                    if deadline is None:
                        await self._invoke_callback_safe(
                            self._callbacks.on_wait_start,
                            model_family=self._limit_config.get_model_family(),
                            preconsumption_capacities=preconsumption,
                            usage=usage,
                        )
                    else:
                        remaining = deadline - callback_started
                        if remaining <= 0:
                            raise TimeoutError("Timed out waiting for capacity")
                        await asyncio.wait_for(
                            self._invoke_callback_safe(
                                self._callbacks.on_wait_start,
                                model_family=self._limit_config.get_model_family(),
                                preconsumption_capacities=preconsumption,
                                usage=usage,
                            ),
                            timeout=remaining,
                        )
                    wait_start_callback_overhead += time.monotonic() - callback_started
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for capacity")

            computed = self._compute_sleep_for_wait(
                usage,
                preconsumption,
                buckets=buckets,
            )
            effective = min(computed, self.MAX_CROSS_WORKER_POLL)
            if deadline is not None:
                effective = min(effective, max(0, deadline - time.monotonic()))
            async with self._local_condition:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._local_condition.wait(),
                        timeout=max(0.001, effective),
                    )

    def _compute_sleep(
        self,
        usage: FrozenUsage,
        preconsumption: Capacities,
        *,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
    ) -> float:
        """Compute max wait across all buckets based on deficit / rate."""
        max_wait = 0.0
        target_buckets = self.sorted_buckets if buckets is None else buckets
        for (metric, per_seconds), current_cap in preconsumption.items():
            if metric not in usage:
                continue
            needed = float(usage[metric])
            deficit = needed - current_cap
            if deficit <= 0:
                continue
            bucket = self._find_bucket(
                target_buckets,
                metric,
                per_seconds,
            )
            if bucket is None:
                raise ValueError(
                    f"No bucket found for metric='{metric}', per_seconds={per_seconds}"
                )
            wait = deficit / bucket._rate_per_sec  # noqa: SLF001
            max_wait = max(max_wait, wait)
        return max_wait if max_wait > 0 else self._sleep_interval

    async def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        await self.refund_capacity_for_buckets(
            reserved_usage,
            actual_usage,
            bucket_ids=self._bucket_ids(),
        )

    async def refund_capacity_for_buckets(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        *,
        bucket_ids: set[tuple[str, int]] | frozenset[tuple[str, int]] | None = None,
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
            bucket_ids: If provided, only refund to buckets matching these
                        (metric, per_seconds) pairs. Used during reconfiguration
                        to refund only to surviving buckets.

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
        buckets = self._snapshot_buckets()
        backend_bucket_ids = self._bucket_ids(buckets)
        refund_bucket_ids = (
            backend_bucket_ids if bucket_ids is None else frozenset(bucket_ids)
        )
        validate_backend_refund_usage_for_bucket_ids(
            reserved_usage,
            actual_usage,
            refund_bucket_ids,
            backend_bucket_ids,
        )
        if not refund_bucket_ids:
            return
        # Calculate how much to refund for each metric
        refund_usage_: dict[str, float] = {}
        for metric, reserved_amount in reserved_usage.items():
            # Key guaranteed to exist: RateLimiter.refund_capacity() calls
            # validate_refund_usage() before reaching the backend.
            actual_amount = actual_usage[metric]
            refund_amount = float(reserved_amount) - float(actual_amount)

            # Check for overuse and log a warning
            if refund_amount < 0:
                warnings.warn(
                    f"Actual usage ({actual_amount}) for {metric} exceeds "
                    f"reserved usage ({reserved_amount}). Applying negative refund.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Include both positive and negative refunds
            refund_usage_[metric] = refund_amount
        refund_usage: frozendict[str, float] = frozendict(refund_usage_)

        fresh_start_buckets: list[RedisBucket] = []
        async with await self._lock(timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets):
            pipeline = self._redis.pipeline()
            current_time = await async_server_time(self._redis)

            # Get current capacities (which already account for time-based refill)
            (
                prerefund_capacities,
                fresh_start_buckets,
            ) = await self._get_capacities_unsafe(
                pipeline=pipeline,
                current_time=current_time,
                buckets=buckets,
            )

            # Apply refund amounts to current capacity
            updated_capacities_: dict[tuple[str, int], float] = dict(
                prerefund_capacities,
            )
            for (
                capability_usage_metric,
                per_seconds,
            ) in prerefund_capacities:
                bucket_id = (capability_usage_metric, int(per_seconds))
                if bucket_id not in refund_bucket_ids:
                    continue
                refund_amount = refund_usage.get(capability_usage_metric)
                if refund_amount is None:
                    continue
                bucket = self._find_bucket(
                    buckets,
                    capability_usage_metric,
                    per_seconds,
                )
                if bucket is None:
                    raise ValueError(
                        f"Bucket '{capability_usage_metric}/{per_seconds}s' not found",
                    )

                # Apply refund (positive or negative), cap at max_capacity.
                # Negative capacity is preserved so the token-bucket refill
                # handles recovery — clamping to 0 here would erase debt
                # from the record_usage (speedometer) path.
                updated_capacities_[(capability_usage_metric, int(per_seconds))] = min(
                    updated_capacities_[(capability_usage_metric, int(per_seconds))]
                    + refund_amount,
                    bucket.max_capacity,  # cached — refreshed by get_max_capacity() in _refresh_capacity
                )
            updated_capacities = frozendict(updated_capacities_)

            # Always update capacities in Redis with the current time
            await self._set_capacities_unsafe(
                frozendict(updated_capacities),
                pipeline=pipeline,
                current_time=current_time,
                allow_negative=True,
                buckets=buckets,
            )
        async with self._local_condition:
            self._local_condition.notify_all()
        await self._fresh_start_buckets_callback(fresh_start_buckets)
        if self._callbacks and self._callbacks.on_capacity_refunded:
            await self._invoke_callback_safe(
                self._callbacks.on_capacity_refunded,
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
        buckets = self._snapshot_buckets()
        bucket = self._find_bucket(
            buckets,
            metric,
            per_seconds,
        )
        if bucket is None:
            raise ValueError(f"Bucket '{metric}/{per_seconds}s' not found")
        async with await self._lock(timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets):
            await bucket.set_max_capacity(value)
        async with self._local_condition:
            self._local_condition.notify_all()

    async def prepare_reconfigured_backend(
        self,
        new_backend: RateLimiterBackend,
        cfg: PerModelConfig,
    ) -> RateLimiterBackend:
        if not isinstance(new_backend, RedisBackend):
            raise TypeError(
                "RedisBackend can only reconfigure into another RedisBackend"
            )

        current_buckets = self._snapshot_buckets()
        new_buckets = tuple(new_backend.sorted_buckets)
        reconfigure_buckets = self._combined_bucket_snapshot(
            current_buckets, new_buckets
        )

        async with await self._lock(
            timeout=LOCK_TIMEOUT_SECONDS,
            buckets=reconfigure_buckets,
        ):
            for quota in cfg.quotas:
                bucket = self._find_bucket(
                    new_buckets,
                    quota.metric,
                    int(quota.per_seconds),
                )
                if bucket is None:  # pragma: no cover
                    raise ValueError(
                        f"Bucket '{quota.metric}/{quota.per_seconds}s' not found",
                    )
                await bucket.set_max_capacity(float(quota.limit))

            self.install_reconfigured_state(
                buckets=list(new_buckets),
                cfg=cfg,
            )
        async with self._local_condition:
            self._local_condition.notify_all()
        return self

    def install_reconfigured_state(
        self,
        *,
        buckets: list[RedisBucket],
        cfg: PerModelConfig,
    ) -> None:
        self.sorted_buckets = sorted(buckets, key=lambda bucket: bucket.full_redis_key)
        self._usage_metric_names = {bucket.usage_metric for bucket in buckets}
        self._limit_config = cfg

    async def _invoke_callback_safe(self, callback, **kwargs) -> None:
        """Fire a user callback, suppressing exceptions to prevent capacity leaks."""
        try:
            await callback(**kwargs)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Rate limiter callback raised {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )

    async def _refund_cancelled_consumption(
        self,
        usage: FrozenUsage,
        *,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
    ) -> None:
        """
        Refund capacity consumed before a CancelledError hit callbacks.

        Uses asyncio.shield() because the refund involves multiple Redis I/O
        await points (lock acquisition, pipeline get, pipeline set).  Shield
        ensures the refund completes even if the task is re-cancelled.
        Fires no callbacks to avoid recursion and another cancellation window.
        """
        target_buckets = self._snapshot_buckets() if buckets is None else tuple(buckets)

        async def _do_refund() -> None:
            async with await self._lock(
                timeout=LOCK_TIMEOUT_SECONDS,
                buckets=target_buckets,
            ):
                pipeline = self._redis.pipeline()
                current_time = await async_server_time(self._redis)
                capacities, _ = await self._get_capacities_unsafe(
                    pipeline=pipeline,
                    current_time=current_time,
                    buckets=target_buckets,
                )
                refunded: dict[tuple[str, int], float] = dict(capacities)
                for (cap_metric, per_seconds), cap_amount in capacities.items():
                    for usage_metric, usage_amount in usage.items():
                        if cap_metric != usage_metric:
                            continue
                        bucket = self._find_bucket(
                            target_buckets,
                            cap_metric,
                            per_seconds,
                        )
                        if bucket is None:  # pragma: no cover
                            raise ValueError(
                                f"Bucket '{cap_metric}/{per_seconds}s' not found",
                            )
                        refunded[(cap_metric, per_seconds)] = min(
                            cap_amount + usage_amount,
                            bucket.max_capacity,
                        )
                await self._set_capacities_unsafe(
                    frozendict(refunded),
                    pipeline=pipeline,
                    current_time=current_time,
                    allow_negative=True,
                    buckets=target_buckets,
                )
            async with self._local_condition:
                self._local_condition.notify_all()

        await asyncio.shield(_do_refund())

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
                await self._invoke_callback_safe(
                    self._callbacks.on_missing_consumption_data,
                    model_family=self._limit_config.get_model_family(),
                    usage_metric=bucket.usage_metric,
                    per_seconds=bucket.per_seconds,
                )
