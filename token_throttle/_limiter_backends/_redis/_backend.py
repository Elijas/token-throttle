import asyncio
import contextlib
import math
import time
import typing
import uuid
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

from token_throttle._interfaces._callable_utils import (
    suppress_current_task_cancellation,
)
from token_throttle._interfaces._callbacks import RateLimiterCallbacks
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


async def _shielded_lock_release(lock: redis.asyncio.lock.Lock) -> None:
    """Release a Redis lock, shielded so the round-trip finishes even under cancel."""
    await asyncio.shield(lock.release())


class _RedisLockStack(AsyncExitStack):
    """
    AsyncExitStack that also holds the acquired Lock objects.

    The parent ``_lock()`` needs to expose the concrete lock instances so
    callers can call ``extend()`` on them right before long-running writes,
    keeping the lock alive across GC pauses or connection stalls that
    otherwise could let the lock's Redis TTL expire mid-operation and
    allow two workers to both commit writes (lost-update race).
    """

    def __init__(self) -> None:
        super().__init__()
        self.locks: list[redis.asyncio.lock.Lock] = []


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
    # Cross-worker poll ceiling; intra-process wakeup is instant via _local_condition. Audited 2026-04.
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
    ) -> _RedisLockStack:
        """
        Acquire distributed Redis locks for a fixed bucket snapshot.

        Returns a ``_RedisLockStack`` that holds the acquired locks.  Use as::

            async with await self._lock(timeout=LOCK_TIMEOUT_SECONDS) as stack:
                ...  # all bucket locks held here
                await self._extend_locks(stack)  # before long writes

        If acquiring lock N fails, locks 0..N-1 are released immediately
        (via ``stack.aclose()``) so we never leak partially-acquired locks.
        """
        stack = _RedisLockStack()
        target_buckets = self.sorted_buckets if buckets is None else buckets
        loop = asyncio.get_running_loop()
        stop_trying_at = (
            None if blocking_timeout is None else loop.time() + blocking_timeout
        )

        try:
            for bucket in target_buckets:
                if stack.locks:
                    await self._extend_locks(stack)
                remaining = (
                    None
                    if stop_trying_at is None
                    else max(0.0, stop_trying_at - loop.time())
                )
                lock = bucket.lock(timeout=timeout)
                # Generate the token ourselves and pass it to acquire().
                # This lets us run LUA_RELEASE directly with a known
                # token if cancel arrives mid-acquire: redis-py's
                # lock.release() first checks self.local.token, which is
                # only assigned AFTER the SET NX succeeds. A cancel in
                # that narrow window leaves the Redis key set but
                # local.token=None, so release() raises LockError before
                # ever talking to Redis and the lock leaks for its TTL.
                token = uuid.uuid4().hex.encode()
                try:
                    acquired = await lock.acquire(
                        blocking_timeout=remaining, token=token
                    )
                except BaseException:
                    # Best-effort compensating release with our known
                    # token. The LUA script is CAS: it only deletes if
                    # stored value matches our token, so even if another
                    # acquirer has since taken the lock we cannot release
                    # theirs. asyncio.shield guards against re-cancel
                    # during the release round-trip.
                    with contextlib.suppress(Exception):
                        await asyncio.shield(
                            lock.lua_release(
                                keys=[lock.name],
                                args=[token],
                                client=self._redis,
                            )
                        )
                    raise
                if not acquired:
                    _raise_lock_timeout_error()
                stack.locks.append(lock)
                stack.push_async_callback(_shielded_lock_release, lock)
        except BaseException:
            await stack.aclose()
            raise

        return stack

    @staticmethod
    async def _extend_locks(stack: _RedisLockStack) -> None:
        """
        Reset each held lock's TTL to its configured timeout (LOCK_TIMEOUT_SECONDS).

        Call this immediately before a long-running write (pipeline exec,
        Lua script) so a GC pause / network stall earlier in the critical
        section cannot let the lock's Redis TTL expire mid-operation,
        which would let a second worker acquire the same lock and race
        on the same write.

        If the lock already expired (or was stolen by another worker),
        ``LockNotOwnedError`` surfaces and we re-raise as ``LockError``
        so the caller aborts the write rather than committing on top of
        another worker's state. The caller's surrounding retry loop
        (await_for_capacity, etc.) will observe the error and retry
        cleanly.
        """
        for lock in stack.locks:
            try:
                await lock.reacquire()
            except redis.exceptions.LockNotOwnedError as exc:
                raise redis.exceptions.LockError(
                    "Lock expired or was stolen mid-operation; aborting write."
                ) from exc

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
            last_checked = results[idx + RedisBucket.PIPELINE_LAST_CHECKED_OFFSET]
            capacity = results[idx + RedisBucket.PIPELINE_CAPACITY_OFFSET]
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

        Implementation note — when the outer task is cancelling, each
        ``await asyncio.shield(task)`` re-raises ``CancelledError`` rather than
        blocking to completion. The loop iterates (yielding to the event loop
        each time, so the shielded task makes progress) until ``task.done()``
        becomes True. Total wait is therefore bounded by the inner task's own
        duration — a single Redis pipeline write — not by the loop.
        """
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.done():
                    break
            except Exception:  # noqa: BLE001
                # Task raised a non-cancellation exception (e.g. Redis
                # connection error). No Redis write succeeded, so reporting
                # consumed=False below is correct and the caller will skip
                # refund. SystemExit/KeyboardInterrupt are intentionally *not*
                # caught here: swallowing them could let the shielded write
                # complete in the background while the caller returns
                # consumed=False, producing phantom consumption.
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
            ) as lock_stack:
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
                        # Uses cached max_capacity — refreshed from the pipeline
                        # result by update_max_capacity_from_result() inside
                        # _get_capacities_unsafe just above.
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
                # Extend lock TTL immediately before the write so a GC
                # pause during get/check above cannot leave the write
                # racing with another worker after the original TTL
                # lapsed.
                await self._extend_locks(lock_stack)
                # `allow_negative=False` is correct here only because the
                # `usage_amount > capacity_amount` gate above guarantees
                # post = pre - usage >= 0. Keep this branch in sync with that
                # gate; consume_capacity (speedometer) writes with
                # `allow_negative=True` because it has no such guarantee.
                write_task = asyncio.create_task(
                    self._set_capacities_unsafe(
                        postconsumption_capacities,
                        pipeline=pipeline,
                        current_time=current_time,
                        allow_negative=False,
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
        async with await self._lock(
            timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets
        ) as lock_stack:
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
            active_metric_names = {metric for metric, _ in preconsumption_capacities}
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

            max_cap = {(b.usage_metric, b.per_seconds): b.max_capacity for b in buckets}
            postconsumption_dict = dict(preconsumption_capacities)
            for (
                capacity_metric_name,
                per_seconds,
            ), capacity_amount in preconsumption_capacities.items():
                usage_amount = usage.get(capacity_metric_name)
                if usage_amount is None:
                    continue
                postconsumption_dict[(capacity_metric_name, per_seconds)] = max(
                    capacity_amount - usage_amount,
                    -max_cap[(capacity_metric_name, per_seconds)],
                )
            postconsumption_capacities = frozendict(postconsumption_dict)
            # Extend lock TTL immediately before the write so a GC pause
            # during get above cannot leave the write racing with another
            # worker after the original TTL lapsed.
            await self._extend_locks(lock_stack)
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
                consumed = await self._wait_for_task_outcome_while_cancelled(write_task)
                if not consumed:
                    raise
                # The shielded Redis write actually landed, so the
                # speedometer reading is already correct. See
                # `suppress_current_task_cancellation` docstring —
                # refunding now would roll back a recorded
                # measurement of real usage.
                #
                # Lock release during the `async with` exit is shielded
                # (via _shielded_lock_release in _lock()), so a re-cancel
                # between suppression and release will not leak the lock.
                suppress_current_task_cancellation()
                return
        # Callbacks fire after the lock is released. Consumption is already
        # durably recorded in Redis, so if CancelledError arrives during
        # callbacks we let it propagate: the caller (e.g. asyncio.timeout)
        # must be informed of the cancel, and Redis state is already
        # correct (speedometer is advanced). Callbacks are best-effort via
        # _invoke_callback_safe.
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
                # When deadline is None (no timeout), LockError propagates
                # raw — this is intentional: an unbounded wait should not
                # silently convert a lock failure into a timeout error.
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
        async with await self._lock(
            timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets
        ) as lock_stack:
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
                refund_amount = max(refund_amount, -bucket.max_capacity)
                updated_capacities_[(capability_usage_metric, int(per_seconds))] = min(
                    updated_capacities_[(capability_usage_metric, int(per_seconds))]
                    + refund_amount,
                    bucket.max_capacity,  # cached — refreshed from pipeline result in _get_capacities_unsafe
                )
            updated_capacities = frozendict(updated_capacities_)

            # Extend lock TTL before committing the write, see _extend_locks.
            await self._extend_locks(lock_stack)
            write_task = asyncio.create_task(
                self._set_capacities_unsafe(
                    frozendict(updated_capacities),
                    pipeline=pipeline,
                    current_time=current_time,
                    allow_negative=True,
                    buckets=buckets,
                )
            )
            try:
                await asyncio.shield(write_task)
            except asyncio.CancelledError:
                refunded = await self._wait_for_task_outcome_while_cancelled(write_task)
                if not refunded:
                    raise
                # Write landed despite cancel — refund is done.
                # Suppress so the caller doesn't retry and double-refund.
                suppress_current_task_cancellation()
                async with self._local_condition:
                    self._local_condition.notify_all()
                return
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

    async def _snapshot_bucket_state(self, bucket: RedisBucket) -> None:
        """
        Freeze ``bucket`` in Redis at its accrued capacity under the CURRENT rate.

        Call before any rate-changing mutation (set_max_capacity,
        clear_max_capacity_override, set_configured_max_capacity when no
        override is present). Caller MUST hold the bucket's distributed lock.

        The anchor is the *uncapped* old-rate integration so that raw stored
        values above ``max_capacity`` are preserved: reads apply
        ``min(max_capacity, …)`` and a later cap raise can re-expose the
        hidden overflow.
        """
        # Refresh the override cache so ``bucket._rate_per_sec`` reflects the
        # current effective rate in Redis before we snapshot under it.
        await bucket.get_max_capacity()
        current_time = await async_server_time(self._redis)
        pipeline = self._redis.pipeline()
        pipeline.get(bucket._last_checked_key)  # noqa: SLF001
        pipeline.get(bucket._capacity_key)  # noqa: SLF001
        last_checked_raw, stored_raw = await pipeline.execute()
        if last_checked_raw is None or stored_raw is None:
            # Partial state (one None) is treated the same as full absence:
            # the bucket will start fresh on the next acquire. This is the
            # correct fallback — anchoring with incomplete data would produce
            # a wrong capacity value.
            return
        try:
            last_checked = float(last_checked_raw)
            stored = float(stored_raw)
        except (TypeError, ValueError):
            return  # unparseable state — leave as-is; a later write will overwrite.
        if not (math.isfinite(last_checked) and math.isfinite(stored)):
            return
        time_passed = current_time - last_checked
        if time_passed < 0:
            time_passed = 0.0  # clock skew — same clamp as calculate_capacity.
        anchored = stored + time_passed * bucket._rate_per_sec  # noqa: SLF001
        await bucket.set_capacity(
            anchored,
            current_time=current_time,
            allow_negative=True,
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
        async with await self._lock(
            timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets
        ) as lock_stack:
            await self._extend_locks(lock_stack)
            await self._snapshot_bucket_state(bucket)
            await bucket.set_max_capacity(value)
        async with self._local_condition:
            self._local_condition.notify_all()

    async def apply_configured_max_capacity(
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
        async with await self._lock(
            timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets
        ) as lock_stack:
            await self._extend_locks(lock_stack)
            await self._snapshot_bucket_state(bucket)
            await bucket.clear_max_capacity_override()
            bucket.set_configured_max_capacity(value)
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
        removed_buckets = tuple(
            bucket
            for bucket in current_buckets
            if self._find_bucket(
                new_buckets,
                bucket.usage_metric,
                int(bucket.per_seconds),
            )
            is None
        )

        async with await self._lock(
            timeout=LOCK_TIMEOUT_SECONDS,
            buckets=reconfigure_buckets,
        ) as lock_stack:
            for bucket in removed_buckets:
                # Snapshot first so the frozen capacity/last_checked in Redis
                # reflect the bucket's state at the moment of removal. Without
                # this, a later re-add with a different rate would retroactively
                # integrate the new rate across the remove→re-add gap (and any
                # time preceding it). Then clear the runtime override so the
                # re-add starts from the callable config's static quota again.
                await self._extend_locks(lock_stack)
                await self._snapshot_bucket_state(bucket)
                await bucket.clear_max_capacity_override()
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
                current_bucket = self._find_bucket(
                    current_buckets,
                    quota.metric,
                    int(quota.per_seconds),
                )
                # Snapshot the bucket under its current effective rate before
                # any mutation could change it. Prefer current_bucket (its
                # _max_capacity_default matches the stored override payload,
                # so the override is accepted and yields the true active rate).
                await self._extend_locks(lock_stack)
                await self._snapshot_bucket_state(
                    current_bucket if current_bucket is not None else bucket
                )
                if current_bucket is not None and float(
                    current_bucket.configured_max_capacity
                ) != float(quota.limit):
                    await bucket.clear_max_capacity_override()
                bucket.set_configured_max_capacity(float(quota.limit))

            await self._extend_locks(lock_stack)
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
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:  # noqa: BLE001
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

        Worst case: if the prior lock release was interrupted (cancel arrived
        before the shielded release completed), the lock may still be held in
        Redis.  Re-acquiring here then blocks for up to LOCK_TIMEOUT_SECONDS
        (30 s) until the TTL expires.  This is inherent to the architecture —
        the lock will eventually expire and the refund will proceed.
        """
        target_buckets = self._snapshot_buckets() if buckets is None else tuple(buckets)

        async def _do_refund() -> None:
            async with await self._lock(
                timeout=LOCK_TIMEOUT_SECONDS,
                buckets=target_buckets,
            ) as lock_stack:
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
                await self._extend_locks(lock_stack)
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
