import contextlib
import math
import threading
import time
import typing
import uuid
import warnings
from contextlib import ExitStack
from typing import ClassVar

try:
    import redis
    import redis.client
    import redis.exceptions
except ImportError as exc:
    raise ImportError(
        'The "redis" package is required for the Redis backend. '
        'Install it with: pip install "token-throttle[redis]"'
    ) from exc
from frozendict import frozendict

from token_throttle._interfaces._callbacks import SyncRateLimiterCallbacks
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    SyncRateLimiterBackend,
    SyncRateLimiterBackendBuilderInterface,
)
from token_throttle._interfaces._models import Capacities, FrozenUsage
from token_throttle._validation import (
    validate_backend_refund_usage_for_bucket_ids,
    validate_backend_usage,
    validate_timeout,
)

from ._server_time import sync_server_time
from ._sync_bucket import SyncRedisBucket


class SyncCapacitiesGetterResult(typing.NamedTuple):
    capacities: Capacities
    fresh_start_buckets: list[SyncRedisBucket]


LOCK_TIMEOUT_SECONDS = 30

# Each bucket enqueues exactly 2 pipeline commands in get_capacity()
# (GET last_checked, GET capacity).  Used to index pipeline results.
_PIPELINE_CMDS_PER_BUCKET = 2


def _raise_lock_timeout_error() -> typing.NoReturn:
    raise redis.exceptions.LockError("Unable to acquire lock within the time specified")


class _SyncRedisLockStack(ExitStack):
    """
    ExitStack that also holds the acquired Lock objects so callers can
    extend their TTLs before long-running writes (see _extend_locks).
    """

    def __init__(self) -> None:
        super().__init__()
        self.locks: list[redis.lock.Lock] = []


class SyncRedisBackendBuilder(SyncRateLimiterBackendBuilderInterface):
    def __init__(
        self,
        redis_client: redis.Redis,
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
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> "SyncRedisBackend":
        redis_buckets = []
        for quota in cfg.quotas:
            b = SyncRedisBucket(
                quota=quota,
                limit_config=cfg,
                redis_client=self._redis,
            )
            redis_buckets.append(b)
        return SyncRedisBackend(
            buckets=redis_buckets,
            redis=self._redis,
            sleep_interval=self._sleep_interval,
            callbacks=callbacks,
            limit_config=cfg,
        )


class SyncRedisBackend(SyncRateLimiterBackend):
    DEFAULT_SLEEP_INTERVAL: ClassVar[float] = 0.1
    MAX_CROSS_WORKER_POLL: ClassVar[float] = 1.0

    def __init__(
        self,
        buckets: list[SyncRedisBucket],
        redis: redis.Redis,
        limit_config: PerModelConfig,
        *,
        sleep_interval: float | None = None,
        callbacks: SyncRateLimiterCallbacks | None = None,
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
        self._local_condition = threading.Condition()

    def supports_metric_set_change(self) -> bool:
        # Surviving bucket state lives in Redis under stable keys, so a rebuilt
        # backend can safely point at the same buckets.
        return True

    def _snapshot_buckets(self) -> tuple[SyncRedisBucket, ...]:
        return tuple(self.sorted_buckets)

    def _validation_metric_names(
        self,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
    ) -> set[str]:
        target_buckets = self._snapshot_buckets() if buckets is None else tuple(buckets)
        if target_buckets:
            return self._usage_metric_names_for(target_buckets)
        return set(self._usage_metric_names)

    @staticmethod
    def _usage_metric_names_for(
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket],
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
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket],
        metric: str,
        per_seconds: int,
    ) -> SyncRedisBucket | None:
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
        *bucket_groups: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket],
    ) -> tuple[SyncRedisBucket, ...]:
        buckets_by_key: dict[str, SyncRedisBucket] = {}
        for bucket_group in bucket_groups:
            for bucket in bucket_group:
                buckets_by_key[bucket.full_redis_key] = bucket
        return tuple(
            sorted(buckets_by_key.values(), key=lambda bucket: bucket.full_redis_key)
        )

    def _lock(
        self,
        *,
        timeout: float,
        blocking_timeout: float | None = None,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
    ) -> _SyncRedisLockStack:
        """Acquire locks for a fixed bucket snapshot in a consistent order."""
        stack = _SyncRedisLockStack()
        target_buckets = self.sorted_buckets if buckets is None else buckets
        stop_trying_at = (
            None if blocking_timeout is None else time.monotonic() + blocking_timeout
        )

        try:
            for bucket in target_buckets:
                if stack.locks:
                    self._extend_locks(stack)
                remaining = (
                    None
                    if stop_trying_at is None
                    else max(0.0, stop_trying_at - time.monotonic())
                )
                lock = bucket.lock(timeout=timeout)
                # Generate the token ourselves so a best-effort CAS
                # release after a KeyboardInterrupt-style cancel can run
                # without relying on lock.local.token (which is only
                # populated AFTER the SET NX succeeds; a cancel in that
                # window would otherwise orphan the lock for its TTL).
                token = uuid.uuid4().hex.encode()
                try:
                    acquired = lock.acquire(blocking_timeout=remaining, token=token)
                except BaseException:
                    # Best-effort CAS release; only deletes the key if
                    # its value still matches our token.
                    with contextlib.suppress(Exception):
                        lock.lua_release(
                            keys=[lock.name], args=[token], client=self._redis
                        )
                    raise
                if not acquired:
                    _raise_lock_timeout_error()
                stack.locks.append(lock)
                stack.callback(lock.release)
        except BaseException:
            stack.close()
            raise

        return stack

    @staticmethod
    def _extend_locks(stack: _SyncRedisLockStack) -> None:
        """
        Reset each held lock's TTL to LOCK_TIMEOUT_SECONDS; see the
        async backend's ``_extend_locks`` for the rationale.
        """
        for lock in stack.locks:
            try:
                lock.reacquire()
            except redis.exceptions.LockNotOwnedError as exc:
                raise redis.exceptions.LockError(
                    "Lock expired or was stolen mid-operation; aborting write."
                ) from exc

    def _get_capacities_unsafe(
        self,
        pipeline: redis.client.Pipeline | None = None,
        current_time: float | None = None,
        *,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
    ) -> SyncCapacitiesGetterResult:
        """Get capacities for all buckets."""
        if pipeline is None:
            pipeline = self._redis.pipeline()
        target_buckets = self.sorted_buckets if buckets is None else buckets

        if current_time is None:
            current_time = sync_server_time(self._redis)

        for bucket in target_buckets:
            bucket.get_capacity(pipeline=pipeline, current_time=current_time)

        # Include max_capacity in the pipeline to avoid extra round-trips
        for bucket in target_buckets:
            pipeline.get(bucket._max_capacity_key)  # noqa: SLF001

        # Execute the pipeline to get all results
        results = pipeline.execute()

        num_buckets = len(target_buckets)
        expected_results = num_buckets * _PIPELINE_CMDS_PER_BUCKET + num_buckets
        if len(results) != expected_results:
            raise RuntimeError(
                f"Pipeline returned {len(results)} results, expected {expected_results} "
                f"({num_buckets} buckets x {_PIPELINE_CMDS_PER_BUCKET} cmds + "
                f"{num_buckets} max_capacity GETs)"
            )

        new_capacities: dict[tuple[str, int], float] = {}
        fresh_start_buckets: list[SyncRedisBucket] = []
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

        return SyncCapacitiesGetterResult(
            capacities=frozendict(new_capacities),
            fresh_start_buckets=fresh_start_buckets,
        )

    def _set_capacities_unsafe(
        self,
        new_capacities: Capacities,
        pipeline: redis.client.Pipeline | None = None,
        current_time: float | None = None,
        *,
        allow_negative: bool = False,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
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
            current_time = sync_server_time(self._redis)

        for (usage_metric, per_seconds), amount in new_capacities.items():
            bucket = self._find_bucket(
                target_buckets,
                usage_metric,
                per_seconds,
            )
            if bucket is None:
                raise ValueError(f"Bucket '{usage_metric}/{per_seconds}s' not found")
            bucket.set_capacity(
                amount,
                pipeline=pipeline,
                current_time=current_time,
                execute=False,
                allow_negative=allow_negative,
            )
        pipeline.execute()

    def _bucket_ids(
        self,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
    ) -> frozenset[tuple[str, int]]:
        target_buckets = self.sorted_buckets if buckets is None else buckets
        return frozenset(
            (bucket.usage_metric, int(bucket.per_seconds)) for bucket in target_buckets
        )

    def _normalize_check_result(
        self,
        result: tuple[bool, Capacities, Capacities, float | None]
        | tuple[
            bool,
            Capacities,
            Capacities,
            float | None,
            tuple[SyncRedisBucket, ...],
        ],
    ) -> tuple[
        bool,
        Capacities,
        Capacities,
        float | None,
        tuple[SyncRedisBucket, ...],
    ]:
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
        buckets: tuple[SyncRedisBucket, ...],
    ) -> float:
        if buckets:
            return self._compute_sleep(usage, preconsumption, buckets=buckets)
        return self._compute_sleep(usage, preconsumption)

    def _check_and_consume_capacity(
        self,
        usage_: FrozenUsage,
        *,
        lock_blocking_timeout: float | None = None,
    ) -> tuple[bool, Capacities, Capacities, float | None, tuple[SyncRedisBucket, ...]]:
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
        fresh_start_buckets: list[SyncRedisBucket] = []
        consumed = False
        try:
            with self._lock(
                timeout=LOCK_TIMEOUT_SECONDS,
                blocking_timeout=lock_blocking_timeout,
                buckets=buckets,
            ) as lock_stack:
                # Pipeline is reused: _get_capacities_unsafe executes it (clearing
                # the command buffer), then _set_capacities_unsafe adds new commands
                # and executes again.  Safe because redis-py clears the buffer on execute().
                pipeline = self._redis.pipeline()
                current_time = sync_server_time(self._redis)

                preconsumption_capacities, fresh_start_buckets = (
                    self._get_capacities_unsafe(
                        pipeline=pipeline,
                        current_time=current_time,
                        buckets=buckets,
                    )
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
                            raise ValueError(  # noqa: TRY301
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
                consumed_monotonic = time.monotonic()
                # Extend lock TTL before write to guard against GC-pause
                # induced lost-update races; see _extend_locks.
                self._extend_locks(lock_stack)
                # `allow_negative=False` is correct here only because the
                # `usage_amount > capacity_amount` gate above guarantees
                # post = pre - usage >= 0. Keep this branch in sync with that
                # gate; consume_capacity (speedometer) writes with
                # `allow_negative=True` because it has no such guarantee.
                self._set_capacities_unsafe(
                    postconsumption_capacities,
                    pipeline=pipeline,
                    current_time=current_time,
                    allow_negative=False,
                    buckets=buckets,
                )
                consumed = True
            self._fresh_start_buckets_callback(fresh_start_buckets)
            if self._callbacks and self._callbacks.on_capacity_consumed:
                self._invoke_callback_safe(
                    self._callbacks.on_capacity_consumed,
                    model_family=self._limit_config.get_model_family(),
                    preconsumption_capacities=preconsumption_capacities,
                    postconsumption_capacities=postconsumption_capacities,
                    usage=usage,
                    current_time=current_time,
                )
        except BaseException:
            if consumed:
                try:  # noqa: SIM105
                    self._refund_cancelled_consumption(usage, buckets=buckets)
                except BaseException:  # noqa: BLE001, S110
                    pass
            raise
        return (
            True,
            preconsumption_capacities,
            postconsumption_capacities,
            consumed_monotonic,
            buckets,
        )

    def consume_capacity(self, usage: FrozenUsage) -> None:
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
        fresh_start_buckets: list[SyncRedisBucket] = []
        with self._lock(timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets) as lock_stack:
            pipeline = self._redis.pipeline()
            current_time = sync_server_time(self._redis)

            preconsumption_capacities, fresh_start_buckets = (
                self._get_capacities_unsafe(
                    pipeline=pipeline,
                    current_time=current_time,
                    buckets=buckets,
                )
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
            # Extend lock TTL before write; see _extend_locks.
            self._extend_locks(lock_stack)
            self._set_capacities_unsafe(
                postconsumption_capacities,
                pipeline=pipeline,
                current_time=current_time,
                allow_negative=True,
                buckets=buckets,
            )
        self._fresh_start_buckets_callback(fresh_start_buckets)
        if self._callbacks and self._callbacks.on_capacity_consumed:
            self._invoke_callback_safe(
                self._callbacks.on_capacity_consumed,
                model_family=self._limit_config.get_model_family(),
                preconsumption_capacities=preconsumption_capacities,
                postconsumption_capacities=postconsumption_capacities,
                usage=usage,
                current_time=current_time,
            )

    def wait_for_capacity(
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
                    self._check_and_consume_capacity(
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
                            self._invoke_callback_safe(
                                self._callbacks.after_wait_end_consumption,
                                model_family=self._limit_config.get_model_family(),
                                preconsumption_capacities=preconsumption,
                                postconsumption_capacities=postconsumption,
                                usage=frozendict(usage),
                                wait_time_s=wait_time_s,
                            )
                except BaseException:
                    try:  # noqa: SIM105
                        self._refund_cancelled_consumption(usage, buckets=buckets)
                    except BaseException:  # noqa: BLE001, S110
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
                    if deadline is not None and callback_started >= deadline:
                        raise TimeoutError("Timed out waiting for capacity")
                    self._invoke_callback_safe(
                        self._callbacks.on_wait_start,
                        model_family=self._limit_config.get_model_family(),
                        preconsumption_capacities=preconsumption,
                        usage=usage,
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
            with self._local_condition:
                self._local_condition.wait(timeout=max(0.001, effective))

    def _compute_sleep(
        self,
        usage: FrozenUsage,
        preconsumption: Capacities,
        *,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
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

    def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        self.refund_capacity_for_buckets(
            reserved_usage,
            actual_usage,
            bucket_ids=self._bucket_ids(),
        )

    def refund_capacity_for_buckets(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        *,
        bucket_ids: set[tuple[str, int]] | frozenset[tuple[str, int]] | None = None,
    ) -> None:
        """Refund unused capacity back to the rate limiter based on actual usage."""
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
            # Key guaranteed to exist: SyncRateLimiter.refund_capacity() calls
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

        fresh_start_buckets: list[SyncRedisBucket] = []
        with self._lock(timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets) as lock_stack:
            pipeline = self._redis.pipeline()
            current_time = sync_server_time(self._redis)

            # Get current capacities (which already account for time-based refill)
            prerefund_capacities, fresh_start_buckets = self._get_capacities_unsafe(
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

            # Extend lock TTL before write; see _extend_locks.
            self._extend_locks(lock_stack)
            # Always update capacities in Redis with the current time
            self._set_capacities_unsafe(
                frozendict(updated_capacities),
                pipeline=pipeline,
                current_time=current_time,
                allow_negative=True,
                buckets=buckets,
            )
        with self._local_condition:
            self._local_condition.notify_all()
        self._fresh_start_buckets_callback(fresh_start_buckets)
        if self._callbacks and self._callbacks.on_capacity_refunded:
            self._invoke_callback_safe(
                self._callbacks.on_capacity_refunded,
                model_family=self._limit_config.get_model_family(),
                reserved_usage=reserved_usage,
                actual_usage=actual_usage,
                refunded_usage=refund_usage,
                prerefund_capacities=prerefund_capacities,
                postrefund_capacities=updated_capacities,
            )

    def _snapshot_bucket_state(self, bucket: SyncRedisBucket) -> None:
        """
        Freeze ``bucket`` in Redis at its accrued capacity under the CURRENT rate.

        Uncapped anchor (stored + elapsed*old_rate) preserves raw values above
        ``max_capacity`` — reads apply ``min(max_capacity, …)`` and a later
        cap raise can re-expose the hidden overflow.
        """
        bucket.get_max_capacity()
        current_time = sync_server_time(self._redis)
        pipeline = self._redis.pipeline()
        pipeline.get(bucket._last_checked_key)  # noqa: SLF001
        pipeline.get(bucket._capacity_key)  # noqa: SLF001
        last_checked_raw, stored_raw = pipeline.execute()
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
            return
        if not (math.isfinite(last_checked) and math.isfinite(stored)):
            return
        time_passed = current_time - last_checked
        if time_passed < 0:
            time_passed = 0.0
        anchored = stored + time_passed * bucket._rate_per_sec  # noqa: SLF001
        bucket.set_capacity(
            anchored,
            current_time=current_time,
            allow_negative=True,
        )

    def set_max_capacity(
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
        with self._lock(timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets) as lock_stack:
            self._extend_locks(lock_stack)
            self._snapshot_bucket_state(bucket)
            bucket.set_max_capacity(value)
        with self._local_condition:
            self._local_condition.notify_all()

    def apply_configured_max_capacity(
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
        with self._lock(timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets) as lock_stack:
            self._extend_locks(lock_stack)
            self._snapshot_bucket_state(bucket)
            bucket.clear_max_capacity_override()
            bucket.set_configured_max_capacity(value)
        with self._local_condition:
            self._local_condition.notify_all()

    def prepare_reconfigured_backend(
        self,
        new_backend: SyncRateLimiterBackend,
        cfg: PerModelConfig,
    ) -> SyncRateLimiterBackend:
        if not isinstance(new_backend, SyncRedisBackend):
            raise TypeError(
                "SyncRedisBackend can only reconfigure into another SyncRedisBackend"
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

        with self._lock(
            timeout=LOCK_TIMEOUT_SECONDS, buckets=reconfigure_buckets
        ) as lock_stack:
            for bucket in removed_buckets:
                # Snapshot first so the frozen capacity/last_checked in Redis
                # reflect the bucket's state at the moment of removal. Without
                # this, a later re-add with a different rate would retroactively
                # integrate the new rate across the remove→re-add gap (and any
                # time preceding it). Then clear the runtime override so the
                # re-add starts from the callable config's static quota again.
                self._extend_locks(lock_stack)
                self._snapshot_bucket_state(bucket)
                bucket.clear_max_capacity_override()
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
                # Snapshot before any mutation could change the effective rate.
                # Prefer current_bucket (its _max_capacity_default matches the
                # stored override payload, so the override is accepted and
                # yields the true active rate).
                self._extend_locks(lock_stack)
                self._snapshot_bucket_state(
                    current_bucket if current_bucket is not None else bucket
                )
                if current_bucket is not None and float(
                    current_bucket.configured_max_capacity
                ) != float(quota.limit):
                    bucket.clear_max_capacity_override()
                bucket.set_configured_max_capacity(float(quota.limit))

            self._extend_locks(lock_stack)
            self.install_reconfigured_state(
                buckets=list(new_buckets),
                cfg=cfg,
            )
        with self._local_condition:
            self._local_condition.notify_all()
        return self

    def install_reconfigured_state(
        self,
        *,
        buckets: list[SyncRedisBucket],
        cfg: PerModelConfig,
    ) -> None:
        self.sorted_buckets = sorted(buckets, key=lambda bucket: bucket.full_redis_key)
        self._usage_metric_names = {bucket.usage_metric for bucket in buckets}
        self._limit_config = cfg

    def _invoke_callback_safe(self, callback, **kwargs) -> None:
        """Fire a user callback, suppressing exceptions to prevent capacity leaks."""
        try:
            callback(**kwargs)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:  # noqa: BLE001
            warnings.warn(
                f"Rate limiter callback raised {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )

    def _refund_cancelled_consumption(
        self,
        usage: FrozenUsage,
        *,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
    ) -> None:
        """
        Refund capacity consumed before a BaseException hit callbacks.

        Fires no callbacks to avoid recursion and another interruption window.
        """
        target_buckets = self._snapshot_buckets() if buckets is None else tuple(buckets)
        with self._lock(
            timeout=LOCK_TIMEOUT_SECONDS,
            buckets=target_buckets,
        ) as lock_stack:
            pipeline = self._redis.pipeline()
            current_time = sync_server_time(self._redis)
            capacities, _ = self._get_capacities_unsafe(
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
            self._extend_locks(lock_stack)
            self._set_capacities_unsafe(
                frozendict(refunded),
                pipeline=pipeline,
                current_time=current_time,
                allow_negative=True,
                buckets=target_buckets,
            )
        with self._local_condition:
            self._local_condition.notify_all()

    def _fresh_start_buckets_callback(
        self,
        fresh_start_buckets: list[SyncRedisBucket],
    ) -> None:
        if (
            fresh_start_buckets
            and self._callbacks
            and self._callbacks.on_missing_consumption_data
        ):
            for bucket in fresh_start_buckets:
                self._invoke_callback_safe(
                    self._callbacks.on_missing_consumption_data,
                    model_family=self._limit_config.get_model_family(),
                    usage_metric=bucket.usage_metric,
                    per_seconds=bucket.per_seconds,
                )
