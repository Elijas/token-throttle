import asyncio
import contextlib
import inspect
import logging
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
    validate_sleep_interval,
    validate_timeout,
)

from ._keys import (
    DEFAULT_REFUND_DEDUP_TTL_SECONDS,
    redis_refund_dedup_key,
    validate_redis_key_prefix,
    validate_refund_dedup_ttl_seconds,
)
from ._server_time import sync_server_time
from ._sync_bucket import (
    SyncRedisBucket,
    _normalize_bucket_state_pair,
    _raise_pipeline_response_error,
    _validate_pipeline_results,
)
from ._ttl import DEFAULT_BUCKET_TTL_SECONDS, validate_redis_ttl_seconds

_logger = logging.getLogger("token_throttle")

_CRITICAL_CALLBACK_EXCEPTION_TYPES = (
    asyncio.CancelledError,
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
)


def _callback_exception_group_contains_critical(exc: BaseException) -> bool:
    if not isinstance(exc, BaseExceptionGroup):
        return False
    critical, _non_critical = exc.split(_CRITICAL_CALLBACK_EXCEPTION_TYPES)
    return critical is not None


class SyncCapacitiesGetterResult(typing.NamedTuple):
    capacities: Capacities
    fresh_start_buckets: list[SyncRedisBucket]


LOCK_TIMEOUT_SECONDS = 30

# Each bucket enqueues 4 pipeline commands in get_capacity()
# (GET last_checked, GET capacity, EXPIRE last_checked, EXPIRE capacity).
# Used to index pipeline results.
_PIPELINE_CMDS_PER_BUCKET = 4

# Each bucket enqueues 2 commands for max-capacity override reads
# (GET override, EXPIRE override). Used to index pipeline results.
_PIPELINE_CMDS_PER_OVERRIDE = 2

DEFAULT_LOCK_BLOCKING_TIMEOUT_SECONDS = 5.0
DEFAULT_LOCK_SLEEP_SECONDS = 0.05
DEFAULT_LOCK_BLOCKING_THREAD_SLEEP_SECONDS = 0.05
_MIN_PRODUCTION_REDIS_POOL_CONNECTIONS = 10


def _raise_lock_timeout_error() -> typing.NoReturn:
    raise redis.exceptions.LockError("Unable to acquire lock within the time specified")


def _validate_positive_seconds(value: object, *, name: str) -> float:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and greater than 0")
    value_float = float(value)
    if not math.isfinite(value_float) or value_float <= 0:
        raise ValueError(f"{name} must be finite and greater than 0")
    return value_float


def _warn_if_small_redis_pool(redis_client: object, *, stacklevel: int = 3) -> None:
    pool = getattr(redis_client, "connection_pool", None)
    max_connections = getattr(pool, "max_connections", None)
    if (
        isinstance(max_connections, int)
        and not isinstance(max_connections, bool)
        and max_connections < _MIN_PRODUCTION_REDIS_POOL_CONNECTIONS
    ):
        warnings.warn(
            "Redis connection_pool.max_connections is less than 10. "
            "This is likely too small for production token-throttle workloads; "
            "prefer a BlockingConnectionPool sized to at least "
            "max_concurrent_acquires plus Redis command headroom.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )


class _SyncRedisLockStack(ExitStack):
    """
    ExitStack that also holds the acquired Lock objects so callers can
    extend their TTLs before long-running writes (see _extend_locks).
    """

    def __init__(self) -> None:
        super().__init__()
        self.locks: list[redis.lock.Lock] = []


class SyncRedisBackendBuilder(SyncRateLimiterBackendBuilderInterface):
    """
    Build synchronous Redis limiter backends.

    ``owns_redis_client`` defaults to ``False`` because Redis clients are often
    shared across limiters. Set it to ``True`` only when this builder is the
    lifecycle owner for ``redis_client``; then limiter ``close()`` cascades to
    ``redis_client.close()``.

    For bounded Redis deployments, prefer ``redis.BlockingConnectionPool`` and
    size ``max_connections`` to at least ``max_concurrent_acquires`` plus
    headroom for lock acquire/release, ``TIME``, and pipeline commands.
    """

    def __init__(  # noqa: PLR0913
        self,
        redis_client: redis.Redis,
        *,
        key_prefix: str,
        sleep_interval: float | None = None,
        bucket_ttl_seconds: int = DEFAULT_BUCKET_TTL_SECONDS,
        override_ttl_seconds: int | None = None,
        refund_dedup_ttl_seconds: int = DEFAULT_REFUND_DEDUP_TTL_SECONDS,
        owns_redis_client: bool = False,
        lock_blocking_timeout_seconds: float = DEFAULT_LOCK_BLOCKING_TIMEOUT_SECONDS,
        lock_sleep_seconds: float = DEFAULT_LOCK_SLEEP_SECONDS,
        lock_blocking_thread_sleep_seconds: float = (
            DEFAULT_LOCK_BLOCKING_THREAD_SLEEP_SECONDS
        ),
    ) -> None:
        super().__init__()
        client_module = type(redis_client).__module__
        if client_module.startswith("redis.asyncio.") or (
            client_module.startswith("redis.")
            and not isinstance(redis_client, redis.Redis)
        ):
            raise TypeError(
                "redis_client must be a redis.Redis instance "
                f"(got {type(redis_client).__name__})"
            )
        self._redis = redis_client
        self._owns_redis_client = owns_redis_client
        self._key_prefix = validate_redis_key_prefix(key_prefix)
        self._sleep_interval = validate_sleep_interval(sleep_interval)
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
        self._refund_dedup_ttl_seconds = validate_refund_dedup_ttl_seconds(
            refund_dedup_ttl_seconds
        )
        self._lock_blocking_timeout_seconds = _validate_positive_seconds(
            lock_blocking_timeout_seconds,
            name="lock_blocking_timeout_seconds",
        )
        self._lock_sleep_seconds = _validate_positive_seconds(
            lock_sleep_seconds,
            name="lock_sleep_seconds",
        )
        self._lock_blocking_thread_sleep_seconds = _validate_positive_seconds(
            lock_blocking_thread_sleep_seconds,
            name="lock_blocking_thread_sleep_seconds",
        )

    def close(self) -> None:
        if not self._owns_redis_client:
            return
        close = getattr(self._redis, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                result.close()

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
                key_prefix=self._key_prefix,
                bucket_ttl_seconds=self._bucket_ttl_seconds,
                override_ttl_seconds=self._override_ttl_seconds,
            )
            redis_buckets.append(b)
        return SyncRedisBackend(
            buckets=redis_buckets,
            redis=self._redis,
            key_prefix=self._key_prefix,
            refund_dedup_ttl_seconds=self._refund_dedup_ttl_seconds,
            sleep_interval=self._sleep_interval,
            lock_blocking_timeout_seconds=self._lock_blocking_timeout_seconds,
            lock_sleep_seconds=self._lock_sleep_seconds,
            lock_blocking_thread_sleep_seconds=self._lock_blocking_thread_sleep_seconds,
            callbacks=callbacks,
            limit_config=cfg,
        )


class SyncRedisBackend(SyncRateLimiterBackend):
    DEFAULT_SLEEP_INTERVAL: ClassVar[float] = 0.1
    MAX_CROSS_WORKER_POLL: ClassVar[float] = 1.0

    def __init__(  # noqa: PLR0913
        self,
        buckets: list[SyncRedisBucket],
        redis: redis.Redis,
        limit_config: PerModelConfig,
        *,
        key_prefix: str,
        refund_dedup_ttl_seconds: int = DEFAULT_REFUND_DEDUP_TTL_SECONDS,
        sleep_interval: float | None = None,
        lock_blocking_timeout_seconds: float = DEFAULT_LOCK_BLOCKING_TIMEOUT_SECONDS,
        lock_sleep_seconds: float = DEFAULT_LOCK_SLEEP_SECONDS,
        lock_blocking_thread_sleep_seconds: float = (
            DEFAULT_LOCK_BLOCKING_THREAD_SLEEP_SECONDS
        ),
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> None:
        super().__init__()
        self.sorted_buckets = sorted(buckets, key=lambda b: b.full_redis_key)
        self._redis = redis
        self._key_prefix = validate_redis_key_prefix(key_prefix)
        self._refund_dedup_ttl_seconds = validate_refund_dedup_ttl_seconds(
            refund_dedup_ttl_seconds
        )
        self._sleep_interval: float = (
            self.DEFAULT_SLEEP_INTERVAL
            if sleep_interval is None
            else validate_sleep_interval(sleep_interval)
        )
        _warn_if_small_redis_pool(redis)
        self._lock_blocking_timeout_seconds = _validate_positive_seconds(
            lock_blocking_timeout_seconds,
            name="lock_blocking_timeout_seconds",
        )
        self._lock_sleep_seconds = _validate_positive_seconds(
            lock_sleep_seconds,
            name="lock_sleep_seconds",
        )
        self._lock_blocking_thread_sleep_seconds = _validate_positive_seconds(
            lock_blocking_thread_sleep_seconds,
            name="lock_blocking_thread_sleep_seconds",
        )
        self._callbacks = callbacks
        self._limit_config = limit_config
        self._usage_metric_names: set[str] = {bucket.usage_metric for bucket in buckets}
        self._local_condition = threading.Condition()

    def _effective_lock_blocking_timeout(
        self,
        blocking_timeout: float | None,
    ) -> float:
        if blocking_timeout is None:
            return self._lock_blocking_timeout_seconds
        requested_timeout = validate_timeout(blocking_timeout)
        if requested_timeout is None:
            return self._lock_blocking_timeout_seconds
        return min(requested_timeout, self._lock_blocking_timeout_seconds)

    def supports_metric_set_change(self) -> bool:
        # Surviving bucket state lives in Redis under stable keys, so a rebuilt
        # backend can safely point at the same buckets.
        return True

    def supports_durable_refund_dedup(self) -> bool:
        return True

    def _claim_refund_dedup(self, reservation_id: str | None) -> bool:
        if reservation_id is None:
            return True
        key = redis_refund_dedup_key(self._key_prefix, reservation_id)
        claimed = self._redis.set(
            key,
            "1",
            ex=self._refund_dedup_ttl_seconds,
            nx=True,
        )
        if claimed:
            return True
        message = (
            f"Reservation {reservation_id} has already been refunded according "
            "to Redis refund dedup. Ignoring duplicate refund to prevent "
            "double-crediting capacity."
        )
        warnings.warn(message, UserWarning, stacklevel=3)
        _logger.warning(message)
        return False

    def _snapshot_buckets(self) -> tuple[SyncRedisBucket, ...]:
        return tuple(self.sorted_buckets)

    def _runtime_max_capacity_for_reconciliation(
        self,
        metric: str,
        per_seconds: int,
    ) -> float | None:
        bucket = self._find_bucket(self._snapshot_buckets(), metric, per_seconds)
        if bucket is None:
            return None
        return bucket.refresh_max_capacity_from_redis()

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
        effective_blocking_timeout = self._effective_lock_blocking_timeout(
            blocking_timeout
        )
        stop_trying_at = time.monotonic() + effective_blocking_timeout

        try:
            for bucket in target_buckets:
                if stack.locks:
                    self._extend_locks(stack)
                remaining = max(0.0, stop_trying_at - time.monotonic())
                lock = bucket.lock(timeout=timeout, sleep=self._lock_sleep_seconds)
                # Generate the token ourselves so a best-effort CAS
                # release after a KeyboardInterrupt-style cancel can run
                # without relying on lock.local.token (which is only
                # populated AFTER the SET NX succeeds; a cancel in that
                # window would otherwise orphan the lock for its TTL).
                token = uuid.uuid4().hex.encode()
                try:
                    acquired = lock.acquire(
                        sleep=self._lock_blocking_thread_sleep_seconds,
                        blocking_timeout=remaining,
                        token=token,
                    )
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

        # Include max_capacity in the pipeline to avoid extra round-trips.
        # Refreshing the override TTL is intentionally separate from the
        # schema-version key, which is a long-lived registry entry.
        for bucket in target_buckets:
            pipeline.get(bucket._max_capacity_key)  # noqa: SLF001
            pipeline.expire(
                bucket._max_capacity_key,  # noqa: SLF001
                bucket._override_ttl_seconds,  # noqa: SLF001
            )

        num_buckets = len(target_buckets)
        expected_results = num_buckets * (
            _PIPELINE_CMDS_PER_BUCKET + _PIPELINE_CMDS_PER_OVERRIDE
        )
        try:
            results = pipeline.execute()
        except redis.exceptions.ResponseError as exc:
            _raise_pipeline_response_error(
                "SyncRedisBackend._get_capacities_unsafe", exc
            )
        results = _validate_pipeline_results(
            results,
            context="SyncRedisBackend._get_capacities_unsafe",
            expected_count=expected_results,
        )

        new_capacities: dict[tuple[str, int], float] = {}
        fresh_start_buckets: list[SyncRedisBucket] = []
        for i, bucket in enumerate(target_buckets):
            idx = i * _PIPELINE_CMDS_PER_BUCKET
            last_checked, capacity = _normalize_bucket_state_pair(
                results[idx + SyncRedisBucket.PIPELINE_LAST_CHECKED_OFFSET],
                results[idx + SyncRedisBucket.PIPELINE_CAPACITY_OFFSET],
                context=(
                    f"SyncRedisBackend._get_capacities_unsafe({bucket.full_redis_key})"
                ),
            )
            # max_capacity GETs come after all per-bucket state commands.
            max_capacity_idx = num_buckets * _PIPELINE_CMDS_PER_BUCKET + (
                i * _PIPELINE_CMDS_PER_OVERRIDE
            )
            bucket.update_max_capacity_from_result(results[max_capacity_idx])
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
        try:
            pipeline.execute()
        except redis.exceptions.ResponseError as exc:
            _raise_pipeline_response_error(
                "SyncRedisBackend._set_capacities_unsafe", exc
            )

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
            # KI/SystemExit are the sync analogue of asyncio.CancelledError:
            # they can interrupt mid-statement and need the same best-effort
            # refund to avoid leaking capacity. Do not narrow to Exception.
            if consumed:
                try:  # noqa: SIM105
                    self._refund_cancelled_consumption(usage, buckets=buckets)
                except BaseException:  # noqa: BLE001, S110
                    # Best-effort: sync interrupts mirror async cancellation
                    # cleanup. Swallow so the original interrupt propagates.
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
                # When deadline is None (no caller timeout), the configured
                # Redis lock blocking timeout still bounds lock polling.
                # Propagating raw LockError preserves that distinction from
                # a caller-level wait timeout.
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
                    # KI/SystemExit are the sync analogue of asyncio.CancelledError:
                    # they can interrupt mid-statement and need the same best-effort
                    # refund to avoid leaking capacity. Do not narrow to Exception.
                    try:  # noqa: SIM105
                        self._refund_cancelled_consumption(usage, buckets=buckets)
                    except BaseException:  # noqa: BLE001, S110
                        # Best-effort: sync interrupts mirror async cancellation
                        # cleanup. Swallow so the original interrupt propagates.
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
            rate_per_sec = bucket._rate_per_sec  # noqa: SLF001
            if not math.isfinite(rate_per_sec) or rate_per_sec <= 0:
                raise ValueError(
                    "Bucket rate is non-positive/non-finite — likely a "
                    "misconfigured max_capacity"
                )
            wait = deficit / rate_per_sec
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
        reservation_id: str | None = None,
    ) -> bool:
        """Refund unused capacity back to the rate limiter based on actual usage."""
        if not self._claim_refund_dedup(reservation_id):
            return False

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
            return True
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
                updated_capacities_[(capability_usage_metric, int(per_seconds))] = max(
                    -bucket.max_capacity,
                    min(
                        updated_capacities_[(capability_usage_metric, int(per_seconds))]
                        + refund_amount,
                        bucket.max_capacity,  # cached — refreshed from pipeline result in _get_capacities_unsafe
                    ),
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
        return True

    def _snapshot_bucket_state(self, bucket: SyncRedisBucket) -> None:
        """
        Freeze ``bucket`` in Redis at its accrued capacity under the CURRENT rate.

        Uncapped anchor (stored + elapsed*old_rate) preserves raw values above
        ``max_capacity`` — reads apply ``min(max_capacity, …)`` and a later
        cap raise can re-expose the hidden overflow.
        """
        refresh_max_capacity = getattr(
            bucket,
            "refresh_max_capacity_from_redis",
            None,
        )
        if callable(refresh_max_capacity):
            refresh_max_capacity()
        else:  # test fakes and compatible custom bucket shims
            bucket.get_max_capacity()
        current_time = sync_server_time(self._redis)
        pipeline = self._redis.pipeline()
        pipeline.get(bucket._last_checked_key)  # noqa: SLF001
        pipeline.get(bucket._capacity_key)  # noqa: SLF001
        pipeline.expire(bucket._last_checked_key, bucket._bucket_ttl_seconds)  # noqa: SLF001
        pipeline.expire(bucket._capacity_key, bucket._bucket_ttl_seconds)  # noqa: SLF001
        try:
            results = pipeline.execute()
        except redis.exceptions.ResponseError as exc:
            _raise_pipeline_response_error(
                "SyncRedisBackend._snapshot_bucket_state", exc
            )
        results = _validate_pipeline_results(
            results,
            context=f"SyncRedisBackend._snapshot_bucket_state({bucket.full_redis_key})",
            expected_count=4,
        )
        last_checked_raw, stored_raw = _normalize_bucket_state_pair(
            results[SyncRedisBucket.PIPELINE_LAST_CHECKED_OFFSET],
            results[SyncRedisBucket.PIPELINE_CAPACITY_OFFSET],
            context=f"SyncRedisBackend._snapshot_bucket_state({bucket.full_redis_key})",
        )
        if last_checked_raw is None or stored_raw is None:
            # Partial state (one None) is treated the same as full absence:
            # the bucket will start fresh on the next acquire. This is the
            # correct fallback — anchoring with incomplete data would produce
            # a wrong capacity value.
            _logger.warning(
                "Bucket %s: snapshot skipped due to missing Redis state "
                "(last_checked=%r, capacity=%r).",
                bucket.full_redis_key,
                last_checked_raw,
                stored_raw,
            )
            return
        try:
            last_checked = float(last_checked_raw)
            stored = float(stored_raw)
        except (TypeError, ValueError) as parse_error:
            _logger.warning(
                "Stale Redis bucket state at %r: %r; "
                "snapshot skipped due to unparseable Redis state "
                "(last_checked=%r, capacity=%r).",
                bucket.full_redis_key,
                parse_error,
                last_checked_raw,
                stored_raw,
            )
            return
        if not (math.isfinite(last_checked) and math.isfinite(stored)):
            _logger.warning(
                "Bucket %s: snapshot skipped due to non-finite Redis state "
                "(last_checked=%r, capacity=%r).",
                bucket.full_redis_key,
                last_checked_raw,
                stored_raw,
            )
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
        """
        Fire a user callback, suppressing exceptions to prevent capacity leaks.

        Audited 2026-05 (R4 L03): exception ladder verified parity-clean across
        all 4 _invoke_callback_safe implementations.
        Audited 2026-05 (R4 L12:C03): non-group exotic exceptions are swallowed
        or propagated by design; warning filters must not reopen the leak path.
        Audited 2026-05 (R4 L12:C01/C02): BaseExceptionGroup containing
        cancellation or process signals must propagate before best-effort logging.
        """
        try:
            callback(**kwargs)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:
            if _callback_exception_group_contains_critical(exc):
                raise
            msg = f"Rate limiter callback raised {type(exc).__name__}: {exc}"
            with contextlib.suppress(Warning):
                warnings.warn(msg, RuntimeWarning, stacklevel=3)
            _logger.warning(msg)

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
