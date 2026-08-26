from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging
import math
import random
import threading
import time
import uuid
import warnings
from typing import TYPE_CHECKING, ClassVar, TypeVar

from frozendict import frozendict

from token_throttle._capacity import _validate_max_capacity_finite_positive
from token_throttle._diagnostic import (
    BackendBucketLimit,
    BackendIntrospectionDiagnostic,
    DiagnosticWaiterState,
    SqliteBackendHealthDiagnostic,
    backend_type_for_object,
    make_bucket_diagnostic,
    wait_bucket_diagnostics,
    waiter_diagnostic_from_state,
)
from token_throttle._exceptions import BackendLockContentionError
from token_throttle._interfaces._callbacks import (
    BACKEND_CALLBACK_CRITICAL_EXCEPTIONS,
    RateLimiterCallbacks,
    current_limiter_callback_context,
    safe_invoke_async_callback,
)
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    RateLimiterBackend,
    RateLimiterBackendBuilderInterface,
)
from token_throttle._validation import (
    _revalidate_dto,
    _validate_key_prefix,
    validate_backend_refund_usage_for_bucket_ids,
    validate_backend_usage,
    validate_sleep_interval,
    validate_timeout,
)

from ._engine import (
    DEFAULT_BUSY_TIMEOUT_MS,
    DEFAULT_PRUNE_BATCH_SIZE,
    BucketSpec,
    CapacityResult,
    SqliteEngine,
    TryConsumeResult,
)
from ._sync_backend import (
    _log_cancellation_refund_failure,
    _normalize_usage,
    _require_marker_metadata,
    _validate_busy_timeout_ms,
    _validate_db_path,
    _validate_marker_refund_scope,
    _validate_prune_batch_size,
)
from ._ttl import (
    DEFAULT_BUCKET_TTL_SECONDS,
    DEFAULT_REFUND_DEDUP_TTL_SECONDS,
    resolve_max_reservation_lifetime_seconds_from_ttls,
    validate_bucket_ttl_covers_quota_windows,
    validate_reservation_lifetime_ttl_invariant,
    validate_sqlite_ttl_seconds,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Callable

    from token_throttle._interfaces._models import BucketId, Capacities, FrozenUsage

_T = TypeVar("_T")

_acquire_logger = logging.getLogger("token_throttle.acquire")
_refund_logger = logging.getLogger("token_throttle.refund")


class SqliteBackendBuilder(RateLimiterBackendBuilderInterface):
    """Build async SQLite backends with one confined worker thread each."""

    def __init__(  # noqa: PLR0913
        self,
        db_path: str | os.PathLike[str],
        *,
        key_prefix: str,
        sleep_interval: float | None = None,
        bucket_ttl_seconds: int = DEFAULT_BUCKET_TTL_SECONDS,
        refund_dedup_ttl_seconds: int = DEFAULT_REFUND_DEDUP_TTL_SECONDS,
        override_ttl_seconds: int | None = None,
        max_reservation_lifetime_seconds: float | None = None,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        prune_batch_size: int = DEFAULT_PRUNE_BATCH_SIZE,
    ) -> None:
        super().__init__()
        self._db_path = _validate_db_path(db_path)
        self._key_prefix = _validate_key_prefix(key_prefix)
        self._sleep_interval = validate_sleep_interval(sleep_interval)
        self._bucket_ttl_seconds = validate_sqlite_ttl_seconds(
            bucket_ttl_seconds, name="bucket_ttl_seconds"
        )
        self._refund_dedup_ttl_seconds = validate_sqlite_ttl_seconds(
            refund_dedup_ttl_seconds, name="refund_dedup_ttl_seconds"
        )
        self._override_ttl_seconds = validate_sqlite_ttl_seconds(
            (
                self._bucket_ttl_seconds
                if override_ttl_seconds is None
                else override_ttl_seconds
            ),
            name="override_ttl_seconds",
        )
        self._max_reservation_lifetime_seconds = (
            resolve_max_reservation_lifetime_seconds_from_ttls(
                max_reservation_lifetime_seconds=max_reservation_lifetime_seconds,
                bucket_ttl_seconds=self._bucket_ttl_seconds,
                refund_dedup_ttl_seconds=self._refund_dedup_ttl_seconds,
            )
        )
        self._busy_timeout_ms = _validate_busy_timeout_ms(busy_timeout_ms)
        self._prune_batch_size = _validate_prune_batch_size(prune_batch_size)
        self._backends: list[SqliteBackend] = []
        self._lock = threading.Lock()

    @property
    def db_path(self) -> str:
        return self._db_path

    def resolve_max_reservation_lifetime_seconds(
        self, max_reservation_lifetime_seconds: float | None
    ) -> float:
        if max_reservation_lifetime_seconds is None:
            return self._max_reservation_lifetime_seconds
        resolved = resolve_max_reservation_lifetime_seconds_from_ttls(
            max_reservation_lifetime_seconds=max_reservation_lifetime_seconds,
            bucket_ttl_seconds=self._bucket_ttl_seconds,
            refund_dedup_ttl_seconds=self._refund_dedup_ttl_seconds,
        )
        if resolved > self._max_reservation_lifetime_seconds:
            raise ValueError(
                "max_reservation_lifetime_seconds exceeds the SQLite builder's "
                "configured maximum"
            )
        return resolved

    def validate_reservation_lifetime_seconds(
        self, max_reservation_lifetime_seconds: float | None
    ) -> None:
        validate_reservation_lifetime_ttl_invariant(
            max_reservation_lifetime_seconds=max_reservation_lifetime_seconds,
            bucket_ttl_seconds=self._bucket_ttl_seconds,
            refund_dedup_ttl_seconds=self._refund_dedup_ttl_seconds,
        )

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> SqliteBackend:
        cfg = _revalidate_dto(cfg)
        if callbacks is not None:
            _revalidate_dto(callbacks)
        validate_bucket_ttl_covers_quota_windows(
            bucket_ttl_seconds=self._bucket_ttl_seconds,
            quotas=cfg.quotas,
        )
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="token-throttle-sqlite",
        )

        def create_engine() -> SqliteEngine:
            engine = SqliteEngine(
                db_path=self._db_path,
                key_prefix=self._key_prefix,
                model_family=cfg.get_model_family(),
                buckets=tuple(
                    BucketSpec(
                        metric=quota.metric,
                        per_seconds=int(quota.per_seconds),
                        configured_max_capacity=float(quota.limit),
                    )
                    for quota in cfg.quotas
                ),
                bucket_ttl_seconds=self._bucket_ttl_seconds,
                refund_dedup_ttl_seconds=self._refund_dedup_ttl_seconds,
                override_ttl_seconds=self._override_ttl_seconds,
                max_reservation_lifetime_seconds=(
                    self._max_reservation_lifetime_seconds
                ),
                busy_timeout_ms=self._busy_timeout_ms,
                prune_batch_size=self._prune_batch_size,
            )
            try:
                engine.initialize_buckets(time.time())
            except BaseException:
                engine.close()
                raise
            return engine

        try:
            engine = executor.submit(create_engine).result()
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        backend = SqliteBackend(
            engine=engine,
            executor=executor,
            callbacks=callbacks,
            limit_config=cfg,
            sleep_interval=self._sleep_interval,
        )
        with self._lock:
            self._backends.append(backend)
        return backend

    async def aclose(self) -> None:
        with self._lock:
            backends = tuple(self._backends)
            self._backends.clear()
        for backend in backends:
            await backend.aclose()

    def close(self) -> None:
        with self._lock:
            backends = tuple(self._backends)
            self._backends.clear()
        for backend in backends:
            backend.close()


class SqliteBackend(RateLimiterBackend):
    DEFAULT_SLEEP_INTERVAL: ClassVar[float] = 0.1
    MAX_CROSS_WORKER_POLL: ClassVar[float] = 1.0
    WAIT_JITTER_RATIO: ClassVar[float] = 0.2

    def __init__(
        self,
        *,
        engine: SqliteEngine,
        executor: concurrent.futures.ThreadPoolExecutor,
        limit_config: PerModelConfig,
        sleep_interval: float | None = None,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._executor = executor
        self._limit_config = _revalidate_dto(limit_config)
        if callbacks is not None:
            _revalidate_dto(callbacks)
        self._callbacks = callbacks
        validated_sleep = validate_sleep_interval(sleep_interval)
        self._sleep_interval = (
            self.DEFAULT_SLEEP_INTERVAL if validated_sleep is None else validated_sleep
        )
        self._diagnostic_waiters: dict[str, DiagnosticWaiterState] = {}
        self._close_lock = threading.Lock()
        self._closed = False

    def supports_metric_set_change(self) -> bool:
        return True

    def supports_durable_refund_dedup(self) -> bool:
        return True

    def supports_acquire_marker_authority(self) -> bool:
        return True

    def _submit(self, callable_: Callable[[], _T]) -> asyncio.Future[_T]:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(self._executor, callable_)

    @staticmethod
    async def _wait_for_future_while_cancelled(
        future: asyncio.Future[_T],
    ) -> bool:
        while not future.done():
            try:
                await asyncio.shield(future)
            # ast-guard: skip — outcome observer; callers own operation cleanup
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001
                break
        return future.done() and not future.cancelled() and future.exception() is None

    async def _run_engine(self, callable_: Callable[[], _T]) -> _T:
        future = self._submit(callable_)
        try:
            return await asyncio.shield(future)
        except BaseException:
            await self._wait_for_future_while_cancelled(future)
            raise

    async def _try_consume_cancellation_safe(  # noqa: PLR0913
        self,
        usage: FrozenUsage,
        *,
        current_time: float,
        reservation_id: str | None,
        reservation_lifetime_seconds: float | None,
        busy_timeout_ms: int,
        timeout_on_busy: bool,
    ) -> TryConsumeResult:
        future = self._submit(
            functools.partial(
                self._engine.try_consume,
                usage,
                current_time=current_time,
                reservation_id=reservation_id,
                reservation_lifetime_seconds=reservation_lifetime_seconds,
                busy_timeout_ms=busy_timeout_ms,
                timeout_on_busy=timeout_on_busy,
            )
        )
        try:
            return await asyncio.shield(future)
        except BaseException:
            settled_successfully = await self._wait_for_future_while_cancelled(future)
            if settled_successfully:
                attempt = future.result()
                if attempt.available:
                    cleanup = self._submit(
                        functools.partial(
                            self._engine.cleanup_consumption,
                            usage,
                            bucket_ids=self._engine.bucket_ids,
                            current_time=time.time(),
                            reservation_id=reservation_id,
                        )
                    )
                    cleanup_ok = await self._wait_for_future_while_cancelled(cleanup)
                    if not cleanup_ok and cleanup.done() and not cleanup.cancelled():
                        try:
                            cleanup.result()
                        except BaseException as exc:  # noqa: BLE001
                            _log_cancellation_refund_failure(
                                exc,
                                reservation_id=reservation_id,
                                usage=usage,
                            )
            raise

    async def introspect(self) -> BackendIntrospectionDiagnostic:
        as_of_monotonic = time.monotonic()
        snapshots, counts = await self._run_engine(
            functools.partial(self._engine.inspect_snapshot, current_time=time.time())
        )
        buckets = tuple(
            make_bucket_diagnostic(
                model_family=self._engine.model_family,
                metric=snapshot.spec.metric,
                per_seconds=snapshot.spec.per_seconds,
                backend_type=backend_type_for_object(self),
                current_capacity=snapshot.current_capacity,
                configured_limit=snapshot.spec.configured_max_capacity,
                effective_max_capacity=snapshot.effective_max_capacity,
                override_source=("backend" if snapshot.override_active else "none"),
                status="fresh_start" if snapshot.is_fresh_start else "ok",
                as_of_monotonic=as_of_monotonic,
            )
            for snapshot in snapshots
        )
        waiters = tuple(
            waiter_diagnostic_from_state(state, as_of_monotonic=as_of_monotonic)
            for state in sorted(
                self._diagnostic_waiters.values(),
                key=lambda item: (item.wait_started_monotonic, item.waiter_id),
            )
        )
        return BackendIntrospectionDiagnostic(
            model_family=self._engine.model_family,
            backend_type=backend_type_for_object(self),
            as_of_monotonic=as_of_monotonic,
            buckets=buckets,
            waits=waiters,
            memory_health=None,
            redis_health=None,
            sqlite_health=SqliteBackendHealthDiagnostic(
                model_family_count=1,
                bucket_count=len(buckets),
                acquire_marker_count=counts["acquire_markers"],
                refund_tombstone_count=counts["refund_tombstones"],
            ),
            issues=(),
        )

    async def prepare_reconfigured_backend(
        self,
        new_backend: RateLimiterBackend,
        cfg: PerModelConfig,
    ) -> RateLimiterBackend:
        if not isinstance(new_backend, SqliteBackend):
            raise TypeError(
                "SqliteBackend can only reconfigure into another SqliteBackend"
            )
        if (
            self._engine.db_path != new_backend._engine.db_path  # noqa: SLF001
            or self._engine.key_prefix != new_backend._engine.key_prefix  # noqa: SLF001
        ):
            raise ValueError(
                "SQLite reconfiguration requires the same db_path and key_prefix"
            )
        old_ids = self._engine.bucket_ids
        new_ids = new_backend._engine.bucket_ids  # noqa: SLF001
        changed_ids = frozenset(
            (quota.metric, int(quota.per_seconds))
            for quota in cfg.quotas
            if (quota.metric, int(quota.per_seconds)) in old_ids
            and not math.isclose(
                self._engine.configured_max_capacity(
                    quota.metric, int(quota.per_seconds)
                ),
                float(quota.limit),
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        )
        await self._run_engine(
            functools.partial(
                self._engine.clear_max_capacity_overrides,
                frozenset(old_ids - new_ids) | changed_ids,
                current_time=time.time(),
            )
        )
        return new_backend

    async def consume_capacity(
        self,
        usage: FrozenUsage,
        *,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        validate_backend_usage(usage, self._engine.metric_names)
        usage = _normalize_usage(usage)
        result = await self._run_engine(
            functools.partial(
                self._engine.consume,
                usage,
                current_time=time.time(),
                reservation_id=reservation_id,
                reservation_lifetime_seconds=reservation_lifetime_seconds,
            )
        )
        self._warn_over_max_consumption(usage, result)
        await self._emit_consumed_callbacks(usage, result)
        return result.current_time

    async def await_for_capacity(  # noqa: PLR0915
        self,
        usage: FrozenUsage,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        validate_backend_usage(usage, self._engine.metric_names)
        timeout = validate_timeout(timeout)
        usage = _normalize_usage(usage)
        deadline = None if timeout is None else time.monotonic() + timeout
        waiter_key = reservation_id or f"sqlite:{uuid.uuid4().hex}"
        has_waited = False
        wait_started_at: float | None = None
        wait_start_callback_overhead = 0.0
        first_failed_pre: Capacities = frozendict()
        result: CapacityResult
        try:
            while True:
                busy_timeout_ms, timeout_on_busy = self._wait_busy_timeout(deadline)
                try:
                    attempt = await self._try_consume_cancellation_safe(
                        usage,
                        current_time=time.time(),
                        reservation_id=reservation_id,
                        reservation_lifetime_seconds=reservation_lifetime_seconds,
                        busy_timeout_ms=busy_timeout_ms,
                        timeout_on_busy=timeout_on_busy,
                    )
                except BackendLockContentionError as exc:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Timed out waiting for SQLite write-lock contention"
                        ) from exc
                    sleep_for = self._jittered_sleep(self._sleep_interval)
                    if deadline is not None:
                        sleep_for = min(
                            sleep_for, max(0.0, deadline - time.monotonic())
                        )
                    await asyncio.sleep(max(0.001, sleep_for))
                    continue
                result = attempt.result
                if attempt.available:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise self._capacity_timeout_error(
                        usage, result.pre_capacities, result.max_capacities
                    )
                if not has_waited:
                    has_waited = True
                    first_failed_pre = result.pre_capacities
                    wait_started_at = time.monotonic()
                    if self._callbacks and self._callbacks.on_wait_start:
                        callback_started = time.monotonic()
                        if deadline is not None and callback_started >= deadline:
                            raise self._capacity_timeout_error(
                                usage, first_failed_pre, result.max_capacities
                            )
                        await self._invoke_callback_safe(
                            self._callbacks.on_wait_start,
                            callback_slot="on_wait_start",
                            model_family=self._engine.model_family,
                            preconsumption_capacities=first_failed_pre,
                            usage=usage,
                            **current_limiter_callback_context(),
                        )
                        wait_start_callback_overhead += (
                            time.monotonic() - callback_started
                        )
                        if deadline is not None and time.monotonic() >= deadline:
                            raise self._capacity_timeout_error(
                                usage, first_failed_pre, result.max_capacities
                            )
                self._upsert_diagnostic_waiter(
                    waiter_key,
                    reservation_id=reservation_id,
                    usage=usage,
                    capacities=result.pre_capacities,
                    max_capacities=result.max_capacities,
                    deadline=deadline,
                    wait_started_at=wait_started_at,
                )
                computed = self._compute_sleep(
                    usage, result.pre_capacities, result.max_capacities
                )
                effective = self._jittered_sleep(
                    min(computed, self.MAX_CROSS_WORKER_POLL)
                )
                effective = min(effective, self.MAX_CROSS_WORKER_POLL)
                if deadline is not None:
                    effective = min(effective, max(0.0, deadline - time.monotonic()))
                await asyncio.sleep(max(0.001, effective))
        finally:
            self._diagnostic_waiters.pop(waiter_key, None)

        consumed_monotonic = time.monotonic()
        try:
            await self._emit_consumed_callbacks(usage, result)
            if (
                has_waited
                and self._callbacks
                and self._callbacks.after_wait_end_consumption
            ):
                wait_time_s = max(
                    0.0,
                    consumed_monotonic
                    - (wait_started_at or consumed_monotonic)
                    - wait_start_callback_overhead,
                )
                await self._invoke_callback_safe(
                    self._callbacks.after_wait_end_consumption,
                    callback_slot="after_wait_end_consumption",
                    model_family=self._engine.model_family,
                    preconsumption_capacities=result.pre_capacities,
                    postconsumption_capacities=result.post_capacities,
                    usage=usage,
                    wait_time_s=wait_time_s,
                    **current_limiter_callback_context(),
                )
        except BaseException:
            try:
                await self._run_engine(
                    functools.partial(
                        self._engine.cleanup_consumption,
                        usage,
                        bucket_ids=self._engine.bucket_ids,
                        current_time=time.time(),
                        reservation_id=reservation_id,
                    )
                )
            except BaseException as refund_exc:  # noqa: BLE001
                _log_cancellation_refund_failure(
                    refund_exc,
                    reservation_id=reservation_id,
                    usage=usage,
                )
            raise
        return result.current_time

    async def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        await self.refund_capacity_for_buckets(
            reserved_usage, actual_usage, bucket_ids=self._engine.bucket_ids
        )

    async def refund_capacity_for_buckets(  # noqa: PLR0913
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        *,
        bucket_ids: set[BucketId] | frozenset[BucketId] | None = None,
        reservation_id: str | None = None,
        reservation_model_family: str | None = None,
        reservation_bucket_ids: set[BucketId] | frozenset[BucketId] | None = None,
        reservation_reserved_usage: FrozenUsage | None = None,
    ) -> bool:
        backend_bucket_ids = self._engine.bucket_ids
        refund_bucket_ids = (
            backend_bucket_ids if bucket_ids is None else frozenset(bucket_ids)
        )
        validate_backend_refund_usage_for_bucket_ids(
            reserved_usage,
            actual_usage,
            refund_bucket_ids,
            backend_bucket_ids,
        )
        reserved_usage = _normalize_usage(reserved_usage)
        actual_usage = _normalize_usage(actual_usage)
        marker_model_family: str | None = None
        marker_bucket_ids: frozenset[BucketId] | None = None
        marker_reserved_usage: FrozenUsage | None = None
        if reservation_id is not None:
            marker_model_family, marker_bucket_ids, marker_reserved_usage = (
                _require_marker_metadata(
                    reservation_model_family=reservation_model_family,
                    reservation_bucket_ids=reservation_bucket_ids,
                    reservation_reserved_usage=reservation_reserved_usage,
                )
            )
            _validate_marker_refund_scope(
                reserved_usage=reserved_usage,
                refund_bucket_ids=refund_bucket_ids,
                marker_bucket_ids=marker_bucket_ids,
                marker_reserved_usage=marker_reserved_usage,
            )
        if not refund_bucket_ids and reservation_id is None:
            return True
        self._warn_refund_overuse(
            reserved_usage,
            actual_usage,
            refund_bucket_ids,
            reservation_id=reservation_id,
        )
        result = await self._run_engine(
            functools.partial(
                self._engine.refund,
                reserved_usage,
                actual_usage,
                refund_bucket_ids=refund_bucket_ids,
                current_time=time.time(),
                reservation_id=reservation_id,
                reservation_model_family=marker_model_family,
                reservation_bucket_ids=marker_bucket_ids,
                reservation_reserved_usage=marker_reserved_usage,
            )
        )
        await self._fresh_start_buckets_callback(result.fresh_bucket_ids)
        if self._callbacks and self._callbacks.on_capacity_refunded:
            await self._invoke_callback_safe(
                self._callbacks.on_capacity_refunded,
                callback_slot="on_capacity_refunded",
                model_family=self._engine.model_family,
                reserved_usage=reserved_usage,
                actual_usage=actual_usage,
                refunded_usage=result.refunded_usage,
                prerefund_capacities=result.pre_capacities,
                postrefund_capacities=result.post_capacities,
            )
        return True

    async def set_max_capacity(
        self, metric: str, per_seconds: int, value: float
    ) -> None:
        value = _validate_max_capacity_finite_positive(value)
        await self._run_engine(
            functools.partial(
                self._engine.set_max_capacity,
                metric,
                per_seconds,
                value,
                current_time=time.time(),
            )
        )

    async def apply_configured_max_capacity(
        self, metric: str, per_seconds: int, value: float
    ) -> None:
        value = _validate_max_capacity_finite_positive(value)
        await self._run_engine(
            functools.partial(
                self._engine.apply_configured_max_capacity,
                metric,
                per_seconds,
                value,
                current_time=time.time(),
            )
        )

    def _wait_busy_timeout(self, deadline: float | None) -> tuple[int, bool]:
        if deadline is None:
            return self._engine.busy_timeout_ms, False
        remaining = max(0.0, deadline - time.monotonic())
        timeout_ms = min(
            self._engine.busy_timeout_ms,
            max(0, math.ceil(remaining * 1000.0)),
        )
        return timeout_ms, timeout_ms == 0

    def _compute_sleep(
        self,
        usage: FrozenUsage,
        capacities: Capacities,
        max_capacities: Capacities,
    ) -> float:
        max_wait = 0.0
        for (metric, per_seconds), current_capacity in capacities.items():
            deficit = float(usage.get(metric, 0.0)) - float(current_capacity)
            if deficit <= 0:
                continue
            rate_per_sec = float(max_capacities[(metric, per_seconds)]) / float(
                per_seconds
            )
            if not math.isfinite(rate_per_sec) or rate_per_sec <= 0:
                raise ValueError(
                    "Bucket rate is non-positive/non-finite — likely a "
                    "misconfigured max_capacity"
                )
            max_wait = max(max_wait, deficit / rate_per_sec)
        return max_wait if max_wait > 0 else self._sleep_interval

    def _jittered_sleep(self, value: float) -> float:
        return value * random.uniform(  # noqa: S311 - scheduling jitter only.
            1.0 - self.WAIT_JITTER_RATIO,
            1.0 + self.WAIT_JITTER_RATIO,
        )

    def _capacity_timeout_error(
        self,
        usage: FrozenUsage,
        capacities: Capacities,
        max_capacities: Capacities,
    ) -> TimeoutError:
        bottleneck_id: BucketId | None = None
        available: float | None = None
        requested: float | None = None
        largest_deficit = 0.0
        for bucket_id, capacity in capacities.items():
            metric, _ = bucket_id
            deficit = float(usage.get(metric, 0.0)) - float(capacity)
            if deficit > largest_deficit:
                largest_deficit = deficit
                bottleneck_id = bucket_id
                available = float(capacity)
                requested = float(usage[metric])
        return TimeoutError(
            "Timed out waiting for capacity "
            f"(bottleneck={bottleneck_id}, available={available}, "
            f"requested={requested}, computed_sleep="
            f"{self._compute_sleep(usage, capacities, max_capacities)})"
        )

    def _upsert_diagnostic_waiter(  # noqa: PLR0913
        self,
        waiter_key: str,
        *,
        reservation_id: str | None,
        usage: FrozenUsage,
        capacities: Capacities,
        max_capacities: Capacities,
        deadline: float | None,
        wait_started_at: float | None,
    ) -> None:
        limits = {
            bucket_id: BackendBucketLimit(
                effective_max_capacity=float(maximum),
                refill_rate_per_second=float(maximum) / bucket_id[1],
            )
            for bucket_id, maximum in max_capacities.items()
        }
        self._diagnostic_waiters[waiter_key] = DiagnosticWaiterState(
            waiter_id=waiter_key,
            reservation_id=reservation_id,
            model_family=self._engine.model_family,
            model=None,
            request_id=None,
            state="waiting_for_capacity",
            usage=usage,
            wait_started_monotonic=wait_started_at or time.monotonic(),
            timeout_deadline_monotonic=deadline,
            blocked_buckets=wait_bucket_diagnostics(
                model_family=self._engine.model_family,
                usage=usage,
                capacities=dict(capacities),
                limits=limits,
            ),
        )

    def _warn_over_max_consumption(
        self, usage: FrozenUsage, result: CapacityResult
    ) -> None:
        for metric, amount in usage.items():
            for bucket_id, maximum in result.max_capacities.items():
                if bucket_id[0] != metric or amount <= maximum:
                    continue
                message = (
                    f"record_usage value for {metric} ({amount}) exceeds bucket "
                    f"max capacity ({maximum}). Capacity will go deeply negative."
                )
                warnings.warn(message, RuntimeWarning, stacklevel=3)
                _acquire_logger.warning(
                    message,
                    extra={
                        "token_throttle_metric": metric,
                        "token_throttle_model_family": self._engine.model_family,
                        "token_throttle_value": amount,
                        "token_throttle_bucket_id": bucket_id,
                    },
                )

    def _warn_refund_overuse(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        refund_bucket_ids: frozenset[BucketId],
        *,
        reservation_id: str | None,
    ) -> None:
        for metric, reserved_amount in reserved_usage.items():
            actual_amount = actual_usage[metric]
            refund_amount = float(reserved_amount) - float(actual_amount)
            if refund_amount >= 0:
                continue
            message = (
                f"Actual usage ({actual_amount}) for {metric} exceeds reserved "
                f"usage ({reserved_amount}). Applying negative refund."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=3)
            _refund_logger.warning(
                message,
                extra={
                    "token_throttle_metric": metric,
                    "token_throttle_model_family": self._engine.model_family,
                    "token_throttle_value": refund_amount,
                    "token_throttle_bucket_ids": sorted(
                        bucket_id
                        for bucket_id in refund_bucket_ids
                        if bucket_id[0] == metric
                    ),
                    "token_throttle_reservation_id": reservation_id,
                },
            )

    async def _emit_consumed_callbacks(
        self, usage: FrozenUsage, result: CapacityResult
    ) -> None:
        await self._fresh_start_buckets_callback(result.fresh_bucket_ids)
        if self._callbacks and self._callbacks.on_capacity_consumed:
            await self._invoke_callback_safe(
                self._callbacks.on_capacity_consumed,
                callback_slot="on_capacity_consumed",
                model_family=self._engine.model_family,
                preconsumption_capacities=result.pre_capacities,
                postconsumption_capacities=result.post_capacities,
                usage=usage,
                current_time=result.current_time,
            )

    async def _fresh_start_buckets_callback(
        self, bucket_ids: tuple[BucketId, ...]
    ) -> None:
        if not (
            bucket_ids
            and self._callbacks
            and self._callbacks.on_missing_consumption_data
        ):
            return
        for metric, per_seconds in bucket_ids:
            await self._invoke_callback_safe(
                self._callbacks.on_missing_consumption_data,
                callback_slot="on_missing_consumption_data",
                model_family=self._engine.model_family,
                usage_metric=metric,
                per_seconds=per_seconds,
            )

    @staticmethod
    async def _invoke_callback_safe(
        callback,
        *,
        callback_slot: str = "callback",
        **kwargs,
    ) -> None:
        await safe_invoke_async_callback(
            callback,
            critical=BACKEND_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter callback",
            callback_slot=callback_slot,
            **kwargs,
        )

    def _begin_close(self) -> bool:
        with self._close_lock:
            if self._closed:
                return False
            self._closed = True
            return True

    async def aclose(self) -> None:
        if not self._begin_close():
            return
        future = self._submit(self._engine.close)
        try:
            await asyncio.shield(future)
        except BaseException:
            await self._wait_for_future_while_cancelled(future)
            raise
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def close(self) -> None:
        if not self._begin_close():
            return
        try:
            self._executor.submit(self._engine.close).result()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
