from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
import warnings
from typing import TYPE_CHECKING, ClassVar

from frozendict import frozendict

from token_throttle._capacity import _validate_max_capacity_finite_positive
from token_throttle._exceptions import BackendLockContentionError
from token_throttle._interfaces._callbacks import (
    BACKEND_CALLBACK_CRITICAL_EXCEPTIONS,
    SyncRateLimiterCallbacks,
    current_limiter_callback_context,
    safe_invoke_sync_callback,
)
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    SyncRateLimiterBackend,
    SyncRateLimiterBackendBuilderInterface,
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
    from token_throttle._interfaces._models import BucketId, Capacities, FrozenUsage

_logger = logging.getLogger("token_throttle")
_acquire_logger = logging.getLogger("token_throttle.acquire")
_refund_logger = logging.getLogger("token_throttle.refund")


def _validate_db_path(value: object) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError("db_path must be a str or os.PathLike[str]")
    try:
        path = os.fspath(value)
    except TypeError as exc:
        raise TypeError("db_path must be a str or os.PathLike[str]") from exc
    if type(path) is not str:
        raise TypeError("db_path must resolve to a str path")
    if not path:
        raise ValueError("db_path must not be empty")
    if "\x00" in path:
        raise ValueError("db_path must not contain NUL characters")
    return os.path.realpath(path)


def _validate_busy_timeout_ms(value: object) -> int:
    if type(value) is not int:
        raise TypeError("busy_timeout_ms must be an exact int number of milliseconds")
    if value < 0:
        raise ValueError("busy_timeout_ms must be greater than or equal to 0")
    if value > 2**31 - 1:
        raise ValueError(f"busy_timeout_ms must be <= {2**31 - 1}")
    return value


def _validate_prune_batch_size(value: object) -> int:
    if type(value) is not int:
        raise TypeError("prune_batch_size must be an exact int")
    if value <= 0:
        raise ValueError("prune_batch_size must be greater than 0")
    return value


def _normalize_usage(usage: FrozenUsage) -> FrozenUsage:
    return frozendict({metric: float(amount) for metric, amount in usage.items()})


def _require_marker_metadata(
    *,
    reservation_model_family: str | None,
    reservation_bucket_ids: set[BucketId] | frozenset[BucketId] | None,
    reservation_reserved_usage: FrozenUsage | None,
) -> tuple[str, frozenset[BucketId], FrozenUsage]:
    if (
        reservation_model_family is None
        or reservation_bucket_ids is None
        or reservation_reserved_usage is None
    ):
        raise ValueError(
            "reservation marker metadata is required for marker-authorized refunds"
        )
    return (
        reservation_model_family,
        frozenset(reservation_bucket_ids),
        _normalize_usage(reservation_reserved_usage),
    )


def _validate_marker_refund_scope(
    *,
    reserved_usage: FrozenUsage,
    refund_bucket_ids: frozenset[BucketId],
    marker_bucket_ids: frozenset[BucketId],
    marker_reserved_usage: FrozenUsage,
) -> None:
    if not refund_bucket_ids.issubset(marker_bucket_ids):
        raise ValueError("refund bucket_ids do not match the acquire marker")
    for metric in frozenset(metric for metric, _ in refund_bucket_ids):
        if metric not in marker_reserved_usage:
            raise ValueError("refund reserved_usage does not match the acquire marker")
        if float(reserved_usage[metric]) != float(marker_reserved_usage[metric]):
            raise ValueError("refund reserved_usage does not match the acquire marker")


def _log_cancellation_refund_failure(
    exc: BaseException,
    *,
    reservation_id: str | None,
    usage: FrozenUsage,
) -> None:
    _refund_logger.warning(
        "SQLite cancellation-path refund failed; reserved capacity for "
        "reservation %s may not be returned until natural refill: %s: %s",
        reservation_id,
        type(exc).__name__,
        exc,
        exc_info=exc,
        extra={
            "token_throttle_reservation_id": reservation_id,
            "token_throttle_usage": dict(usage),
        },
    )


class SyncSqliteBackendBuilder(SyncRateLimiterBackendBuilderInterface):
    """Build synchronous SQLite backends scoped by a database and key prefix."""

    def __init__(  # noqa: PLR0913
        self,
        db_path: str | os.PathLike[str],
        *,
        key_prefix: str,
        sleep_interval: float | None = None,
        bucket_ttl_seconds: int = DEFAULT_BUCKET_TTL_SECONDS,
        refund_dedup_ttl_seconds: int = DEFAULT_REFUND_DEDUP_TTL_SECONDS,
        max_reservation_lifetime_seconds: float | None = None,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        prune_batch_size: int = DEFAULT_PRUNE_BATCH_SIZE,
    ) -> None:
        super().__init__()
        self._db_path = _validate_db_path(db_path)
        self._key_prefix = _validate_key_prefix(key_prefix)
        self._sleep_interval = validate_sleep_interval(sleep_interval)
        self._bucket_ttl_seconds = validate_sqlite_ttl_seconds(
            bucket_ttl_seconds,
            name="bucket_ttl_seconds",
        )
        self._refund_dedup_ttl_seconds = validate_sqlite_ttl_seconds(
            refund_dedup_ttl_seconds,
            name="refund_dedup_ttl_seconds",
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
        self._engines: list[SqliteEngine] = []
        self._lock = threading.Lock()

    @property
    def db_path(self) -> str:
        return self._db_path

    def resolve_max_reservation_lifetime_seconds(
        self,
        max_reservation_lifetime_seconds: float | None,
    ) -> float:
        if max_reservation_lifetime_seconds is None:
            return self._max_reservation_lifetime_seconds
        return resolve_max_reservation_lifetime_seconds_from_ttls(
            max_reservation_lifetime_seconds=max_reservation_lifetime_seconds,
            bucket_ttl_seconds=self._bucket_ttl_seconds,
            refund_dedup_ttl_seconds=self._refund_dedup_ttl_seconds,
        )

    def validate_reservation_lifetime_seconds(
        self,
        max_reservation_lifetime_seconds: float | None,
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
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> SyncSqliteBackend:
        cfg = _revalidate_dto(cfg)
        if callbacks is not None:
            _revalidate_dto(callbacks)
        validate_bucket_ttl_covers_quota_windows(
            bucket_ttl_seconds=self._bucket_ttl_seconds,
            quotas=cfg.quotas,
        )
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
            max_reservation_lifetime_seconds=(self._max_reservation_lifetime_seconds),
            busy_timeout_ms=self._busy_timeout_ms,
            prune_batch_size=self._prune_batch_size,
        )
        try:
            engine.initialize_buckets(time.time())
        except BaseException:
            engine.close()
            raise
        with self._lock:
            self._engines.append(engine)
        return SyncSqliteBackend(
            engine=engine,
            callbacks=callbacks,
            limit_config=cfg,
            sleep_interval=self._sleep_interval,
        )

    def close(self) -> None:
        with self._lock:
            engines = tuple(self._engines)
            self._engines.clear()
        for engine in engines:
            engine.close()


class SyncSqliteBackend(SyncRateLimiterBackend):
    DEFAULT_SLEEP_INTERVAL: ClassVar[float] = 0.1
    MAX_CROSS_WORKER_POLL: ClassVar[float] = 1.0
    WAIT_JITTER_RATIO: ClassVar[float] = 0.2

    def __init__(
        self,
        *,
        engine: SqliteEngine,
        limit_config: PerModelConfig,
        sleep_interval: float | None = None,
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._limit_config = _revalidate_dto(limit_config)
        if callbacks is not None:
            _revalidate_dto(callbacks)
        self._callbacks = callbacks
        validated_sleep = validate_sleep_interval(sleep_interval)
        self._sleep_interval = (
            self.DEFAULT_SLEEP_INTERVAL if validated_sleep is None else validated_sleep
        )

    def supports_metric_set_change(self) -> bool:
        return True

    def supports_durable_refund_dedup(self) -> bool:
        return True

    def supports_acquire_marker_authority(self) -> bool:
        return True

    def prepare_reconfigured_backend(
        self,
        new_backend: SyncRateLimiterBackend,
        _cfg: PerModelConfig,
    ) -> SyncRateLimiterBackend:
        if not isinstance(new_backend, SyncSqliteBackend):
            raise TypeError(
                "SyncSqliteBackend can only reconfigure into another SyncSqliteBackend"
            )
        if (
            self._engine.db_path != new_backend._engine.db_path  # noqa: SLF001
            or self._engine.key_prefix != new_backend._engine.key_prefix  # noqa: SLF001
        ):
            raise ValueError(
                "SQLite reconfiguration requires the same db_path and key_prefix"
            )
        return new_backend

    def consume_capacity(
        self,
        usage: FrozenUsage,
        *,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        validate_backend_usage(usage, self._engine.metric_names)
        usage = _normalize_usage(usage)
        result = self._engine.consume(
            usage,
            current_time=time.time(),
            reservation_id=reservation_id,
            reservation_lifetime_seconds=reservation_lifetime_seconds,
        )
        self._warn_over_max_consumption(usage, result)
        self._emit_consumed_callbacks(usage, result)
        return result.current_time

    def wait_for_capacity(  # noqa: PLR0915
        self,
        usage: FrozenUsage,
        *,
        timeout: float | None = None,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        validate_backend_usage(usage, self._engine.metric_names)
        timeout = validate_timeout(timeout)
        usage = _normalize_usage(usage)
        deadline = None if timeout is None else time.monotonic() + timeout
        has_waited = False
        wait_started_at: float | None = None
        wait_start_callback_overhead = 0.0
        first_failed_pre: Capacities = frozendict()
        result: CapacityResult

        while True:
            try:
                attempt = self._engine.try_consume(
                    usage,
                    current_time=time.time(),
                    reservation_id=reservation_id,
                    reservation_lifetime_seconds=reservation_lifetime_seconds,
                )
            except BackendLockContentionError as exc:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for SQLite write-lock contention"
                    ) from exc
                sleep_for = self._jittered_sleep(self._sleep_interval)
                if deadline is not None:
                    sleep_for = min(
                        sleep_for,
                        max(0.0, deadline - time.monotonic()),
                    )
                time.sleep(max(0.001, sleep_for))
                continue

            result = attempt.result
            if attempt.available:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise self._capacity_timeout_error(
                    usage,
                    result.pre_capacities,
                    result.max_capacities,
                )
            if not has_waited:
                has_waited = True
                first_failed_pre = result.pre_capacities
                wait_started_at = time.monotonic()
                if self._callbacks and self._callbacks.on_wait_start:
                    callback_started = time.monotonic()
                    if deadline is not None and callback_started >= deadline:
                        raise self._capacity_timeout_error(
                            usage,
                            first_failed_pre,
                            result.max_capacities,
                        )
                    self._invoke_callback_safe(
                        self._callbacks.on_wait_start,
                        callback_slot="on_wait_start",
                        model_family=self._limit_config.get_model_family(),
                        preconsumption_capacities=first_failed_pre,
                        usage=usage,
                        **current_limiter_callback_context(),
                    )
                    wait_start_callback_overhead += time.monotonic() - callback_started
                    if deadline is not None and time.monotonic() >= deadline:
                        raise self._capacity_timeout_error(
                            usage,
                            first_failed_pre,
                            result.max_capacities,
                        )
            computed = self._compute_sleep(
                usage,
                result.pre_capacities,
                result.max_capacities,
            )
            effective = self._jittered_sleep(min(computed, self.MAX_CROSS_WORKER_POLL))
            effective = min(effective, self.MAX_CROSS_WORKER_POLL)
            if deadline is not None:
                effective = min(effective, max(0.0, deadline - time.monotonic()))
            time.sleep(max(0.001, effective))

        consumed_monotonic = time.monotonic()
        consumed_bucket_ids = self._engine.bucket_ids
        try:
            self._emit_consumed_callbacks(usage, result)
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
                self._invoke_callback_safe(
                    self._callbacks.after_wait_end_consumption,
                    callback_slot="after_wait_end_consumption",
                    model_family=self._limit_config.get_model_family(),
                    preconsumption_capacities=result.pre_capacities,
                    postconsumption_capacities=result.post_capacities,
                    usage=usage,
                    wait_time_s=wait_time_s,
                    **current_limiter_callback_context(),
                )
        except BaseException:
            try:
                self._engine.cleanup_consumption(
                    usage,
                    bucket_ids=consumed_bucket_ids,
                    current_time=time.time(),
                    reservation_id=reservation_id,
                )
            except BaseException as refund_exc:  # noqa: BLE001
                _log_cancellation_refund_failure(
                    refund_exc,
                    reservation_id=reservation_id,
                    usage=usage,
                )
            raise
        return result.current_time

    def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        self.refund_capacity_for_buckets(
            reserved_usage,
            actual_usage,
            bucket_ids=self._engine.bucket_ids,
        )

    def refund_capacity_for_buckets(  # noqa: PLR0913
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
            (
                marker_model_family,
                marker_bucket_ids,
                marker_reserved_usage,
            ) = _require_marker_metadata(
                reservation_model_family=reservation_model_family,
                reservation_bucket_ids=reservation_bucket_ids,
                reservation_reserved_usage=reservation_reserved_usage,
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
        result = self._engine.refund(
            reserved_usage,
            actual_usage,
            refund_bucket_ids=refund_bucket_ids,
            current_time=time.time(),
            reservation_id=reservation_id,
            reservation_model_family=marker_model_family,
            reservation_bucket_ids=marker_bucket_ids,
            reservation_reserved_usage=marker_reserved_usage,
        )
        self._fresh_start_buckets_callback(result.fresh_bucket_ids)
        if self._callbacks and self._callbacks.on_capacity_refunded:
            self._invoke_callback_safe(
                self._callbacks.on_capacity_refunded,
                callback_slot="on_capacity_refunded",
                model_family=self._limit_config.get_model_family(),
                reserved_usage=reserved_usage,
                actual_usage=actual_usage,
                refunded_usage=result.refunded_usage,
                prerefund_capacities=result.pre_capacities,
                postrefund_capacities=result.post_capacities,
            )
        return True

    def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        value = _validate_max_capacity_finite_positive(value)
        self._engine.set_max_capacity(
            metric,
            per_seconds,
            value,
            current_time=time.time(),
        )

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
            max_capacity = float(max_capacities[(metric, per_seconds)])
            rate_per_sec = max_capacity / float(per_seconds)
            if not math.isfinite(rate_per_sec) or rate_per_sec <= 0:
                raise ValueError(
                    "Bucket rate is non-positive/non-finite — likely a "
                    "misconfigured max_capacity"
                )
            max_wait = max(max_wait, deficit / rate_per_sec)
        return max_wait if max_wait > 0 else self._sleep_interval

    def _jittered_sleep(self, value: float) -> float:
        return value * random.uniform(  # noqa: S311 - scheduling jitter, not security.
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
        computed_sleep = self._compute_sleep(usage, capacities, max_capacities)
        return TimeoutError(
            "Timed out waiting for capacity "
            f"(bottleneck={bottleneck_id}, available={available}, "
            f"requested={requested}, computed_sleep={computed_sleep})"
        )

    def _warn_over_max_consumption(
        self,
        usage: FrozenUsage,
        result: CapacityResult,
    ) -> None:
        for metric, amount in usage.items():
            matching = [
                (bucket_id, maximum)
                for bucket_id, maximum in result.max_capacities.items()
                if bucket_id[0] == metric and amount > maximum
            ]
            for bucket_id, maximum in matching:
                message = (
                    f"record_usage value for {metric} ({amount}) exceeds bucket "
                    f"max capacity ({maximum}). Capacity will go deeply negative."
                )
                warnings.warn(message, RuntimeWarning, stacklevel=3)
                _acquire_logger.warning(
                    message,
                    extra={
                        "token_throttle_metric": metric,
                        "token_throttle_model_family": (
                            self._limit_config.get_model_family()
                        ),
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
                    "token_throttle_model_family": (
                        self._limit_config.get_model_family()
                    ),
                    "token_throttle_value": refund_amount,
                    "token_throttle_bucket_ids": sorted(
                        bucket_id
                        for bucket_id in refund_bucket_ids
                        if bucket_id[0] == metric
                    ),
                    "token_throttle_reservation_id": reservation_id,
                },
            )

    def _emit_consumed_callbacks(
        self,
        usage: FrozenUsage,
        result: CapacityResult,
    ) -> None:
        self._fresh_start_buckets_callback(result.fresh_bucket_ids)
        if self._callbacks and self._callbacks.on_capacity_consumed:
            self._invoke_callback_safe(
                self._callbacks.on_capacity_consumed,
                callback_slot="on_capacity_consumed",
                model_family=self._limit_config.get_model_family(),
                preconsumption_capacities=result.pre_capacities,
                postconsumption_capacities=result.post_capacities,
                usage=usage,
                current_time=result.current_time,
            )

    def _fresh_start_buckets_callback(
        self,
        bucket_ids: tuple[BucketId, ...],
    ) -> None:
        if not (
            bucket_ids
            and self._callbacks
            and self._callbacks.on_missing_consumption_data
        ):
            return
        for metric, per_seconds in bucket_ids:
            self._invoke_callback_safe(
                self._callbacks.on_missing_consumption_data,
                callback_slot="on_missing_consumption_data",
                model_family=self._limit_config.get_model_family(),
                usage_metric=metric,
                per_seconds=per_seconds,
            )

    @staticmethod
    def _invoke_callback_safe(
        callback,
        *,
        callback_slot: str = "callback",
        **kwargs,
    ) -> None:
        safe_invoke_sync_callback(
            callback,
            critical=BACKEND_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter callback",
            callback_slot=callback_slot,
            **kwargs,
        )
