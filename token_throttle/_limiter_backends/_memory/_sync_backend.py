import threading
import time
import warnings
from typing import ClassVar

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

from ._bucket import MemoryBucket


class SyncMemoryBackendBuilder(SyncRateLimiterBackendBuilderInterface):
    def __init__(
        self,
        *,
        sleep_interval: float | None = None,
    ) -> None:
        super().__init__()
        self._sleep_interval = sleep_interval

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> "SyncMemoryBackend":
        buckets = []
        for quota in cfg.quotas:
            b = MemoryBucket(
                metric=quota.metric,
                per_seconds=quota.per_seconds,
                limit=float(quota.limit),
                model_family=cfg.get_model_family(),
            )
            buckets.append(b)
        return SyncMemoryBackend(
            buckets=buckets,
            sleep_interval=self._sleep_interval,
            callbacks=callbacks,
            limit_config=cfg,
        )


class SyncMemoryBackend(SyncRateLimiterBackend):
    DEFAULT_SLEEP_INTERVAL: ClassVar[float] = 0.1

    def __init__(
        self,
        buckets: list[MemoryBucket],
        limit_config: PerModelConfig,
        *,
        sleep_interval: float | None = None,
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> None:
        super().__init__()
        self._buckets = buckets
        self._condition = threading.Condition()
        self._sleep_interval: float = (
            self.DEFAULT_SLEEP_INTERVAL if sleep_interval is None else sleep_interval
        )
        self._callbacks = callbacks
        self._limit_config = limit_config
        self._usage_metric_names: set[str] = {bucket.usage_metric for bucket in buckets}
        # Keep retired buckets around so a later metric re-add can resume from
        # the last known capacity instead of starting full again.
        self._bucket_registry: dict[tuple[str, int], MemoryBucket] = {
            (bucket.usage_metric, int(bucket.per_seconds)): bucket for bucket in buckets
        }

    def supports_metric_set_change(self) -> bool:
        return True

    def _get_capacities(
        self,
        current_time: float,
    ) -> tuple[Capacities, list[MemoryBucket]]:
        """Get capacities for all buckets. Must be called under lock."""
        caps: dict[tuple[str, int], float] = {}
        fresh_start_buckets: list[MemoryBucket] = []
        for bucket in self._buckets:
            result = bucket.get_capacity(current_time)
            if result.is_fresh_start:
                fresh_start_buckets.append(bucket)
            caps[(bucket.usage_metric, int(bucket.per_seconds))] = result.amount
        return frozendict(caps), fresh_start_buckets

    def _bucket_ids(self) -> frozenset[tuple[str, int]]:
        return frozenset(
            (bucket.usage_metric, int(bucket.per_seconds)) for bucket in self._buckets
        )

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

    def _set_capacities(
        self,
        new_capacities: Capacities,
        current_time: float,
        *,
        allow_negative: bool = False,
    ) -> None:
        """
        Set capacities for all buckets. Must be called under lock.

        allow_negative=True is required for consume_capacity (speedometer)
        and refund_capacity (preserves negative debt for natural refill).
        """
        for (usage_metric, per_seconds), amount in new_capacities.items():
            bucket = next(
                (
                    b
                    for b in self._buckets
                    if b.usage_metric == usage_metric and b.per_seconds == per_seconds
                ),
                None,
            )
            if bucket is None:
                raise ValueError(f"Bucket '{usage_metric}/{per_seconds}s' not found")
            bucket.set_capacity(amount, current_time, allow_negative=allow_negative)

    def _try_consume_locked(
        self,
        usage: FrozenUsage,
        preconsumption: Capacities,
        current_time: float,
    ) -> tuple[bool, Capacities]:
        """Check and consume atomically. Caller MUST hold self._condition."""
        active_metric_names = {metric for metric, _ in preconsumption}
        self._ensure_usage_metrics_are_active(usage, active_metric_names)

        # Fail fast: if usage exceeds any bucket's max_capacity, it can
        # never be satisfied (capacity is capped at max_capacity).
        for usage_metric, usage_amount in usage.items():
            for bucket in self._buckets:
                if bucket.usage_metric != usage_metric:
                    continue
                if usage_amount > bucket.max_capacity:
                    raise ValueError(
                        f"Usage value for {usage_metric} ({usage_amount}) "
                        f"exceeds bucket max capacity ({bucket.max_capacity})",
                    )

        # All-or-nothing: check every bucket for the relevant metric
        for usage_metric, usage_amount in usage.items():
            for (cap_metric, _), cap_amount in preconsumption.items():
                if usage_metric != cap_metric:
                    continue
                if usage_amount > cap_amount:
                    return False, frozendict()

        # Sufficient capacity — subtract usage from each matching bucket.
        postconsumption_dict: dict[tuple[str, int], float] = dict(preconsumption)
        for (
            cap_metric,
            per_seconds,
        ), cap_amount in preconsumption.items():
            usage_amount = usage.get(cap_metric)
            if usage_amount is None:
                continue
            postconsumption_dict[(cap_metric, per_seconds)] = cap_amount - usage_amount
        postconsumption = frozendict(postconsumption_dict)
        self._set_capacities(postconsumption, current_time)
        return True, postconsumption

    def consume_capacity(self, usage: FrozenUsage) -> None:
        """
        Consume capacity unconditionally.

        Capacity may go negative by design (speedometer pattern); this tracks
        overshoot rather than blocking.
        """
        validate_backend_usage(usage, self._usage_metric_names)
        usage = frozendict({metric: float(amount) for metric, amount in usage.items()})
        fresh_start_buckets: list[MemoryBucket] = []

        with self._condition:
            current_time = time.time()
            preconsumption_capacities, fresh_start_buckets = self._get_capacities(
                current_time,
            )
            active_metric_names = {
                metric for metric, _ in preconsumption_capacities
            }
            self._ensure_usage_metrics_are_active(usage, active_metric_names)

            for usage_metric, usage_amount in usage.items():
                for bucket in self._buckets:
                    if bucket.usage_metric != usage_metric:
                        continue
                    if usage_amount > bucket.max_capacity:
                        warnings.warn(
                            f"record_usage value for {usage_metric} ({usage_amount}) exceeds "
                            f"bucket max capacity ({bucket.max_capacity}). "
                            f"Capacity will go deeply negative.",
                            RuntimeWarning,
                            stacklevel=2,
                        )

            postconsumption_dict: dict[tuple[str, int], float] = dict(
                preconsumption_capacities
            )
            for (
                cap_metric,
                per_seconds,
            ), cap_amount in preconsumption_capacities.items():
                usage_amount = usage.get(cap_metric)
                if usage_amount is None:
                    continue
                postconsumption_dict[(cap_metric, per_seconds)] = (
                    cap_amount - usage_amount
                )
            postconsumption_capacities = frozendict(postconsumption_dict)
            self._set_capacities(
                postconsumption_capacities,
                current_time,
                allow_negative=True,
            )

        # Callbacks fired outside the lock
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
        validate_backend_usage(usage, self._usage_metric_names)
        timeout = validate_timeout(timeout)
        usage = frozendict({metric: float(amount) for metric, amount in usage.items()})
        deadline = None if timeout is None else time.monotonic() + timeout
        has_waited = False
        first_failed_pre: Capacities = frozendict()
        wait_started_at: float | None = None
        wait_start_callback_overhead = 0.0
        postconsumption: Capacities = frozendict()
        fresh: list[MemoryBucket] = []
        preconsumption: Capacities = frozendict()
        current_time = time.time()
        consumed_monotonic = time.monotonic()

        while True:
            should_fire_wait_start = False
            with self._condition:
                while True:
                    current_time = time.time()
                    preconsumption, fresh = self._get_capacities(current_time)
                    ok, postconsumption = self._try_consume_locked(
                        usage,
                        preconsumption,
                        current_time,
                    )
                    if ok:
                        consumed_monotonic = time.monotonic()
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for capacity")
                    if not has_waited:
                        has_waited = True
                        first_failed_pre = preconsumption
                        wait_started_at = time.monotonic()
                        should_fire_wait_start = True
                        break
                    computed = self._compute_sleep(usage, preconsumption)
                    if deadline is not None:
                        computed = min(computed, max(0, deadline - time.monotonic()))
                    self._condition.wait(timeout=max(0.001, computed))
                if ok:
                    break

            if (
                should_fire_wait_start
                and self._callbacks
                and self._callbacks.on_wait_start
            ):
                callback_started = time.monotonic()
                self._invoke_callback_safe(
                    self._callbacks.on_wait_start,
                    model_family=self._limit_config.get_model_family(),
                    preconsumption_capacities=first_failed_pre,
                    usage=usage,
                )
                wait_start_callback_overhead += time.monotonic() - callback_started

        # All callbacks fired outside the lock
        self._fresh_start_buckets_callback(fresh)
        if self._callbacks and self._callbacks.on_capacity_consumed:
            self._invoke_callback_safe(
                self._callbacks.on_capacity_consumed,
                model_family=self._limit_config.get_model_family(),
                preconsumption_capacities=preconsumption,
                postconsumption_capacities=postconsumption,
                usage=usage,
                current_time=current_time,
            )
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
                model_family=self._limit_config.get_model_family(),
                preconsumption_capacities=preconsumption,
                postconsumption_capacities=postconsumption,
                usage=frozendict(usage),
                wait_time_s=wait_time_s,
            )

    def _compute_sleep(self, usage: FrozenUsage, preconsumption: Capacities) -> float:
        """Compute max wait across all buckets based on deficit / rate."""
        max_wait = 0.0
        for (metric, per_seconds), current_cap in preconsumption.items():
            if metric not in usage:
                continue
            needed = usage[metric]
            deficit = needed - current_cap
            if deficit <= 0:
                continue
            bucket = next(
                (
                    b
                    for b in self._buckets
                    if b.usage_metric == metric and b.per_seconds == per_seconds
                ),
                None,
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
        """
        Refund unused capacity back to the rate limiter based on actual usage.

        Handles both positive refunds (used less than reserved) and negative
        refunds (used more than reserved, i.e. overuse).
        """
        backend_bucket_ids = self._bucket_ids()
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
        # Calculate refund amounts per metric
        refund_usage_: dict[str, float] = {}
        for metric, reserved_amount in reserved_usage.items():
            # Key guaranteed to exist: SyncRateLimiter.refund_capacity() calls
            # validate_refund_usage() before reaching the backend.
            actual_amount = actual_usage[metric]
            refund_amount = float(reserved_amount) - float(actual_amount)

            if refund_amount < 0:
                warnings.warn(
                    f"Actual usage ({actual_amount}) for {metric} exceeds "
                    f"reserved usage ({reserved_amount}). Applying negative refund.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            refund_usage_[metric] = refund_amount
        refund_usage: frozendict[str, float] = frozendict(refund_usage_)

        with self._condition:
            current_time = time.time()
            prerefund_capacities, fresh_start_buckets = self._get_capacities(
                current_time,
            )

            # Apply refund amounts to current capacity
            updated_capacities_: dict[tuple[str, int], float] = dict(
                prerefund_capacities,
            )
            for cap_metric, per_seconds in prerefund_capacities:
                bucket_id = (cap_metric, int(per_seconds))
                if bucket_id not in refund_bucket_ids:
                    continue
                refund_amount = refund_usage.get(cap_metric)
                if refund_amount is None:
                    continue
                bucket = next(
                    (
                        b
                        for b in self._buckets
                        if b.usage_metric == cap_metric and b.per_seconds == per_seconds
                    ),
                    None,
                )
                if bucket is None:  # pragma: no cover
                    raise ValueError(f"Bucket '{cap_metric}/{per_seconds}s' not found")
                updated_capacities_[(cap_metric, int(per_seconds))] = min(
                    updated_capacities_[(cap_metric, int(per_seconds))] + refund_amount,
                    bucket.max_capacity,
                )
            updated_capacities = frozendict(updated_capacities_)

            self._set_capacities(updated_capacities, current_time, allow_negative=True)
            self._condition.notify_all()

        # Callbacks fired outside the lock
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

    def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        bucket = next(
            (
                b
                for b in self._buckets
                if b.usage_metric == metric and int(b.per_seconds) == per_seconds
            ),
            None,
        )
        if bucket is None:
            raise ValueError(f"Bucket '{metric}/{per_seconds}s' not found")
        with self._condition:
            bucket.set_max_capacity(value)
            self._condition.notify_all()

    def prepare_reconfigured_backend(
        self,
        new_backend: SyncRateLimiterBackend,
        cfg: PerModelConfig,
    ) -> SyncRateLimiterBackend:
        if not isinstance(new_backend, SyncMemoryBackend):
            raise TypeError(
                "SyncMemoryBackend can only reconfigure into another SyncMemoryBackend"
            )

        existing_buckets = dict(
            getattr(self, "_bucket_registry", {})
        ) or {
            (bucket.usage_metric, int(bucket.per_seconds)): bucket
            for bucket in self._buckets
        }

        with self._condition:
            prepared_buckets: list[MemoryBucket] = []
            for quota in cfg.quotas:
                bucket_id = (quota.metric, int(quota.per_seconds))
                bucket = existing_buckets.get(bucket_id)
                if bucket is None:
                    bucket = MemoryBucket(
                        metric=quota.metric,
                        per_seconds=quota.per_seconds,
                        limit=float(quota.limit),
                        model_family=cfg.get_model_family(),
                    )
                else:
                    bucket.set_max_capacity(float(quota.limit))
                prepared_buckets.append(bucket)
                existing_buckets[bucket_id] = bucket

            self._bucket_registry = existing_buckets

            self.install_reconfigured_state(
                condition=self._condition,
                buckets=prepared_buckets,
                cfg=cfg,
            )
            self._condition.notify_all()

        return self

    def install_reconfigured_state(
        self,
        *,
        condition: threading.Condition,
        buckets: list[MemoryBucket],
        cfg: PerModelConfig,
    ) -> None:
        self._condition = condition
        self._buckets = buckets
        self._usage_metric_names = {bucket.usage_metric for bucket in buckets}
        if not hasattr(self, "_bucket_registry"):
            self._bucket_registry = {}
        self._bucket_registry.update(
            {
                (bucket.usage_metric, int(bucket.per_seconds)): bucket
                for bucket in buckets
            }
        )
        self._limit_config = cfg

    def _invoke_callback_safe(self, callback, **kwargs) -> None:
        """Fire a user callback, suppressing exceptions to prevent capacity leaks."""
        try:
            callback(**kwargs)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Rate limiter callback raised {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )

    def _fresh_start_buckets_callback(
        self,
        fresh_start_buckets: list[MemoryBucket],
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
