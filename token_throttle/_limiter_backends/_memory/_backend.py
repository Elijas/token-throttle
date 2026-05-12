import asyncio
import contextlib
import logging
import math
import time
import warnings
from typing import ClassVar

from frozendict import frozendict

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
    validate_sleep_interval,
    validate_timeout,
)

from ._bucket import MemoryBucket

_logger = logging.getLogger("token_throttle")


class MemoryBackendBuilder(RateLimiterBackendBuilderInterface):
    def __init__(
        self,
        *,
        sleep_interval: float | None = None,
    ) -> None:
        super().__init__()
        self._sleep_interval = validate_sleep_interval(sleep_interval)

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> "RateLimiterBackend":
        buckets = []
        for quota in cfg.quotas:
            b = MemoryBucket(
                metric=quota.metric,
                per_seconds=quota.per_seconds,
                limit=float(quota.limit),
                model_family=cfg.get_model_family(),
            )
            buckets.append(b)
        return MemoryBackend(
            buckets=buckets,
            sleep_interval=self._sleep_interval,
            callbacks=callbacks,
            limit_config=cfg,
        )


class MemoryBackend(RateLimiterBackend):
    DEFAULT_SLEEP_INTERVAL: ClassVar[float] = 0.1
    MAX_CROSS_WORKER_POLL: ClassVar[float] = 1.0

    def __init__(
        self,
        buckets: list[MemoryBucket],
        limit_config: PerModelConfig,
        *,
        sleep_interval: float | None = None,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> None:
        super().__init__()
        self._buckets = buckets
        self._condition = asyncio.Condition()  # Lazily binds to event loop on first use (3.10+), safe before loop starts. Audited 2026-04.
        self._sleep_interval: float = (
            self.DEFAULT_SLEEP_INTERVAL
            if sleep_interval is None
            else validate_sleep_interval(sleep_interval)
        )
        self._callbacks = callbacks
        self._limit_config = limit_config
        self._usage_metric_names: set[str] = {bucket.usage_metric for bucket in buckets}
        self._bucket_registry: dict[tuple[str, int], MemoryBucket] = {
            (bucket.usage_metric, int(bucket.per_seconds)): bucket for bucket in buckets
        }

    def supports_metric_set_change(self) -> bool:
        return True

    def _get_capacities(
        self,
        current_time: float,
        *,
        buckets: list[MemoryBucket] | None = None,
    ) -> tuple[Capacities, list[MemoryBucket]]:
        """Get capacities for all buckets. Must be called under lock."""
        target = self._buckets if buckets is None else buckets
        caps: dict[tuple[str, int], float] = {}
        fresh_start_buckets: list[MemoryBucket] = []
        for bucket in target:
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
        buckets: list[MemoryBucket] | None = None,
    ) -> None:
        """
        Set capacities for all buckets. Must be called under lock.

        allow_negative=True is required for consume_capacity (speedometer)
        and refund_capacity (preserves negative debt for natural refill).
        """
        # O(N*M) linear scan per bucket -- acceptable because N (buckets)
        # and M (usage metrics) are typically 2-5 in practice.
        target = self._buckets if buckets is None else buckets
        for (usage_metric, per_seconds), amount in new_capacities.items():
            bucket = next(
                (
                    b
                    for b in target
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

    async def consume_capacity(self, usage: FrozenUsage) -> None:
        """
        Consume capacity unconditionally.

        Capacity may go negative by design (speedometer pattern); this tracks
        overshoot rather than blocking.
        """
        validate_backend_usage(usage, self._usage_metric_names)
        usage = frozendict({metric: float(amount) for metric, amount in usage.items()})
        fresh_start_buckets: list[MemoryBucket] = []

        async with self._condition:
            current_time = time.time()
            preconsumption_capacities, fresh_start_buckets = self._get_capacities(
                current_time,
            )
            active_metric_names = {metric for metric, _ in preconsumption_capacities}
            self._ensure_usage_metrics_are_active(usage, active_metric_names)

            # stacklevel=2 points to the backend caller, not the user's code.
            # The correct user-facing level varies by call path (3-5 frames up)
            # and isn't worth computing dynamically for a non-fatal warning.
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

            max_cap = {
                (b.usage_metric, b.per_seconds): b.max_capacity for b in self._buckets
            }
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
                postconsumption_dict[(cap_metric, per_seconds)] = max(
                    cap_amount - usage_amount,
                    -max_cap[(cap_metric, per_seconds)],
                )
            postconsumption_capacities = frozendict(postconsumption_dict)
            self._set_capacities(
                postconsumption_capacities,
                current_time,
                allow_negative=True,
            )

        # Callbacks fired outside the lock. Consumption has already been
        # durably recorded in bucket state above, so if a CancelledError
        # arrives during callbacks, we let it propagate: the caller
        # (e.g. asyncio.timeout) must be informed of the cancel, and
        # bucket state is already correct (speedometer is advanced).
        # Callbacks themselves are best-effort via _invoke_callback_safe.
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
        consumed_buckets: list[MemoryBucket] | None = None

        while True:
            should_fire_wait_start = False
            async with self._condition:
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
                        consumed_buckets = list(self._buckets)
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for capacity")
                    if not has_waited:
                        has_waited = True
                        first_failed_pre = preconsumption
                        wait_started_at = time.monotonic()
                        should_fire_wait_start = True
                        break
                    computed = min(
                        self._compute_sleep(usage, preconsumption),
                        self.MAX_CROSS_WORKER_POLL,
                    )
                    if deadline is not None:
                        computed = min(computed, max(0, deadline - time.monotonic()))
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=max(0.001, computed),
                        )
                if ok:
                    break

            if (
                should_fire_wait_start
                and self._callbacks
                and self._callbacks.on_wait_start
            ):
                callback_started = time.monotonic()
                if deadline is None:
                    await self._invoke_callback_safe(
                        self._callbacks.on_wait_start,
                        model_family=self._limit_config.get_model_family(),
                        preconsumption_capacities=first_failed_pre,
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
                            preconsumption_capacities=first_failed_pre,
                            usage=usage,
                        ),
                        timeout=remaining,
                    )
                wait_start_callback_overhead += time.monotonic() - callback_started
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for capacity")

        # All callbacks fired outside the lock.  If CancelledError arrives
        # during any callback await, refund the consumed capacity so it is
        # not permanently lost (the caller never receives a reservation).
        try:
            await self._fresh_start_buckets_callback(fresh)
            if self._callbacks and self._callbacks.on_capacity_consumed:
                await self._invoke_callback_safe(
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
                await self._refund_cancelled_consumption(
                    usage, buckets=consumed_buckets
                )
            except BaseException:  # noqa: BLE001, S110
                # Best-effort refund: asyncio.shield() inside the refund
                # ensures the coroutine runs to completion even under
                # re-cancel.  BaseException (not just Exception) is caught
                # because the shielded refund itself may raise CancelledError
                # or unexpected errors from the condition lock.  Safe to
                # swallow: the original CancelledError is re-raised below,
                # preserving structured-concurrency (TaskGroup) semantics.
                pass
            raise

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
            rate_per_sec = bucket._rate_per_sec  # noqa: SLF001
            if not math.isfinite(rate_per_sec) or rate_per_sec <= 0:
                raise ValueError(
                    "Bucket rate is non-positive/non-finite — likely a "
                    "misconfigured max_capacity"
                )
            wait = deficit / rate_per_sec
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
            # Key guaranteed to exist: RateLimiter.refund_capacity() calls
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

        async with self._condition:
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
                refund_amount = max(refund_amount, -bucket.max_capacity)
                updated_capacities_[(cap_metric, int(per_seconds))] = max(
                    -bucket.max_capacity,
                    min(
                        updated_capacities_[(cap_metric, int(per_seconds))]
                        + refund_amount,
                        bucket.max_capacity,
                    ),
                )
            updated_capacities = frozendict(updated_capacities_)

            self._set_capacities(updated_capacities, current_time, allow_negative=True)
            self._condition.notify_all()

        # Callbacks fired outside the lock
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
        async with self._condition:
            bucket.set_max_capacity(value, time.time())
            self._condition.notify_all()

    async def prepare_reconfigured_backend(
        self,
        new_backend: RateLimiterBackend,
        cfg: PerModelConfig,
    ) -> RateLimiterBackend:
        if not isinstance(new_backend, MemoryBackend):
            raise TypeError(
                "MemoryBackend can only reconfigure into another MemoryBackend"
            )

        async with self._condition:
            if hasattr(self, "_bucket_registry"):
                existing_buckets = dict(self._bucket_registry)
            else:
                existing_buckets = {
                    (bucket.usage_metric, int(bucket.per_seconds)): bucket
                    for bucket in self._buckets
                }
            current_time = time.time()
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
                    bucket.set_max_capacity(float(quota.limit), current_time)
                prepared_buckets.append(bucket)
                existing_buckets[bucket_id] = bucket

            active_ids = {(q.metric, int(q.per_seconds)) for q in cfg.quotas}
            self._bucket_registry = {
                k: v for k, v in existing_buckets.items() if k in active_ids
            }

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
        condition: asyncio.Condition,
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

    async def _invoke_callback_safe(self, callback, **kwargs) -> None:
        """
        Fire a user callback, suppressing exceptions to prevent capacity leaks.

        Audited 2026-05 (R4 L03): exception ladder verified parity-clean across
        all 4 _invoke_callback_safe implementations.
        Audited 2026-05 (R4 L12:C03): non-group exotic exceptions are swallowed
        or propagated by design; warning filters must not reopen the leak path.
        """
        try:
            await callback(**kwargs)
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:  # noqa: BLE001
            msg = f"Rate limiter callback raised {type(exc).__name__}: {exc}"
            with contextlib.suppress(Warning):
                warnings.warn(msg, RuntimeWarning, stacklevel=3)
            _logger.warning(msg)

    async def _refund_cancelled_consumption(
        self,
        usage: FrozenUsage,
        *,
        buckets: list[MemoryBucket] | None = None,
    ) -> None:
        """
        Refund capacity consumed before a CancelledError hit callbacks.

        Uses asyncio.shield() because the refund must complete even if the
        task is re-cancelled (e.g. in structured concurrency / TaskGroups).
        Fires no callbacks to avoid recursion and another cancellation window.

        ``buckets`` should be the snapshot captured at consumption time so
        that a concurrent reconfiguration cannot redirect the refund to a
        different bucket set.
        """
        target_buckets = self._buckets if buckets is None else buckets

        async def _do_refund() -> None:
            async with self._condition:
                current_time = time.time()
                capacities, _ = self._get_capacities(
                    current_time, buckets=target_buckets
                )
                refunded: dict[tuple[str, int], float] = dict(capacities)
                for (cap_metric, per_seconds), cap_amount in capacities.items():
                    for usage_metric, usage_amount in usage.items():
                        if cap_metric != usage_metric:
                            continue
                        bucket = next(
                            (
                                b
                                for b in target_buckets
                                if b.usage_metric == cap_metric
                                and b.per_seconds == per_seconds
                            ),
                            None,
                        )
                        if bucket is None:
                            continue
                        refunded[(cap_metric, per_seconds)] = min(
                            cap_amount + usage_amount,
                            bucket.max_capacity,
                        )
                self._set_capacities(
                    frozendict(refunded),
                    current_time,
                    allow_negative=True,
                    buckets=target_buckets,
                )
                self._condition.notify_all()

        await asyncio.shield(_do_refund())

    async def _fresh_start_buckets_callback(
        self,
        fresh_start_buckets: list[MemoryBucket],
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
