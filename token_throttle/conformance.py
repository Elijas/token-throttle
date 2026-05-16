from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import inspect
import math
import time
import uuid
import warnings
from typing import TYPE_CHECKING, cast

from token_throttle._exceptions import (
    AcquireRefundFailedError,
    BackendConformanceError,
    DuplicateRefundError,
    UnknownReservationError,
)
from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    RateLimiterBackend,
    RateLimiterBackendBuilderInterface,
    SyncRateLimiterBackend,
    SyncRateLimiterBackendBuilderInterface,
    backend_uses_default_prepare_reconfigured_backend,
    backend_uses_default_refund_capacity_for_buckets,
    sync_backend_uses_default_prepare_reconfigured_backend,
    sync_backend_uses_default_refund_capacity_for_buckets,
)
from token_throttle._interfaces._models import (
    BucketId,
    CapacityReservation,
    FrozenUsage,
    Quota,
    UsageQuotas,
    frozen_usage,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_BUILDER_DEADLINE_SECONDS = 5.0
_OPERATION_DEADLINE_SECONDS = 10.0
_PROMPT_DEADLINE_SECONDS = 1.0
_WAIT_BUDGET_SECONDS = 5.0
_WAIT_TIMEOUT_SECONDS = 2.0
_TRY_ACQUIRE_TIMEOUT_SECONDS = 0.0
_SHORT_WINDOW_SECONDS = 1
_FAST_LIMIT = 10.0
_CALLBACK_LIMIT = 4.0
_RESERVATION_LIFETIME_SECONDS = 30.0
_REQUESTS_BUCKET_ID = ("requests", _SHORT_WINDOW_SECONDS)
_TOKENS_BUCKET_ID = ("tokens", _SHORT_WINDOW_SECONDS)
_PUBLIC_MODEL_NAME = "conformance-model"
_RETURN_TIMESTAMP_SKEW_SECONDS = 24 * 60 * 60


def _family(label: str) -> str:
    return f"conformance/{label}/{uuid.uuid4().hex[:12]}"


def _config(
    label: str,
    *,
    limit: float = _FAST_LIMIT,
    extra_quotas: tuple[Quota, ...] = (),
) -> PerModelConfig:
    quotas = [
        Quota(
            metric="requests",
            limit=limit,
            per_seconds=_SHORT_WINDOW_SECONDS,
        )
    ]
    quotas.extend(extra_quotas)
    return PerModelConfig(
        model_family=_family(label),
        quotas=UsageQuotas(quotas),
    )


def _two_metric_config(label: str, *, limit: float = 2.0) -> PerModelConfig:
    return _config(
        label,
        limit=limit,
        extra_quotas=(
            Quota(
                metric="tokens",
                limit=limit,
                per_seconds=_SHORT_WINDOW_SECONDS,
            ),
        ),
    )


def _fail(message: str) -> None:
    raise BackendConformanceError(message)


def _check(condition: object, message: str) -> None:
    if not condition:
        _fail(message)


def _matches_allowed_exception(
    exc: BaseException,
    allowed_exceptions: tuple[type[BaseException], ...],
) -> bool:
    return bool(allowed_exceptions) and isinstance(exc, allowed_exceptions)


def _run_sync_step(
    label: str,
    callable_: Callable[[], object],
    *,
    deadline: float,
    allowed_exceptions: tuple[type[BaseException], ...] = (),
) -> object:
    """
    Run a synchronous callable under a wall-clock deadline; normalize exceptions.

    KNOWN LIMITATION: Python cannot safely kill an in-process thread that
    ignores its task. A timeout reports the hang and shuts down the executor
    without waiting, but the worker thread may continue running.
    """
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="token-throttle-conformance",
    )
    future = executor.submit(callable_)
    timed_out = False
    try:
        return future.result(timeout=deadline)
    except TimeoutError as exc:
        if future.done():
            if _matches_allowed_exception(exc, allowed_exceptions):
                raise
            raise BackendConformanceError(
                f"{label} raised {type(exc).__name__}: {exc}"
            ) from exc
        timed_out = True
        message = (
            f"{label} did not return within {deadline}s; thread may still be "
            "running (cannot kill in-process)"
        )
        warnings.warn(message, RuntimeWarning)
        raise BackendConformanceError(message) from exc
    except BackendConformanceError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        if _matches_allowed_exception(exc, allowed_exceptions):
            raise
        raise BackendConformanceError(
            f"{label} raised {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        executor.shutdown(wait=not timed_out, cancel_futures=True)


def _task_is_being_cancelled() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _consume_abandoned_task_result(task: asyncio.Future[object]) -> None:
    with contextlib.suppress(BaseException):
        task.result()


async def _run_async_step(
    label: str,
    awaitable_or_coro_fn: Awaitable[object] | Callable[[], object] | object,
    *,
    deadline: float,
    expect_awaitable: bool = False,
    allowed_exceptions: tuple[type[BaseException], ...] = (),
) -> object:
    """Run an awaitable under a wall-clock deadline; normalize exceptions."""
    try:
        value: object
        if inspect.isawaitable(awaitable_or_coro_fn):
            value = awaitable_or_coro_fn
        elif callable(awaitable_or_coro_fn):
            value = _run_sync_step(
                label,
                awaitable_or_coro_fn,
                deadline=deadline,
                allowed_exceptions=allowed_exceptions,
            )
        else:
            value = awaitable_or_coro_fn

        if not inspect.isawaitable(value):
            if expect_awaitable:
                _fail(f"{label} returned non-awaitable {type(value).__name__}")
            return value

        task = asyncio.ensure_future(cast("Awaitable[object]", value))
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=deadline)
        except TimeoutError as exc:
            if task.done() and not task.cancelled():
                if _matches_allowed_exception(exc, allowed_exceptions):
                    raise
                raise BackendConformanceError(
                    f"{label} raised {type(exc).__name__}: {exc}"
                ) from exc
            task.add_done_callback(_consume_abandoned_task_result)
            task.cancel()
            raise BackendConformanceError(
                f"{label} did not return within {deadline}s"
            ) from exc
    except BackendConformanceError:
        raise
    except asyncio.CancelledError as exc:
        if _task_is_being_cancelled():
            raise
        raise BackendConformanceError(
            f"{label} raised {type(exc).__name__}: {exc}"
        ) from exc
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        if _matches_allowed_exception(exc, allowed_exceptions):
            raise
        raise BackendConformanceError(
            f"{label} raised {type(exc).__name__}: {exc}"
        ) from exc


def _close_awaitable(value: object) -> None:
    if not inspect.isawaitable(value):
        return
    close = getattr(value, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


def _check_bool_claim(value: object, method_name: str) -> bool:
    if inspect.isawaitable(value):
        _close_awaitable(value)
        _fail(f"{method_name}() must be synchronous and return bool")
    if type(value) is not bool:
        _fail(f"{method_name}() must return bool, got {type(value).__name__}")
    return cast("bool", value)


def _validate_capacity_op_return(
    label: str,
    value: object,
    *,
    allow_timestamp: bool,
) -> None:
    if value is None:
        return
    if not allow_timestamp:
        _fail(f"{label} must return None, got {type(value).__name__}")
    if type(value) not in {int, float}:
        _fail(
            f"{label} must return None or a finite timestamp, "
            f"got {type(value).__name__}"
        )
    timestamp = float(cast("float | int", value))
    if not math.isfinite(timestamp):
        _fail(f"{label} returned a non-finite timestamp")
    now = time.time()
    if timestamp > now + _RETURN_TIMESTAMP_SKEW_SECONDS:
        _fail(f"{label} returned a timestamp more than 24h in the future")
    if timestamp < now - _RETURN_TIMESTAMP_SKEW_SECONDS:
        _fail(f"{label} returned a timestamp more than 24h in the past")


def _validate_backend_step_return(label: str, value: object) -> None:
    if label.startswith(
        (
            "await_for_capacity(",
            "wait_for_capacity(",
            "consume_capacity(",
        )
    ):
        _validate_capacity_op_return(label, value, allow_timestamp=True)
    elif label.startswith(
        (
            "refund_capacity(",
            "set_max_capacity(",
            "apply_configured_max_capacity(",
        )
    ):
        _validate_capacity_op_return(label, value, allow_timestamp=False)


def _check_sync_backend_result_not_awaitable(label: str, value: object) -> None:
    if inspect.isawaitable(value):
        _close_awaitable(value)
        method_name = label.split("(", 1)[0]
        _fail(
            f"{method_name}() returned an awaitable; sync methods must be synchronous"
        )


def _check_runtime_protocols(
    backend_builder: object,
    backend: object,
    *,
    sync: bool,
) -> None:
    if sync:
        _check(
            isinstance(backend_builder, SyncRateLimiterBackendBuilderInterface),
            "sync backend builder does not satisfy SyncRateLimiterBackendBuilderInterface",
        )
        _check(
            isinstance(backend, SyncRateLimiterBackend),
            "sync backend does not satisfy SyncRateLimiterBackend",
        )
        return

    _check(
        isinstance(backend_builder, RateLimiterBackendBuilderInterface),
        "async backend builder does not satisfy RateLimiterBackendBuilderInterface",
    )
    _check(
        isinstance(backend, RateLimiterBackend),
        "async backend does not satisfy RateLimiterBackend",
    )


def _build_async_backend(
    builder: RateLimiterBackendBuilderInterface,
    cfg: PerModelConfig,
    *,
    label: str,
    callbacks: RateLimiterCallbacks | None = None,
) -> RateLimiterBackend:
    backend = _run_sync_step(
        label,
        lambda: builder.build(cfg, callbacks=callbacks),
        deadline=_BUILDER_DEADLINE_SECONDS,
    )
    _check_runtime_protocols(builder, backend, sync=False)
    return cast("RateLimiterBackend", backend)


def _build_sync_backend(
    builder: SyncRateLimiterBackendBuilderInterface,
    cfg: PerModelConfig,
    *,
    label: str,
    callbacks: SyncRateLimiterCallbacks | None = None,
) -> SyncRateLimiterBackend:
    backend = _run_sync_step(
        label,
        lambda: builder.build(cfg, callbacks=callbacks),
        deadline=_BUILDER_DEADLINE_SECONDS,
    )
    _check_runtime_protocols(builder, backend, sync=True)
    return cast("SyncRateLimiterBackend", backend)


async def _async_backend_step(
    label: str,
    awaitable_fn: Callable[[], object],
    *,
    deadline: float | None = None,
    allowed_exceptions: tuple[type[BaseException], ...] = (),
) -> object:
    result = await _run_async_step(
        label,
        awaitable_fn,
        deadline=_OPERATION_DEADLINE_SECONDS if deadline is None else deadline,
        expect_awaitable=True,
        allowed_exceptions=allowed_exceptions,
    )
    _validate_backend_step_return(label, result)
    return result


def _sync_backend_step(
    label: str,
    callable_: Callable[[], object],
    *,
    deadline: float | None = None,
    allowed_exceptions: tuple[type[BaseException], ...] = (),
) -> object:
    result = _run_sync_step(
        label,
        callable_,
        deadline=_OPERATION_DEADLINE_SECONDS if deadline is None else deadline,
        allowed_exceptions=allowed_exceptions,
    )
    _check_sync_backend_result_not_awaitable(label, result)
    _validate_backend_step_return(label, result)
    return result


class _AsyncRefundFailureBackend(RateLimiterBackend):
    def __init__(
        self,
        backend: RateLimiterBackend,
        refund_error: RuntimeError,
    ) -> None:
        self._backend = backend
        self._refund_error = refund_error

    async def await_for_capacity(
        self,
        usage: FrozenUsage,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        return await self._backend.await_for_capacity(
            usage,
            timeout=timeout,
            reservation_id=reservation_id,
            reservation_lifetime_seconds=reservation_lifetime_seconds,
        )

    async def consume_capacity(
        self,
        usage: FrozenUsage,
        *,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        return await self._backend.consume_capacity(
            usage,
            reservation_id=reservation_id,
            reservation_lifetime_seconds=reservation_lifetime_seconds,
        )

    async def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        await self._backend.refund_capacity(reserved_usage, actual_usage)

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
        _ = (
            reserved_usage,
            actual_usage,
            bucket_ids,
            reservation_id,
            reservation_model_family,
            reservation_bucket_ids,
            reservation_reserved_usage,
        )
        raise self._refund_error

    def supports_durable_refund_dedup(self) -> bool:
        return self._backend.supports_durable_refund_dedup()

    def supports_acquire_marker_authority(self) -> bool:
        return self._backend.supports_acquire_marker_authority()

    async def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        await self._backend.set_max_capacity(metric, per_seconds, value)

    async def apply_configured_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        await self._backend.apply_configured_max_capacity(metric, per_seconds, value)

    def supports_metric_set_change(self) -> bool:
        return self._backend.supports_metric_set_change()

    async def prepare_reconfigured_backend(
        self,
        new_backend: RateLimiterBackend,
        cfg: PerModelConfig,
    ) -> RateLimiterBackend:
        prepared = await self._backend.prepare_reconfigured_backend(new_backend, cfg)
        if prepared is new_backend:
            return self
        self._backend = prepared
        return self


class _AsyncRefundFailureBuilder(RateLimiterBackendBuilderInterface):
    def __init__(
        self,
        builder: RateLimiterBackendBuilderInterface,
        refund_error: RuntimeError,
    ) -> None:
        self._builder = builder
        self._refund_error = refund_error

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> RateLimiterBackend:
        return _AsyncRefundFailureBackend(
            self._builder.build(cfg, callbacks=callbacks),
            self._refund_error,
        )

    async def aclose(self) -> None:
        await self._builder.aclose()

    def close(self) -> None:
        self._builder.close()


class _SyncRefundFailureBackend(SyncRateLimiterBackend):
    def __init__(
        self,
        backend: SyncRateLimiterBackend,
        refund_error: RuntimeError,
    ) -> None:
        self._backend = backend
        self._refund_error = refund_error

    def wait_for_capacity(
        self,
        usage: FrozenUsage,
        *,
        timeout: float | None = None,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        return self._backend.wait_for_capacity(
            usage,
            timeout=timeout,
            reservation_id=reservation_id,
            reservation_lifetime_seconds=reservation_lifetime_seconds,
        )

    def consume_capacity(
        self,
        usage: FrozenUsage,
        *,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        return self._backend.consume_capacity(
            usage,
            reservation_id=reservation_id,
            reservation_lifetime_seconds=reservation_lifetime_seconds,
        )

    def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        self._backend.refund_capacity(reserved_usage, actual_usage)

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
        _ = (
            reserved_usage,
            actual_usage,
            bucket_ids,
            reservation_id,
            reservation_model_family,
            reservation_bucket_ids,
            reservation_reserved_usage,
        )
        raise self._refund_error

    def supports_durable_refund_dedup(self) -> bool:
        return self._backend.supports_durable_refund_dedup()

    def supports_acquire_marker_authority(self) -> bool:
        return self._backend.supports_acquire_marker_authority()

    def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        self._backend.set_max_capacity(metric, per_seconds, value)

    def apply_configured_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        self._backend.apply_configured_max_capacity(metric, per_seconds, value)

    def supports_metric_set_change(self) -> bool:
        return self._backend.supports_metric_set_change()

    def prepare_reconfigured_backend(
        self,
        new_backend: SyncRateLimiterBackend,
        cfg: PerModelConfig,
    ) -> SyncRateLimiterBackend:
        prepared = self._backend.prepare_reconfigured_backend(new_backend, cfg)
        if prepared is new_backend:
            return self
        self._backend = prepared
        return self


class _SyncRefundFailureBuilder(SyncRateLimiterBackendBuilderInterface):
    def __init__(
        self,
        builder: SyncRateLimiterBackendBuilderInterface,
        refund_error: RuntimeError,
    ) -> None:
        self._builder = builder
        self._refund_error = refund_error

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> SyncRateLimiterBackend:
        return _SyncRefundFailureBackend(
            self._builder.build(cfg, callbacks=callbacks),
            self._refund_error,
        )

    def close(self) -> None:
        self._builder.close()


class _SyncAcquireInterrupted(BaseException):
    """Internal sync control-flow exception for the FIX-50 probe."""


def _check_async_claims(backend: RateLimiterBackend) -> None:
    marker_authority = _check_bool_claim(
        _run_sync_step(
            "supports_acquire_marker_authority()",
            backend.supports_acquire_marker_authority,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_acquire_marker_authority",
    )
    durable_dedup = _check_bool_claim(
        _run_sync_step(
            "supports_durable_refund_dedup()",
            backend.supports_durable_refund_dedup,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_durable_refund_dedup",
    )
    metric_set_change = _check_bool_claim(
        _run_sync_step(
            "supports_metric_set_change()",
            backend.supports_metric_set_change,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_metric_set_change",
    )

    if marker_authority and backend_uses_default_refund_capacity_for_buckets(backend):
        _fail(
            "supports_acquire_marker_authority=True requires an override of "
            "refund_capacity_for_buckets()"
        )
    if durable_dedup and backend_uses_default_refund_capacity_for_buckets(backend):
        _fail(
            "supports_durable_refund_dedup=True requires an override of "
            "refund_capacity_for_buckets()"
        )
    if metric_set_change and backend_uses_default_prepare_reconfigured_backend(backend):
        _fail(
            "supports_metric_set_change=True requires an override of "
            "prepare_reconfigured_backend()"
        )


def _check_sync_claims(backend: SyncRateLimiterBackend) -> None:
    marker_authority = _check_bool_claim(
        _run_sync_step(
            "supports_acquire_marker_authority()",
            backend.supports_acquire_marker_authority,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_acquire_marker_authority",
    )
    durable_dedup = _check_bool_claim(
        _run_sync_step(
            "supports_durable_refund_dedup()",
            backend.supports_durable_refund_dedup,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_durable_refund_dedup",
    )
    metric_set_change = _check_bool_claim(
        _run_sync_step(
            "supports_metric_set_change()",
            backend.supports_metric_set_change,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_metric_set_change",
    )

    if marker_authority and sync_backend_uses_default_refund_capacity_for_buckets(
        backend
    ):
        _fail(
            "supports_acquire_marker_authority=True requires an override of "
            "refund_capacity_for_buckets()"
        )
    if durable_dedup and sync_backend_uses_default_refund_capacity_for_buckets(backend):
        _fail(
            "supports_durable_refund_dedup=True requires an override of "
            "refund_capacity_for_buckets()"
        )
    if metric_set_change and sync_backend_uses_default_prepare_reconfigured_backend(
        backend
    ):
        _fail(
            "supports_metric_set_change=True requires an override of "
            "prepare_reconfigured_backend()"
        )


async def _expect_async_value_error(
    label: str,
    awaitable_fn: Callable[[], object],
    message: str,
) -> None:
    try:
        await _run_async_step(
            label,
            awaitable_fn,
            deadline=_OPERATION_DEADLINE_SECONDS,
            expect_awaitable=True,
            allowed_exceptions=(ValueError,),
        )
    except ValueError:
        return
    _fail(message)


async def _expect_async_timeout(
    label: str,
    awaitable_fn: Callable[[], object],
    message: str,
    *,
    promptness_deadline: float = _PROMPT_DEADLINE_SECONDS,
) -> None:
    start = time.monotonic()
    try:
        await _run_async_step(
            label,
            awaitable_fn,
            deadline=_WAIT_BUDGET_SECONDS,
            expect_awaitable=True,
            allowed_exceptions=(TimeoutError,),
        )
    except TimeoutError:
        elapsed = time.monotonic() - start
        if elapsed < promptness_deadline:
            return
        _fail(
            f"{label} raised TimeoutError after {elapsed:.2f}s; "
            "expected prompt try-acquire"
        )
    _fail(message)


def _expect_value_error(
    label: str,
    fn: Callable[[], object],
    message: str,
) -> None:
    try:
        _sync_backend_step(
            label,
            fn,
            allowed_exceptions=(ValueError,),
        )
    except ValueError:
        return
    _fail(message)


def _expect_timeout(
    label: str,
    fn: Callable[[], object],
    message: str,
    *,
    promptness_deadline: float = _PROMPT_DEADLINE_SECONDS,
) -> None:
    start = time.monotonic()
    try:
        _sync_backend_step(
            label,
            fn,
            deadline=_WAIT_BUDGET_SECONDS,
            allowed_exceptions=(TimeoutError,),
        )
    except TimeoutError:
        elapsed = time.monotonic() - start
        if elapsed < promptness_deadline:
            return
        _fail(
            f"{label} raised TimeoutError after {elapsed:.2f}s; "
            "expected prompt try-acquire"
        )
    _fail(message)


def _check_public_reservation_fields(
    reservation: object,
    *,
    expected_usage: FrozenUsage,
    expected_model_family: str,
    expected_model: str,
    expected_bucket_ids: frozenset[BucketId],
) -> CapacityReservation:
    _check(
        type(reservation) is CapacityReservation,
        "public limiter acquire must return exact CapacityReservation",
    )
    reservation = cast("CapacityReservation", reservation)
    _check(
        reservation.usage == expected_usage,
        "CapacityReservation.usage did not match acquired usage",
    )
    _check(
        reservation.get_usage() == expected_usage,
        "CapacityReservation.get_usage() did not match acquired usage",
    )
    _check(
        reservation.model_family == expected_model_family,
        "CapacityReservation.model_family did not match limiter config",
    )
    _check(
        reservation.model == expected_model,
        "CapacityReservation.model did not match acquire model",
    )
    _check(
        reservation.bucket_ids == expected_bucket_ids,
        "CapacityReservation.bucket_ids did not match configured quota buckets",
    )
    _check(
        type(reservation.reservation_id) is str and bool(reservation.reservation_id),
        "CapacityReservation.reservation_id must be a non-empty str",
    )
    _check(
        type(reservation.limiter_instance_id) is str
        and bool(reservation.limiter_instance_id),
        "CapacityReservation.limiter_instance_id must be a non-empty str",
    )
    return reservation


def _check_acquire_refund_failed_payload(
    exc: BaseException,
    *,
    refund_error: RuntimeError,
    interrupted_by: BaseException,
) -> None:
    _check(
        isinstance(exc, AcquireRefundFailedError),
        "interrupted acquire cleanup must raise AcquireRefundFailedError",
    )
    exc = cast("AcquireRefundFailedError", exc)
    _check(
        isinstance(exc, Exception),
        "AcquireRefundFailedError must be catchable as Exception",
    )
    _check(
        not isinstance(exc, asyncio.CancelledError),
        "AcquireRefundFailedError must not be an asyncio.CancelledError",
    )
    _check(
        exc.refund_error is refund_error,
        "AcquireRefundFailedError.refund_error must be the backend refund failure",
    )
    _check(
        exc.interrupted_by is interrupted_by,
        "AcquireRefundFailedError.interrupted_by must be the acquire interruption",
    )
    _check(
        exc.__cause__ is refund_error,
        "AcquireRefundFailedError.__cause__ must chain to refund_error",
    )
    _check(
        type(exc.reservation) is CapacityReservation,
        "AcquireRefundFailedError.reservation must be an exact CapacityReservation",
    )


async def _check_async_basic_capacity(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    backend = _build_async_backend(
        builder,
        _config("async-basic"),
        label="build(async-basic)",
    )
    _check_async_claims(backend)

    start = time.monotonic()
    await _async_backend_step(
        "await_for_capacity(requests=1)",
        lambda: backend.await_for_capacity(frozen_usage({"requests": 1})),
    )
    _check(
        time.monotonic() - start < _PROMPT_DEADLINE_SECONDS,
        "await_for_capacity() did not return promptly when capacity was available",
    )

    await _expect_async_value_error(
        "await_for_capacity(requests=-1)",
        lambda: backend.await_for_capacity(frozen_usage({"requests": -1})),
        "await_for_capacity() must reject negative usage",
    )

    exhausted = _build_async_backend(
        builder,
        _config("async-exhaust"),
        label="build(async-exhaust)",
    )
    await _async_backend_step(
        "await_for_capacity(requests=10)",
        lambda: exhausted.await_for_capacity(frozen_usage({"requests": _FAST_LIMIT})),
    )
    await _expect_async_timeout(
        "await_for_capacity(requests=1, timeout=0)",
        lambda: exhausted.await_for_capacity(
            frozen_usage({"requests": 1}), timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS
        ),
        "await_for_capacity(timeout=0) must raise TimeoutError when capacity is unavailable",
    )
    await _async_backend_step(
        "refund_capacity(requests=10, actual=5)",
        lambda: exhausted.refund_capacity(
            frozen_usage({"requests": _FAST_LIMIT}),
            frozen_usage({"requests": _FAST_LIMIT / 2}),
        ),
    )
    start = time.monotonic()
    await _async_backend_step(
        "await_for_capacity(requests=5)",
        lambda: exhausted.await_for_capacity(
            frozen_usage({"requests": _FAST_LIMIT / 2})
        ),
    )
    _check(
        time.monotonic() - start < _PROMPT_DEADLINE_SECONDS,
        "refund_capacity() did not restore unused capacity",
    )


async def _check_async_all_or_nothing(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    backend = _build_async_backend(
        builder,
        _config(
            "async-all-or-nothing",
            limit=1.0,
            extra_quotas=(
                Quota(
                    metric="tokens",
                    limit=1.0,
                    per_seconds=_SHORT_WINDOW_SECONDS,
                ),
            ),
        ),
        label="build(async-all-or-nothing)",
    )
    await _async_backend_step(
        "await_for_capacity(tokens=1, requests=0)",
        lambda: backend.await_for_capacity(frozen_usage({"tokens": 1, "requests": 0})),
    )
    await _expect_async_timeout(
        "await_for_capacity(tokens=1, requests=1, timeout=0)",
        lambda: backend.await_for_capacity(
            frozen_usage({"tokens": 1, "requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "await_for_capacity() must not partially consume when one metric lacks capacity",
    )
    await _async_backend_step(
        "await_for_capacity(tokens=0, requests=1, timeout=0)",
        lambda: backend.await_for_capacity(
            frozen_usage({"tokens": 0, "requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
    )


async def _check_async_refund_and_overuse(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    invalid_refund = _build_async_backend(
        builder,
        _config("async-invalid-refund"),
        label="build(async-invalid-refund)",
    )
    await _async_backend_step(
        "await_for_capacity(requests=1)",
        lambda: invalid_refund.await_for_capacity(frozen_usage({"requests": 1})),
    )
    await _expect_async_value_error(
        "refund_capacity(requests=1, actual=-1)",
        lambda: invalid_refund.refund_capacity(
            frozen_usage({"requests": 1}),
            frozen_usage({"requests": -1}),
        ),
        "refund_capacity() must reject negative actual usage",
    )

    overuse = _build_async_backend(
        builder,
        _config("async-overuse"),
        label="build(async-overuse)",
    )
    await _async_backend_step(
        "await_for_capacity(requests=1)",
        lambda: overuse.await_for_capacity(frozen_usage({"requests": 1})),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await _async_backend_step(
            "refund_capacity(requests=1, actual=2)",
            lambda: overuse.refund_capacity(
                frozen_usage({"requests": 1}),
                frozen_usage({"requests": 2}),
            ),
        )
    _check(
        any(issubclass(item.category, RuntimeWarning) for item in caught),
        "refund_capacity() must warn with RuntimeWarning when actual usage exceeds reserved usage",
    )


async def _check_async_consume_and_capacity_updates(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    consume = _build_async_backend(
        builder,
        _config("async-consume", limit=5.0),
        label="build(async-consume)",
    )
    await _async_backend_step(
        "await_for_capacity(requests=5)",
        lambda: consume.await_for_capacity(frozen_usage({"requests": 5})),
    )
    await _async_backend_step(
        "consume_capacity(requests=5)",
        lambda: consume.consume_capacity(frozen_usage({"requests": 5})),
    )
    await _expect_async_timeout(
        "await_for_capacity(requests=1, timeout=0)",
        lambda: consume.await_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "consume_capacity() must allow debt that later blocking acquires observe",
    )

    capacity = _build_async_backend(
        builder,
        _config("async-max-capacity", limit=5.0),
        label="build(async-max-capacity)",
    )
    await _async_backend_step(
        "apply_configured_max_capacity(requests, 1, 3)",
        lambda: capacity.apply_configured_max_capacity(
            "requests", _SHORT_WINDOW_SECONDS, 3.0
        ),
    )
    await _expect_async_value_error(
        "await_for_capacity(requests=4, timeout=0)",
        lambda: capacity.await_for_capacity(
            frozen_usage({"requests": 4}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "apply_configured_max_capacity() must update the live bucket max_capacity",
    )
    await _async_backend_step(
        "set_max_capacity(requests, 1, 4)",
        lambda: capacity.set_max_capacity("requests", _SHORT_WINDOW_SECONDS, 4.0),
    )
    await _async_backend_step(
        "await_for_capacity(requests=4, timeout=0)",
        lambda: capacity.await_for_capacity(
            frozen_usage({"requests": 4}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
    )


async def _check_async_callbacks(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    events: list[str] = []

    async def on_wait_start(**_kwargs) -> None:
        events.append("wait_start")

    async def after_wait_end_consumption(**_kwargs) -> None:
        events.append("wait_end")

    async def on_capacity_consumed(**_kwargs) -> None:
        events.append("consumed")

    async def on_capacity_refunded(**_kwargs) -> None:
        events.append("refunded")

    async def on_missing_consumption_data(**_kwargs) -> None:
        events.append("missing")

    backend = _build_async_backend(
        builder,
        _config("async-callbacks", limit=_CALLBACK_LIMIT),
        label="build(async-callbacks)",
        callbacks=RateLimiterCallbacks(
            on_wait_start=on_wait_start,
            after_wait_end_consumption=after_wait_end_consumption,
            on_capacity_consumed=on_capacity_consumed,
            on_capacity_refunded=on_capacity_refunded,
            on_missing_consumption_data=on_missing_consumption_data,
        ),
    )
    await _async_backend_step(
        "await_for_capacity(requests=4)",
        lambda: backend.await_for_capacity(frozen_usage({"requests": _CALLBACK_LIMIT})),
    )
    await _async_backend_step(
        "await_for_capacity(requests=1, timeout=2)",
        lambda: backend.await_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_WAIT_TIMEOUT_SECONDS,
        ),
        deadline=_WAIT_BUDGET_SECONDS,
    )
    await _async_backend_step(
        "refund_capacity(requests=1, actual=0)",
        lambda: backend.refund_capacity(
            frozen_usage({"requests": 1}),
            frozen_usage({"requests": 0}),
        ),
    )

    for event in ("missing", "consumed", "wait_start", "wait_end", "refunded"):
        _check(event in events, f"callback event {event!r} was not emitted")


def _metric_set_configs(label: str) -> tuple[PerModelConfig, PerModelConfig]:
    family = _family(label)
    old_cfg = PerModelConfig(
        model_family=family,
        quotas=UsageQuotas(
            [
                Quota(
                    metric="requests",
                    limit=2.0,
                    per_seconds=_SHORT_WINDOW_SECONDS,
                ),
                Quota(
                    metric="tokens",
                    limit=2.0,
                    per_seconds=_SHORT_WINDOW_SECONDS,
                ),
            ]
        ),
    )
    new_cfg = PerModelConfig(
        model_family=family,
        quotas=UsageQuotas(
            [
                Quota(
                    metric="requests",
                    limit=2.0,
                    per_seconds=_SHORT_WINDOW_SECONDS,
                ),
                Quota(
                    metric="images",
                    limit=2.0,
                    per_seconds=_SHORT_WINDOW_SECONDS,
                ),
            ]
        ),
    )
    return old_cfg, new_cfg


async def _check_async_metric_set_change(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    old_cfg, new_cfg = _metric_set_configs("async-metric-set")
    old_backend = _build_async_backend(
        builder,
        old_cfg,
        label="build(async-metric-set-old)",
    )
    if not _check_bool_claim(
        _run_sync_step(
            "supports_metric_set_change()",
            old_backend.supports_metric_set_change,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_metric_set_change",
    ):
        return

    await _async_backend_step(
        "await_for_capacity(requests=2, tokens=0)",
        lambda: old_backend.await_for_capacity(
            frozen_usage({"requests": 2, "tokens": 0})
        ),
    )
    new_backend = _build_async_backend(
        builder,
        new_cfg,
        label="build(async-metric-set-new)",
    )
    prepared = await _async_backend_step(
        "prepare_reconfigured_backend(async-metric-set)",
        lambda: old_backend.prepare_reconfigured_backend(new_backend, new_cfg),
    )
    _check(
        isinstance(prepared, RateLimiterBackend),
        "prepare_reconfigured_backend() must return a RateLimiterBackend",
    )
    prepared_backend = cast("RateLimiterBackend", prepared)
    await _expect_async_timeout(
        "await_for_capacity(requests=1, images=0, timeout=0)",
        lambda: prepared_backend.await_for_capacity(
            frozen_usage({"requests": 1, "images": 0}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "prepare_reconfigured_backend() must preserve surviving bucket consumption",
    )
    await _async_backend_step(
        "await_for_capacity(requests=0, images=1, timeout=0)",
        lambda: prepared_backend.await_for_capacity(
            frozen_usage({"requests": 0, "images": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
    )


async def _check_async_per_build_isolation(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    backend_a = _build_async_backend(
        builder,
        _config("async-isolation-a", limit=2.0),
        label="build(async-isolation-a)",
    )
    await _async_backend_step(
        "await_for_capacity(requests=2)",
        lambda: backend_a.await_for_capacity(frozen_usage({"requests": 2})),
    )
    _build_async_backend(
        builder,
        _config("async-isolation-b", limit=5.0),
        label="build(async-isolation-b)",
    )
    await _expect_async_value_error(
        "await_for_capacity(requests=3, timeout=0)",
        lambda: backend_a.await_for_capacity(
            frozen_usage({"requests": 3}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "building another backend must not change an existing backend's quota limits",
    )
    await _expect_async_timeout(
        "await_for_capacity(requests=1, timeout=0)",
        lambda: backend_a.await_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "building another backend must not reset an existing backend's consumed state",
    )


async def _async_refund_for_buckets(  # noqa: PLR0913
    label: str,
    backend: RateLimiterBackend,
    reserved_usage,
    actual_usage,
    *,
    reservation_id: str,
    reservation_model_family: str,
    reservation_bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
    reservation_reserved_usage=None,
) -> object:
    return await _async_backend_step(
        label,
        lambda: backend.refund_capacity_for_buckets(
            reserved_usage,
            actual_usage,
            bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
            reservation_id=reservation_id,
            reservation_model_family=reservation_model_family,
            reservation_bucket_ids=reservation_bucket_ids,
            reservation_reserved_usage=(
                reserved_usage
                if reservation_reserved_usage is None
                else reservation_reserved_usage
            ),
        ),
        allowed_exceptions=(UnknownReservationError, DuplicateRefundError, ValueError),
    )


async def _check_async_no_double_refund_credit(
    backend: RateLimiterBackend,
    message: str,
) -> None:
    await _expect_async_timeout(
        "await_for_capacity(requests=2, timeout=0)",
        lambda: backend.await_for_capacity(
            frozen_usage({"requests": 2}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        message,
    )


async def _check_async_durable_refund_dedup(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    cfg = _config("async-durable-dedup", limit=2.0)
    backend = _build_async_backend(
        builder,
        cfg,
        label="build(async-durable-dedup)",
    )
    if not _check_bool_claim(
        _run_sync_step(
            "supports_durable_refund_dedup()",
            backend.supports_durable_refund_dedup,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_durable_refund_dedup",
    ):
        return

    reservation_id = f"conformance-{uuid.uuid4().hex}"
    reserved_usage = frozen_usage({"requests": 2})
    await _async_backend_step(
        "await_for_capacity(requests=2, reservation_id=...)",
        lambda: backend.await_for_capacity(
            reserved_usage,
            reservation_id=reservation_id,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        ),
    )
    await _async_refund_for_buckets(
        "refund_capacity_for_buckets(durable-first)",
        backend,
        reserved_usage,
        frozen_usage({"requests": 1}),
        reservation_id=reservation_id,
        reservation_model_family=cfg.get_model_family(),
    )
    try:
        await _async_refund_for_buckets(
            "refund_capacity_for_buckets(durable-duplicate)",
            backend,
            reserved_usage,
            frozen_usage({"requests": 1}),
            reservation_id=reservation_id,
            reservation_model_family=cfg.get_model_family(),
        )
    except DuplicateRefundError:
        return
    await _check_async_no_double_refund_credit(
        backend,
        "supports_durable_refund_dedup=True must not credit duplicate refunds twice",
    )


async def _check_async_marker_authority(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    cfg = _config("async-marker")
    backend = _build_async_backend(
        builder,
        cfg,
        label="build(async-marker)",
    )
    if not _check_bool_claim(
        _run_sync_step(
            "supports_acquire_marker_authority()",
            backend.supports_acquire_marker_authority,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_acquire_marker_authority",
    ):
        return

    reservation_id = f"conformance-{uuid.uuid4().hex}"
    reserved_usage = frozen_usage({"requests": 1})
    await _async_backend_step(
        "await_for_capacity(requests=1, reservation_id=...)",
        lambda: backend.await_for_capacity(
            reserved_usage,
            reservation_id=reservation_id,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        ),
    )
    result = await _async_backend_step(
        "refund_capacity_for_buckets(requests=1, actual=0)",
        lambda: backend.refund_capacity_for_buckets(
            reserved_usage,
            frozen_usage({"requests": 0}),
            bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
            reservation_id=reservation_id,
            reservation_model_family=cfg.get_model_family(),
            reservation_bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
            reservation_reserved_usage=reserved_usage,
        ),
    )
    _check(
        result is True,
        "refund_capacity_for_buckets() must return True after a marker-authorized refund",
    )

    unknown_cfg = _config("async-marker-unknown", limit=2.0)
    unknown_backend = _build_async_backend(
        builder,
        unknown_cfg,
        label="build(async-marker-unknown)",
    )
    await _async_backend_step(
        "await_for_capacity(requests=2)",
        lambda: unknown_backend.await_for_capacity(frozen_usage({"requests": 2})),
    )
    try:
        await _async_refund_for_buckets(
            "refund_capacity_for_buckets(unknown-reservation)",
            unknown_backend,
            frozen_usage({"requests": 2}),
            frozen_usage({"requests": 1}),
            reservation_id=f"unknown-{uuid.uuid4().hex}",
            reservation_model_family=unknown_cfg.get_model_family(),
        )
    except UnknownReservationError:
        pass
    else:
        await _expect_async_timeout(
            "await_for_capacity(requests=1, timeout=0)",
            lambda: unknown_backend.await_for_capacity(
                frozen_usage({"requests": 1}),
                timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
            ),
            "marker-authority backends must fail closed for unknown reservations",
        )

    duplicate_cfg = _config("async-marker-duplicate", limit=2.0)
    duplicate_backend = _build_async_backend(
        builder,
        duplicate_cfg,
        label="build(async-marker-duplicate)",
    )
    duplicate_reservation_id = f"conformance-{uuid.uuid4().hex}"
    duplicate_reserved_usage = frozen_usage({"requests": 2})
    await _async_backend_step(
        "await_for_capacity(requests=2, reservation_id=...)",
        lambda: duplicate_backend.await_for_capacity(
            duplicate_reserved_usage,
            reservation_id=duplicate_reservation_id,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        ),
    )
    await _async_refund_for_buckets(
        "refund_capacity_for_buckets(duplicate-first)",
        duplicate_backend,
        duplicate_reserved_usage,
        frozen_usage({"requests": 1}),
        reservation_id=duplicate_reservation_id,
        reservation_model_family=duplicate_cfg.get_model_family(),
    )
    try:
        await _async_refund_for_buckets(
            "refund_capacity_for_buckets(duplicate-second)",
            duplicate_backend,
            duplicate_reserved_usage,
            frozen_usage({"requests": 1}),
            reservation_id=duplicate_reservation_id,
            reservation_model_family=duplicate_cfg.get_model_family(),
        )
    except DuplicateRefundError:
        pass
    else:
        await _check_async_no_double_refund_credit(
            duplicate_backend,
            "marker-authority backends must not credit duplicate refunds twice",
        )

    mismatch_cfg = _config("async-marker-mismatch", limit=2.0)
    mismatch_backend = _build_async_backend(
        builder,
        mismatch_cfg,
        label="build(async-marker-mismatch)",
    )
    mismatch_reservation_id = f"conformance-{uuid.uuid4().hex}"
    mismatch_reserved_usage = frozen_usage({"requests": 2})
    await _async_backend_step(
        "await_for_capacity(requests=2, reservation_id=...)",
        lambda: mismatch_backend.await_for_capacity(
            mismatch_reserved_usage,
            reservation_id=mismatch_reservation_id,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        ),
    )
    try:
        mismatch_result = await _async_refund_for_buckets(
            "refund_capacity_for_buckets(metadata-mismatch)",
            mismatch_backend,
            mismatch_reserved_usage,
            frozen_usage({"requests": 1}),
            reservation_id=mismatch_reservation_id,
            reservation_model_family=_family("forged-marker-family"),
        )
    except (UnknownReservationError, DuplicateRefundError, ValueError):
        return
    if mismatch_result is False:
        return
    await _expect_async_timeout(
        "await_for_capacity(requests=1, timeout=0)",
        lambda: mismatch_backend.await_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "marker-authority backends must fail closed for reservation metadata mismatch",
    )


async def _check_async_public_reservation_round_trip(
    backend_builder: RateLimiterBackendBuilderInterface,
) -> None:
    cfg = _two_metric_config("async-public-round-trip")
    usage = frozen_usage({"requests": 1.0, "tokens": 1.0})
    full_usage = frozen_usage({"requests": 2.0, "tokens": 2.0})
    zero_usage = frozen_usage({"requests": 0.0, "tokens": 0.0})
    bucket_ids = frozenset({_REQUESTS_BUCKET_ID, _TOKENS_BUCKET_ID})
    limiter = RateLimiter(cfg, backend=backend_builder)
    try:
        reservation = _check_public_reservation_fields(
            await _run_async_step(
                "RateLimiter.acquire_capacity(public reservation)",
                lambda: limiter.acquire_capacity(usage, _PUBLIC_MODEL_NAME),
                deadline=_OPERATION_DEADLINE_SECONDS,
                expect_awaitable=True,
            ),
            expected_usage=usage,
            expected_model_family=cfg.get_model_family(),
            expected_model=_PUBLIC_MODEL_NAME,
            expected_bucket_ids=bucket_ids,
        )
        await _run_async_step(
            "RateLimiter.refund_capacity(public reservation)",
            lambda: limiter.refund_capacity(zero_usage, reservation),
            deadline=_OPERATION_DEADLINE_SECONDS,
            expect_awaitable=True,
        )

        restored = _check_public_reservation_fields(
            await _run_async_step(
                "RateLimiter.acquire_capacity(restored capacity)",
                lambda: limiter.acquire_capacity(
                    full_usage,
                    _PUBLIC_MODEL_NAME,
                    timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
                ),
                deadline=_OPERATION_DEADLINE_SECONDS,
                expect_awaitable=True,
            ),
            expected_usage=full_usage,
            expected_model_family=cfg.get_model_family(),
            expected_model=_PUBLIC_MODEL_NAME,
            expected_bucket_ids=bucket_ids,
        )
        await _run_async_step(
            "RateLimiter.refund_capacity(restored reservation)",
            lambda: limiter.refund_capacity(zero_usage, restored),
            deadline=_OPERATION_DEADLINE_SECONDS,
            expect_awaitable=True,
        )

        mutated = restored.model_copy(update={"usage": zero_usage})
        _check(
            type(mutated) is CapacityReservation,
            "CapacityReservation.model_copy() must preserve exact public type",
        )
        snapshot_probe = _check_public_reservation_fields(
            await _run_async_step(
                "RateLimiter.acquire_capacity(snapshot authority)",
                lambda: limiter.acquire_capacity(
                    full_usage,
                    _PUBLIC_MODEL_NAME,
                    timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
                ),
                deadline=_OPERATION_DEADLINE_SECONDS,
                expect_awaitable=True,
            ),
            expected_usage=full_usage,
            expected_model_family=cfg.get_model_family(),
            expected_model=_PUBLIC_MODEL_NAME,
            expected_bucket_ids=bucket_ids,
        )
        mutated_snapshot_probe = snapshot_probe.model_copy(update={"usage": zero_usage})
        await _run_async_step(
            "RateLimiter.refund_capacity(mutated reservation snapshot authority)",
            lambda: limiter.refund_capacity(zero_usage, mutated_snapshot_probe),
            deadline=_OPERATION_DEADLINE_SECONDS,
            expect_awaitable=True,
        )
        final_reservation = _check_public_reservation_fields(
            await _run_async_step(
                "RateLimiter.acquire_capacity(after mutated reservation refund)",
                lambda: limiter.acquire_capacity(
                    full_usage,
                    _PUBLIC_MODEL_NAME,
                    timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
                ),
                deadline=_OPERATION_DEADLINE_SECONDS,
                expect_awaitable=True,
            ),
            expected_usage=full_usage,
            expected_model_family=cfg.get_model_family(),
            expected_model=_PUBLIC_MODEL_NAME,
            expected_bucket_ids=bucket_ids,
        )
        await _run_async_step(
            "RateLimiter.refund_capacity(final public reservation)",
            lambda: limiter.refund_capacity(zero_usage, final_reservation),
            deadline=_OPERATION_DEADLINE_SECONDS,
            expect_awaitable=True,
        )
    finally:
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(
                limiter.aclose(),
                timeout=_OPERATION_DEADLINE_SECONDS,
            )


async def _check_async_acquire_refund_failed_error(
    backend_builder: RateLimiterBackendBuilderInterface,
) -> None:
    refund_error = RuntimeError("fault-injection")
    interrupted_by = asyncio.CancelledError("fault-injection")
    cfg = _config("async-acquire-refund-failed", limit=2.0)
    limiter = RateLimiter(
        cfg,
        backend=_AsyncRefundFailureBuilder(backend_builder, refund_error),
    )
    original_complete_acquire_state_update = cast(
        "Callable[[Awaitable[object]], Awaitable[asyncio.CancelledError | None]]",
        limiter._complete_acquire_state_update,  # noqa: SLF001
    )

    async def complete_acquire_state_update_with_cancel(
        awaitable: Awaitable[object],
    ) -> asyncio.CancelledError | None:
        result = await original_complete_acquire_state_update(awaitable)
        if result is not None:
            return result
        return interrupted_by

    limiter._complete_acquire_state_update = (  # type: ignore[method-assign]  # noqa: SLF001
        complete_acquire_state_update_with_cancel
    )
    try:
        try:
            await _run_async_step(
                "RateLimiter.acquire_capacity(FIX-50 refund failure)",
                lambda: limiter.acquire_capacity(
                    frozen_usage({"requests": 1.0}),
                    _PUBLIC_MODEL_NAME,
                ),
                deadline=_OPERATION_DEADLINE_SECONDS,
                expect_awaitable=True,
                allowed_exceptions=(AcquireRefundFailedError,),
            )
        except AcquireRefundFailedError as exc:
            _check_acquire_refund_failed_payload(
                exc,
                refund_error=refund_error,
                interrupted_by=interrupted_by,
            )
            return
        _fail("interrupted acquire cleanup must raise AcquireRefundFailedError")
    finally:
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(
                limiter.aclose(),
                timeout=_OPERATION_DEADLINE_SECONDS,
            )


async def conformance_test_for(
    backend_builder: RateLimiterBackendBuilderInterface,
) -> None:
    """
    Run the public async backend conformance checks for one backend builder.

    The builder should point at isolated backend state: use a disposable Redis
    key prefix, database, or in-memory instance so these tests can consume and
    refund capacity freely.

    Backend operations are bounded by helper-owned deadlines. KNOWN LIMITATION:
    if a synchronous backend call hangs in its worker thread, Python cannot
    safely kill that thread; the helper reports the hang and continues without
    waiting for that thread to finish.
    """
    _build_async_backend(
        backend_builder,
        _config("async-protocol-probe"),
        label="build(async-protocol-probe)",
    )
    await _check_async_basic_capacity(backend_builder)
    await _check_async_all_or_nothing(backend_builder)
    await _check_async_refund_and_overuse(backend_builder)
    await _check_async_consume_and_capacity_updates(backend_builder)
    await _check_async_callbacks(backend_builder)
    await _check_async_metric_set_change(backend_builder)
    await _check_async_per_build_isolation(backend_builder)
    await _check_async_durable_refund_dedup(backend_builder)
    await _check_async_marker_authority(backend_builder)
    await _check_async_public_reservation_round_trip(backend_builder)
    await _check_async_acquire_refund_failed_error(backend_builder)


def _check_sync_basic_capacity(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    backend = _build_sync_backend(
        builder,
        _config("sync-basic"),
        label="build(sync-basic)",
    )
    _check_sync_claims(backend)

    start = time.monotonic()
    _sync_backend_step(
        "wait_for_capacity(requests=1)",
        lambda: backend.wait_for_capacity(frozen_usage({"requests": 1})),
    )
    _check(
        time.monotonic() - start < _PROMPT_DEADLINE_SECONDS,
        "wait_for_capacity() did not return promptly when capacity was available",
    )

    _expect_value_error(
        "wait_for_capacity(requests=-1)",
        lambda: backend.wait_for_capacity(frozen_usage({"requests": -1})),
        "wait_for_capacity() must reject negative usage",
    )

    exhausted = _build_sync_backend(
        builder,
        _config("sync-exhaust"),
        label="build(sync-exhaust)",
    )
    _sync_backend_step(
        "wait_for_capacity(requests=10)",
        lambda: exhausted.wait_for_capacity(frozen_usage({"requests": _FAST_LIMIT})),
    )
    _expect_timeout(
        "wait_for_capacity(requests=1, timeout=0)",
        lambda: exhausted.wait_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "wait_for_capacity(timeout=0) must raise TimeoutError when capacity is unavailable",
    )
    _sync_backend_step(
        "refund_capacity(requests=10, actual=5)",
        lambda: exhausted.refund_capacity(
            frozen_usage({"requests": _FAST_LIMIT}),
            frozen_usage({"requests": _FAST_LIMIT / 2}),
        ),
    )
    start = time.monotonic()
    _sync_backend_step(
        "wait_for_capacity(requests=5)",
        lambda: exhausted.wait_for_capacity(
            frozen_usage({"requests": _FAST_LIMIT / 2})
        ),
    )
    _check(
        time.monotonic() - start < _PROMPT_DEADLINE_SECONDS,
        "refund_capacity() did not restore unused capacity",
    )


def _check_sync_all_or_nothing(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    backend = _build_sync_backend(
        builder,
        _config(
            "sync-all-or-nothing",
            limit=1.0,
            extra_quotas=(
                Quota(
                    metric="tokens",
                    limit=1.0,
                    per_seconds=_SHORT_WINDOW_SECONDS,
                ),
            ),
        ),
        label="build(sync-all-or-nothing)",
    )
    _sync_backend_step(
        "wait_for_capacity(tokens=1, requests=0)",
        lambda: backend.wait_for_capacity(frozen_usage({"tokens": 1, "requests": 0})),
    )
    _expect_timeout(
        "wait_for_capacity(tokens=1, requests=1, timeout=0)",
        lambda: backend.wait_for_capacity(
            frozen_usage({"tokens": 1, "requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "wait_for_capacity() must not partially consume when one metric lacks capacity",
    )
    _sync_backend_step(
        "wait_for_capacity(tokens=0, requests=1, timeout=0)",
        lambda: backend.wait_for_capacity(
            frozen_usage({"tokens": 0, "requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
    )


def _check_sync_refund_and_overuse(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    invalid_refund = _build_sync_backend(
        builder,
        _config("sync-invalid-refund"),
        label="build(sync-invalid-refund)",
    )
    _sync_backend_step(
        "wait_for_capacity(requests=1)",
        lambda: invalid_refund.wait_for_capacity(frozen_usage({"requests": 1})),
    )
    _expect_value_error(
        "refund_capacity(requests=1, actual=-1)",
        lambda: invalid_refund.refund_capacity(
            frozen_usage({"requests": 1}),
            frozen_usage({"requests": -1}),
        ),
        "refund_capacity() must reject negative actual usage",
    )

    overuse = _build_sync_backend(
        builder,
        _config("sync-overuse"),
        label="build(sync-overuse)",
    )
    _sync_backend_step(
        "wait_for_capacity(requests=1)",
        lambda: overuse.wait_for_capacity(frozen_usage({"requests": 1})),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _sync_backend_step(
            "refund_capacity(requests=1, actual=2)",
            lambda: overuse.refund_capacity(
                frozen_usage({"requests": 1}),
                frozen_usage({"requests": 2}),
            ),
        )
    _check(
        any(issubclass(item.category, RuntimeWarning) for item in caught),
        "refund_capacity() must warn with RuntimeWarning when actual usage exceeds reserved usage",
    )


def _check_sync_consume_and_capacity_updates(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    consume = _build_sync_backend(
        builder,
        _config("sync-consume", limit=5.0),
        label="build(sync-consume)",
    )
    _sync_backend_step(
        "wait_for_capacity(requests=5)",
        lambda: consume.wait_for_capacity(frozen_usage({"requests": 5})),
    )
    _sync_backend_step(
        "consume_capacity(requests=5)",
        lambda: consume.consume_capacity(frozen_usage({"requests": 5})),
    )
    _expect_timeout(
        "wait_for_capacity(requests=1, timeout=0)",
        lambda: consume.wait_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "consume_capacity() must allow debt that later blocking acquires observe",
    )

    capacity = _build_sync_backend(
        builder,
        _config("sync-max-capacity", limit=5.0),
        label="build(sync-max-capacity)",
    )
    _sync_backend_step(
        "apply_configured_max_capacity(requests, 1, 3)",
        lambda: capacity.apply_configured_max_capacity(
            "requests", _SHORT_WINDOW_SECONDS, 3.0
        ),
    )
    _expect_value_error(
        "wait_for_capacity(requests=4, timeout=0)",
        lambda: capacity.wait_for_capacity(
            frozen_usage({"requests": 4}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "apply_configured_max_capacity() must update the live bucket max_capacity",
    )
    _sync_backend_step(
        "set_max_capacity(requests, 1, 4)",
        lambda: capacity.set_max_capacity("requests", _SHORT_WINDOW_SECONDS, 4.0),
    )
    _sync_backend_step(
        "wait_for_capacity(requests=4, timeout=0)",
        lambda: capacity.wait_for_capacity(
            frozen_usage({"requests": 4}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
    )


def _check_sync_callbacks(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    events: list[str] = []

    def on_wait_start(**_kwargs) -> None:
        events.append("wait_start")

    def after_wait_end_consumption(**_kwargs) -> None:
        events.append("wait_end")

    def on_capacity_consumed(**_kwargs) -> None:
        events.append("consumed")

    def on_capacity_refunded(**_kwargs) -> None:
        events.append("refunded")

    def on_missing_consumption_data(**_kwargs) -> None:
        events.append("missing")

    backend = _build_sync_backend(
        builder,
        _config("sync-callbacks", limit=_CALLBACK_LIMIT),
        label="build(sync-callbacks)",
        callbacks=SyncRateLimiterCallbacks(
            on_wait_start=on_wait_start,
            after_wait_end_consumption=after_wait_end_consumption,
            on_capacity_consumed=on_capacity_consumed,
            on_capacity_refunded=on_capacity_refunded,
            on_missing_consumption_data=on_missing_consumption_data,
        ),
    )
    _sync_backend_step(
        "wait_for_capacity(requests=4)",
        lambda: backend.wait_for_capacity(frozen_usage({"requests": _CALLBACK_LIMIT})),
    )
    _sync_backend_step(
        "wait_for_capacity(requests=1, timeout=2)",
        lambda: backend.wait_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_WAIT_TIMEOUT_SECONDS,
        ),
        deadline=_WAIT_BUDGET_SECONDS,
    )
    _sync_backend_step(
        "refund_capacity(requests=1, actual=0)",
        lambda: backend.refund_capacity(
            frozen_usage({"requests": 1}),
            frozen_usage({"requests": 0}),
        ),
    )

    for event in ("missing", "consumed", "wait_start", "wait_end", "refunded"):
        _check(event in events, f"callback event {event!r} was not emitted")


def _check_sync_metric_set_change(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    old_cfg, new_cfg = _metric_set_configs("sync-metric-set")
    old_backend = _build_sync_backend(
        builder,
        old_cfg,
        label="build(sync-metric-set-old)",
    )
    if not _check_bool_claim(
        _run_sync_step(
            "supports_metric_set_change()",
            old_backend.supports_metric_set_change,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_metric_set_change",
    ):
        return

    _sync_backend_step(
        "wait_for_capacity(requests=2, tokens=0)",
        lambda: old_backend.wait_for_capacity(
            frozen_usage({"requests": 2, "tokens": 0})
        ),
    )
    new_backend = _build_sync_backend(
        builder,
        new_cfg,
        label="build(sync-metric-set-new)",
    )
    prepared = _sync_backend_step(
        "prepare_reconfigured_backend(sync-metric-set)",
        lambda: old_backend.prepare_reconfigured_backend(new_backend, new_cfg),
    )
    _check(
        isinstance(prepared, SyncRateLimiterBackend),
        "prepare_reconfigured_backend() must return a SyncRateLimiterBackend",
    )
    prepared_backend = cast("SyncRateLimiterBackend", prepared)
    _expect_timeout(
        "wait_for_capacity(requests=1, images=0, timeout=0)",
        lambda: prepared_backend.wait_for_capacity(
            frozen_usage({"requests": 1, "images": 0}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "prepare_reconfigured_backend() must preserve surviving bucket consumption",
    )
    _sync_backend_step(
        "wait_for_capacity(requests=0, images=1, timeout=0)",
        lambda: prepared_backend.wait_for_capacity(
            frozen_usage({"requests": 0, "images": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
    )


def _check_sync_per_build_isolation(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    backend_a = _build_sync_backend(
        builder,
        _config("sync-isolation-a", limit=2.0),
        label="build(sync-isolation-a)",
    )
    _sync_backend_step(
        "wait_for_capacity(requests=2)",
        lambda: backend_a.wait_for_capacity(frozen_usage({"requests": 2})),
    )
    _build_sync_backend(
        builder,
        _config("sync-isolation-b", limit=5.0),
        label="build(sync-isolation-b)",
    )
    _expect_value_error(
        "wait_for_capacity(requests=3, timeout=0)",
        lambda: backend_a.wait_for_capacity(
            frozen_usage({"requests": 3}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "building another backend must not change an existing backend's quota limits",
    )
    _expect_timeout(
        "wait_for_capacity(requests=1, timeout=0)",
        lambda: backend_a.wait_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "building another backend must not reset an existing backend's consumed state",
    )


def _sync_refund_for_buckets(  # noqa: PLR0913
    label: str,
    backend: SyncRateLimiterBackend,
    reserved_usage,
    actual_usage,
    *,
    reservation_id: str,
    reservation_model_family: str,
    reservation_bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
    reservation_reserved_usage=None,
) -> object:
    return _sync_backend_step(
        label,
        lambda: backend.refund_capacity_for_buckets(
            reserved_usage,
            actual_usage,
            bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
            reservation_id=reservation_id,
            reservation_model_family=reservation_model_family,
            reservation_bucket_ids=reservation_bucket_ids,
            reservation_reserved_usage=(
                reserved_usage
                if reservation_reserved_usage is None
                else reservation_reserved_usage
            ),
        ),
        allowed_exceptions=(UnknownReservationError, DuplicateRefundError, ValueError),
    )


def _check_sync_no_double_refund_credit(
    backend: SyncRateLimiterBackend,
    message: str,
) -> None:
    _expect_timeout(
        "wait_for_capacity(requests=2, timeout=0)",
        lambda: backend.wait_for_capacity(
            frozen_usage({"requests": 2}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        message,
    )


def _check_sync_durable_refund_dedup(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    cfg = _config("sync-durable-dedup", limit=2.0)
    backend = _build_sync_backend(
        builder,
        cfg,
        label="build(sync-durable-dedup)",
    )
    if not _check_bool_claim(
        _run_sync_step(
            "supports_durable_refund_dedup()",
            backend.supports_durable_refund_dedup,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_durable_refund_dedup",
    ):
        return

    reservation_id = f"conformance-{uuid.uuid4().hex}"
    reserved_usage = frozen_usage({"requests": 2})
    _sync_backend_step(
        "wait_for_capacity(requests=2, reservation_id=...)",
        lambda: backend.wait_for_capacity(
            reserved_usage,
            reservation_id=reservation_id,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        ),
    )
    _sync_refund_for_buckets(
        "refund_capacity_for_buckets(durable-first)",
        backend,
        reserved_usage,
        frozen_usage({"requests": 1}),
        reservation_id=reservation_id,
        reservation_model_family=cfg.get_model_family(),
    )
    try:
        _sync_refund_for_buckets(
            "refund_capacity_for_buckets(durable-duplicate)",
            backend,
            reserved_usage,
            frozen_usage({"requests": 1}),
            reservation_id=reservation_id,
            reservation_model_family=cfg.get_model_family(),
        )
    except DuplicateRefundError:
        return
    _check_sync_no_double_refund_credit(
        backend,
        "supports_durable_refund_dedup=True must not credit duplicate refunds twice",
    )


def _check_sync_marker_authority(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    cfg = _config("sync-marker")
    backend = _build_sync_backend(
        builder,
        cfg,
        label="build(sync-marker)",
    )
    if not _check_bool_claim(
        _run_sync_step(
            "supports_acquire_marker_authority()",
            backend.supports_acquire_marker_authority,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_acquire_marker_authority",
    ):
        return

    reservation_id = f"conformance-{uuid.uuid4().hex}"
    reserved_usage = frozen_usage({"requests": 1})
    _sync_backend_step(
        "wait_for_capacity(requests=1, reservation_id=...)",
        lambda: backend.wait_for_capacity(
            reserved_usage,
            reservation_id=reservation_id,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        ),
    )
    result = _sync_backend_step(
        "refund_capacity_for_buckets(requests=1, actual=0)",
        lambda: backend.refund_capacity_for_buckets(
            reserved_usage,
            frozen_usage({"requests": 0}),
            bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
            reservation_id=reservation_id,
            reservation_model_family=cfg.get_model_family(),
            reservation_bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
            reservation_reserved_usage=reserved_usage,
        ),
    )
    _check(
        result is True,
        "refund_capacity_for_buckets() must return True after a marker-authorized refund",
    )

    unknown_cfg = _config("sync-marker-unknown", limit=2.0)
    unknown_backend = _build_sync_backend(
        builder,
        unknown_cfg,
        label="build(sync-marker-unknown)",
    )
    _sync_backend_step(
        "wait_for_capacity(requests=2)",
        lambda: unknown_backend.wait_for_capacity(frozen_usage({"requests": 2})),
    )
    try:
        _sync_refund_for_buckets(
            "refund_capacity_for_buckets(unknown-reservation)",
            unknown_backend,
            frozen_usage({"requests": 2}),
            frozen_usage({"requests": 1}),
            reservation_id=f"unknown-{uuid.uuid4().hex}",
            reservation_model_family=unknown_cfg.get_model_family(),
        )
    except UnknownReservationError:
        pass
    else:
        _expect_timeout(
            "wait_for_capacity(requests=1, timeout=0)",
            lambda: unknown_backend.wait_for_capacity(
                frozen_usage({"requests": 1}),
                timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
            ),
            "marker-authority backends must fail closed for unknown reservations",
        )

    duplicate_cfg = _config("sync-marker-duplicate", limit=2.0)
    duplicate_backend = _build_sync_backend(
        builder,
        duplicate_cfg,
        label="build(sync-marker-duplicate)",
    )
    duplicate_reservation_id = f"conformance-{uuid.uuid4().hex}"
    duplicate_reserved_usage = frozen_usage({"requests": 2})
    _sync_backend_step(
        "wait_for_capacity(requests=2, reservation_id=...)",
        lambda: duplicate_backend.wait_for_capacity(
            duplicate_reserved_usage,
            reservation_id=duplicate_reservation_id,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        ),
    )
    _sync_refund_for_buckets(
        "refund_capacity_for_buckets(duplicate-first)",
        duplicate_backend,
        duplicate_reserved_usage,
        frozen_usage({"requests": 1}),
        reservation_id=duplicate_reservation_id,
        reservation_model_family=duplicate_cfg.get_model_family(),
    )
    try:
        _sync_refund_for_buckets(
            "refund_capacity_for_buckets(duplicate-second)",
            duplicate_backend,
            duplicate_reserved_usage,
            frozen_usage({"requests": 1}),
            reservation_id=duplicate_reservation_id,
            reservation_model_family=duplicate_cfg.get_model_family(),
        )
    except DuplicateRefundError:
        pass
    else:
        _check_sync_no_double_refund_credit(
            duplicate_backend,
            "marker-authority backends must not credit duplicate refunds twice",
        )

    mismatch_cfg = _config("sync-marker-mismatch", limit=2.0)
    mismatch_backend = _build_sync_backend(
        builder,
        mismatch_cfg,
        label="build(sync-marker-mismatch)",
    )
    mismatch_reservation_id = f"conformance-{uuid.uuid4().hex}"
    mismatch_reserved_usage = frozen_usage({"requests": 2})
    _sync_backend_step(
        "wait_for_capacity(requests=2, reservation_id=...)",
        lambda: mismatch_backend.wait_for_capacity(
            mismatch_reserved_usage,
            reservation_id=mismatch_reservation_id,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        ),
    )
    try:
        mismatch_result = _sync_refund_for_buckets(
            "refund_capacity_for_buckets(metadata-mismatch)",
            mismatch_backend,
            mismatch_reserved_usage,
            frozen_usage({"requests": 1}),
            reservation_id=mismatch_reservation_id,
            reservation_model_family=_family("forged-marker-family"),
        )
    except (UnknownReservationError, DuplicateRefundError, ValueError):
        return
    if mismatch_result is False:
        return
    _expect_timeout(
        "wait_for_capacity(requests=1, timeout=0)",
        lambda: mismatch_backend.wait_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "marker-authority backends must fail closed for reservation metadata mismatch",
    )


def _check_sync_public_reservation_round_trip(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    cfg = _two_metric_config("sync-public-round-trip")
    usage = frozen_usage({"requests": 1.0, "tokens": 1.0})
    full_usage = frozen_usage({"requests": 2.0, "tokens": 2.0})
    zero_usage = frozen_usage({"requests": 0.0, "tokens": 0.0})
    bucket_ids = frozenset({_REQUESTS_BUCKET_ID, _TOKENS_BUCKET_ID})
    limiter = SyncRateLimiter(cfg, backend=builder)
    try:
        reservation = _check_public_reservation_fields(
            _run_sync_step(
                "SyncRateLimiter.acquire_capacity(public reservation)",
                lambda: limiter.acquire_capacity(usage, _PUBLIC_MODEL_NAME),
                deadline=_OPERATION_DEADLINE_SECONDS,
            ),
            expected_usage=usage,
            expected_model_family=cfg.get_model_family(),
            expected_model=_PUBLIC_MODEL_NAME,
            expected_bucket_ids=bucket_ids,
        )
        _run_sync_step(
            "SyncRateLimiter.refund_capacity(public reservation)",
            lambda: limiter.refund_capacity(zero_usage, reservation),
            deadline=_OPERATION_DEADLINE_SECONDS,
        )

        restored = _check_public_reservation_fields(
            _run_sync_step(
                "SyncRateLimiter.acquire_capacity(restored capacity)",
                lambda: limiter.acquire_capacity(
                    full_usage,
                    _PUBLIC_MODEL_NAME,
                    timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
                ),
                deadline=_OPERATION_DEADLINE_SECONDS,
            ),
            expected_usage=full_usage,
            expected_model_family=cfg.get_model_family(),
            expected_model=_PUBLIC_MODEL_NAME,
            expected_bucket_ids=bucket_ids,
        )
        _run_sync_step(
            "SyncRateLimiter.refund_capacity(restored reservation)",
            lambda: limiter.refund_capacity(zero_usage, restored),
            deadline=_OPERATION_DEADLINE_SECONDS,
        )

        mutated = restored.model_copy(update={"usage": zero_usage})
        _check(
            type(mutated) is CapacityReservation,
            "CapacityReservation.model_copy() must preserve exact public type",
        )
        snapshot_probe = _check_public_reservation_fields(
            _run_sync_step(
                "SyncRateLimiter.acquire_capacity(snapshot authority)",
                lambda: limiter.acquire_capacity(
                    full_usage,
                    _PUBLIC_MODEL_NAME,
                    timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
                ),
                deadline=_OPERATION_DEADLINE_SECONDS,
            ),
            expected_usage=full_usage,
            expected_model_family=cfg.get_model_family(),
            expected_model=_PUBLIC_MODEL_NAME,
            expected_bucket_ids=bucket_ids,
        )
        mutated_snapshot_probe = snapshot_probe.model_copy(update={"usage": zero_usage})
        _run_sync_step(
            "SyncRateLimiter.refund_capacity(mutated reservation snapshot authority)",
            lambda: limiter.refund_capacity(zero_usage, mutated_snapshot_probe),
            deadline=_OPERATION_DEADLINE_SECONDS,
        )
        final_reservation = _check_public_reservation_fields(
            _run_sync_step(
                "SyncRateLimiter.acquire_capacity(after mutated reservation refund)",
                lambda: limiter.acquire_capacity(
                    full_usage,
                    _PUBLIC_MODEL_NAME,
                    timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
                ),
                deadline=_OPERATION_DEADLINE_SECONDS,
            ),
            expected_usage=full_usage,
            expected_model_family=cfg.get_model_family(),
            expected_model=_PUBLIC_MODEL_NAME,
            expected_bucket_ids=bucket_ids,
        )
        _run_sync_step(
            "SyncRateLimiter.refund_capacity(final public reservation)",
            lambda: limiter.refund_capacity(zero_usage, final_reservation),
            deadline=_OPERATION_DEADLINE_SECONDS,
        )
    finally:
        with contextlib.suppress(BaseException):
            _run_sync_step(
                "SyncRateLimiter.close()",
                limiter.close,
                deadline=_OPERATION_DEADLINE_SECONDS,
            )


def _check_sync_acquire_refund_failed_error(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    refund_error = RuntimeError("fault-injection")
    interrupted_by = _SyncAcquireInterrupted("fault-injection")
    cfg = _config("sync-acquire-refund-failed", limit=2.0)
    limiter = SyncRateLimiter(
        cfg,
        backend=_SyncRefundFailureBuilder(builder, refund_error),
    )
    original_finalize_pending_acquire = cast(
        "Callable[[CapacityReservation, str], None]",
        limiter._finalize_pending_acquire,  # noqa: SLF001
    )
    first_finalize = True

    def finalize_pending_acquire_with_interruption(
        reservation: CapacityReservation,
        model: str,
    ) -> None:
        nonlocal first_finalize
        if first_finalize:
            first_finalize = False
            raise interrupted_by
        original_finalize_pending_acquire(reservation, model)

    limiter._finalize_pending_acquire = (  # type: ignore[method-assign]  # noqa: SLF001
        finalize_pending_acquire_with_interruption
    )
    try:
        try:
            _run_sync_step(
                "SyncRateLimiter.acquire_capacity(FIX-50 refund failure)",
                lambda: limiter.acquire_capacity(
                    frozen_usage({"requests": 1.0}),
                    _PUBLIC_MODEL_NAME,
                ),
                deadline=_OPERATION_DEADLINE_SECONDS,
                allowed_exceptions=(AcquireRefundFailedError,),
            )
        except AcquireRefundFailedError as exc:
            _check_acquire_refund_failed_payload(
                exc,
                refund_error=refund_error,
                interrupted_by=interrupted_by,
            )
            return
        _fail("interrupted acquire cleanup must raise AcquireRefundFailedError")
    finally:
        with contextlib.suppress(BaseException):
            _run_sync_step(
                "SyncRateLimiter.close()",
                limiter.close,
                deadline=_OPERATION_DEADLINE_SECONDS,
            )


def sync_conformance_test_for(
    backend_builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    """
    Run the public sync backend conformance checks for one backend builder.

    The builder should point at isolated backend state: use a disposable Redis
    key prefix, database, or in-memory instance so these tests can consume and
    refund capacity freely.

    Backend operations are bounded by helper-owned deadlines. KNOWN LIMITATION:
    if a synchronous backend call hangs in its worker thread, Python cannot
    safely kill that thread; the helper reports the hang and continues without
    waiting for that thread to finish.
    """
    _build_sync_backend(
        backend_builder,
        _config("sync-protocol-probe"),
        label="build(sync-protocol-probe)",
    )
    _check_sync_basic_capacity(backend_builder)
    _check_sync_all_or_nothing(backend_builder)
    _check_sync_refund_and_overuse(backend_builder)
    _check_sync_consume_and_capacity_updates(backend_builder)
    _check_sync_callbacks(backend_builder)
    _check_sync_metric_set_change(backend_builder)
    _check_sync_per_build_isolation(backend_builder)
    _check_sync_durable_refund_dedup(backend_builder)
    _check_sync_marker_authority(backend_builder)
    _check_sync_public_reservation_round_trip(backend_builder)
    _check_sync_acquire_refund_failed_error(backend_builder)


def run_conformance_test_for(
    backend_builder: RateLimiterBackendBuilderInterface,
) -> None:
    """Run async backend conformance checks from synchronous test suites."""
    asyncio.run(conformance_test_for(backend_builder))
