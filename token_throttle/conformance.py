from __future__ import annotations

import asyncio
import asyncio.tasks as _asyncio_tasks
import concurrent.futures
import contextlib
import contextvars
import inspect
import math
import os
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn, SupportsFloat, SupportsIndex, cast

from frozendict import frozendict

from token_throttle._exceptions import (
    AcquireRefundFailedError,
    BackendConformanceError,
    DuplicateRefundError,
    UnknownReservationError,
)
from token_throttle._interfaces._callbacks import (
    LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
    LifecycleEvent,
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
    Capacities,
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
_RESERVATION_TIMESTAMP_TOLERANCE_SECONDS = 1.0
_BUCKET_ID_SIZE = 2
_TIMING_SCALE_ENV = "TOKEN_THROTTLE_CONFORMANCE_TIMING_SCALE"
_OMIT_MARKER_METADATA = object()
_DEFAULT_MARKER_BUCKET_IDS = object()
_DEFAULT_MARKER_RESERVED_USAGE = object()


@dataclass(frozen=True)
class ConformanceTiming:
    """
    Wall-clock deadlines used by the public conformance helpers.

    When ``timing=`` is passed, the env var is ignored; set every desired field
    on the dataclass. Scales below ~0.1 may cause correct backends to fail
    conformance because internal probe deadlines have an implicit floor.
    """

    builder_deadline_seconds: float = _BUILDER_DEADLINE_SECONDS
    operation_deadline_seconds: float = _OPERATION_DEADLINE_SECONDS
    prompt_deadline_seconds: float = _PROMPT_DEADLINE_SECONDS
    wait_budget_seconds: float = _WAIT_BUDGET_SECONDS


_TIMING_CONTEXT: contextvars.ContextVar[ConformanceTiming | None] = (
    contextvars.ContextVar("token_throttle_conformance_timing", default=None)
)
_CALLBACK_FAILURES: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "token_throttle_conformance_callback_failures", default=None
)


@dataclass(frozen=True)
class _AsyncStepContext:
    label: str
    deadline: float
    expiry: float
    cancelling_at_entry: int
    allowed_exceptions: tuple[type[BaseException], ...]


def _validate_timing_value(field_name: str, value: object) -> float:
    if isinstance(value, bool | str | bytes | bytearray | complex):
        raise ValueError(  # noqa: TRY004 - public validator preserves ValueError.
            f"{field_name} must be a positive finite number"
        )
    try:
        value = float(cast("SupportsFloat | SupportsIndex", value))
    except Exception:  # noqa: BLE001 - hostile __float__ errors must be sanitized.
        raise ValueError(f"{field_name} must be a positive finite number") from None
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a positive finite number")
    return value


def _scale_timing_value(field_name: str, value: float, scale: float) -> float:
    scaled_value = value * scale
    if not math.isfinite(scaled_value):
        raise ValueError(f"{field_name} must be a positive finite number")
    return scaled_value


def _resolve_timing(timing: ConformanceTiming | None) -> ConformanceTiming:
    if timing is None:
        default_timing = ConformanceTiming()
        scale_text = os.environ.get(_TIMING_SCALE_ENV)
        if scale_text is None:
            timing = default_timing
        else:
            try:
                scale = float(scale_text)
            except ValueError:
                raise ValueError(
                    f"{_TIMING_SCALE_ENV} must be a positive finite number"
                ) from None
            _validate_timing_value(_TIMING_SCALE_ENV, scale)
            timing = ConformanceTiming(
                builder_deadline_seconds=_scale_timing_value(
                    _TIMING_SCALE_ENV,
                    default_timing.builder_deadline_seconds,
                    scale,
                ),
                operation_deadline_seconds=_scale_timing_value(
                    _TIMING_SCALE_ENV,
                    default_timing.operation_deadline_seconds,
                    scale,
                ),
                prompt_deadline_seconds=_scale_timing_value(
                    _TIMING_SCALE_ENV,
                    default_timing.prompt_deadline_seconds,
                    scale,
                ),
                wait_budget_seconds=_scale_timing_value(
                    _TIMING_SCALE_ENV,
                    default_timing.wait_budget_seconds,
                    scale,
                ),
            )
    elif not isinstance(timing, ConformanceTiming):
        raise TypeError("timing must be a ConformanceTiming instance")

    return ConformanceTiming(
        builder_deadline_seconds=_validate_timing_value(
            "builder_deadline_seconds", timing.builder_deadline_seconds
        ),
        operation_deadline_seconds=_validate_timing_value(
            "operation_deadline_seconds", timing.operation_deadline_seconds
        ),
        prompt_deadline_seconds=_validate_timing_value(
            "prompt_deadline_seconds", timing.prompt_deadline_seconds
        ),
        wait_budget_seconds=_validate_timing_value(
            "wait_budget_seconds", timing.wait_budget_seconds
        ),
    )


def _timing() -> ConformanceTiming:
    return _TIMING_CONTEXT.get() or _resolve_timing(None)


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


def _check_callback_model_family(slot: str, model_family: object) -> None:
    _check_callback_payload(
        isinstance(model_family, str) and model_family.startswith("conformance/"),
        f"{slot} callback passed invalid model_family {model_family!r}",
    )


def _check_callback_usage(slot: str, field_name: str, value: object) -> None:
    _check_callback_payload(
        isinstance(value, frozendict)
        and all(isinstance(key, str) for key in value)
        and all(isinstance(amount, int | float) for amount in value.values()),
        f"{slot} callback passed invalid {field_name}",
    )


def _check_callback_capacities(slot: str, field_name: str, value: object) -> None:
    _check_callback_payload(
        isinstance(value, frozendict)
        and all(
            isinstance(bucket_id, tuple)
            and len(bucket_id) == _BUCKET_ID_SIZE
            and isinstance(bucket_id[0], str)
            and isinstance(bucket_id[1], int)
            and isinstance(amount, int | float)
            for bucket_id, amount in value.items()
        ),
        f"{slot} callback passed invalid {field_name}",
    )


def _check_callback_float(slot: str, field_name: str, value: object) -> None:
    _check_callback_payload(
        isinstance(value, int | float) and math.isfinite(float(value)),
        f"{slot} callback passed invalid {field_name}",
    )


def _check_callback_payload(condition: object, message: str) -> None:
    if condition:
        return
    failures = _CALLBACK_FAILURES.get()
    if failures is not None:
        failures.append(message)
    _fail(message)


def _matches_allowed_exception(
    exc: BaseException,
    allowed_exceptions: tuple[type[BaseException], ...],
) -> bool:
    return (
        bool(allowed_exceptions)
        and not isinstance(exc, BaseExceptionGroup)
        and isinstance(exc, allowed_exceptions)
    )


# Derived from the canonical critical-exception set so this harness cannot
# drift behind it. This was previously a hand-maintained literal that silently
# lagged the canonical set (for example, RecursionError was added to the
# canonical set but left missing here, and the group set omitted
# concurrent.futures.CancelledError), so it is now derived instead.
#
# Both CancelledError types are filtered out of the non-group set — but their
# actual non-group treatment differs, because their MROs differ:
#   - asyncio.CancelledError is BaseException-but-NOT-Exception. The async
#     runners catch it in a dedicated cancellation-aware `except` clause; the
#     sync runner has none, so it falls through `except Exception` untouched
#     and propagates raw. Either way it must stay out of the non-group set —
#     listing it would preempt the dedicated async handler.
#   - concurrent.futures.CancelledError IS an Exception subclass, so a bare
#     one is caught by `except Exception` and normalized into a
#     BackendConformanceError: a backend raising it bare is treated as a
#     conformance failure (consistent with a spurious backend
#     asyncio.CancelledError). Keeping it OUT of the non-group set is what
#     preserves that normalization — adding it would force a raw re-raise.
#
# Group-leaf classification instead uses the full canonical set directly: any
# critical exception that appears as a BaseExceptionGroup leaf is control flow
# and must propagate, never be normalized. This is where
# concurrent.futures.CancelledError reaches parity with asyncio.CancelledError.
_CANCELLED_ERROR_TYPES = (asyncio.CancelledError, concurrent.futures.CancelledError)
_NON_NORMALIZED_EXCEPTION_TYPES: tuple[type[BaseException], ...] = tuple(
    exc
    for exc in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS
    if not issubclass(exc, _CANCELLED_ERROR_TYPES)
)


def _is_non_normalized_group_exception(exc: BaseException) -> bool:
    return isinstance(exc, LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS)


def _step_exception_message(label: str, exc: BaseException) -> str:
    return f"{label} raised {type(exc).__name__}: {exc}"


def _raise_normalized_step_exception(label: str, exc: BaseException) -> NoReturn:
    raise BackendConformanceError(_step_exception_message(label, exc)) from exc


def _raise_split_exception_group(
    label: str,
    exc: BaseExceptionGroup[BaseException],
) -> NoReturn:
    """
    Propagate control/catastrophic grouped leaves; normalize grouped failures.

    ``allowed_exceptions`` intentionally does not match exception groups. A
    grouped backend failure is normalized unless it contains a control-flow or
    catastrophic leaf, in which case that subgroup is propagated. For mixed
    groups, the non-control remainder is preserved as a labeled
    ``BackendConformanceError`` inside a new group.
    """
    control_group, backend_group = exc.split(_is_non_normalized_group_exception)
    if control_group is None:
        _raise_normalized_step_exception(label, exc)
    if backend_group is None:
        raise control_group
    raise BaseExceptionGroup(
        exc.message,
        [
            control_group,
            BackendConformanceError(_step_exception_message(label, backend_group)),
        ],
    ) from exc


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
    context = contextvars.copy_context()
    future = executor.submit(lambda: context.run(callable_))
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
    except _NON_NORMALIZED_EXCEPTION_TYPES:
        raise
    except BaseExceptionGroup as exc:
        _raise_split_exception_group(label, exc)
    except Exception as exc:
        if _matches_allowed_exception(exc, allowed_exceptions):
            raise
        _raise_normalized_step_exception(label, exc)
    finally:
        executor.shutdown(wait=not timed_out, cancel_futures=True)


def _task_is_being_cancelled(cancelling_at_entry: int = 0) -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > cancelling_at_entry


def _current_task_cancelling_count() -> int:
    task = asyncio.current_task()
    if task is None:
        return 0
    return task.cancelling()


def _consume_abandoned_task_result(task: asyncio.Future[object]) -> None:
    with contextlib.suppress(BaseException):
        task.exception()
    with contextlib.suppress(BaseException):
        task.result()


def _remove_asyncio_shield_exception_logger(task: asyncio.Future[object]) -> None:
    shield_logger = getattr(_asyncio_tasks, "_log_on_exception", None)
    if shield_logger is not None:
        task.remove_done_callback(shield_logger)


def _cancel_and_consume_abandoned_future(task: asyncio.Future[object]) -> None:
    _remove_asyncio_shield_exception_logger(task)
    task.add_done_callback(_consume_abandoned_task_result)
    task.cancel()


async def _cancel_and_drain_abandoned_future(task: asyncio.Future[object]) -> None:
    _cancel_and_consume_abandoned_future(task)
    current = asyncio.current_task()
    uncancel_count = 0
    if current is not None:
        while current.cancelling():
            current.uncancel()
            uncancel_count += 1
    try:
        for _ in range(3):
            if task.done():
                break
            await asyncio.sleep(0)
        if task.done():
            _consume_abandoned_task_result(task)
    finally:
        if current is not None:
            for _ in range(uncancel_count):
                current.cancel()


def _remaining_deadline_seconds(expiry: float) -> float:
    return expiry - time.monotonic()


async def _materialize_async_callable(
    step: _AsyncStepContext,
    callable_: Callable[[], object],
) -> object:
    remaining = _remaining_deadline_seconds(step.expiry)
    if remaining <= 0:
        raise BackendConformanceError(
            f"{step.label} did not return within {step.deadline}s"
        )

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="token-throttle-conformance",
    )
    context = contextvars.copy_context()
    future: concurrent.futures.Future[object] = executor.submit(
        lambda: context.run(callable_)
    )
    async_future: asyncio.Future[object] = asyncio.wrap_future(future)
    abandoned = False
    shielded: asyncio.Future[object] = asyncio.shield(async_future)
    try:
        return await asyncio.wait_for(shielded, timeout=remaining)
    except TimeoutError as exc:
        if async_future.done():
            if _matches_allowed_exception(exc, step.allowed_exceptions):
                raise
            _raise_normalized_step_exception(step.label, exc)
        abandoned = True
        message = (
            f"{step.label} did not return within {step.deadline}s; thread may still be "
            "running (cannot kill in-process)"
        )
        warnings.warn(message, RuntimeWarning)
        _cancel_and_consume_abandoned_future(shielded)
        _cancel_and_consume_abandoned_future(async_future)
        future.cancel()
        raise BackendConformanceError(message) from exc
    # ast-guard: skip — step cancellation is normalized; executor cleanup is in finally
    except asyncio.CancelledError as exc:
        if _task_is_being_cancelled(step.cancelling_at_entry):
            abandoned = True
            _cancel_and_consume_abandoned_future(shielded)
            _cancel_and_consume_abandoned_future(async_future)
            future.cancel()
            raise
        if _matches_allowed_exception(exc, step.allowed_exceptions):
            raise
        _raise_normalized_step_exception(step.label, exc)
    except _NON_NORMALIZED_EXCEPTION_TYPES:
        raise
    except BaseExceptionGroup as exc:
        _raise_split_exception_group(step.label, exc)
    except Exception as exc:
        if _matches_allowed_exception(exc, step.allowed_exceptions):
            raise
        _raise_normalized_step_exception(step.label, exc)
    finally:
        executor.shutdown(wait=not abandoned, cancel_futures=True)


def _ensure_async_step_task(
    step: _AsyncStepContext,
    value: object,
) -> asyncio.Future[object]:
    try:
        task = asyncio.ensure_future(cast("Awaitable[object]", value))
        task.add_done_callback(_consume_abandoned_task_result)
        return task
    except _NON_NORMALIZED_EXCEPTION_TYPES:
        raise
    except BaseExceptionGroup as exc:
        _raise_split_exception_group(step.label, exc)
    except Exception as exc:
        if _matches_allowed_exception(exc, step.allowed_exceptions):
            raise
        _raise_normalized_step_exception(step.label, exc)


async def _await_async_step_task(
    step: _AsyncStepContext,
    task: asyncio.Future[object],
) -> object:
    remaining = _remaining_deadline_seconds(step.expiry)
    if remaining <= 0:
        _cancel_and_consume_abandoned_future(task)
        raise BackendConformanceError(
            f"{step.label} did not return within {step.deadline}s"
        )

    try:
        done, _pending = await asyncio.wait({task}, timeout=remaining)
    # ast-guard: skip — caller cancellation drains the abandoned step task
    except asyncio.CancelledError as exc:
        if _task_is_being_cancelled(step.cancelling_at_entry):
            await _cancel_and_drain_abandoned_future(task)
            raise
        if _matches_allowed_exception(exc, step.allowed_exceptions):
            raise
        _raise_normalized_step_exception(step.label, exc)

    if not done:
        await _cancel_and_drain_abandoned_future(task)
        raise BackendConformanceError(
            f"{step.label} did not return within {step.deadline}s"
        )

    try:
        return task.result()
    # ast-guard: skip — task.result() re-raises step cancellation for normalization
    except asyncio.CancelledError as exc:
        if _task_is_being_cancelled(step.cancelling_at_entry):
            raise
        if _matches_allowed_exception(exc, step.allowed_exceptions):
            raise
        _raise_normalized_step_exception(step.label, exc)
    except _NON_NORMALIZED_EXCEPTION_TYPES:
        raise
    except BaseExceptionGroup as exc:
        _raise_split_exception_group(step.label, exc)
    except Exception as exc:
        if _matches_allowed_exception(exc, step.allowed_exceptions):
            raise
        _raise_normalized_step_exception(step.label, exc)


async def _run_async_step(
    label: str,
    awaitable_or_coro_fn: Awaitable[object] | Callable[[], object] | object,
    *,
    deadline: float,
    expect_awaitable: bool = False,
    allowed_exceptions: tuple[type[BaseException], ...] = (),
) -> object:
    """
    Run an awaitable under one wall-clock deadline; normalize exceptions.

    ``allowed_exceptions`` matches only plain, non-grouped exceptions. Exception
    groups are split first: embedded control-flow/catastrophic leaves propagate,
    while non-control grouped backend failures are normalized.
    """
    step = _AsyncStepContext(
        label=label,
        deadline=deadline,
        expiry=time.monotonic() + deadline,
        cancelling_at_entry=_current_task_cancelling_count(),
        allowed_exceptions=allowed_exceptions,
    )
    if inspect.isawaitable(awaitable_or_coro_fn):
        value: object = awaitable_or_coro_fn
    elif callable(awaitable_or_coro_fn):
        value = await _materialize_async_callable(
            step,
            awaitable_or_coro_fn,
        )
    else:
        value = awaitable_or_coro_fn

    if not inspect.isawaitable(value):
        if expect_awaitable:
            _fail(f"{label} returned non-awaitable {type(value).__name__}")
        return value

    return await _await_async_step_task(step, _ensure_async_step_task(step, value))


def _close_awaitable(value: object) -> None:
    if not inspect.isawaitable(value):
        return
    if isinstance(value, asyncio.Future):
        _cancel_and_consume_abandoned_future(value)
        return
    close = getattr(value, "close", None)
    if callable(close):
        with contextlib.suppress(Exception, asyncio.CancelledError):
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
        deadline=_timing().builder_deadline_seconds,
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
        deadline=_timing().builder_deadline_seconds,
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
        deadline=_timing().operation_deadline_seconds if deadline is None else deadline,
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
        deadline=_timing().operation_deadline_seconds if deadline is None else deadline,
        allowed_exceptions=allowed_exceptions,
    )
    _check_sync_backend_result_not_awaitable(label, result)
    _validate_backend_step_return(label, result)
    return result


async def _cleanup_async_builder(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    aclose = getattr(builder, "aclose", None)
    if callable(aclose):

        def call_aclose() -> object:
            return aclose()

        await _run_async_step(
            "builder.aclose()",
            call_aclose,
            deadline=_timing().builder_deadline_seconds,
            expect_awaitable=True,
        )

    close = getattr(builder, "close", None)
    if callable(close):
        _run_sync_step(
            "builder.close()",
            close,
            deadline=_timing().builder_deadline_seconds,
        )


def _cleanup_sync_builder(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    close = getattr(builder, "close", None)
    if callable(close):
        _run_sync_step(
            "builder.close()",
            close,
            deadline=_timing().builder_deadline_seconds,
        )


def _warn_cleanup_failure(exc: BaseException, *, target: str = "builder") -> None:
    warnings.warn(
        f"{target} cleanup raised {type(exc).__name__}: {exc}",
        RuntimeWarning,
        stacklevel=3,
    )


def _raise_limiter_cleanup_failure(exc: BaseException) -> NoReturn:
    raise BackendConformanceError(
        f"limiter cleanup raised {type(exc).__name__}: {exc}"
    ) from exc


async def _cleanup_async_limiter(
    limiter: RateLimiter,
    *,
    body_exc: BaseException | None = None,
) -> None:
    try:
        await asyncio.wait_for(
            limiter.aclose(),
            timeout=_timing().operation_deadline_seconds,
        )
    # ast-guard: skip — limiter.aclose() cancellation must propagate
    except asyncio.CancelledError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 - suppress non-control cleanup failures.
        if body_exc is None:
            _raise_limiter_cleanup_failure(exc)
        _warn_cleanup_failure(exc, target="limiter")


def _cleanup_sync_limiter(
    limiter: SyncRateLimiter,
    *,
    body_exc: BaseException | None = None,
) -> None:
    try:
        _run_sync_step(
            "SyncRateLimiter.close()",
            limiter.close,
            deadline=_timing().operation_deadline_seconds,
            allowed_exceptions=(asyncio.CancelledError,),
        )
    # ast-guard: skip — sync limiter close cancellation must propagate
    except asyncio.CancelledError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 - suppress non-control cleanup failures.
        if body_exc is None:
            _raise_limiter_cleanup_failure(exc)
        _warn_cleanup_failure(exc, target="limiter")


class _AsyncRefundFailureBackend(RateLimiterBackend):
    def __init__(
        self,
        backend: RateLimiterBackend,
        refund_error: RuntimeError,
    ) -> None:
        self._backend = backend
        self._refund_error = refund_error
        self.last_refund_reserved_usage: FrozenUsage | None = None
        self.last_refund_actual_usage: FrozenUsage | None = None
        self.last_refund_bucket_ids: frozenset[BucketId] | None = None
        self.last_refund_reservation_id: str | None = None
        self.last_refund_reservation_model_family: str | None = None
        self.last_refund_reservation_bucket_ids: frozenset[BucketId] | None = None
        self.last_refund_reservation_reserved_usage: FrozenUsage | None = None

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
        self.last_refund_reserved_usage = reserved_usage
        self.last_refund_actual_usage = actual_usage
        self.last_refund_bucket_ids = (
            None if bucket_ids is None else frozenset(bucket_ids)
        )
        self.last_refund_reservation_id = reservation_id
        self.last_refund_reservation_model_family = reservation_model_family
        self.last_refund_reservation_bucket_ids = (
            None
            if reservation_bucket_ids is None
            else frozenset(reservation_bucket_ids)
        )
        self.last_refund_reservation_reserved_usage = reservation_reserved_usage
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
        self.last_backend: _AsyncRefundFailureBackend | None = None

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> RateLimiterBackend:
        backend = _AsyncRefundFailureBackend(
            self._builder.build(cfg, callbacks=callbacks),
            self._refund_error,
        )
        self.last_backend = backend
        return backend

    def __getattr__(self, name: str) -> object:
        return getattr(self._builder, name)

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
        self.last_refund_reserved_usage: FrozenUsage | None = None
        self.last_refund_actual_usage: FrozenUsage | None = None
        self.last_refund_bucket_ids: frozenset[BucketId] | None = None
        self.last_refund_reservation_id: str | None = None
        self.last_refund_reservation_model_family: str | None = None
        self.last_refund_reservation_bucket_ids: frozenset[BucketId] | None = None
        self.last_refund_reservation_reserved_usage: FrozenUsage | None = None

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
        self.last_refund_reserved_usage = reserved_usage
        self.last_refund_actual_usage = actual_usage
        self.last_refund_bucket_ids = (
            None if bucket_ids is None else frozenset(bucket_ids)
        )
        self.last_refund_reservation_id = reservation_id
        self.last_refund_reservation_model_family = reservation_model_family
        self.last_refund_reservation_bucket_ids = (
            None
            if reservation_bucket_ids is None
            else frozenset(reservation_bucket_ids)
        )
        self.last_refund_reservation_reserved_usage = reservation_reserved_usage
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
        self.last_backend: _SyncRefundFailureBackend | None = None

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> SyncRateLimiterBackend:
        backend = _SyncRefundFailureBackend(
            self._builder.build(cfg, callbacks=callbacks),
            self._refund_error,
        )
        self.last_backend = backend
        return backend

    def __getattr__(self, name: str) -> object:
        return getattr(self._builder, name)

    def close(self) -> None:
        self._builder.close()


class _SyncAcquireInterruptedError(Exception):
    """Internal sync interruption exception for the acquire-refund-failed probe."""


def _check_async_claims(backend: RateLimiterBackend) -> None:
    marker_authority = _check_bool_claim(
        _run_sync_step(
            "supports_acquire_marker_authority()",
            backend.supports_acquire_marker_authority,
            deadline=_timing().builder_deadline_seconds,
        ),
        "supports_acquire_marker_authority",
    )
    durable_dedup = _check_bool_claim(
        _run_sync_step(
            "supports_durable_refund_dedup()",
            backend.supports_durable_refund_dedup,
            deadline=_timing().builder_deadline_seconds,
        ),
        "supports_durable_refund_dedup",
    )
    metric_set_change = _check_bool_claim(
        _run_sync_step(
            "supports_metric_set_change()",
            backend.supports_metric_set_change,
            deadline=_timing().builder_deadline_seconds,
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
            deadline=_timing().builder_deadline_seconds,
        ),
        "supports_acquire_marker_authority",
    )
    durable_dedup = _check_bool_claim(
        _run_sync_step(
            "supports_durable_refund_dedup()",
            backend.supports_durable_refund_dedup,
            deadline=_timing().builder_deadline_seconds,
        ),
        "supports_durable_refund_dedup",
    )
    metric_set_change = _check_bool_claim(
        _run_sync_step(
            "supports_metric_set_change()",
            backend.supports_metric_set_change,
            deadline=_timing().builder_deadline_seconds,
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
            deadline=_timing().operation_deadline_seconds,
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
    promptness_deadline: float | None = None,
) -> None:
    if promptness_deadline is None:
        promptness_deadline = _timing().prompt_deadline_seconds
    start = time.monotonic()
    try:
        await _run_async_step(
            label,
            awaitable_fn,
            deadline=_timing().wait_budget_seconds,
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
            deadline=_timing().operation_deadline_seconds,
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
    promptness_deadline: float | None = None,
) -> None:
    if promptness_deadline is None:
        promptness_deadline = _timing().prompt_deadline_seconds
    start = time.monotonic()
    try:
        _sync_backend_step(
            label,
            fn,
            deadline=_timing().wait_budget_seconds,
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


def _check_public_reservation_fields(  # noqa: PLR0913
    reservation: object,
    *,
    expected_usage: FrozenUsage,
    expected_model_family: str,
    expected_model: str,
    expected_bucket_ids: frozenset[BucketId],
    expected_limiter_instance_id: str,
    expected_reservation_id: str | None = None,
    acquired_after_seconds: float | None = None,
    acquired_before_seconds: float | None = None,
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
    if expected_reservation_id is not None:
        _check(
            reservation.reservation_id == expected_reservation_id,
            "CapacityReservation.reservation_id did not match acquired reservation",
        )
    _check(
        type(reservation.limiter_instance_id) is str
        and bool(reservation.limiter_instance_id),
        "CapacityReservation.limiter_instance_id must be a non-empty str",
    )
    _check(
        reservation.limiter_instance_id == expected_limiter_instance_id,
        "CapacityReservation.limiter_instance_id did not match issuing limiter",
    )
    created_at_seconds = reservation.created_at_seconds
    if type(created_at_seconds) not in {int, float}:
        _fail("CapacityReservation.created_at_seconds must be a finite timestamp")
    created_at = float(cast("float | int", created_at_seconds))
    _check(
        math.isfinite(created_at),
        "CapacityReservation.created_at_seconds must be a finite timestamp",
    )
    if acquired_after_seconds is not None:
        _check(
            created_at
            >= acquired_after_seconds - _RESERVATION_TIMESTAMP_TOLERANCE_SECONDS,
            "CapacityReservation.created_at_seconds predates the acquire attempt",
        )
    if acquired_before_seconds is not None:
        _check(
            created_at
            <= acquired_before_seconds + _RESERVATION_TIMESTAMP_TOLERANCE_SECONDS,
            "CapacityReservation.created_at_seconds postdates the acquire attempt",
        )
    return reservation


def _check_acquire_refund_failed_payload(  # noqa: PLR0913
    exc: BaseException,
    *,
    refund_error: RuntimeError,
    interrupted_by: BaseException,
    expected_usage: FrozenUsage,
    expected_model_family: str,
    expected_model: str,
    expected_bucket_ids: frozenset[BucketId],
    expected_limiter_instance_id: str,
    expected_reservation_id: str,
    acquired_after_seconds: float,
    acquired_before_seconds: float,
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
        exc.args == (AcquireRefundFailedError._MESSAGE,),  # noqa: SLF001
        "AcquireRefundFailedError.args must contain the stable public message",
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
        exc.__suppress_context__ is True,
        "AcquireRefundFailedError must suppress implicit context when chained",
    )
    _check_public_reservation_fields(
        exc.reservation,
        expected_usage=expected_usage,
        expected_model_family=expected_model_family,
        expected_model=expected_model,
        expected_bucket_ids=expected_bucket_ids,
        expected_limiter_instance_id=expected_limiter_instance_id,
        expected_reservation_id=expected_reservation_id,
        acquired_after_seconds=acquired_after_seconds,
        acquired_before_seconds=acquired_before_seconds,
    )


def _check_refund_failure_backend_payload(
    backend: object,
    *,
    expected_usage: FrozenUsage,
    expected_actual_usage: FrozenUsage,
    expected_model_family: str,
    expected_bucket_ids: frozenset[BucketId],
) -> str:
    _check(
        backend is not None,
        "refund failure probe must build the fault-injection backend",
    )
    reservation_id = getattr(backend, "last_refund_reservation_id", None)
    _check(
        type(reservation_id) is str and bool(reservation_id),
        "refund failure probe did not capture the acquired reservation id",
    )
    _check(
        getattr(backend, "last_refund_reserved_usage", None) == expected_usage,
        "refund failure probe reserved_usage did not match acquired usage",
    )
    _check(
        getattr(backend, "last_refund_actual_usage", None) == expected_actual_usage,
        "refund failure probe actual_usage did not match zero cleanup usage",
    )
    _check(
        getattr(backend, "last_refund_bucket_ids", None) == expected_bucket_ids,
        "refund failure probe bucket_ids did not match acquired buckets",
    )
    _check(
        getattr(backend, "last_refund_reservation_model_family", None)
        == expected_model_family,
        "refund failure probe model_family did not match acquired reservation",
    )
    refund_reservation_bucket_ids = getattr(
        backend,
        "last_refund_reservation_bucket_ids",
        None,
    )
    if refund_reservation_bucket_ids is not None:
        _check(
            refund_reservation_bucket_ids == expected_bucket_ids,
            "refund failure probe reservation bucket_ids did not match acquired buckets",
        )
    refund_reservation_reserved_usage = getattr(
        backend,
        "last_refund_reservation_reserved_usage",
        None,
    )
    if refund_reservation_reserved_usage is not None:
        _check(
            refund_reservation_reserved_usage == expected_usage,
            "refund failure probe reservation usage did not match acquired usage",
        )
    return cast("str", reservation_id)


async def _construct_async_public_limiter(
    label: str,
    cfg: PerModelConfig,
    builder: RateLimiterBackendBuilderInterface,
) -> RateLimiter:
    try:
        return cast(
            "RateLimiter",
            _run_sync_step(
                label,
                lambda: RateLimiter(cfg, backend=builder),
                deadline=_timing().builder_deadline_seconds,
            ),
        )
    except BaseException:
        try:
            await _cleanup_async_builder(builder)
        # ast-guard: skip — builder cleanup cancellation must propagate
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as cleanup_exc:  # noqa: BLE001
            _warn_cleanup_failure(
                cleanup_exc,
                target="builder after failed limiter construction",
            )
        raise


def _construct_sync_public_limiter(
    label: str,
    cfg: PerModelConfig,
    builder: SyncRateLimiterBackendBuilderInterface,
) -> SyncRateLimiter:
    try:
        return cast(
            "SyncRateLimiter",
            _run_sync_step(
                label,
                lambda: SyncRateLimiter(cfg, backend=builder),
                deadline=_timing().builder_deadline_seconds,
            ),
        )
    except BaseException:
        try:
            _cleanup_sync_builder(builder)
        # ast-guard: skip — builder cleanup cancellation must propagate
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as cleanup_exc:  # noqa: BLE001
            _warn_cleanup_failure(
                cleanup_exc,
                target="builder after failed limiter construction",
            )
        raise


async def _cleanup_pending_async_public_refunds(
    limiter: RateLimiter,
    reservations: list[CapacityReservation],
    *,
    zero_usage: FrozenUsage,
    body_exc: BaseException | None,
) -> None:
    for reservation in reversed(reservations[:]):
        try:

            def call_refund(
                reservation: CapacityReservation = reservation,
            ) -> object:
                return limiter.refund_capacity(
                    zero_usage,
                    reservation,
                )

            await _run_async_step(
                "RateLimiter.refund_capacity(public reservation cleanup)",
                call_refund,
                deadline=_timing().operation_deadline_seconds,
                expect_awaitable=True,
            )
            _discard_pending_public_refund(reservations, reservation)
        # ast-guard: skip — public reservation refund cancellation must propagate
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            if body_exc is None:
                raise
            _warn_cleanup_failure(exc, target="public reservation refund")


def _cleanup_pending_sync_public_refunds(
    limiter: SyncRateLimiter,
    reservations: list[CapacityReservation],
    *,
    zero_usage: FrozenUsage,
    body_exc: BaseException | None,
) -> None:
    for reservation in reversed(reservations[:]):
        try:

            def call_refund(
                reservation: CapacityReservation = reservation,
            ) -> object:
                limiter.refund_capacity(
                    zero_usage,
                    reservation,
                )
                return None

            _run_sync_step(
                "SyncRateLimiter.refund_capacity(public reservation cleanup)",
                call_refund,
                deadline=_timing().operation_deadline_seconds,
            )
            _discard_pending_public_refund(reservations, reservation)
        # ast-guard: skip — public reservation refund cancellation must propagate
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            if body_exc is None:
                raise
            _warn_cleanup_failure(exc, target="public reservation refund")


def _discard_pending_public_refund(
    reservations: list[CapacityReservation],
    reservation: CapacityReservation,
) -> None:
    for index, pending in enumerate(reservations):
        if pending is reservation:
            del reservations[index]
            return


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
        time.monotonic() - start < _timing().prompt_deadline_seconds,
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
        time.monotonic() - start < _timing().prompt_deadline_seconds,
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
    callback_failures: list[str] = []
    callback_token = _CALLBACK_FAILURES.set(callback_failures)

    async def on_wait_start(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
    ) -> None:
        _check_callback_model_family("on_wait_start", model_family)
        _check_callback_usage("on_wait_start", "usage", usage)
        _check_callback_capacities(
            "on_wait_start", "preconsumption_capacities", preconsumption_capacities
        )
        events.append("wait_start")

    async def after_wait_end_consumption(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        wait_time_s: float,
    ) -> None:
        _check_callback_model_family("after_wait_end_consumption", model_family)
        _check_callback_usage("after_wait_end_consumption", "usage", usage)
        _check_callback_capacities(
            "after_wait_end_consumption",
            "preconsumption_capacities",
            preconsumption_capacities,
        )
        _check_callback_capacities(
            "after_wait_end_consumption",
            "postconsumption_capacities",
            postconsumption_capacities,
        )
        _check_callback_float("after_wait_end_consumption", "wait_time_s", wait_time_s)
        events.append("wait_end")

    async def on_capacity_consumed(
        *,
        model_family: str,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        usage: FrozenUsage,
        current_time: float,
    ) -> None:
        _check_callback_model_family("on_capacity_consumed", model_family)
        _check_callback_capacities(
            "on_capacity_consumed",
            "preconsumption_capacities",
            preconsumption_capacities,
        )
        _check_callback_capacities(
            "on_capacity_consumed",
            "postconsumption_capacities",
            postconsumption_capacities,
        )
        _check_callback_usage("on_capacity_consumed", "usage", usage)
        _check_callback_float("on_capacity_consumed", "current_time", current_time)
        events.append("consumed")

    async def on_capacity_refunded(  # noqa: PLR0913 - mirrors public callback protocol
        *,
        model_family: str,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        refunded_usage: FrozenUsage,
        prerefund_capacities: Capacities,
        postrefund_capacities: Capacities,
    ) -> None:
        _check_callback_model_family("on_capacity_refunded", model_family)
        _check_callback_usage("on_capacity_refunded", "reserved_usage", reserved_usage)
        _check_callback_usage("on_capacity_refunded", "actual_usage", actual_usage)
        _check_callback_usage("on_capacity_refunded", "refunded_usage", refunded_usage)
        _check_callback_capacities(
            "on_capacity_refunded", "prerefund_capacities", prerefund_capacities
        )
        _check_callback_capacities(
            "on_capacity_refunded", "postrefund_capacities", postrefund_capacities
        )
        events.append("refunded")

    async def on_missing_consumption_data(
        *,
        model_family: str,
        usage_metric: str,
        per_seconds: int,
    ) -> None:
        _check_callback_model_family("on_missing_consumption_data", model_family)
        _check_callback_payload(
            isinstance(usage_metric, str) and usage_metric == "requests",
            "on_missing_consumption_data callback passed invalid usage_metric",
        )
        _check_callback_payload(
            type(per_seconds) is int and per_seconds > 0,
            "on_missing_consumption_data callback passed invalid per_seconds",
        )
        events.append("missing")

    async def on_lifecycle_event(
        *,
        event: LifecycleEvent,
    ) -> None:
        _check_callback_payload(
            isinstance(event, LifecycleEvent)
            and event.model_family.startswith("conformance/"),
            "on_lifecycle_event callback passed invalid event",
        )
        events.append("lifecycle")

    try:
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
                on_lifecycle_event=on_lifecycle_event,
            ),
        )
        await _async_backend_step(
            "await_for_capacity(requests=4)",
            lambda: backend.await_for_capacity(
                frozen_usage({"requests": _CALLBACK_LIMIT})
            ),
        )
        await _async_backend_step(
            "await_for_capacity(requests=1, timeout=2)",
            lambda: backend.await_for_capacity(
                frozen_usage({"requests": 1}),
                timeout=_WAIT_TIMEOUT_SECONDS,
            ),
            deadline=_timing().wait_budget_seconds,
        )
        await _async_backend_step(
            "refund_capacity(requests=1, actual=0)",
            lambda: backend.refund_capacity(
                frozen_usage({"requests": 1}),
                frozen_usage({"requests": 0}),
            ),
        )
    finally:
        _CALLBACK_FAILURES.reset(callback_token)

    if callback_failures:
        _fail(callback_failures[0])

    for event in ("missing", "consumed", "wait_start", "wait_end", "refunded"):
        _check(
            event in events,
            f"callback slot {event!r} was not emitted with a valid payload",
        )


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
            deadline=_timing().builder_deadline_seconds,
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
    bucket_ids: set[BucketId] | frozenset[BucketId] | None = frozenset(
        {_REQUESTS_BUCKET_ID}
    ),
    reservation_id: str,
    reservation_model_family: object,
    reservation_bucket_ids: object = _DEFAULT_MARKER_BUCKET_IDS,
    reservation_reserved_usage: object = _DEFAULT_MARKER_RESERVED_USAGE,
) -> object:
    kwargs: dict[str, Any] = {
        "bucket_ids": bucket_ids,
        "reservation_id": reservation_id,
    }
    if reservation_model_family is not _OMIT_MARKER_METADATA:
        kwargs["reservation_model_family"] = reservation_model_family
    if reservation_bucket_ids is _DEFAULT_MARKER_BUCKET_IDS:
        kwargs["reservation_bucket_ids"] = frozenset({_REQUESTS_BUCKET_ID})
    elif reservation_bucket_ids is not _OMIT_MARKER_METADATA:
        kwargs["reservation_bucket_ids"] = reservation_bucket_ids
    if reservation_reserved_usage is _DEFAULT_MARKER_RESERVED_USAGE:
        kwargs["reservation_reserved_usage"] = reserved_usage
    elif reservation_reserved_usage is not _OMIT_MARKER_METADATA:
        kwargs["reservation_reserved_usage"] = reservation_reserved_usage
    return await _async_backend_step(
        label,
        lambda: backend.refund_capacity_for_buckets(
            reserved_usage,
            actual_usage,
            **kwargs,
        ),
        allowed_exceptions=(UnknownReservationError, DuplicateRefundError, ValueError),
    )


async def _check_async_no_capacity(
    backend: RateLimiterBackend,
    usage: FrozenUsage,
    label: str,
    message: str,
) -> None:
    await _expect_async_timeout(
        label,
        lambda: backend.await_for_capacity(
            usage,
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        message,
    )


async def _check_async_no_double_refund_credit(
    backend: RateLimiterBackend,
    message: str,
) -> None:
    await _check_async_no_capacity(
        backend,
        frozen_usage({"requests": 2}),
        "await_for_capacity(requests=2, timeout=0)",
        message,
    )


async def _check_async_marker_metadata_mismatch(
    builder: RateLimiterBackendBuilderInterface,
    mismatch_field: str,
    *,
    reservation_model_family: str | None = None,
    reservation_bucket_ids: frozenset[BucketId] | None = None,
    reservation_reserved_usage: object = _DEFAULT_MARKER_RESERVED_USAGE,
) -> None:
    mismatch_cfg = _config(f"async-marker-mismatch-{mismatch_field}", limit=2.0)
    mismatch_backend = _build_async_backend(
        builder,
        mismatch_cfg,
        label=f"build(async-marker-mismatch-{mismatch_field})",
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
    with contextlib.suppress(UnknownReservationError, DuplicateRefundError, ValueError):
        await _async_refund_for_buckets(
            f"refund_capacity_for_buckets(metadata-mismatch-{mismatch_field})",
            mismatch_backend,
            mismatch_reserved_usage,
            frozen_usage({"requests": 1}),
            reservation_id=mismatch_reservation_id,
            reservation_model_family=(
                mismatch_cfg.get_model_family()
                if reservation_model_family is None
                else reservation_model_family
            ),
            reservation_bucket_ids=(
                frozenset({_REQUESTS_BUCKET_ID})
                if reservation_bucket_ids is None
                else reservation_bucket_ids
            ),
            reservation_reserved_usage=reservation_reserved_usage,
        )
    await _check_async_no_capacity(
        mismatch_backend,
        frozen_usage({"requests": 1}),
        "await_for_capacity(requests=1, timeout=0)",
        "marker-authority backends must fail closed for "
        f"reservation metadata mismatch ({mismatch_field})",
    )


async def _check_async_marker_metadata_omission(
    builder: RateLimiterBackendBuilderInterface,
    omitted_field: str,
) -> None:
    omission_cfg = _config(f"async-marker-omitted-{omitted_field}", limit=2.0)
    omission_backend = _build_async_backend(
        builder,
        omission_cfg,
        label=f"build(async-marker-omitted-{omitted_field})",
    )
    omission_reservation_id = f"conformance-{uuid.uuid4().hex}"
    omission_reserved_usage = frozen_usage({"requests": 2})
    reservation_model_family: object = omission_cfg.get_model_family()
    reservation_bucket_ids: object = frozenset({_REQUESTS_BUCKET_ID})
    reservation_reserved_usage: object = omission_reserved_usage
    if omitted_field == "reservation_model_family":
        reservation_model_family = _OMIT_MARKER_METADATA
    elif omitted_field == "reservation_bucket_ids":
        reservation_bucket_ids = _OMIT_MARKER_METADATA
    elif omitted_field == "reservation_reserved_usage":
        reservation_reserved_usage = _OMIT_MARKER_METADATA
    else:
        _fail(f"unknown marker metadata omission field {omitted_field!r}")
    await _async_backend_step(
        "await_for_capacity(requests=2, reservation_id=...)",
        lambda: omission_backend.await_for_capacity(
            omission_reserved_usage,
            reservation_id=omission_reservation_id,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        ),
    )
    with contextlib.suppress(UnknownReservationError, DuplicateRefundError, ValueError):
        await _async_refund_for_buckets(
            f"refund_capacity_for_buckets(metadata-omitted-{omitted_field})",
            omission_backend,
            omission_reserved_usage,
            frozen_usage({"requests": 1}),
            reservation_id=omission_reservation_id,
            reservation_model_family=reservation_model_family,
            reservation_bucket_ids=reservation_bucket_ids,
            reservation_reserved_usage=reservation_reserved_usage,
        )
    await _check_async_no_capacity(
        omission_backend,
        frozen_usage({"requests": 1}),
        "await_for_capacity(requests=1, timeout=0)",
        "marker-authority backends must fail closed when "
        f"{omitted_field} marker metadata is omitted",
    )


async def _check_async_marker_refund_scope_forgery(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    scope_cfg = _two_metric_config("async-marker-refund-scope", limit=2.0)
    scope_backend = _build_async_backend(
        builder,
        scope_cfg,
        label="build(async-marker-refund-scope)",
    )
    scope_reservation_id = f"conformance-{uuid.uuid4().hex}"
    marker_reserved_usage = frozen_usage({"requests": 2, "tokens": 0})
    await _async_backend_step(
        "await_for_capacity(requests=2, tokens=0, reservation_id=...)",
        lambda: scope_backend.await_for_capacity(
            marker_reserved_usage,
            reservation_id=scope_reservation_id,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        ),
    )
    await _async_backend_step(
        "await_for_capacity(requests=0, tokens=2)",
        lambda: scope_backend.await_for_capacity(
            frozen_usage({"requests": 0, "tokens": 2})
        ),
    )
    with contextlib.suppress(UnknownReservationError, DuplicateRefundError, ValueError):
        await _async_refund_for_buckets(
            "refund_capacity_for_buckets(forged-refund-scope)",
            scope_backend,
            frozen_usage({"tokens": 2}),
            frozen_usage({"tokens": 1}),
            bucket_ids=frozenset({_TOKENS_BUCKET_ID}),
            reservation_id=scope_reservation_id,
            reservation_model_family=scope_cfg.get_model_family(),
            reservation_bucket_ids=frozenset({_REQUESTS_BUCKET_ID, _TOKENS_BUCKET_ID}),
            reservation_reserved_usage=marker_reserved_usage,
        )
    await _check_async_no_capacity(
        scope_backend,
        frozen_usage({"requests": 0, "tokens": 1}),
        "await_for_capacity(requests=0, tokens=1, timeout=0)",
        "marker-authority backends must fail closed for forged refund bucket_ids",
    )
    await _check_async_no_capacity(
        scope_backend,
        frozen_usage({"requests": 1, "tokens": 0}),
        "await_for_capacity(requests=1, tokens=0, timeout=0)",
        "marker-authority backends must not credit the genuine bucket for "
        "forged refund bucket_ids",
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
            deadline=_timing().builder_deadline_seconds,
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
    with contextlib.suppress(DuplicateRefundError):
        await _async_refund_for_buckets(
            "refund_capacity_for_buckets(durable-duplicate)",
            backend,
            reserved_usage,
            frozen_usage({"requests": 1}),
            reservation_id=reservation_id,
            reservation_model_family=cfg.get_model_family(),
        )
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
            deadline=_timing().builder_deadline_seconds,
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
    with contextlib.suppress(UnknownReservationError):
        await _async_refund_for_buckets(
            "refund_capacity_for_buckets(unknown-reservation)",
            unknown_backend,
            frozen_usage({"requests": 2}),
            frozen_usage({"requests": 1}),
            reservation_id=f"unknown-{uuid.uuid4().hex}",
            reservation_model_family=unknown_cfg.get_model_family(),
        )
    await _check_async_no_capacity(
        unknown_backend,
        frozen_usage({"requests": 1}),
        "await_for_capacity(requests=1, timeout=0)",
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
    with contextlib.suppress(DuplicateRefundError):
        await _async_refund_for_buckets(
            "refund_capacity_for_buckets(duplicate-second)",
            duplicate_backend,
            duplicate_reserved_usage,
            frozen_usage({"requests": 1}),
            reservation_id=duplicate_reservation_id,
            reservation_model_family=duplicate_cfg.get_model_family(),
        )
    await _check_async_no_double_refund_credit(
        duplicate_backend,
        "marker-authority backends must not credit duplicate refunds twice",
    )

    await _check_async_marker_metadata_mismatch(
        builder,
        "model_family",
        reservation_model_family=_family("forged-marker-family"),
    )
    await _check_async_marker_metadata_mismatch(
        builder,
        "bucket_ids",
        reservation_bucket_ids=frozenset(
            {_REQUESTS_BUCKET_ID, ("forged-bucket", _SHORT_WINDOW_SECONDS)}
        ),
    )
    await _check_async_marker_metadata_mismatch(
        builder,
        "reserved_usage",
        reservation_reserved_usage=frozen_usage({"requests": 1}),
    )
    await _check_async_marker_refund_scope_forgery(builder)
    for omitted_field in (
        "reservation_model_family",
        "reservation_bucket_ids",
        "reservation_reserved_usage",
    ):
        await _check_async_marker_metadata_omission(builder, omitted_field)


async def _check_async_public_reservation_round_trip(
    backend_builder: RateLimiterBackendBuilderInterface,
) -> None:
    cfg = _two_metric_config("async-public-round-trip")
    usage = frozen_usage({"requests": 1.0, "tokens": 1.0})
    full_usage = frozen_usage({"requests": 2.0, "tokens": 2.0})
    zero_usage = frozen_usage({"requests": 0.0, "tokens": 0.0})
    bucket_ids = frozenset({_REQUESTS_BUCKET_ID, _TOKENS_BUCKET_ID})
    limiter = await _construct_async_public_limiter(
        "RateLimiter construction",
        cfg,
        backend_builder,
    )
    expected_limiter_instance_id = cast("str", limiter._limiter_instance_id)  # noqa: SLF001
    pending_refunds: list[CapacityReservation] = []
    body_exc: BaseException | None = None

    async def acquire_public_reservation(
        label: str,
        expected_usage: FrozenUsage,
        *,
        acquire_timeout: float | None = None,
    ) -> CapacityReservation:
        started_at = time.time()
        raw_reservation = await _run_async_step(
            label,
            lambda: limiter.acquire_capacity(
                expected_usage,
                _PUBLIC_MODEL_NAME,
                timeout=acquire_timeout,
            ),
            deadline=_timing().operation_deadline_seconds,
            expect_awaitable=True,
        )
        finished_at = time.time()
        if type(raw_reservation) is CapacityReservation:
            pending_refunds.append(cast("CapacityReservation", raw_reservation))
        return _check_public_reservation_fields(
            raw_reservation,
            expected_usage=expected_usage,
            expected_model_family=cfg.get_model_family(),
            expected_model=_PUBLIC_MODEL_NAME,
            expected_bucket_ids=bucket_ids,
            expected_limiter_instance_id=expected_limiter_instance_id,
            acquired_after_seconds=started_at,
            acquired_before_seconds=finished_at,
        )

    async def refund_public_reservation(
        label: str,
        reservation: CapacityReservation,
    ) -> None:
        await _run_async_step(
            label,
            lambda: limiter.refund_capacity(zero_usage, reservation),
            deadline=_timing().operation_deadline_seconds,
            expect_awaitable=True,
        )
        _discard_pending_public_refund(pending_refunds, reservation)

    try:
        reservation = await acquire_public_reservation(
            "RateLimiter.acquire_capacity(public reservation)",
            usage,
        )
        await refund_public_reservation(
            "RateLimiter.refund_capacity(public reservation)",
            reservation,
        )

        restored = await acquire_public_reservation(
            "RateLimiter.acquire_capacity(restored capacity)",
            full_usage,
            acquire_timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        )
        await refund_public_reservation(
            "RateLimiter.refund_capacity(restored reservation)",
            restored,
        )

        mutated = restored.model_copy(update={"usage": zero_usage})
        _check(
            type(mutated) is CapacityReservation,
            "CapacityReservation.model_copy() must preserve exact public type",
        )
        snapshot_probe = await acquire_public_reservation(
            "RateLimiter.acquire_capacity(snapshot authority)",
            full_usage,
            acquire_timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        )
        mutated_snapshot_probe = snapshot_probe.model_copy(update={"usage": zero_usage})
        await _run_async_step(
            "RateLimiter.refund_capacity(mutated reservation snapshot authority)",
            lambda: limiter.refund_capacity(zero_usage, mutated_snapshot_probe),
            deadline=_timing().operation_deadline_seconds,
            expect_awaitable=True,
        )
        _discard_pending_public_refund(pending_refunds, snapshot_probe)
        final_reservation = await acquire_public_reservation(
            "RateLimiter.acquire_capacity(after mutated reservation refund)",
            full_usage,
            acquire_timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        )
        await refund_public_reservation(
            "RateLimiter.refund_capacity(final public reservation)",
            final_reservation,
        )
    except BaseException as exc:
        body_exc = exc
        raise
    finally:
        cleanup_body_exc = body_exc
        try:
            await _cleanup_pending_async_public_refunds(
                limiter,
                pending_refunds,
                zero_usage=zero_usage,
                body_exc=body_exc,
            )
        except BaseException as exc:
            cleanup_body_exc = exc
            raise
        finally:
            await _cleanup_async_limiter(limiter, body_exc=cleanup_body_exc)


async def _check_async_acquire_refund_failed_error(
    backend_builder: RateLimiterBackendBuilderInterface,
) -> None:
    refund_error = RuntimeError("fault-injection")
    interrupted_by = asyncio.CancelledError("fault-injection")
    cfg = _config("async-acquire-refund-failed", limit=2.0)
    usage = frozen_usage({"requests": 1.0})
    zero_usage = frozen_usage({"requests": 0.0})
    bucket_ids = frozenset({_REQUESTS_BUCKET_ID})
    wrapped_builder = _AsyncRefundFailureBuilder(backend_builder, refund_error)
    limiter = await _construct_async_public_limiter(
        "RateLimiter construction",
        cfg,
        wrapped_builder,
    )
    expected_limiter_instance_id = cast("str", limiter._limiter_instance_id)  # noqa: SLF001
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
    body_exc: BaseException | None = None
    try:
        try:
            started_at = time.time()
            await _run_async_step(
                "RateLimiter.acquire_capacity(acquire-refund-failed probe)",
                lambda: limiter.acquire_capacity(
                    usage,
                    _PUBLIC_MODEL_NAME,
                ),
                deadline=_timing().operation_deadline_seconds,
                expect_awaitable=True,
                allowed_exceptions=(AcquireRefundFailedError,),
            )
        except AcquireRefundFailedError as exc:
            finished_at = time.time()
            expected_reservation_id = _check_refund_failure_backend_payload(
                wrapped_builder.last_backend,
                expected_usage=usage,
                expected_actual_usage=zero_usage,
                expected_model_family=cfg.get_model_family(),
                expected_bucket_ids=bucket_ids,
            )
            _check_acquire_refund_failed_payload(
                exc,
                refund_error=refund_error,
                interrupted_by=interrupted_by,
                expected_usage=usage,
                expected_model_family=cfg.get_model_family(),
                expected_model=_PUBLIC_MODEL_NAME,
                expected_bucket_ids=bucket_ids,
                expected_limiter_instance_id=expected_limiter_instance_id,
                expected_reservation_id=expected_reservation_id,
                acquired_after_seconds=started_at,
                acquired_before_seconds=finished_at,
            )
            return
        _fail("interrupted acquire cleanup must raise AcquireRefundFailedError")
    except BaseException as exc:
        body_exc = exc
        raise
    finally:
        await _cleanup_async_limiter(limiter, body_exc=body_exc)


async def conformance_test_for(
    backend_builder: RateLimiterBackendBuilderInterface,
    *,
    timing: ConformanceTiming | None = None,
) -> None:
    """
    Run the public async backend conformance checks for one backend builder.

    The builder should point at isolated backend state: use a disposable Redis
    key prefix, database, or in-memory instance so these tests can consume and
    refund capacity freely.

    Backend operations are bounded by helper-owned deadlines. Pass
    ``ConformanceTiming`` to tune those deadlines, or set
    ``TOKEN_THROTTLE_CONFORMANCE_TIMING_SCALE`` to multiply the defaults for
    slow runners. When ``timing=`` is passed, the env var is ignored; set every
    desired field on the dataclass.

    Unlike ``RateLimiter``, which tolerates a builder that omits ``aclose()``/
    ``close()`` entirely, this helper requires the builder to satisfy the full
    ``RateLimiterBackendBuilderInterface`` protocol — including those two
    hooks — since it type-checks the builder before running any test. Subclass
    the interface (it provides no-op defaults) or otherwise define both
    methods; the helper then calls them during cleanup.

    KNOWN LIMITATION: if a synchronous backend call hangs in its worker thread,
    Python cannot safely kill that thread; the helper reports the hang and
    continues without waiting for that thread to finish.
    """
    resolved_timing = _resolve_timing(timing)
    token = _TIMING_CONTEXT.set(resolved_timing)
    body_exc: BaseException | None = None
    try:
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
    except BaseException as exc:
        body_exc = exc
        raise
    finally:
        try:
            await _cleanup_async_builder(backend_builder)
        # ast-guard: skip — builder cleanup cancellation must propagate
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            if body_exc is None:
                raise
            _warn_cleanup_failure(exc)
        finally:
            _TIMING_CONTEXT.reset(token)


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
        time.monotonic() - start < _timing().prompt_deadline_seconds,
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
        time.monotonic() - start < _timing().prompt_deadline_seconds,
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
    callback_failures: list[str] = []
    callback_token = _CALLBACK_FAILURES.set(callback_failures)

    def on_wait_start(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
    ) -> None:
        _check_callback_model_family("on_wait_start", model_family)
        _check_callback_usage("on_wait_start", "usage", usage)
        _check_callback_capacities(
            "on_wait_start", "preconsumption_capacities", preconsumption_capacities
        )
        events.append("wait_start")

    def after_wait_end_consumption(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        wait_time_s: float,
    ) -> None:
        _check_callback_model_family("after_wait_end_consumption", model_family)
        _check_callback_usage("after_wait_end_consumption", "usage", usage)
        _check_callback_capacities(
            "after_wait_end_consumption",
            "preconsumption_capacities",
            preconsumption_capacities,
        )
        _check_callback_capacities(
            "after_wait_end_consumption",
            "postconsumption_capacities",
            postconsumption_capacities,
        )
        _check_callback_float("after_wait_end_consumption", "wait_time_s", wait_time_s)
        events.append("wait_end")

    def on_capacity_consumed(
        *,
        model_family: str,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        usage: FrozenUsage,
        current_time: float,
    ) -> None:
        _check_callback_model_family("on_capacity_consumed", model_family)
        _check_callback_capacities(
            "on_capacity_consumed",
            "preconsumption_capacities",
            preconsumption_capacities,
        )
        _check_callback_capacities(
            "on_capacity_consumed",
            "postconsumption_capacities",
            postconsumption_capacities,
        )
        _check_callback_usage("on_capacity_consumed", "usage", usage)
        _check_callback_float("on_capacity_consumed", "current_time", current_time)
        events.append("consumed")

    def on_capacity_refunded(  # noqa: PLR0913 - mirrors public callback protocol
        *,
        model_family: str,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        refunded_usage: FrozenUsage,
        prerefund_capacities: Capacities,
        postrefund_capacities: Capacities,
    ) -> None:
        _check_callback_model_family("on_capacity_refunded", model_family)
        _check_callback_usage("on_capacity_refunded", "reserved_usage", reserved_usage)
        _check_callback_usage("on_capacity_refunded", "actual_usage", actual_usage)
        _check_callback_usage("on_capacity_refunded", "refunded_usage", refunded_usage)
        _check_callback_capacities(
            "on_capacity_refunded", "prerefund_capacities", prerefund_capacities
        )
        _check_callback_capacities(
            "on_capacity_refunded", "postrefund_capacities", postrefund_capacities
        )
        events.append("refunded")

    def on_missing_consumption_data(
        *,
        model_family: str,
        usage_metric: str,
        per_seconds: int,
    ) -> None:
        _check_callback_model_family("on_missing_consumption_data", model_family)
        _check_callback_payload(
            isinstance(usage_metric, str) and usage_metric == "requests",
            "on_missing_consumption_data callback passed invalid usage_metric",
        )
        _check_callback_payload(
            type(per_seconds) is int and per_seconds > 0,
            "on_missing_consumption_data callback passed invalid per_seconds",
        )
        events.append("missing")

    def on_lifecycle_event(
        *,
        event: LifecycleEvent,
    ) -> None:
        _check_callback_payload(
            isinstance(event, LifecycleEvent)
            and event.model_family.startswith("conformance/"),
            "on_lifecycle_event callback passed invalid event",
        )
        events.append("lifecycle")

    try:
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
                on_lifecycle_event=on_lifecycle_event,
            ),
        )
        _sync_backend_step(
            "wait_for_capacity(requests=4)",
            lambda: backend.wait_for_capacity(
                frozen_usage({"requests": _CALLBACK_LIMIT})
            ),
        )
        _sync_backend_step(
            "wait_for_capacity(requests=1, timeout=2)",
            lambda: backend.wait_for_capacity(
                frozen_usage({"requests": 1}),
                timeout=_WAIT_TIMEOUT_SECONDS,
            ),
            deadline=_timing().wait_budget_seconds,
        )
        _sync_backend_step(
            "refund_capacity(requests=1, actual=0)",
            lambda: backend.refund_capacity(
                frozen_usage({"requests": 1}),
                frozen_usage({"requests": 0}),
            ),
        )
    finally:
        _CALLBACK_FAILURES.reset(callback_token)

    if callback_failures:
        _fail(callback_failures[0])

    for event in ("missing", "consumed", "wait_start", "wait_end", "refunded"):
        _check(
            event in events,
            f"callback slot {event!r} was not emitted with a valid payload",
        )


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
            deadline=_timing().builder_deadline_seconds,
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
    bucket_ids: set[BucketId] | frozenset[BucketId] | None = frozenset(
        {_REQUESTS_BUCKET_ID}
    ),
    reservation_id: str,
    reservation_model_family: object,
    reservation_bucket_ids: object = _DEFAULT_MARKER_BUCKET_IDS,
    reservation_reserved_usage: object = _DEFAULT_MARKER_RESERVED_USAGE,
) -> object:
    kwargs: dict[str, Any] = {
        "bucket_ids": bucket_ids,
        "reservation_id": reservation_id,
    }
    if reservation_model_family is not _OMIT_MARKER_METADATA:
        kwargs["reservation_model_family"] = reservation_model_family
    if reservation_bucket_ids is _DEFAULT_MARKER_BUCKET_IDS:
        kwargs["reservation_bucket_ids"] = frozenset({_REQUESTS_BUCKET_ID})
    elif reservation_bucket_ids is not _OMIT_MARKER_METADATA:
        kwargs["reservation_bucket_ids"] = reservation_bucket_ids
    if reservation_reserved_usage is _DEFAULT_MARKER_RESERVED_USAGE:
        kwargs["reservation_reserved_usage"] = reserved_usage
    elif reservation_reserved_usage is not _OMIT_MARKER_METADATA:
        kwargs["reservation_reserved_usage"] = reservation_reserved_usage
    return _sync_backend_step(
        label,
        lambda: backend.refund_capacity_for_buckets(
            reserved_usage,
            actual_usage,
            **kwargs,
        ),
        allowed_exceptions=(UnknownReservationError, DuplicateRefundError, ValueError),
    )


def _check_sync_no_capacity(
    backend: SyncRateLimiterBackend,
    usage: FrozenUsage,
    label: str,
    message: str,
) -> None:
    _expect_timeout(
        label,
        lambda: backend.wait_for_capacity(
            usage,
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        message,
    )


def _check_sync_no_double_refund_credit(
    backend: SyncRateLimiterBackend,
    message: str,
) -> None:
    _check_sync_no_capacity(
        backend,
        frozen_usage({"requests": 2}),
        "wait_for_capacity(requests=2, timeout=0)",
        message,
    )


def _check_sync_marker_metadata_mismatch(
    builder: SyncRateLimiterBackendBuilderInterface,
    mismatch_field: str,
    *,
    reservation_model_family: str | None = None,
    reservation_bucket_ids: frozenset[BucketId] | None = None,
    reservation_reserved_usage: object = _DEFAULT_MARKER_RESERVED_USAGE,
) -> None:
    mismatch_cfg = _config(f"sync-marker-mismatch-{mismatch_field}", limit=2.0)
    mismatch_backend = _build_sync_backend(
        builder,
        mismatch_cfg,
        label=f"build(sync-marker-mismatch-{mismatch_field})",
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
    with contextlib.suppress(UnknownReservationError, DuplicateRefundError, ValueError):
        _sync_refund_for_buckets(
            f"refund_capacity_for_buckets(metadata-mismatch-{mismatch_field})",
            mismatch_backend,
            mismatch_reserved_usage,
            frozen_usage({"requests": 1}),
            reservation_id=mismatch_reservation_id,
            reservation_model_family=(
                mismatch_cfg.get_model_family()
                if reservation_model_family is None
                else reservation_model_family
            ),
            reservation_bucket_ids=(
                frozenset({_REQUESTS_BUCKET_ID})
                if reservation_bucket_ids is None
                else reservation_bucket_ids
            ),
            reservation_reserved_usage=reservation_reserved_usage,
        )
    _check_sync_no_capacity(
        mismatch_backend,
        frozen_usage({"requests": 1}),
        "wait_for_capacity(requests=1, timeout=0)",
        "marker-authority backends must fail closed for "
        f"reservation metadata mismatch ({mismatch_field})",
    )


def _check_sync_marker_metadata_omission(
    builder: SyncRateLimiterBackendBuilderInterface,
    omitted_field: str,
) -> None:
    omission_cfg = _config(f"sync-marker-omitted-{omitted_field}", limit=2.0)
    omission_backend = _build_sync_backend(
        builder,
        omission_cfg,
        label=f"build(sync-marker-omitted-{omitted_field})",
    )
    omission_reservation_id = f"conformance-{uuid.uuid4().hex}"
    omission_reserved_usage = frozen_usage({"requests": 2})
    reservation_model_family: object = omission_cfg.get_model_family()
    reservation_bucket_ids: object = frozenset({_REQUESTS_BUCKET_ID})
    reservation_reserved_usage: object = omission_reserved_usage
    if omitted_field == "reservation_model_family":
        reservation_model_family = _OMIT_MARKER_METADATA
    elif omitted_field == "reservation_bucket_ids":
        reservation_bucket_ids = _OMIT_MARKER_METADATA
    elif omitted_field == "reservation_reserved_usage":
        reservation_reserved_usage = _OMIT_MARKER_METADATA
    else:
        _fail(f"unknown marker metadata omission field {omitted_field!r}")
    _sync_backend_step(
        "wait_for_capacity(requests=2, reservation_id=...)",
        lambda: omission_backend.wait_for_capacity(
            omission_reserved_usage,
            reservation_id=omission_reservation_id,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        ),
    )
    with contextlib.suppress(UnknownReservationError, DuplicateRefundError, ValueError):
        _sync_refund_for_buckets(
            f"refund_capacity_for_buckets(metadata-omitted-{omitted_field})",
            omission_backend,
            omission_reserved_usage,
            frozen_usage({"requests": 1}),
            reservation_id=omission_reservation_id,
            reservation_model_family=reservation_model_family,
            reservation_bucket_ids=reservation_bucket_ids,
            reservation_reserved_usage=reservation_reserved_usage,
        )
    _check_sync_no_capacity(
        omission_backend,
        frozen_usage({"requests": 1}),
        "wait_for_capacity(requests=1, timeout=0)",
        "marker-authority backends must fail closed when "
        f"{omitted_field} marker metadata is omitted",
    )


def _check_sync_marker_refund_scope_forgery(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    scope_cfg = _two_metric_config("sync-marker-refund-scope", limit=2.0)
    scope_backend = _build_sync_backend(
        builder,
        scope_cfg,
        label="build(sync-marker-refund-scope)",
    )
    scope_reservation_id = f"conformance-{uuid.uuid4().hex}"
    marker_reserved_usage = frozen_usage({"requests": 2, "tokens": 0})
    _sync_backend_step(
        "wait_for_capacity(requests=2, tokens=0, reservation_id=...)",
        lambda: scope_backend.wait_for_capacity(
            marker_reserved_usage,
            reservation_id=scope_reservation_id,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        ),
    )
    _sync_backend_step(
        "wait_for_capacity(requests=0, tokens=2)",
        lambda: scope_backend.wait_for_capacity(
            frozen_usage({"requests": 0, "tokens": 2})
        ),
    )
    with contextlib.suppress(UnknownReservationError, DuplicateRefundError, ValueError):
        _sync_refund_for_buckets(
            "refund_capacity_for_buckets(forged-refund-scope)",
            scope_backend,
            frozen_usage({"tokens": 2}),
            frozen_usage({"tokens": 1}),
            bucket_ids=frozenset({_TOKENS_BUCKET_ID}),
            reservation_id=scope_reservation_id,
            reservation_model_family=scope_cfg.get_model_family(),
            reservation_bucket_ids=frozenset({_REQUESTS_BUCKET_ID, _TOKENS_BUCKET_ID}),
            reservation_reserved_usage=marker_reserved_usage,
        )
    _check_sync_no_capacity(
        scope_backend,
        frozen_usage({"requests": 0, "tokens": 1}),
        "wait_for_capacity(requests=0, tokens=1, timeout=0)",
        "marker-authority backends must fail closed for forged refund bucket_ids",
    )
    _check_sync_no_capacity(
        scope_backend,
        frozen_usage({"requests": 1, "tokens": 0}),
        "wait_for_capacity(requests=1, tokens=0, timeout=0)",
        "marker-authority backends must not credit the genuine bucket for "
        "forged refund bucket_ids",
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
            deadline=_timing().builder_deadline_seconds,
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
    with contextlib.suppress(DuplicateRefundError):
        _sync_refund_for_buckets(
            "refund_capacity_for_buckets(durable-duplicate)",
            backend,
            reserved_usage,
            frozen_usage({"requests": 1}),
            reservation_id=reservation_id,
            reservation_model_family=cfg.get_model_family(),
        )
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
            deadline=_timing().builder_deadline_seconds,
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
    with contextlib.suppress(UnknownReservationError):
        _sync_refund_for_buckets(
            "refund_capacity_for_buckets(unknown-reservation)",
            unknown_backend,
            frozen_usage({"requests": 2}),
            frozen_usage({"requests": 1}),
            reservation_id=f"unknown-{uuid.uuid4().hex}",
            reservation_model_family=unknown_cfg.get_model_family(),
        )
    _check_sync_no_capacity(
        unknown_backend,
        frozen_usage({"requests": 1}),
        "wait_for_capacity(requests=1, timeout=0)",
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
    with contextlib.suppress(DuplicateRefundError):
        _sync_refund_for_buckets(
            "refund_capacity_for_buckets(duplicate-second)",
            duplicate_backend,
            duplicate_reserved_usage,
            frozen_usage({"requests": 1}),
            reservation_id=duplicate_reservation_id,
            reservation_model_family=duplicate_cfg.get_model_family(),
        )
    _check_sync_no_double_refund_credit(
        duplicate_backend,
        "marker-authority backends must not credit duplicate refunds twice",
    )

    _check_sync_marker_metadata_mismatch(
        builder,
        "model_family",
        reservation_model_family=_family("forged-marker-family"),
    )
    _check_sync_marker_metadata_mismatch(
        builder,
        "bucket_ids",
        reservation_bucket_ids=frozenset(
            {_REQUESTS_BUCKET_ID, ("forged-bucket", _SHORT_WINDOW_SECONDS)}
        ),
    )
    _check_sync_marker_metadata_mismatch(
        builder,
        "reserved_usage",
        reservation_reserved_usage=frozen_usage({"requests": 1}),
    )
    _check_sync_marker_refund_scope_forgery(builder)
    for omitted_field in (
        "reservation_model_family",
        "reservation_bucket_ids",
        "reservation_reserved_usage",
    ):
        _check_sync_marker_metadata_omission(builder, omitted_field)


def _check_sync_public_reservation_round_trip(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    cfg = _two_metric_config("sync-public-round-trip")
    usage = frozen_usage({"requests": 1.0, "tokens": 1.0})
    full_usage = frozen_usage({"requests": 2.0, "tokens": 2.0})
    zero_usage = frozen_usage({"requests": 0.0, "tokens": 0.0})
    bucket_ids = frozenset({_REQUESTS_BUCKET_ID, _TOKENS_BUCKET_ID})
    limiter = _construct_sync_public_limiter(
        "SyncRateLimiter construction",
        cfg,
        builder,
    )
    expected_limiter_instance_id = cast("str", limiter._limiter_instance_id)  # noqa: SLF001
    pending_refunds: list[CapacityReservation] = []
    body_exc: BaseException | None = None

    def acquire_public_reservation(
        label: str,
        expected_usage: FrozenUsage,
        *,
        timeout: float | None = None,
    ) -> CapacityReservation:
        started_at = time.time()
        raw_reservation = _run_sync_step(
            label,
            lambda: limiter.acquire_capacity(
                expected_usage,
                _PUBLIC_MODEL_NAME,
                timeout=timeout,
            ),
            deadline=_timing().operation_deadline_seconds,
        )
        finished_at = time.time()
        if type(raw_reservation) is CapacityReservation:
            pending_refunds.append(cast("CapacityReservation", raw_reservation))
        return _check_public_reservation_fields(
            raw_reservation,
            expected_usage=expected_usage,
            expected_model_family=cfg.get_model_family(),
            expected_model=_PUBLIC_MODEL_NAME,
            expected_bucket_ids=bucket_ids,
            expected_limiter_instance_id=expected_limiter_instance_id,
            acquired_after_seconds=started_at,
            acquired_before_seconds=finished_at,
        )

    def refund_public_reservation(
        label: str,
        reservation: CapacityReservation,
    ) -> None:
        _run_sync_step(
            label,
            lambda: limiter.refund_capacity(zero_usage, reservation),
            deadline=_timing().operation_deadline_seconds,
        )
        _discard_pending_public_refund(pending_refunds, reservation)

    try:
        reservation = acquire_public_reservation(
            "SyncRateLimiter.acquire_capacity(public reservation)",
            usage,
        )
        refund_public_reservation(
            "SyncRateLimiter.refund_capacity(public reservation)",
            reservation,
        )

        restored = acquire_public_reservation(
            "SyncRateLimiter.acquire_capacity(restored capacity)",
            full_usage,
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        )
        refund_public_reservation(
            "SyncRateLimiter.refund_capacity(restored reservation)",
            restored,
        )

        mutated = restored.model_copy(update={"usage": zero_usage})
        _check(
            type(mutated) is CapacityReservation,
            "CapacityReservation.model_copy() must preserve exact public type",
        )
        snapshot_probe = acquire_public_reservation(
            "SyncRateLimiter.acquire_capacity(snapshot authority)",
            full_usage,
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        )
        mutated_snapshot_probe = snapshot_probe.model_copy(update={"usage": zero_usage})
        _run_sync_step(
            "SyncRateLimiter.refund_capacity(mutated reservation snapshot authority)",
            lambda: limiter.refund_capacity(zero_usage, mutated_snapshot_probe),
            deadline=_timing().operation_deadline_seconds,
        )
        _discard_pending_public_refund(pending_refunds, snapshot_probe)
        final_reservation = acquire_public_reservation(
            "SyncRateLimiter.acquire_capacity(after mutated reservation refund)",
            full_usage,
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        )
        refund_public_reservation(
            "SyncRateLimiter.refund_capacity(final public reservation)",
            final_reservation,
        )
    except BaseException as exc:
        body_exc = exc
        raise
    finally:
        cleanup_body_exc = body_exc
        try:
            _cleanup_pending_sync_public_refunds(
                limiter,
                pending_refunds,
                zero_usage=zero_usage,
                body_exc=body_exc,
            )
        except BaseException as exc:
            cleanup_body_exc = exc
            raise
        finally:
            _cleanup_sync_limiter(limiter, body_exc=cleanup_body_exc)


def _check_sync_acquire_refund_failed_error(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    refund_error = RuntimeError("fault-injection")
    interrupted_by = _SyncAcquireInterruptedError("fault-injection")
    cfg = _config("sync-acquire-refund-failed", limit=2.0)
    usage = frozen_usage({"requests": 1.0})
    zero_usage = frozen_usage({"requests": 0.0})
    bucket_ids = frozenset({_REQUESTS_BUCKET_ID})
    wrapped_builder = _SyncRefundFailureBuilder(builder, refund_error)
    limiter = _construct_sync_public_limiter(
        "SyncRateLimiter construction",
        cfg,
        wrapped_builder,
    )
    expected_limiter_instance_id = cast("str", limiter._limiter_instance_id)  # noqa: SLF001
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
    body_exc: BaseException | None = None
    try:
        try:
            started_at = time.time()
            _run_sync_step(
                "SyncRateLimiter.acquire_capacity(acquire-refund-failed probe)",
                lambda: limiter.acquire_capacity(
                    usage,
                    _PUBLIC_MODEL_NAME,
                ),
                deadline=_timing().operation_deadline_seconds,
                allowed_exceptions=(AcquireRefundFailedError,),
            )
        except AcquireRefundFailedError as exc:
            finished_at = time.time()
            expected_reservation_id = _check_refund_failure_backend_payload(
                wrapped_builder.last_backend,
                expected_usage=usage,
                expected_actual_usage=zero_usage,
                expected_model_family=cfg.get_model_family(),
                expected_bucket_ids=bucket_ids,
            )
            _check_acquire_refund_failed_payload(
                exc,
                refund_error=refund_error,
                interrupted_by=interrupted_by,
                expected_usage=usage,
                expected_model_family=cfg.get_model_family(),
                expected_model=_PUBLIC_MODEL_NAME,
                expected_bucket_ids=bucket_ids,
                expected_limiter_instance_id=expected_limiter_instance_id,
                expected_reservation_id=expected_reservation_id,
                acquired_after_seconds=started_at,
                acquired_before_seconds=finished_at,
            )
            return
        _fail("interrupted acquire cleanup must raise AcquireRefundFailedError")
    except BaseException as exc:
        body_exc = exc
        raise
    finally:
        _cleanup_sync_limiter(limiter, body_exc=body_exc)


def sync_conformance_test_for(
    backend_builder: SyncRateLimiterBackendBuilderInterface,
    *,
    timing: ConformanceTiming | None = None,
) -> None:
    """
    Run the public sync backend conformance checks for one backend builder.

    The builder should point at isolated backend state: use a disposable Redis
    key prefix, database, or in-memory instance so these tests can consume and
    refund capacity freely.

    Backend operations are bounded by helper-owned deadlines. Pass
    ``ConformanceTiming`` to tune those deadlines, or set
    ``TOKEN_THROTTLE_CONFORMANCE_TIMING_SCALE`` to multiply the defaults for
    slow runners. When ``timing=`` is passed, the env var is ignored; set every
    desired field on the dataclass.

    Unlike ``SyncRateLimiter``, which tolerates a builder that omits
    ``close()`` entirely, this helper requires the builder to satisfy the full
    ``SyncRateLimiterBackendBuilderInterface`` protocol — including that hook
    — since it type-checks the builder before running any test. Subclass the
    interface (it provides a no-op default) or otherwise define ``close()``;
    the helper then calls it during cleanup.

    KNOWN LIMITATION: if a synchronous backend call hangs in its worker thread,
    Python cannot safely kill that thread; the helper reports the hang and
    continues without waiting for that thread to finish.
    """
    resolved_timing = _resolve_timing(timing)
    token = _TIMING_CONTEXT.set(resolved_timing)
    body_exc: BaseException | None = None
    try:
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
    except BaseException as exc:
        body_exc = exc
        raise
    finally:
        try:
            _cleanup_sync_builder(backend_builder)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            if body_exc is None:
                raise
            _warn_cleanup_failure(exc)
        finally:
            _TIMING_CONTEXT.reset(token)


def run_conformance_test_for(
    backend_builder: RateLimiterBackendBuilderInterface,
    *,
    timing: ConformanceTiming | None = None,
) -> None:
    """Run async backend conformance checks from synchronous test suites."""
    asyncio.run(conformance_test_for(backend_builder, timing=timing))
