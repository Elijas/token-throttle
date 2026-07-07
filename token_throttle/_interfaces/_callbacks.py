"""Callback infrastructure for the rate limiter."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
import inspect
import logging
import threading
import warnings
from typing import Protocol, runtime_checkable

from frozendict import frozendict
from pydantic import Field, ValidationInfo, field_validator

from token_throttle._dto import StrictDTO
from token_throttle._exceptions import AcquireRefundFailedError
from token_throttle._interfaces._callable_utils import (
    close_awaitable_if_possible,
    is_async_callable,
)

type MetricName = str
type PerSeconds = int
type BucketId = tuple[MetricName, PerSeconds]
type FrozenUsage = frozendict[MetricName, float]
type Capacities = frozendict[BucketId, float]

_stdlib_logger = logging.getLogger("token_throttle")
_IN_LIMITER_CALLBACK: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "token_throttle_in_limiter_callback",
    default=False,
)
_LIMITER_CALLBACK_CONTEXT: contextvars.ContextVar[dict[str, object] | None] = (
    contextvars.ContextVar("token_throttle_limiter_callback_context", default=None)
)


_EXPECTED_CALLBACK_PARAMS: dict[str, frozenset[str]] = {
    "on_wait_start": frozenset({"model_family", "usage", "preconsumption_capacities"}),
    "after_wait_end_consumption": frozenset(
        {
            "model_family",
            "usage",
            "preconsumption_capacities",
            "postconsumption_capacities",
            "wait_time_s",
        }
    ),
    "on_capacity_consumed": frozenset(
        {
            "model_family",
            "preconsumption_capacities",
            "postconsumption_capacities",
            "usage",
            "current_time",
        }
    ),
    "on_capacity_refunded": frozenset(
        {
            "model_family",
            "reserved_usage",
            "actual_usage",
            "refunded_usage",
            "prerefund_capacities",
            "postrefund_capacities",
        }
    ),
    "on_missing_consumption_data": frozenset(
        {"model_family", "usage_metric", "per_seconds"}
    ),
    "on_lifecycle_event": frozenset({"event"}),
}


def _validate_callback_signature(value: object, field_name: str) -> None:
    if not callable(value):
        return
    expected = _EXPECTED_CALLBACK_PARAMS[field_name]
    try:
        sig = inspect.signature(value)
    except (TypeError, ValueError):
        return
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    missing = expected - accepted
    if missing:
        raise ValueError(
            f"{field_name} is missing required keyword parameters: {sorted(missing)}"
        )


def _is_generator_callback(value: object) -> bool:
    if inspect.isgeneratorfunction(value) or inspect.isasyncgenfunction(value):
        return True
    if not callable(value):
        return False
    return inspect.isgeneratorfunction(value.__call__) or inspect.isasyncgenfunction(
        value.__call__
    )


def _validate_not_generator_callback(value: object, field_name: str) -> None:
    if _is_generator_callback(value):
        raise ValueError(
            f"{field_name} must not be a generator or async-generator callback; "
            "callback bodies must run during dispatch and return None"
        )


_STDLIB_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def limiter_callback_context_active() -> bool:
    """Return whether execution is currently inside limiter callback dispatch."""
    return _IN_LIMITER_CALLBACK.get()


def set_limiter_callback_context(
    **context: object,
) -> contextvars.Token[dict[str, object] | None]:
    """Set request/reservation context propagated to backend callback payloads."""
    return _LIMITER_CALLBACK_CONTEXT.set(
        {key: value for key, value in context.items() if value is not None}
    )


def reset_limiter_callback_context(
    token: contextvars.Token[dict[str, object] | None],
) -> None:
    """Restore the previous request/reservation callback context."""
    _LIMITER_CALLBACK_CONTEXT.reset(token)


def current_limiter_callback_context() -> dict[str, object]:
    """Return the current request/reservation callback context, if any."""
    return dict(_LIMITER_CALLBACK_CONTEXT.get() or {})


def _validate_log_level(level: str | None, param_name: str) -> None:
    if level is None:
        return
    if not isinstance(level, str):
        raise TypeError(
            f"{param_name} must be a string or None (got {type(level).__name__})"
        )
    if not level.strip():
        raise ValueError(f"{param_name} must not be empty or whitespace-only")
    if level.upper() not in _STDLIB_LEVEL_MAP:
        raise ValueError(
            f"Unknown log level {level!r} for {param_name}; "
            f"valid levels: {sorted(_STDLIB_LEVEL_MAP)}"
        )


def _log_context(kwargs: dict[str, object]) -> str:
    return " ".join(f"{key}={value!r}" for key, value in kwargs.items())


def _log(level: str, message: str, **kwargs) -> None:
    """Log rate-limiter callback events through stdlib logging only."""
    if not isinstance(level, str):
        raise TypeError(f"_log level must be str, got {type(level).__name__}")
    # Intentional KeyError on unknown levels: callers validate public inputs.
    stdlib_level = _STDLIB_LEVEL_MAP[level.upper()]
    if kwargs:
        _stdlib_logger.log(
            stdlib_level,
            "%s | %s",
            message,
            _log_context(kwargs),
            extra=kwargs,
        )
    else:
        _stdlib_logger.log(stdlib_level, message)


def _callback_log_extra(
    callback_slot: str | None,
    kwargs: dict[str, object],
) -> dict[str, object]:
    extra = current_limiter_callback_context()
    event = kwargs.get("event")
    for name in ("model_family", "reservation_id"):
        value = kwargs.get(name)
        if value is None and event is not None:
            value = getattr(event, name, None)
        if value is not None:
            extra[name] = value
    if "bucket_id" in kwargs and kwargs["bucket_id"] is not None:
        extra["bucket_id"] = kwargs["bucket_id"]
    elif "usage_metric" in kwargs and "per_seconds" in kwargs:
        extra["bucket_id"] = (kwargs["usage_metric"], kwargs["per_seconds"])
    elif event is not None:
        bucket_ids = getattr(event, "bucket_ids", None)
        if bucket_ids is not None and len(bucket_ids) == 1:
            extra["bucket_id"] = next(iter(bucket_ids))
    if callback_slot is not None:
        extra["callback_slot"] = callback_slot
    return extra


def _log_callback_timeout(
    timeout: float,
    *,
    callback_slot: str | None,
    kwargs: dict[str, object],
) -> None:
    extra = _callback_log_extra(callback_slot, kwargs)
    _stdlib_logger.warning(
        "Rate limiter callback exceeded %.3fs timeout; skipping",
        timeout,
        extra=extra,
    )


def _log_late_callback_exception(
    exc: BaseException,
    *,
    callback_slot: str | None,
    kwargs: dict[str, object],
) -> None:
    msg = (
        "Rate limiter callback raised after callback_timeout elapsed "
        f"{type(exc).__name__}: {exc}"
    )
    with contextlib.suppress(Warning):
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    _stdlib_logger.warning(msg, extra=_callback_log_extra(callback_slot, kwargs))


def _log_callback_error_during_cancellation(
    exc: BaseException,
    *,
    callback_slot: str | None,
    kwargs: dict[str, object],
) -> None:
    msg = (
        "Rate limiter callback raised while the caller was being cancelled "
        f"{type(exc).__name__}: {exc}; re-raising CancelledError"
    )
    with contextlib.suppress(Warning):
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    _stdlib_logger.warning(msg, extra=_callback_log_extra(callback_slot, kwargs))


def _masks_caller_cancellation(exc: BaseException) -> bool:
    """
    Return whether ``exc`` would silently replace the caller's CancelledError.

    Once the caller has been cancelled, only exceptions that both safe-dispatch
    ladders treat as critical may take CancelledError's place: anything else
    would be logged-and-swallowed by ``safe_invoke_async_callback``, letting a
    cancelled ``acquire_capacity`` return normally and defeating
    ``asyncio.timeout()`` / ``TaskGroup`` aborts.
    """
    if isinstance(exc, BACKEND_CALLBACK_CRITICAL_EXCEPTIONS):
        return False
    return not _exception_group_contains_critical(
        exc, BACKEND_CALLBACK_CRITICAL_EXCEPTIONS
    )


def _caller_cancellation_pending() -> bool:
    task = _running_task_or_none()
    return task is not None and task.cancelling() > 0


def _invoke_sync_callback_checked(callback, **kwargs) -> None:
    result = callback(**_accepted_callback_kwargs(callback, kwargs))
    if inspect.isawaitable(result):
        close_awaitable_if_possible(result)
        raise TypeError(
            "Synchronous rate limiter callback returned an awaitable; "
            "use async RateLimiterCallbacks with RateLimiter instead"
        )


class _CallbackCriticalCarrierError(Exception):
    """
    Carry a callback's ``KeyboardInterrupt``/``SystemExit``/``GeneratorExit`` out
    of its task.

    A task's step re-raises ``KeyboardInterrupt``/``SystemExit`` into the event
    loop rather than the awaiting frame, which would bypass the limiter's inline
    critical-exception handling (and any acquire/close cleanup). ``GeneratorExit``
    has a separate hazard: retrieving it from a *different* task re-enters the
    awaiting coroutine via ``Task.__wakeup`` -> ``coro.throw()``, and per PEP 380,
    throwing ``GeneratorExit`` into a suspended ``await``/``yield from`` chain is
    delegated as ``close()`` on the innermost suspended awaitable. If any cleanup
    handler downstream (e.g. the refund-on-raise context manager) does further
    real awaiting before the exception is done propagating, Python raises
    ``RuntimeError: coroutine ignored GeneratorExit`` instead of letting
    ``GeneratorExit`` through. Wrapping all three in this carrier and re-raising
    the original via a plain ``raise`` statement in the awaiting frame sidesteps
    both hazards: the sync path's helper thread captures every ``BaseException``
    into a future and gets the same treatment there.
    """

    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self.exc = exc


def _running_task_or_none() -> asyncio.Task | None:
    try:
        return asyncio.current_task()
    except RuntimeError:  # no running event loop (e.g. GC-driven teardown)
        return None


async def _run_timed_callback(callback, callback_kwargs: dict[str, object]) -> None:
    own_task = _running_task_or_none()
    try:
        await callback(**callback_kwargs)
    except GeneratorExit as exc:
        # The carrier is only needed while this coroutine runs inside its own
        # task step (the Task.__wakeup / coro.throw hazard described on the
        # carrier). When ``coroutine.close()`` tears the invocation down -- GC
        # of an abandoned detached task, explicit close(), loop shutdown --
        # GeneratorExit is delivered outside that step and must propagate
        # unwrapped, or close() surfaces the carrier as "Exception ignored"
        # unraisable noise.
        if own_task is None or _running_task_or_none() is not own_task:
            raise
        raise _CallbackCriticalCarrierError(exc) from exc
    except (KeyboardInterrupt, SystemExit) as exc:
        raise _CallbackCriticalCarrierError(exc) from exc


# Strong references to callbacks abandoned after ``callback_timeout``. Without
# this the event loop only holds a weak reference to a detached task and may
# garbage-collect it mid-flight. Entries are discarded from the done-callback,
# except tasks whose loop closed before they finished: their done-callback can
# never run, so they are pruned when the next detachment happens.
_DETACHED_CALLBACK_TASKS: set[asyncio.Future[None]] = set()


def _prune_detached_tasks_on_closed_loops() -> None:
    for detached in list(_DETACHED_CALLBACK_TASKS):
        if detached.get_loop().is_closed():
            _DETACHED_CALLBACK_TASKS.discard(detached)


def _log_detached_callback_outcome(
    task: asyncio.Future[None],
    *,
    callback_slot: str | None,
    kwargs: dict[str, object],
) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        return
    if isinstance(exc, _CallbackCriticalCarrierError):
        exc = exc.exc
    _log_late_callback_exception(exc, callback_slot=callback_slot, kwargs=kwargs)


def _detach_async_callback(
    task: asyncio.Future[None],
    *,
    callback_slot: str | None,
    kwargs: dict[str, object],
) -> None:
    """
    Let a timed-out callback finish in the background and log its outcome.

    Mirrors the sync path, which abandons its helper thread after the timeout
    and only surfaces an exception the callback raises later. ``asyncio.shield``
    already retrieved the task result, so logging is the only remaining work.
    """
    if task.done():
        _log_detached_callback_outcome(task, callback_slot=callback_slot, kwargs=kwargs)
        return
    _prune_detached_tasks_on_closed_loops()
    _DETACHED_CALLBACK_TASKS.add(task)

    def on_done(done: asyncio.Future[None]) -> None:
        _DETACHED_CALLBACK_TASKS.discard(done)
        _log_detached_callback_outcome(done, callback_slot=callback_slot, kwargs=kwargs)

    task.add_done_callback(on_done)


async def _invoke_async_callback_with_timeout(
    callback,
    callback_timeout: float | None,
    *,
    callback_slot: str | None = None,
    **kwargs,
) -> None:
    callback_kwargs = _accepted_callback_kwargs(callback, kwargs)
    if callback_timeout is None:
        try:
            await callback(**callback_kwargs)
        except BaseException as exc:
            # The caller's CancelledError is delivered straight into the
            # callback coroutine here. If its unwind swallowed that into an
            # ordinary exception, restore CancelledError so cancellation is
            # never silently lost.
            if _masks_caller_cancellation(exc) and _caller_cancellation_pending():
                _log_callback_error_during_cancellation(
                    exc,
                    callback_slot=callback_slot,
                    kwargs=kwargs,
                )
                raise asyncio.CancelledError from exc
            raise
        return
    # Run the callback as its own task and shield it from the deadline: awaiting
    # the raw coroutine via wait_for would, on timeout, cancel it and then block
    # on its completion, so a callback that swallows CancelledError (or awaits in
    # cleanup) would stall acquire/refund for its full runtime -- silently, since
    # a swallowed cancellation makes wait_for return without raising TimeoutError.
    # Shielding gives a deterministic timeout and lets us abandon the task, the
    # async analogue of the sync path's abandoned helper thread.
    task: asyncio.Future[None] = asyncio.ensure_future(
        _run_timed_callback(callback, callback_kwargs)
    )
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=callback_timeout)
    except _CallbackCriticalCarrierError as carrier:
        raise carrier.exc from None
    except TimeoutError:
        # ``wait_for`` raising TimeoutError does not always mean the deadline
        # expired: a callback that itself raises TimeoutError surfaces through
        # the shield identically. That case is an ordinary callback error for
        # the safe-dispatch ladder, not deadline expiry.
        callback_exc = (
            task.exception() if task.done() and not task.cancelled() else None
        )
        if callback_exc is not None:
            if isinstance(callback_exc, _CallbackCriticalCarrierError):
                raise callback_exc.exc from None
            raise callback_exc  # noqa: B904 - the callback's own error, unchanged
        _log_callback_timeout(
            callback_timeout,
            callback_slot=callback_slot,
            kwargs=kwargs,
        )
        _detach_async_callback(task, callback_slot=callback_slot, kwargs=kwargs)
    # ast-guard: skip — narrow cancel composition; deadline branch owns cleanup.
    except asyncio.CancelledError:
        # Caller cancellation, not the deadline: cancel the callback and let
        # critical exceptions from its unwind surface, but never let an
        # ordinary unwind error replace the caller's CancelledError -- the
        # safe-dispatch ladder would swallow it, and a cancelled acquire would
        # return normally. Awaiting here still blocks on a callback that
        # swallows cancellation, exactly as before; the deadline branch above
        # is what bounds that case.
        task.cancel()
        try:
            await task
        except _CallbackCriticalCarrierError as carrier:
            raise carrier.exc from None
        except BaseException as unwind_exc:
            if not _masks_caller_cancellation(unwind_exc):
                raise
            _log_callback_error_during_cancellation(
                unwind_exc,
                callback_slot=callback_slot,
                kwargs=kwargs,
            )
        raise


def _invoke_sync_callback_with_timeout(
    callback,
    callback_timeout: float | None,
    *,
    callback_slot: str | None = None,
    **kwargs,
) -> None:
    kwargs = _accepted_callback_kwargs(callback, kwargs)
    if callback_timeout is None:
        _invoke_sync_callback_checked(callback, **kwargs)
        return

    future: concurrent.futures.Future[None] = concurrent.futures.Future()
    context = contextvars.copy_context()

    def run_callback() -> None:
        try:
            _invoke_sync_callback_checked(callback, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised on caller thread below
            with contextlib.suppress(concurrent.futures.InvalidStateError):
                future.set_exception(exc)
        else:
            with contextlib.suppress(concurrent.futures.InvalidStateError):
                future.set_result(None)

    def log_late_exception(done: concurrent.futures.Future[None]) -> None:
        try:
            done.result()
        except BaseException as exc:  # noqa: BLE001 - cannot propagate after timeout
            _log_late_callback_exception(
                exc,
                callback_slot=callback_slot,
                kwargs=kwargs,
            )

    # Timeout-wrapped sync callbacks run in a helper thread.  Copy the caller's
    # contextvars context so ambient tracing/request state survives dispatch.
    thread = threading.Thread(target=lambda: context.run(run_callback), daemon=True)
    thread.start()
    try:
        future.result(timeout=callback_timeout)
    except concurrent.futures.TimeoutError:
        # ``concurrent.futures.TimeoutError`` is the builtin TimeoutError, so a
        # callback that itself raises TimeoutError lands here too. Mirror the
        # async path: that is an ordinary callback error, not deadline expiry.
        callback_exc = future.exception() if future.done() else None
        if callback_exc is not None:
            raise callback_exc  # noqa: B904 - the callback's own error, unchanged
        _log_callback_timeout(
            callback_timeout,
            callback_slot=callback_slot,
            kwargs=kwargs,
        )
        if future.done():
            log_late_exception(future)
        else:
            future.add_done_callback(log_late_exception)


# ---------------------------------------------------------------------------
# Critical-exception protected callback dispatch
#
# Single source of truth for the "ladder" pattern formerly hand-rolled at
# six dispatch sites (two `_emit_lifecycle_event` methods on the rate
# limiters, four `_invoke_callback_safe` methods on the backends). Both
# the critical-type tuples and the group-aware dispatcher live here so
# the set of "must propagate" exceptions cannot drift between sync and
# async, or between rate-limiter and backend, paths.
#
# Callback dispatch remains best-effort for ordinary exceptions, but
# cancellation/termination signals and severe process-health failures
# (out-of-memory or runaway recursion) propagate instead of being buried
# behind RuntimeWarning.
# ---------------------------------------------------------------------------

LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.CancelledError,
    concurrent.futures.CancelledError,
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
    MemoryError,
    RecursionError,
)

BACKEND_CALLBACK_CRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AcquireRefundFailedError,
    *LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
)


def _exception_group_contains_critical(
    exc: BaseException,
    critical: tuple[type[BaseException], ...],
) -> bool:
    if not isinstance(exc, BaseExceptionGroup):
        return False
    critical_part, _non_critical = exc.split(critical)
    return critical_part is not None


def _accepted_callback_kwargs(callback, kwargs: dict[str, object]) -> dict[str, object]:
    """Keep optional callback context additive for exact-signature callbacks."""
    try:
        sig = inspect.signature(callback)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    return {name: value for name, value in kwargs.items() if name in accepted}


async def safe_invoke_async_callback(
    callback,
    *,
    critical: tuple[type[BaseException], ...],
    log_label: str,
    callback_slot: str | None = None,
    **kwargs,
) -> None:
    token = _IN_LIMITER_CALLBACK.set(True)
    try:
        await callback(**_accepted_callback_kwargs(callback, kwargs))
    except critical:
        raise
    except BaseException as exc:
        if _exception_group_contains_critical(exc, critical):
            raise
        msg = f"{log_label} raised {type(exc).__name__}: {exc}"
        with contextlib.suppress(Warning):
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
        _stdlib_logger.warning(
            msg,
            extra=_callback_log_extra(callback_slot, kwargs),
        )
    finally:
        _IN_LIMITER_CALLBACK.reset(token)


def safe_invoke_sync_callback(
    callback,
    *,
    critical: tuple[type[BaseException], ...],
    log_label: str,
    callback_slot: str | None = None,
    **kwargs,
) -> None:
    token = _IN_LIMITER_CALLBACK.set(True)
    try:
        _invoke_sync_callback_checked(
            callback, **_accepted_callback_kwargs(callback, kwargs)
        )
    except critical:
        raise
    except BaseException as exc:
        if _exception_group_contains_critical(exc, critical):
            raise
        msg = f"{log_label} raised {type(exc).__name__}: {exc}"
        with contextlib.suppress(Warning):
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
        _stdlib_logger.warning(
            msg,
            extra=_callback_log_extra(callback_slot, kwargs),
        )
    finally:
        _IN_LIMITER_CALLBACK.reset(token)


def with_callback_timeout(
    callbacks: RateLimiterCallbacks | None,
    timeout: float | None,
) -> RateLimiterCallbacks | None:
    if callbacks is None:
        return None
    callbacks.revalidate()

    def wrap(callback, callback_slot: str):
        if callback is None:
            return None

        async def wrapped(**kwargs) -> None:
            await _invoke_async_callback_with_timeout(
                callback,
                timeout,
                callback_slot=callback_slot,
                **kwargs,
            )

        return wrapped

    return RateLimiterCallbacks(
        on_wait_start=wrap(callbacks.on_wait_start, "on_wait_start"),
        after_wait_end_consumption=wrap(
            callbacks.after_wait_end_consumption,
            "after_wait_end_consumption",
        ),
        on_capacity_consumed=wrap(
            callbacks.on_capacity_consumed,
            "on_capacity_consumed",
        ),
        on_capacity_refunded=wrap(
            callbacks.on_capacity_refunded,
            "on_capacity_refunded",
        ),
        on_missing_consumption_data=wrap(
            callbacks.on_missing_consumption_data,
            "on_missing_consumption_data",
        ),
        on_lifecycle_event=wrap(callbacks.on_lifecycle_event, "on_lifecycle_event"),
    )


def with_sync_callback_timeout(
    callbacks: SyncRateLimiterCallbacks | None,
    timeout: float | None,
) -> SyncRateLimiterCallbacks | None:
    if callbacks is None:
        return None
    callbacks.revalidate()

    def wrap(callback, callback_slot: str):
        if callback is None:
            return None

        def wrapped(**kwargs) -> None:
            _invoke_sync_callback_with_timeout(
                callback,
                timeout,
                callback_slot=callback_slot,
                **kwargs,
            )

        return wrapped

    return SyncRateLimiterCallbacks(
        on_wait_start=wrap(callbacks.on_wait_start, "on_wait_start"),
        after_wait_end_consumption=wrap(
            callbacks.after_wait_end_consumption,
            "after_wait_end_consumption",
        ),
        on_capacity_consumed=wrap(
            callbacks.on_capacity_consumed,
            "on_capacity_consumed",
        ),
        on_capacity_refunded=wrap(
            callbacks.on_capacity_refunded,
            "on_capacity_refunded",
        ),
        on_missing_consumption_data=wrap(
            callbacks.on_missing_consumption_data,
            "on_missing_consumption_data",
        ),
        on_lifecycle_event=wrap(callbacks.on_lifecycle_event, "on_lifecycle_event"),
    )


# ---------------------------------------------------------------------------
# Async callback protocols
# ---------------------------------------------------------------------------


class LifecycleEvent(StrictDTO):
    """
    Structured, additive lifecycle event for correlation and metrics collectors.

    ``model_alias`` and ``request_id`` are caller-controlled values and may
    contain application identifiers. ``model_family``, ``bucket_ids``, and
    ``usage`` come from limiter configuration and reservation state.
    """

    event_type: str = Field(description="Lifecycle event kind")
    reservation_id: str | None = Field(default=None)
    request_id: str | None = Field(default=None)
    model_family: str = Field(description="Resolved limiter model family")
    model_alias: str | None = Field(default=None, description="Public model alias")
    bucket_ids: frozenset[BucketId] | None = Field(default=None)
    usage: FrozenUsage | None = Field(default=None)
    timestamp: float = Field(description="Unix timestamp in seconds")


@runtime_checkable
class OnWaitStartCallback(Protocol):
    async def __call__(
        self,
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
    ) -> None:
        """Called when capacity required waiting."""


@runtime_checkable
class OnWaitEndCallback(Protocol):
    async def __call__(
        self,
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        wait_time_s: float,
    ) -> None:
        """Called after successfully acquiring capacity if there is a wait time."""


@runtime_checkable
class OnCapacityConsumedCallback(Protocol):
    async def __call__(
        self,
        *,
        model_family: str,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        usage: FrozenUsage,
        current_time: float,
    ) -> None:
        """
        Called when capacity is consumed.

        Not 100% delivery-guaranteed under task cancellation: if the
        calling task is cancelled while a shielded backend write is
        in flight and that write commits, the cancellation is suppressed
        and this callback is skipped (the cancel context is already
        stripped, so firing user callbacks would be misleading).
        """


@runtime_checkable
class OnCapacityRefundedCallback(Protocol):
    async def __call__(  # noqa: PLR0913
        self,
        *,
        model_family: str,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        refunded_usage: FrozenUsage,
        prerefund_capacities: Capacities,
        postrefund_capacities: Capacities,
    ) -> None:
        """Called when capacity is refunded (unused tokens or errors)"""


@runtime_checkable
class OnMissingConsumptionDataCallback(Protocol):
    async def __call__(
        self,
        *,
        model_family: str,
        usage_metric: str,
        per_seconds: int,
    ) -> None:
        """Called when bucket consumption data is missing or partially missing."""


@runtime_checkable
class OnLifecycleEventCallback(Protocol):
    async def __call__(
        self,
        *,
        event: LifecycleEvent,
    ) -> None:
        """Called with structured limiter lifecycle events."""


class RateLimiterCallbacks(StrictDTO):
    """
    Exact-type immutable async callback bundle.

    v2.0.0 contract: ``RateLimiterCallbacks`` is a data-transfer object, not
    a subclass extension point. Construction, assignment, copy, pickle
    restore, ``model_copy()``, and ``model_construct()`` all preserve the
    async-callable validators; ``model_construct()`` is disabled.
    """

    on_wait_start: OnWaitStartCallback | None = Field(
        default=None,
        description="Called when capacity required waiting",
    )
    after_wait_end_consumption: OnWaitEndCallback | None = Field(
        default=None,
        description="Called after successfully acquiring capacity",
    )
    on_capacity_consumed: OnCapacityConsumedCallback | None = Field(
        default=None,
        description="Called when capacity is consumed",
    )
    on_capacity_refunded: OnCapacityRefundedCallback | None = Field(
        default=None,
        description="Called when capacity is refunded (e.g., unused tokens, errors)",
    )
    on_missing_consumption_data: OnMissingConsumptionDataCallback | None = Field(
        default=None,
        description="Called when no previous consumption data is detected, assuming full quota",
    )
    on_lifecycle_event: OnLifecycleEventCallback | None = Field(
        default=None,
        description="Called with structured limiter lifecycle events",
    )

    @field_validator(
        "on_wait_start",
        "after_wait_end_consumption",
        "on_capacity_consumed",
        "on_capacity_refunded",
        "on_missing_consumption_data",
        "on_lifecycle_event",
        mode="after",
    )
    @classmethod
    def _validate_async_callbacks(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if value is not None and info.field_name is not None:
            _validate_not_generator_callback(value, info.field_name)
        if value is not None and not is_async_callable(value):
            raise ValueError(f"{info.field_name} must be an async callable")
        if value is not None and info.field_name is not None:
            _validate_callback_signature(value, info.field_name)
        return value


def _merge_rate_limiter_callbacks(
    user_callbacks: RateLimiterCallbacks | None,
    default_callbacks: RateLimiterCallbacks,
) -> RateLimiterCallbacks:
    default_callbacks = default_callbacks.revalidate()
    if user_callbacks is None:
        return default_callbacks
    user_callbacks = user_callbacks.revalidate()

    merged = {
        field_name: (
            user_value
            if (user_value := getattr(user_callbacks, field_name)) is not None
            else getattr(default_callbacks, field_name)
        )
        for field_name in RateLimiterCallbacks.model_fields
    }
    return RateLimiterCallbacks(**merged)


# ---------------------------------------------------------------------------
# Sync callback protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SyncOnWaitStartCallback(Protocol):
    def __call__(
        self,
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
    ) -> None:
        """Called by ``SyncRateLimiter`` when capacity required waiting."""


@runtime_checkable
class SyncOnWaitEndCallback(Protocol):
    def __call__(
        self,
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        wait_time_s: float,
    ) -> None:
        """Called by ``SyncRateLimiter`` after a waited acquire succeeds."""


@runtime_checkable
class SyncOnCapacityConsumedCallback(Protocol):
    def __call__(
        self,
        *,
        model_family: str,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        usage: FrozenUsage,
        current_time: float,
    ) -> None:
        """
        Called by ``SyncRateLimiter`` when capacity is consumed.

        Sync callbacks are invoked after the backend write returns. If callback
        timeout wrapping is enabled and the callback exceeds the timeout, the
        limiter logs the timeout and continues; the helper thread may finish
        later, but its result no longer affects limiter control flow.
        """


@runtime_checkable
class SyncOnCapacityRefundedCallback(Protocol):
    def __call__(  # noqa: PLR0913
        self,
        *,
        model_family: str,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        refunded_usage: FrozenUsage,
        prerefund_capacities: Capacities,
        postrefund_capacities: Capacities,
    ) -> None:
        """Called by ``SyncRateLimiter`` when unused capacity is refunded."""


@runtime_checkable
class SyncOnMissingConsumptionDataCallback(Protocol):
    def __call__(
        self,
        *,
        model_family: str,
        usage_metric: str,
        per_seconds: int,
    ) -> None:
        """Called when sync bucket consumption data is missing or partially missing."""


@runtime_checkable
class SyncOnLifecycleEventCallback(Protocol):
    def __call__(
        self,
        *,
        event: LifecycleEvent,
    ) -> None:
        """Called with structured sync limiter lifecycle events."""


class SyncRateLimiterCallbacks(StrictDTO):
    """
    Exact-type immutable sync callback bundle.

    v2.0.0 contract: ``SyncRateLimiterCallbacks`` is a data-transfer object,
    not a subclass extension point. Construction, assignment, copy, pickle
    restore, ``model_copy()``, and ``model_construct()`` all preserve the
    sync-callable validators; ``model_construct()`` is disabled.
    """

    on_wait_start: SyncOnWaitStartCallback | None = Field(
        default=None,
        description="Called when capacity required waiting",
    )
    after_wait_end_consumption: SyncOnWaitEndCallback | None = Field(
        default=None,
        description="Called after successfully acquiring capacity",
    )
    on_capacity_consumed: SyncOnCapacityConsumedCallback | None = Field(
        default=None,
        description="Called when capacity is consumed",
    )
    on_capacity_refunded: SyncOnCapacityRefundedCallback | None = Field(
        default=None,
        description="Called when capacity is refunded (e.g., unused tokens, errors)",
    )
    on_missing_consumption_data: SyncOnMissingConsumptionDataCallback | None = Field(
        default=None,
        description="Called when no previous consumption data is detected, assuming full quota",
    )
    on_lifecycle_event: SyncOnLifecycleEventCallback | None = Field(
        default=None,
        description="Called with structured limiter lifecycle events",
    )

    @field_validator(
        "on_wait_start",
        "after_wait_end_consumption",
        "on_capacity_consumed",
        "on_capacity_refunded",
        "on_missing_consumption_data",
        "on_lifecycle_event",
        mode="after",
    )
    @classmethod
    def _validate_sync_callbacks(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if value is not None and info.field_name is not None:
            _validate_not_generator_callback(value, info.field_name)
        if value is not None and is_async_callable(value):
            raise ValueError(f"{info.field_name} must be a synchronous callable")
        if value is not None and info.field_name is not None:
            _validate_callback_signature(value, info.field_name)
        return value


def _merge_sync_rate_limiter_callbacks(
    user_callbacks: SyncRateLimiterCallbacks | None,
    default_callbacks: SyncRateLimiterCallbacks,
) -> SyncRateLimiterCallbacks:
    default_callbacks = default_callbacks.revalidate()
    if user_callbacks is None:
        return default_callbacks
    user_callbacks = user_callbacks.revalidate()

    merged = {
        field_name: (
            user_value
            if (user_value := getattr(user_callbacks, field_name)) is not None
            else getattr(default_callbacks, field_name)
        )
        for field_name in SyncRateLimiterCallbacks.model_fields
    }
    return SyncRateLimiterCallbacks(**merged)


# ---------------------------------------------------------------------------
# Stdlib logging callback factories
# ---------------------------------------------------------------------------


def create_logging_callbacks(
    *,
    wait_start: str | None = "DEBUG",
    wait_end_consumption: str | None = "DEBUG",
    capacity_consumed: str | None = "DEBUG",
    capacity_refunded: str | None = "DEBUG",
    missing_consumption_data: str | None = "DEBUG",
) -> RateLimiterCallbacks:
    """
    Create async callbacks that log rate-limiter events.

    Emits through the stdlib ``token_throttle`` logger. Each keyword selects
    the log level for one callback slot. Pass ``None`` to leave that slot
    unset. The returned callbacks are suitable for ``RateLimiter`` and async
    backends.
    """
    for _name, _val in (
        ("wait_start", wait_start),
        ("wait_end_consumption", wait_end_consumption),
        ("capacity_consumed", capacity_consumed),
        ("capacity_refunded", capacity_refunded),
        ("missing_consumption_data", missing_consumption_data),
    ):
        _validate_log_level(_val, _name)
    wait_start_level = wait_start
    wait_end_consumption_level = wait_end_consumption
    capacity_consumed_level = capacity_consumed
    capacity_refunded_level = capacity_refunded
    missing_consumption_data_level = missing_consumption_data

    async def on_wait_start(  # noqa: PLR0913
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        model_alias: str | None = None,
        request_id: str | None = None,
        reservation_id: str | None = None,
    ) -> None:
        assert wait_start_level is not None  # noqa: S101
        _log(
            wait_start_level,
            "Rate limiter wait starting",
            model_family=model_family,
            model_alias=model_alias,
            request_id=request_id,
            reservation_id=reservation_id,
            usage=usage,
            preconsumption_capacities=preconsumption_capacities,
        )

    async def after_wait_end_consumption(  # noqa: PLR0913
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        wait_time_s: float,
        model_alias: str | None = None,
        request_id: str | None = None,
        reservation_id: str | None = None,
    ) -> None:
        assert wait_end_consumption_level is not None  # noqa: S101
        _log(
            wait_end_consumption_level,
            "Rate limiter wait complete",
            model_family=model_family,
            model_alias=model_alias,
            request_id=request_id,
            reservation_id=reservation_id,
            usage=usage,
            preconsumption_capacities=preconsumption_capacities,
            postconsumption_capacities=postconsumption_capacities,
            wait_time_s=wait_time_s,
        )

    async def on_capacity_consumed(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        current_time: float,
    ) -> None:
        assert capacity_consumed_level is not None  # noqa: S101
        _log(
            capacity_consumed_level,
            "Rate limiter capacity consumed",
            model_family=model_family,
            usage=usage,
            preconsumption_capacities=preconsumption_capacities,
            postconsumption_capacities=postconsumption_capacities,
            current_time=current_time,
        )

    async def on_capacity_refunded(  # noqa: PLR0913
        *,
        model_family: str,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        refunded_usage: FrozenUsage,
        prerefund_capacities: Capacities,
        postrefund_capacities: Capacities,
    ) -> None:
        assert capacity_refunded_level is not None  # noqa: S101
        _log(
            capacity_refunded_level,
            "Rate limiter capacity refunded",
            model_family=model_family,
            reserved_usage=reserved_usage,
            actual_usage=actual_usage,
            refunded_usage=refunded_usage,
            prerefund_capacities=prerefund_capacities,
            postrefund_capacities=postrefund_capacities,
        )

    async def on_missing_consumption_data(  # noqa: PLR0913
        *,
        model_family: str,
        usage_metric: str,
        per_seconds: int,
        missing_state_reason: str | None = None,
        missing_state_keys: tuple[str, ...] | None = None,
        present_state_keys: tuple[str, ...] | None = None,
    ) -> None:
        assert missing_consumption_data_level is not None  # noqa: S101
        _log(
            missing_consumption_data_level,
            "Rate limiter missing consumption data",
            model_family=model_family,
            usage_metric=usage_metric,
            per_seconds=per_seconds,
            missing_state_reason=missing_state_reason,
            missing_state_keys=missing_state_keys,
            present_state_keys=present_state_keys,
        )

    return RateLimiterCallbacks(
        on_wait_start=on_wait_start if wait_start else None,
        after_wait_end_consumption=(
            after_wait_end_consumption if wait_end_consumption else None
        ),
        on_capacity_consumed=on_capacity_consumed if capacity_consumed else None,
        on_capacity_refunded=on_capacity_refunded if capacity_refunded else None,
        on_missing_consumption_data=(
            on_missing_consumption_data if missing_consumption_data else None
        ),
    )


def create_sync_logging_callbacks(
    *,
    wait_start: str | None = "DEBUG",
    wait_end_consumption: str | None = "DEBUG",
    capacity_consumed: str | None = "DEBUG",
    capacity_refunded: str | None = "DEBUG",
    missing_consumption_data: str | None = "DEBUG",
) -> SyncRateLimiterCallbacks:
    """
    Create synchronous callbacks that log rate-limiter events.

    Emits through the stdlib ``token_throttle`` logger. Each keyword selects
    the log level for one callback slot. Pass ``None`` to leave that slot
    unset. The returned callbacks are suitable for ``SyncRateLimiter`` and
    synchronous backends.
    """
    for _name, _val in (
        ("wait_start", wait_start),
        ("wait_end_consumption", wait_end_consumption),
        ("capacity_consumed", capacity_consumed),
        ("capacity_refunded", capacity_refunded),
        ("missing_consumption_data", missing_consumption_data),
    ):
        _validate_log_level(_val, _name)
    wait_start_level = wait_start
    wait_end_consumption_level = wait_end_consumption
    capacity_consumed_level = capacity_consumed
    capacity_refunded_level = capacity_refunded
    missing_consumption_data_level = missing_consumption_data

    def on_wait_start(  # noqa: PLR0913
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        model_alias: str | None = None,
        request_id: str | None = None,
        reservation_id: str | None = None,
    ) -> None:
        assert wait_start_level is not None  # noqa: S101
        _log(
            wait_start_level,
            "Rate limiter wait starting",
            model_family=model_family,
            model_alias=model_alias,
            request_id=request_id,
            reservation_id=reservation_id,
            usage=usage,
            preconsumption_capacities=preconsumption_capacities,
        )

    def after_wait_end_consumption(  # noqa: PLR0913
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        wait_time_s: float,
        model_alias: str | None = None,
        request_id: str | None = None,
        reservation_id: str | None = None,
    ) -> None:
        assert wait_end_consumption_level is not None  # noqa: S101
        _log(
            wait_end_consumption_level,
            "Rate limiter wait complete",
            model_family=model_family,
            model_alias=model_alias,
            request_id=request_id,
            reservation_id=reservation_id,
            usage=usage,
            preconsumption_capacities=preconsumption_capacities,
            postconsumption_capacities=postconsumption_capacities,
            wait_time_s=wait_time_s,
        )

    def on_capacity_consumed(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        current_time: float,
    ) -> None:
        assert capacity_consumed_level is not None  # noqa: S101
        _log(
            capacity_consumed_level,
            "Rate limiter capacity consumed",
            model_family=model_family,
            usage=usage,
            preconsumption_capacities=preconsumption_capacities,
            postconsumption_capacities=postconsumption_capacities,
            current_time=current_time,
        )

    def on_capacity_refunded(  # noqa: PLR0913
        *,
        model_family: str,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        refunded_usage: FrozenUsage,
        prerefund_capacities: Capacities,
        postrefund_capacities: Capacities,
    ) -> None:
        assert capacity_refunded_level is not None  # noqa: S101
        _log(
            capacity_refunded_level,
            "Rate limiter capacity refunded",
            model_family=model_family,
            reserved_usage=reserved_usage,
            actual_usage=actual_usage,
            refunded_usage=refunded_usage,
            prerefund_capacities=prerefund_capacities,
            postrefund_capacities=postrefund_capacities,
        )

    def on_missing_consumption_data(  # noqa: PLR0913
        *,
        model_family: str,
        usage_metric: str,
        per_seconds: int,
        missing_state_reason: str | None = None,
        missing_state_keys: tuple[str, ...] | None = None,
        present_state_keys: tuple[str, ...] | None = None,
    ) -> None:
        assert missing_consumption_data_level is not None  # noqa: S101
        _log(
            missing_consumption_data_level,
            "Rate limiter missing consumption data",
            model_family=model_family,
            usage_metric=usage_metric,
            per_seconds=per_seconds,
            missing_state_reason=missing_state_reason,
            missing_state_keys=missing_state_keys,
            present_state_keys=present_state_keys,
        )

    return SyncRateLimiterCallbacks(
        on_wait_start=on_wait_start if wait_start else None,
        after_wait_end_consumption=(
            after_wait_end_consumption if wait_end_consumption else None
        ),
        on_capacity_consumed=on_capacity_consumed if capacity_consumed else None,
        on_capacity_refunded=on_capacity_refunded if capacity_refunded else None,
        on_missing_consumption_data=(
            on_missing_consumption_data if missing_consumption_data else None
        ),
    )
