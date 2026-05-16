"""
Callback infrastructure for the rate limiter.

Loguru detection staleness contract
-----------------------------------
``_probe_loguru()`` runs once per process and caches its result. If
loguru becomes available or breaks AFTER the first probe, the cache
will not auto-invalidate. To force a re-probe, either:

- call :func:`_reset_loguru_cache` programmatically, or
- set the ``TOKEN_THROTTLE_LOGURU_DETECT_AGAIN`` environment variable
  to a non-empty value before the next callback fires.

The cache stores a *factory* callable (or an unavailability sentinel),
not a resolved logger. Each ``_log()`` invocation calls the factory,
which re-reads ``loguru.logger`` from ``sys.modules``. This means
in-process loguru API drift (monkey-patching, hot reload, a future
loguru release renaming ``.log()``) surfaces as a clear ``AttributeError``
at the call site instead of being masked by a stale poisoned reference.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
import inspect
import logging
import os
import threading
import warnings
from typing import Protocol, runtime_checkable

from frozendict import frozendict
from pydantic import Field, ValidationInfo, field_validator

from token_throttle._dto import StrictDTO
from token_throttle._interfaces._callable_utils import (
    close_awaitable_if_possible,
    is_async_callable,
)

type MetricName = str
type PerSeconds = int
type BucketId = tuple[MetricName, PerSeconds]
type FrozenUsage = frozendict[MetricName, float]
type Capacities = frozendict[BucketId, float]

# ---------------------------------------------------------------------------
# Auto-detect loguru vs stdlib logging
# ---------------------------------------------------------------------------

_stdlib_logger = logging.getLogger("token_throttle")

# Sentinel: the probe has run and concluded loguru is not usable for any
# reason (missing, broken install, API drifted at probe time, etc.).
_LOGURU_UNAVAILABLE: object = object()

# Module-level detection cache.
#
# Schema: ``{"factory": <zero-arg callable> | _LOGURU_UNAVAILABLE}``.
# Missing key → not yet probed.
#
# Storing a factory rather than a resolved logger means each ``_log()``
# call re-reads ``loguru.logger`` (cheap via ``sys.modules``) so any
# post-probe API drift (e.g. ``.log`` renamed) surfaces as a clear
# ``AttributeError`` at the call site instead of being masked by a stale
# cached reference.
_loguru_cache: dict[str, object] = {}

_LOGURU_DETECT_AGAIN_ENV = "TOKEN_THROTTLE_LOGURU_DETECT_AGAIN"


def _resolve_loguru_logger():
    """
    Fresh resolution + API smoke-test of the loguru logger.

    Imports ``loguru.logger`` (cached via ``sys.modules`` after the first
    real import) and verifies that ``.log`` is callable. Any exception
    that import or attribute lookup raises propagates — callers handle
    the catching.
    """
    from loguru import logger

    if not callable(getattr(logger, "log", None)):
        # ImportError signals "the loguru API we need is not importable"
        # so the upstream `except Exception` in _probe_loguru routes us
        # to the stdlib fallback uniformly with other import failures.
        raise ImportError(  # noqa: TRY004 - ImportError is the bridge's contract
            "loguru.logger.log is not callable; loguru API has drifted."
        )
    return logger


def _probe_loguru():
    """
    Detect loguru and return a logger factory, or ``None``.

    On first call (or after a reset), attempts to import loguru and
    smoke-test the API. Caches a factory callable on success or the
    ``_LOGURU_UNAVAILABLE`` sentinel on any failure (including non-
    ``ImportError`` failures from broken installs). Subsequent calls
    return the cached value.

    Returns a zero-arg callable that yields a working ``loguru.Logger``
    when loguru is usable, otherwise ``None``.
    """
    if os.environ.get(_LOGURU_DETECT_AGAIN_ENV):
        _loguru_cache.pop("factory", None)

    if "factory" not in _loguru_cache:
        try:
            _resolve_loguru_logger()
        except Exception as exc:  # noqa: BLE001 - any import-time failure → fall back
            # Broad on purpose. Broken loguru installs raise
            # ImportError / ModuleNotFoundError (most common) plus
            # AttributeError / RuntimeError / OSError / TypeError /
            # ValueError on import side effects, and the API smoke-test
            # raises ImportError on .log drift. Excludes BaseException
            # so KeyboardInterrupt and SystemExit propagate.
            _stdlib_logger.warning(
                "loguru not usable (%s: %s); falling back to stdlib logging",
                type(exc).__name__,
                exc,
            )
            _loguru_cache["factory"] = _LOGURU_UNAVAILABLE
        else:
            _loguru_cache["factory"] = _resolve_loguru_logger

    factory = _loguru_cache["factory"]
    return None if factory is _LOGURU_UNAVAILABLE else factory


def _reset_loguru_cache() -> None:
    """
    Drop the cached loguru detection result so the next probe re-runs.

    Call this after a runtime install (e.g. ``%pip install loguru`` in
    a notebook, plugin loaders) or in tests that toggle availability.
    The cache is shared by sync and async paths, so a single reset
    suffices for both. See module docstring for the staleness contract.
    """
    _loguru_cache.pop("factory", None)


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
    "TRACE": logging.DEBUG,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "SUCCESS": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


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
        loguru_factory = _probe_loguru()
        if loguru_factory is None:
            raise ValueError(
                f"Unknown log level {level!r} for {param_name}; "
                f"valid levels: {sorted(_STDLIB_LEVEL_MAP)}"
            )


def _log(level: str, message: str, **kwargs) -> None:
    """
    Log using loguru if available, otherwise stdlib logging.

    ``level`` must be a non-None string.  Callers are closures inside the
    ``create_*_callbacks`` factories; those closures are only registered when
    their level parameter is truthy (see e.g. ``on_wait_start if wait_start
    else None`` guards), so ``_log`` is never reachable with ``level=None``.

    Resolves the loguru logger fresh per call (via the cached factory)
    so post-probe API drift surfaces as a clear error at the call site
    rather than from a poisoned cached reference.
    """
    if not isinstance(level, str):
        raise TypeError(f"_log level must be str, got {type(level).__name__}")
    loguru_factory = _probe_loguru()
    if loguru_factory is not None:
        loguru_factory().log(level, message, **kwargs)
    else:
        # Intentional KeyError on unknown levels — _log() is private and only
        # called from create_*_callbacks() factories with known level strings.
        stdlib_level = _STDLIB_LEVEL_MAP[level.upper()]
        if kwargs:
            extra = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
            _stdlib_logger.log(stdlib_level, "%s | %s", message, extra)
        else:
            _stdlib_logger.log(stdlib_level, message)


def _log_callback_timeout(timeout: float) -> None:
    _stdlib_logger.warning(
        "Rate limiter callback exceeded %.3fs timeout; skipping", timeout
    )


def _log_late_callback_exception(exc: BaseException) -> None:
    msg = (
        "Rate limiter callback raised after callback_timeout elapsed "
        f"{type(exc).__name__}: {exc}"
    )
    with contextlib.suppress(Warning):
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    _stdlib_logger.warning(msg)


def _invoke_sync_callback_checked(callback, **kwargs) -> None:
    result = callback(**kwargs)
    if inspect.isawaitable(result):
        close_awaitable_if_possible(result)
        raise TypeError(
            "Synchronous rate limiter callback returned an awaitable; "
            "use async RateLimiterCallbacks with RateLimiter instead"
        )


async def _invoke_async_callback_with_timeout(
    callback,
    callback_timeout: float | None,
    **kwargs,
) -> None:
    if callback_timeout is None:
        await callback(**kwargs)
        return
    try:
        await asyncio.wait_for(callback(**kwargs), timeout=callback_timeout)
    except TimeoutError:
        _log_callback_timeout(callback_timeout)


def _invoke_sync_callback_with_timeout(
    callback,
    callback_timeout: float | None,
    **kwargs,
) -> None:
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
            _log_late_callback_exception(exc)

    # Timeout-wrapped sync callbacks run in a helper thread.  Copy the caller's
    # contextvars context so ambient tracing/request state survives dispatch.
    thread = threading.Thread(target=lambda: context.run(run_callback), daemon=True)
    thread.start()
    try:
        future.result(timeout=callback_timeout)
    except concurrent.futures.TimeoutError:
        _log_callback_timeout(callback_timeout)
        if future.done():
            log_late_exception(future)
        else:
            future.add_done_callback(log_late_exception)


def with_callback_timeout(
    callbacks: RateLimiterCallbacks | None,
    timeout: float | None,
) -> RateLimiterCallbacks | None:
    if callbacks is None:
        return None
    callbacks.revalidate()

    def wrap(callback):
        if callback is None:
            return None

        async def wrapped(**kwargs) -> None:
            await _invoke_async_callback_with_timeout(callback, timeout, **kwargs)

        return wrapped

    return RateLimiterCallbacks(
        on_wait_start=wrap(callbacks.on_wait_start),
        after_wait_end_consumption=wrap(callbacks.after_wait_end_consumption),
        on_capacity_consumed=wrap(callbacks.on_capacity_consumed),
        on_capacity_refunded=wrap(callbacks.on_capacity_refunded),
        on_missing_consumption_data=wrap(callbacks.on_missing_consumption_data),
        on_lifecycle_event=wrap(callbacks.on_lifecycle_event),
    )


def with_sync_callback_timeout(
    callbacks: SyncRateLimiterCallbacks | None,
    timeout: float | None,
) -> SyncRateLimiterCallbacks | None:
    if callbacks is None:
        return None
    callbacks.revalidate()

    def wrap(callback):
        if callback is None:
            return None

        def wrapped(**kwargs) -> None:
            _invoke_sync_callback_with_timeout(callback, timeout, **kwargs)

        return wrapped

    return SyncRateLimiterCallbacks(
        on_wait_start=wrap(callbacks.on_wait_start),
        after_wait_end_consumption=wrap(callbacks.after_wait_end_consumption),
        on_capacity_consumed=wrap(callbacks.on_capacity_consumed),
        on_capacity_refunded=wrap(callbacks.on_capacity_refunded),
        on_missing_consumption_data=wrap(callbacks.on_missing_consumption_data),
        on_lifecycle_event=wrap(callbacks.on_lifecycle_event),
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
        """Called when no previous consumption data is detected, assuming full quota"""


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
        """Called when sync bucket state is first initialized at full quota."""


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
# Auto-detect callback factories (loguru if available, else stdlib logging)
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

    Uses loguru when it is importable and passes a small API smoke test;
    otherwise falls back to the stdlib ``token_throttle`` logger. Each keyword
    selects the log level for one callback slot. Pass ``None`` to leave that
    slot unset. The returned callbacks are suitable for ``RateLimiter`` and
    async backends.
    """
    for _name, _val in (
        ("wait_start", wait_start),
        ("wait_end_consumption", wait_end_consumption),
        ("capacity_consumed", capacity_consumed),
        ("capacity_refunded", capacity_refunded),
        ("missing_consumption_data", missing_consumption_data),
    ):
        _validate_log_level(_val, _name)

    async def on_wait_start(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
    ) -> None:
        _log(
            wait_start,
            "Rate limiter wait starting",
            model_family=model_family,
            usage=usage,
            preconsumption_capacities=preconsumption_capacities,
        )

    async def after_wait_end_consumption(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        wait_time_s: float,
    ) -> None:
        _log(
            wait_end_consumption,
            "Rate limiter wait complete",
            model_family=model_family,
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
        _log(
            capacity_consumed,
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
        _log(
            capacity_refunded,
            "Rate limiter capacity refunded",
            model_family=model_family,
            reserved_usage=reserved_usage,
            actual_usage=actual_usage,
            refunded_usage=refunded_usage,
            prerefund_capacities=prerefund_capacities,
            postrefund_capacities=postrefund_capacities,
        )

    async def on_missing_consumption_data(
        *,
        model_family: str,
        usage_metric: str,
        per_seconds: int,
    ) -> None:
        _log(
            missing_consumption_data,
            "Rate limiter missing consumption data",
            model_family=model_family,
            usage_metric=usage_metric,
            per_seconds=per_seconds,
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

    Uses loguru when it is importable and passes a small API smoke test;
    otherwise falls back to the stdlib ``token_throttle`` logger. Each keyword
    selects the log level for one callback slot. Pass ``None`` to leave that
    slot unset. The returned callbacks are suitable for ``SyncRateLimiter`` and
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

    def on_wait_start(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
    ) -> None:
        _log(
            wait_start,
            "Rate limiter wait starting",
            model_family=model_family,
            usage=usage,
            preconsumption_capacities=preconsumption_capacities,
        )

    def after_wait_end_consumption(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        wait_time_s: float,
    ) -> None:
        _log(
            wait_end_consumption,
            "Rate limiter wait complete",
            model_family=model_family,
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
        _log(
            capacity_consumed,
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
        _log(
            capacity_refunded,
            "Rate limiter capacity refunded",
            model_family=model_family,
            reserved_usage=reserved_usage,
            actual_usage=actual_usage,
            refunded_usage=refunded_usage,
            prerefund_capacities=prerefund_capacities,
            postrefund_capacities=postrefund_capacities,
        )

    def on_missing_consumption_data(
        *,
        model_family: str,
        usage_metric: str,
        per_seconds: int,
    ) -> None:
        _log(
            missing_consumption_data,
            "Rate limiter missing consumption data",
            model_family=model_family,
            usage_metric=usage_metric,
            per_seconds=per_seconds,
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


# ---------------------------------------------------------------------------
# Loguru-only callback factories (backward compatibility)
# ---------------------------------------------------------------------------


def _get_loguru_logger():
    """
    Resolve a working loguru logger or raise ``ImportError``.

    Used by the explicit ``create_loguru_callbacks`` factories. Calls
    the cached factory each invocation so post-probe API drift surfaces
    via the live attribute lookup at the call site.
    """
    loguru_factory = _probe_loguru()
    if loguru_factory is None:
        raise ImportError(
            'The "loguru" package is required for loguru callbacks. '
            'Install it with: pip install "token-throttle[loguru]"'
        )
    return loguru_factory()


def create_loguru_callbacks(
    *,
    # Defaults are None (opt-in), unlike create_logging_callbacks which defaults
    # to "DEBUG" (opt-out).  This is intentional: the loguru factories predate
    # create_logging_callbacks and changing their defaults would break callers.
    wait_start: str | None = None,
    wait_end_consumption: str | None = None,
    capacity_consumed: str | None = None,
    capacity_refunded: str | None = None,
    missing_consumption_data: str | None = None,
) -> RateLimiterCallbacks:
    """
    Create async callbacks that emit only through loguru.

    Unlike ``create_logging_callbacks``, this factory does not fall back to
    stdlib logging. It raises ``ImportError`` when loguru is unavailable or its
    logger API is not usable. Each keyword selects the loguru level for one
    callback slot; pass ``None`` to leave that slot unset.
    """
    for _name, _val in (
        ("wait_start", wait_start),
        ("wait_end_consumption", wait_end_consumption),
        ("capacity_consumed", capacity_consumed),
        ("capacity_refunded", capacity_refunded),
        ("missing_consumption_data", missing_consumption_data),
    ):
        _validate_log_level(_val, _name)

    async def on_wait_start(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
    ) -> None:
        logger = _get_loguru_logger()
        logger.log(
            wait_start,
            "Rate limiter wait starting",
            model_family=model_family,
            usage=usage,
            preconsumption_capacities=preconsumption_capacities,
        )

    async def after_wait_end_consumption(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        wait_time_s: float,
    ) -> None:
        logger = _get_loguru_logger()
        logger.log(
            wait_end_consumption,
            "Rate limiter wait complete",
            model_family=model_family,
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
        logger = _get_loguru_logger()
        logger.log(
            capacity_consumed,
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
        logger = _get_loguru_logger()
        logger.log(
            capacity_refunded,
            "Rate limiter capacity refunded",
            model_family=model_family,
            reserved_usage=reserved_usage,
            actual_usage=actual_usage,
            refunded_usage=refunded_usage,
            prerefund_capacities=prerefund_capacities,
            postrefund_capacities=postrefund_capacities,
        )

    async def on_missing_consumption_data(
        *,
        model_family: str,
        usage_metric: str,
        per_seconds: int,
    ) -> None:
        logger = _get_loguru_logger()
        logger.log(
            missing_consumption_data,
            "Rate limiter missing consumption data",
            model_family=model_family,
            usage_metric=usage_metric,
            per_seconds=per_seconds,
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


def create_sync_loguru_callbacks(
    *,
    wait_start: str | None = None,
    wait_end_consumption: str | None = None,
    capacity_consumed: str | None = None,
    capacity_refunded: str | None = None,
    missing_consumption_data: str | None = None,
) -> SyncRateLimiterCallbacks:
    """
    Create synchronous callbacks that emit only through loguru.

    Unlike ``create_sync_logging_callbacks``, this factory does not fall back
    to stdlib logging. It raises ``ImportError`` when loguru is unavailable or
    its logger API is not usable. Each keyword selects the loguru level for one
    callback slot; pass ``None`` to leave that slot unset.
    """
    for _name, _val in (
        ("wait_start", wait_start),
        ("wait_end_consumption", wait_end_consumption),
        ("capacity_consumed", capacity_consumed),
        ("capacity_refunded", capacity_refunded),
        ("missing_consumption_data", missing_consumption_data),
    ):
        _validate_log_level(_val, _name)

    def on_wait_start(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
    ) -> None:
        logger = _get_loguru_logger()
        logger.log(
            wait_start,
            "Rate limiter wait starting",
            model_family=model_family,
            usage=usage,
            preconsumption_capacities=preconsumption_capacities,
        )

    def after_wait_end_consumption(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        wait_time_s: float,
    ) -> None:
        logger = _get_loguru_logger()
        logger.log(
            wait_end_consumption,
            "Rate limiter wait complete",
            model_family=model_family,
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
        logger = _get_loguru_logger()
        logger.log(
            capacity_consumed,
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
        logger = _get_loguru_logger()
        logger.log(
            capacity_refunded,
            "Rate limiter capacity refunded",
            model_family=model_family,
            reserved_usage=reserved_usage,
            actual_usage=actual_usage,
            refunded_usage=refunded_usage,
            prerefund_capacities=prerefund_capacities,
            postrefund_capacities=postrefund_capacities,
        )

    def on_missing_consumption_data(
        *,
        model_family: str,
        usage_metric: str,
        per_seconds: int,
    ) -> None:
        logger = _get_loguru_logger()
        logger.log(
            missing_consumption_data,
            "Rate limiter missing consumption data",
            model_family=model_family,
            usage_metric=usage_metric,
            per_seconds=per_seconds,
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
