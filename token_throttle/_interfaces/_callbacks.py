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

import inspect
import logging
import os
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from token_throttle._interfaces._callable_utils import is_async_callable
from token_throttle._interfaces._models import Capacities, FrozenUsage

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


# ---------------------------------------------------------------------------
# Async callback protocols
# ---------------------------------------------------------------------------


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


class RateLimiterCallbacks(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    @field_validator(
        "on_wait_start",
        "after_wait_end_consumption",
        "on_capacity_consumed",
        "on_capacity_refunded",
        "on_missing_consumption_data",
        mode="after",
    )
    @classmethod
    def _validate_async_callbacks(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if value is not None and not is_async_callable(value):
            raise ValueError(f"{info.field_name} must be an async callable")
        if value is not None and info.field_name is not None:
            _validate_callback_signature(value, info.field_name)
        return value


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
    ) -> None: ...


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
    ) -> None: ...


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
        Called when capacity is consumed.

        Not 100% delivery-guaranteed under task cancellation: if the
        calling task is cancelled while a shielded backend write is
        in flight and that write commits, the cancellation is suppressed
        and this callback is skipped.
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
    ) -> None: ...


@runtime_checkable
class SyncOnMissingConsumptionDataCallback(Protocol):
    def __call__(
        self,
        *,
        model_family: str,
        usage_metric: str,
        per_seconds: int,
    ) -> None: ...


class SyncRateLimiterCallbacks(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    @field_validator(
        "on_wait_start",
        "after_wait_end_consumption",
        "on_capacity_consumed",
        "on_capacity_refunded",
        "on_missing_consumption_data",
        mode="after",
    )
    @classmethod
    def _validate_sync_callbacks(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if value is not None and is_async_callable(value):
            raise ValueError(f"{info.field_name} must be a synchronous callable")
        if value is not None and info.field_name is not None:
            _validate_callback_signature(value, info.field_name)
        return value


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
