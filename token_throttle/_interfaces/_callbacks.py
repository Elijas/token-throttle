import logging
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from token_throttle._interfaces._callable_utils import is_async_callable
from token_throttle._interfaces._models import Capacities, FrozenUsage

# ---------------------------------------------------------------------------
# Auto-detect loguru vs stdlib logging
# ---------------------------------------------------------------------------

_loguru_cache: dict[str, object] = {}


def _probe_loguru():
    if "logger" not in _loguru_cache:
        try:
            from loguru import logger

            _loguru_cache["logger"] = logger
        except ImportError:
            _loguru_cache["logger"] = None
    return _loguru_cache["logger"]


_stdlib_logger = logging.getLogger("token_throttle")

_STDLIB_LEVEL_MAP: dict[str, int] = {
    "TRACE": logging.DEBUG,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "SUCCESS": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _log(level: str, message: str, **kwargs) -> None:
    """
    Log using loguru if available, otherwise stdlib logging.

    ``level`` must be a non-None string.  Callers are closures inside the
    ``create_*_callbacks`` factories; those closures are only registered when
    their level parameter is truthy (see e.g. ``on_wait_start if wait_start
    else None`` guards), so ``_log`` is never reachable with ``level=None``.
    """
    loguru = _probe_loguru()
    if loguru is not None:
        loguru.log(level, message, **kwargs)
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
        """Called when capacity is consumed"""


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
    ) -> None: ...


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
            raise ValueError(
                f"{info.field_name} must be a synchronous callable"
            )
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
    loguru = _probe_loguru()
    if loguru is None:
        raise ImportError(
            'The "loguru" package is required for loguru callbacks. '
            'Install it with: pip install "token-throttle[loguru]"'
        )
    return loguru


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
