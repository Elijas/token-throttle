from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from token_throttle._interfaces._models import Capacities, FrozenUsage


@runtime_checkable
class OnWaitStartCallback(Protocol):
    async def __call__(
        self,
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
    ) -> None:
        """Called before waiting for capacity."""


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
        """Called after successfully acquiring capacit if there is a wait time"""


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
        description="Called before waiting for capacity",
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


def _get_loguru_logger():
    try:
        from loguru import logger

        return logger
    except ImportError as exc:
        raise ImportError(
            'The "loguru" package is required for logging callbacks. '
            "Install it with: pip install loguru"
        ) from exc


def create_loguru_callbacks(
    *,
    wait_start: str | None = None,
    wait_end_consumption: str | None = None,
    capacity_consumed: str | None = None,
    capacity_refunded: str | None = None,
    missing_consumption_data: str | None = None,
) -> RateLimiterCallbacks:
    default = "DEBUG"

    async def on_wait_start(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
    ) -> None:
        logger = _get_loguru_logger()
        logger.log(
            wait_start or default,
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
            wait_end_consumption or default,
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
            capacity_consumed or default,
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
            capacity_refunded or default,
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
            missing_consumption_data or default,
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


# ---------------------------------------------------------------------------
# Sync counterparts
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
        description="Called before waiting for capacity",
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


def create_sync_loguru_callbacks(
    *,
    wait_start: str | None = None,
    wait_end_consumption: str | None = None,
    capacity_consumed: str | None = None,
    capacity_refunded: str | None = None,
    missing_consumption_data: str | None = None,
) -> SyncRateLimiterCallbacks:
    default = "DEBUG"

    def on_wait_start(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
    ) -> None:
        logger = _get_loguru_logger()
        logger.log(
            wait_start or default,
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
            wait_end_consumption or default,
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
            capacity_consumed or default,
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
            capacity_refunded or default,
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
            missing_consumption_data or default,
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
