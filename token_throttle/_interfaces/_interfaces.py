from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._models import (
    CapacityReservation,
    FrozenUsage,
    Usage,
    UsageQuotas,
)


@runtime_checkable
class UsageCounter(Protocol):
    def __call__(self, **request) -> FrozenUsage: ...


class PerModelConfig(BaseModel):
    """Configuration for limiting API requests to a model."""

    quotas: UsageQuotas = Field(
        ...,
        description=(
            "Defines the maximum usage per minute. "
            "Allows tracking of resources like requests and tokens per minute."
        ),
    )
    usage_counter: UsageCounter | None = Field(
        default=None,
        description="Optional function to count usage tokens.",
    )

    model_family: str | None = Field(
        default=None,
        description="Optional identifier for rate limiting purposes. Multiple model versions can share the same model_family to count against the same quota. Defaults to the model name if not specified.",
    )

    def get_model_family(self) -> str:
        if not self.model_family:
            raise ValueError("model_family must be defined")
        return self.model_family

    @property
    def is_unlimited(self) -> bool:
        return self.quotas.is_unlimited

    # Note: in "model_config", "model" means Pydantic Model, not LLM Model like in other fields of this class
    model_config = ConfigDict(arbitrary_types_allowed=True)


@runtime_checkable
class PerModelConfigGetter(Protocol):
    def __call__(self, model_name: str, /) -> PerModelConfig:
        """model_name: The model identifier used in API requests (e.g., 'gpt-4o')."""
        ...


class RateLimiterBackendBuilderInterface(ABC):
    @abstractmethod
    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> RateLimiterBackend: ...


class RateLimiterBackend(ABC):
    @abstractmethod
    async def await_for_capacity(self, usage: FrozenUsage) -> None: ...

    @abstractmethod
    async def consume_capacity(self, usage: FrozenUsage) -> None:
        """Consume capacity unconditionally. Capacity may go negative."""

    @abstractmethod
    async def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None: ...

    @abstractmethod
    async def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        """Dynamically change the max capacity for a specific bucket."""


class BaseRateLimiter(ABC):
    @abstractmethod
    async def acquire_capacity(
        self,
        usage: Usage,
        model: str,
    ) -> CapacityReservation: ...

    @abstractmethod
    async def acquire_capacity_for_request(self, **kwargs) -> CapacityReservation: ...

    @abstractmethod
    async def refund_capacity(
        self,
        actual_usage: Usage,
        reservation: CapacityReservation,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Sync counterparts
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from token_throttle._interfaces._callbacks import SyncRateLimiterCallbacks


class SyncRateLimiterBackend(ABC):
    @abstractmethod
    def wait_for_capacity(self, usage: FrozenUsage) -> None: ...

    @abstractmethod
    def consume_capacity(self, usage: FrozenUsage) -> None:
        """Consume capacity unconditionally. Capacity may go negative."""

    @abstractmethod
    def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None: ...

    @abstractmethod
    def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        """Dynamically change the max capacity for a specific bucket."""


class SyncRateLimiterBackendBuilderInterface(ABC):
    @abstractmethod
    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> SyncRateLimiterBackend: ...
