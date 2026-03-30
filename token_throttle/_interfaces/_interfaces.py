from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from token_throttle._interfaces._callable_utils import is_async_callable
from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._models import (
    BucketId,
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
            "Defines the maximum usage per time window. "
            "Allows tracking of resources like requests and tokens per minute, hour, etc."
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

    @field_validator("model_family", mode="before")
    @classmethod
    def _reject_empty_string(cls, value: object) -> object:
        if isinstance(value, str) and not value:
            raise ValueError("model_family must not be an empty string")
        return value

    @field_validator("usage_counter", mode="after")
    @classmethod
    def _reject_async_usage_counter(
        cls,
        value: UsageCounter | None,
        info: ValidationInfo,
    ) -> UsageCounter | None:
        if value is not None and is_async_callable(value):
            raise ValueError(f"{info.field_name} must be a synchronous callable")
        return value

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
    """
    Per-model-family backend that owns a set of token buckets.

    Consumption is all-or-nothing: either every bucket for the requested
    metrics has sufficient capacity and they are all decremented atomically,
    or nothing is consumed.

    Capacity is checked against the live bucket ``max_capacity`` (which may
    differ from the static ``quota.limit`` after a ``set_max_capacity`` call),
    not the original quota value.

    Implementation note — callbacks are fired *outside* the lock so that
    user callback code cannot deadlock the backend.  This means callback
    data is a point-in-time snapshot; other requests may have changed
    capacity between lock release and callback invocation.
    """

    @abstractmethod
    async def await_for_capacity(
        self, usage: FrozenUsage, *, timeout: float | None = None
    ) -> None:
        """
        Wait until all buckets can satisfy *usage*, then consume atomically.

        *timeout* controls how long to wait:
        - ``None`` (default): block indefinitely (current behaviour).
        - ``0``: try-acquire — return immediately or raise ``TimeoutError``.
        - ``N > 0``: wait up to *N* seconds, then raise ``TimeoutError``.

        Raises ``ValueError`` immediately (fail-fast) if any single metric
        in *usage* exceeds that bucket's ``max_capacity``, because waiting
        would be infinite.
        """

    @abstractmethod
    async def consume_capacity(self, usage: FrozenUsage) -> None:
        """
        Consume capacity unconditionally.

        Capacity may go negative by design (speedometer pattern); this tracks
        overshoot rather than blocking.
        """

    @abstractmethod
    async def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        """
        Return unused capacity after a request completes.

        *reserved_usage* is what ``acquire_capacity`` originally reserved.
        *actual_usage* is what the request actually consumed (e.g. from the
        API response).  The difference is added back to each bucket, capped
        at ``max_capacity``.

        If actual > reserved the refund is negative (increases debt).
        Negative capacity is preserved so the token-bucket refill handles
        recovery naturally — clamping to zero would silently erase debt
        created by the speedometer / ``record_usage`` path.
        """

    async def refund_capacity_for_buckets(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        *,
        bucket_ids: set[BucketId] | frozenset[BucketId] | None = None,
    ) -> None:
        """
        Return unused capacity to a specific subset of buckets.

        Backends that support metric-set reconfiguration should override this
        so refunds created before a config rebuild only touch surviving bucket
        ids. The default falls back to ``refund_capacity()``.
        """
        await self.refund_capacity(reserved_usage, actual_usage)

    @abstractmethod
    async def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        """
        Dynamically change the max capacity for a specific bucket.

        Also recalculates the refill rate (``max_capacity / per_seconds``)
        so that a full refill still takes exactly one time window.
        """

    def supports_metric_set_change(self) -> bool:
        """
        Whether the backend can safely handle callable config metric-set changes.

        The default is ``False`` because rebuilding a backend that keeps its
        state only in local memory will otherwise lose or split accounting.
        Backends with shared external state (for example Redis) or custom
        rebuild logic should override this and, when needed, also override
        :meth:`prepare_reconfigured_backend`.
        """
        return False

    async def prepare_reconfigured_backend(
        self,
        new_backend: RateLimiterBackend,
        _cfg: PerModelConfig,
    ) -> RateLimiterBackend:
        """
        Finalize a rebuilt backend before the limiter installs it.

        Called only for metric-set changes after ``new_backend`` has been
        constructed from the new config. The default implementation is a no-op.
        Backends that keep local state can override this to share or migrate
        live state into the rebuilt backend.
        """
        return new_backend


class BaseRateLimiter(ABC):
    @abstractmethod
    async def acquire_capacity(
        self,
        usage: Usage,
        model: str,
        *,
        timeout: float | None = None,
    ) -> CapacityReservation: ...

    @abstractmethod
    async def acquire_capacity_for_request(
        self, *, timeout: float | None = None, **kwargs
    ) -> CapacityReservation: ...

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
    """Synchronous counterpart of ``RateLimiterBackend`` — same contract."""

    @abstractmethod
    def wait_for_capacity(
        self, usage: FrozenUsage, *, timeout: float | None = None
    ) -> None: ...

    @abstractmethod
    def consume_capacity(self, usage: FrozenUsage) -> None:
        """
        Consume capacity unconditionally.

        Capacity may go negative by design (speedometer pattern); this tracks
        overshoot rather than blocking.
        """

    @abstractmethod
    def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None: ...

    def refund_capacity_for_buckets(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        *,
        bucket_ids: set[BucketId] | frozenset[BucketId] | None = None,
    ) -> None:
        """
        Synchronous counterpart of ``refund_capacity_for_buckets``.

        The default falls back to ``refund_capacity()`` for backwards
        compatibility with custom backends.
        """
        self.refund_capacity(reserved_usage, actual_usage)

    @abstractmethod
    def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        """Dynamically change the max capacity for a specific bucket."""

    def supports_metric_set_change(self) -> bool:
        """
        Whether the backend can safely handle callable config metric-set changes.

        The default is ``False`` because rebuilding a backend that keeps its
        state only in local memory will otherwise lose or split accounting.
        Backends with shared external state (for example Redis) or custom
        rebuild logic should override this and, when needed, also override
        :meth:`prepare_reconfigured_backend`.
        """
        return False

    def prepare_reconfigured_backend(
        self,
        new_backend: SyncRateLimiterBackend,
        _cfg: PerModelConfig,
    ) -> SyncRateLimiterBackend:
        """
        Finalize a rebuilt backend before the limiter installs it.

        Called only for metric-set changes after ``new_backend`` has been
        constructed from the new config. The default implementation is a no-op.
        Backends that keep local state can override this to share or migrate
        live state into the rebuilt backend.
        """
        return new_backend


class SyncRateLimiterBackendBuilderInterface(ABC):
    @abstractmethod
    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> SyncRateLimiterBackend: ...
