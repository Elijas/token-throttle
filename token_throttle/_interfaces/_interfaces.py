from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
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
    _validate_key_segment,
)

UsageCounter = Callable[..., FrozenUsage]
"""Synchronous callable that derives a usage mapping from request kwargs.

Counters should accept ``**kwargs`` so they receive the full request payload.
Fixed-signature counters are still supported for compatibility, but that
dispatch path is deprecated because request fields not named in the signature
are filtered out before the counter is called.
"""


class PerModelConfig(BaseModel):
    """
    Configuration for limiting API requests to a model.

    Use ``PerModelConfig(quotas=UsageQuotas.unlimited(), ...)`` to disable
    rate limiting for a model while preserving the normal limiter API.
    """

    quotas: UsageQuotas = Field(
        ...,
        description=(
            "Defines the maximum usage per time window. "
            "Allows tracking of resources like requests and tokens per minute, hour, etc."
        ),
    )
    usage_counter: UsageCounter | None = Field(
        default=None,
        description=(
            "Optional synchronous callable that derives usage from request kwargs. "
            "Counter contract: accept **kwargs to receive the full request payload; "
            "fixed-signature counters are deprecated because unmatched request "
            "fields are filtered before invocation. The callable must not be "
            "async, must not be an async generator, and must return a usage "
            "mapping. Async RateLimiter invokes the counter inline on the event "
            "loop, so expensive CPU work, blocking I/O, or sleep calls block "
            "concurrent rate-limited work unless the caller wraps the counter "
            "explicitly, for example with asyncio.to_thread."
        ),
    )

    model_family: str | None = Field(
        default=None,
        description=(
            "Optional identifier for rate limiting purposes. Multiple model "
            "versions can share the same model_family to count against the same "
            "quota, but all models that resolve to the same model_family must "
            "expose identical quotas and unlimited-vs-limited behavior within "
            "a limiter instance. Defaults to the model name if not specified. "
            "When set, the value must be non-empty, NFC normalized, printable, "
            "contain no whitespace/control characters, and cannot contain ':'; "
            "the portable recommended character set is ^[A-Za-z0-9_./-]+$."
        ),
    )

    @field_validator("model_family", mode="before")
    @classmethod
    def _reject_empty_string(cls, value: object) -> object:
        return _validate_key_segment(value, field_name="model_family", allow_none=True)

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
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)


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

        *timeout* controls how long to wait for capacity:
        - ``None`` (default): block indefinitely (current behaviour).
        - ``0``: try-acquire — return immediately or raise ``TimeoutError``.
        - ``N > 0``: wait up to *N* seconds, then raise ``TimeoutError``.

        The timeout is not a total wall-clock deadline: backend operation
        latency and callback dispatch are outside this budget. Public limiters
        bound callback dispatch separately with ``callback_timeout``.

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

    async def apply_configured_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        """
        Apply a config-driven max-capacity update.

        Distinct from :meth:`set_max_capacity`, which is the explicit runtime
        override API. Backends that do not need separate persistence semantics
        can keep the default behaviour and delegate to :meth:`set_max_capacity`.
        """
        await self.set_max_capacity(metric, per_seconds, value)

    def supports_metric_set_change(self) -> bool:
        """
        Whether the backend can safely handle callable config metric-set changes.

        The default is ``False`` because rebuilding a backend that keeps its
        state only in local memory will otherwise lose or split accounting.

        Override this to return ``True`` only when metric additions/removals
        can preserve accounting for surviving buckets. To do so you must
        satisfy one of two contracts: store live state in stable external
        storage keyed by metric/window, or override
        :meth:`prepare_reconfigured_backend` to migrate/share in-process state.
        Returning ``True`` while inheriting the no-op migration can silently
        reset surviving metrics' consumption state.

        L18 H07 deliberately keeps this default ``False``: flipping it to
        ``True`` would make naive custom backends opt into silent state loss.
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


def backend_uses_default_prepare_reconfigured_backend(
    backend: RateLimiterBackend,
) -> bool:
    """Return whether *backend* inherits the ABC's no-op reconfiguration hook."""
    return (
        type(backend).prepare_reconfigured_backend
        is RateLimiterBackend.prepare_reconfigured_backend
    )


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
    ) -> None:
        """
        Wait until all buckets can satisfy *usage*, then consume atomically.

        *timeout* controls how long capacity waiting may block:
        - ``None`` (default): block indefinitely.
        - ``0``: try-acquire — return immediately or raise ``TimeoutError``.
        - ``N > 0``: wait up to *N* seconds, then raise ``TimeoutError``.

        The timeout is not a total wall-clock deadline: backend operation
        latency and callback dispatch are outside this budget. Public limiters
        bound callback dispatch separately with ``callback_timeout``.
        """

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

    def apply_configured_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        """
        Synchronous counterpart of ``apply_configured_max_capacity``.

        The default falls back to ``set_max_capacity()`` for backwards
        compatibility with custom backends.
        """
        self.set_max_capacity(metric, per_seconds, value)

    def supports_metric_set_change(self) -> bool:
        """
        Whether the backend can safely handle callable config metric-set changes.

        The default is ``False`` because rebuilding a backend that keeps its
        state only in local memory will otherwise lose or split accounting.

        Override this to return ``True`` only when metric additions/removals
        can preserve accounting for surviving buckets. If your backend keeps
        any live state outside stable shared storage, you must also override
        :meth:`prepare_reconfigured_backend` to migrate or share that state.
        Returning ``True`` while inheriting the no-op
        ``prepare_reconfigured_backend`` can silently reset surviving metrics'
        consumption state.

        L18 H07 deliberately keeps this default ``False``: flipping it to
        ``True`` would make naive custom backends opt into silent state loss.
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


def sync_backend_uses_default_prepare_reconfigured_backend(
    backend: SyncRateLimiterBackend,
) -> bool:
    """Return whether *backend* inherits the ABC's no-op reconfiguration hook."""
    return (
        type(backend).prepare_reconfigured_backend
        is SyncRateLimiterBackend.prepare_reconfigured_backend
    )


class SyncRateLimiterBackendBuilderInterface(ABC):
    @abstractmethod
    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> SyncRateLimiterBackend: ...
