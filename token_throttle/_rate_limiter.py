import asyncio

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import (
    BaseRateLimiter,
    PerModelConfig,
    PerModelConfigGetter,
    RateLimiterBackend,
    RateLimiterBackendBuilderInterface,
)
from token_throttle._interfaces._models import (
    CapacityReservation,
    FrozenUsage,
    Usage,
    frozen_usage,
)
from token_throttle._validation import (
    resolve_config,
    validate_acquire_usage,
    validate_refund_usage,
)

_UNLIMITED_FLAG = "__rate_limiting_disabled__"


class RateLimiter(BaseRateLimiter):
    """
    Top-level async rate limiter — the main public entry point.

    Architecture:
      1. A *config* (static ``PerModelConfig`` or callable ``PerModelConfigGetter``)
         resolves a model name to quotas, a ``model_family``, and an optional
         ``usage_counter``.
      2. Each unique ``model_family`` gets its own ``RateLimiterBackend``,
         built lazily on first use and cached for the limiter's lifetime.
      3. ``acquire_capacity`` blocks until capacity is available;
         ``record_usage`` consumes immediately (capacity may go negative).
         Both return a ``CapacityReservation`` that must be passed to
         ``refund_capacity`` after the API call completes.
    """

    def __init__(
        self,
        cfg: PerModelConfig | PerModelConfigGetter,
        /,
        backend: RateLimiterBackendBuilderInterface,
        *,
        callbacks: RateLimiterCallbacks | None = None,
    ):
        self._backend = backend
        self._lock = asyncio.Lock()
        self._callbacks = callbacks
        self._cfg = cfg
        self._config_getter = lambda model_name: resolve_config(cfg, model_name)
        self._model_family_to_backend: dict[str, RateLimiterBackend] = {}

    async def acquire_capacity(self, usage: Usage, model: str) -> CapacityReservation:
        return await self._acquire_or_record(usage, model, _block=True)

    async def record_usage(self, usage: Usage, model: str) -> CapacityReservation:
        """
        Record usage without blocking.

        Capacity may go negative by design (speedometer pattern); this tracks
        overshoot rather than blocking.
        """
        return await self._acquire_or_record(usage, model, _block=False)

    async def _acquire_or_record(
        self,
        usage: Usage,
        model: str,
        *,
        _block: bool,
    ) -> CapacityReservation:
        usage = frozen_usage(usage)
        limit_config = self._config_getter(model)
        if limit_config.is_unlimited:
            if usage:
                raise ValueError("Usage must be empty for unlimited capacity")
            return CapacityReservation(
                usage={},
                model_family=_UNLIMITED_FLAG,
            )
        return await self._acquire_capacity(usage, limit_config, _block=_block)

    async def acquire_capacity_for_request(
        self,
        *,
        extra_usage: dict | None,
        **kwargs,
    ) -> CapacityReservation:
        model = kwargs.get("model")
        if not model:
            raise ValueError("'model' parameter is required")

        limit_config = self._config_getter(model)
        if limit_config.is_unlimited:
            return CapacityReservation(
                usage={},
                model_family=_UNLIMITED_FLAG,
            )
        if limit_config.usage_counter is None:
            raise ValueError("limit_config.usage_counter cannot be None")

        usage = dict(limit_config.usage_counter(**kwargs))
        if extra_usage:
            for k, v in extra_usage.items():
                if k not in usage:
                    raise ValueError(
                        f"Usage key '{k}' not found in usage counter",
                    )
                usage[k] += v
        return await self._acquire_capacity(frozen_usage(usage), limit_config)

    async def _acquire_capacity(
        self,
        usage: FrozenUsage,
        limit_config: PerModelConfig,
        *,
        _block: bool = True,
    ) -> CapacityReservation:
        validate_acquire_usage(usage, limit_config.quotas)

        backend = await self._get_backend(limit_config)
        if _block:
            await backend.await_for_capacity(usage)
        else:
            await backend.consume_capacity(usage)
        return CapacityReservation(
            usage=usage,
            model_family=limit_config.get_model_family(),
        )

    async def refund_capacity(
        self,
        actual_usage: Usage,
        reservation: CapacityReservation,
    ) -> None:
        if reservation.model_family == _UNLIMITED_FLAG:
            if actual_usage:
                raise ValueError(
                    "Usage must be empty for unlimited capacity reservations",
                )
            return
        validate_refund_usage(actual_usage, set(reservation.usage))
        await self._refund_capacity(
            actual_usage,
            reservation,
        )

    async def refund_capacity_from_response(
        self,
        reservation: CapacityReservation,
        response=None,
        **kwargs,
    ) -> None:
        if reservation.model_family == _UNLIMITED_FLAG:
            return
        if response is not None:
            # Pydantic model (OpenAI SDK v1+) or any object with .usage
            usage = response.usage
            total_tokens = (
                usage.total_tokens
                if hasattr(usage, "total_tokens")
                else usage["total_tokens"]
            )
        else:
            # Dict-based kwargs (legacy or manual)
            total_tokens = kwargs["usage"]["total_tokens"]
        actual_usage = {"tokens": total_tokens, "requests": 1}
        validate_refund_usage(actual_usage, set(reservation.usage))
        await self._refund_capacity(
            actual_usage,
            reservation,
        )

    async def set_max_capacity(
        self,
        model: str,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        """Dynamically change the max capacity for a specific bucket."""
        limit_config = self._config_getter(model)
        model_family = limit_config.get_model_family()
        backend = self._model_family_to_backend.get(model_family)
        if backend is None:
            raise ValueError(
                f"No backend for model family '{model_family}'. "
                "Call acquire_capacity or record_usage first."
            )
        await backend.set_max_capacity(metric, per_seconds, value)

    async def _refund_capacity(
        self,
        actual_usage: Usage,
        reservation: CapacityReservation,
    ) -> None:
        actual_usage = frozen_usage(actual_usage)
        # No need to call _config_getter since we already have the model_family
        # Just get the backend directly
        backend = self._model_family_to_backend.get(reservation.model_family)
        if backend is None:
            raise ValueError(
                f"Backend not found for model family {reservation.model_family}",
            )
        await backend.refund_capacity(reservation.get_usage(), actual_usage)

    async def _get_backend(self, cfg: PerModelConfig) -> RateLimiterBackend:
        if not cfg.model_family:
            raise ValueError("cfg.model_family cannot be empty")
        if cfg.model_family in self._model_family_to_backend:
            return self._model_family_to_backend[cfg.model_family]

        async with self._lock:
            # Check again after acquiring lock
            if cfg.model_family in self._model_family_to_backend:
                return self._model_family_to_backend[cfg.model_family]

            backend = self._backend.build(cfg, callbacks=self._callbacks)

            self._model_family_to_backend[cfg.model_family] = backend
            return backend
