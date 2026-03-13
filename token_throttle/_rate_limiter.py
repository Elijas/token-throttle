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
    merge_extra_usage,
    resolve_config,
    validate_acquire_usage,
    validate_refund_usage,
)

_UNLIMITED_FLAG = "__rate_limiting_disabled__"


def _is_unlimited_reservation(reservation: CapacityReservation) -> bool:
    return reservation.model_family == _UNLIMITED_FLAG and not reservation.usage


def _extract_total_tokens(usage: object) -> int | float:
    """Extract total_tokens from a usage object (attribute or dict access)."""
    if hasattr(usage, "total_tokens"):
        total_tokens = usage.total_tokens
    elif isinstance(usage, dict):
        try:
            total_tokens = usage["total_tokens"]
        except KeyError:
            raise ValueError(
                "'total_tokens' key not found in usage data — "
                "pass actual usage via refund_capacity() instead."
            ) from None
    else:
        raise ValueError(
            "usage must be an object with total_tokens attribute or a dict"
        )
    if total_tokens is None:
        raise ValueError(
            "total_tokens is None — cannot compute refund. "
            "Pass actual usage via refund_capacity() instead."
        )
    return total_tokens


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
        self._config_getter = lambda model_name: resolve_config(cfg, model_name)
        self._model_family_to_backend: dict[str, RateLimiterBackend] = {}

    async def acquire_capacity(
        self, usage: Usage, model: str, *, timeout: float | None = None
    ) -> CapacityReservation:
        return await self._acquire_or_record(usage, model, _block=True, timeout=timeout)

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
        timeout: float | None = None,
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
        return await self._acquire_capacity(
            usage, limit_config, _block=_block, timeout=timeout
        )

    async def acquire_capacity_for_request(
        self,
        *,
        extra_usage: dict | None = None,
        timeout: float | None = None,
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

        usage = merge_extra_usage(
            frozen_usage(limit_config.usage_counter(**kwargs)),
            extra_usage,
        )
        return await self._acquire_capacity(
            usage, limit_config, timeout=timeout
        )

    async def _acquire_capacity(
        self,
        usage: FrozenUsage,
        limit_config: PerModelConfig,
        *,
        _block: bool = True,
        timeout: float | None = None,
    ) -> CapacityReservation:
        validate_acquire_usage(usage, limit_config.quotas)

        backend = await self._get_backend(limit_config)
        if _block:
            await backend.await_for_capacity(usage, timeout=timeout)
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
        if _is_unlimited_reservation(reservation):
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
        """
        Convenience for OpenAI-style responses with ``total_tokens``.

        Requires metric names ``"tokens"`` and ``"requests"`` (as configured by
        ``create_openai_*`` factories).  For custom metric names, use
        :meth:`refund_capacity` directly.
        """
        if _is_unlimited_reservation(reservation):
            return
        if response is not None:
            # Pydantic model (OpenAI SDK v1+) or any object with .usage
            usage = response.usage
            if usage is None:
                raise ValueError(
                    "response.usage is None — cannot extract token counts. "
                    "Streaming responses may not include usage data; "
                    "pass actual usage via refund_capacity() instead."
                )
            total_tokens = _extract_total_tokens(usage)
        else:
            if "usage" not in kwargs:
                raise ValueError(
                    "Either 'response' or 'usage' keyword argument is required"
                )
            total_tokens = _extract_total_tokens(kwargs["usage"])
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
        # Lock-free read is safe: same reasoning as _get_backend (asyncio is
        # single-threaded; values are set once and never mutated/removed).
        backend = self._model_family_to_backend.get(reservation.model_family)
        if backend is None:
            raise ValueError(
                f"Backend not found for model family {reservation.model_family}",
            )
        await backend.refund_capacity(reservation.get_usage(), actual_usage)

    async def _get_backend(self, cfg: PerModelConfig) -> RateLimiterBackend:
        if not cfg.model_family:
            raise ValueError("cfg.model_family cannot be empty")
        # Lock-free read is safe: asyncio is single-threaded, and the dict
        # value is set once (never mutated/removed after insertion).
        if cfg.model_family in self._model_family_to_backend:
            return self._model_family_to_backend[cfg.model_family]

        async with self._lock:
            # Check again after acquiring lock
            if cfg.model_family in self._model_family_to_backend:  # pragma: no cover
                return self._model_family_to_backend[cfg.model_family]

            backend = self._backend.build(cfg, callbacks=self._callbacks)

            self._model_family_to_backend[cfg.model_family] = backend
            return backend
