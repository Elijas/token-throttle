import threading

from token_throttle._interfaces._callbacks import SyncRateLimiterCallbacks
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    PerModelConfigGetter,
    SyncRateLimiterBackend,
    SyncRateLimiterBackendBuilderInterface,
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


class SyncRateLimiter:
    """Synchronous counterpart of ``RateLimiter`` — same architecture and contract."""

    def __init__(
        self,
        cfg: PerModelConfig | PerModelConfigGetter,
        /,
        backend: SyncRateLimiterBackendBuilderInterface,
        *,
        callbacks: SyncRateLimiterCallbacks | None = None,
    ):
        self._backend = backend
        self._lock = threading.Lock()
        self._callbacks = callbacks
        self._config_getter = lambda model_name: resolve_config(cfg, model_name)
        self._model_family_to_backend: dict[str, SyncRateLimiterBackend] = {}

    def acquire_capacity(
        self, usage: Usage, model: str, *, timeout: float | None = None
    ) -> CapacityReservation:
        return self._acquire_or_record(usage, model, _block=True, timeout=timeout)

    def record_usage(self, usage: Usage, model: str) -> CapacityReservation:
        """
        Record usage without blocking.

        Capacity may go negative by design (speedometer pattern); this tracks
        overshoot rather than blocking.
        """
        return self._acquire_or_record(usage, model, _block=False)

    def _acquire_or_record(
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
        return self._acquire_capacity(
            usage, limit_config, _block=_block, timeout=timeout
        )

    def acquire_capacity_for_request(
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

        usage = dict(limit_config.usage_counter(**kwargs))
        if extra_usage:
            for k, v in extra_usage.items():
                if k not in usage:
                    raise ValueError(
                        f"Usage key '{k}' not found in usage counter",
                    )
                usage[k] += v
        return self._acquire_capacity(
            frozen_usage(usage), limit_config, timeout=timeout
        )

    def _acquire_capacity(
        self,
        usage: FrozenUsage,
        limit_config: PerModelConfig,
        *,
        _block: bool = True,
        timeout: float | None = None,
    ) -> CapacityReservation:
        validate_acquire_usage(usage, limit_config.quotas)

        backend = self._get_backend(limit_config)
        if _block:
            backend.wait_for_capacity(usage, timeout=timeout)
        else:
            backend.consume_capacity(usage)
        return CapacityReservation(
            usage=usage,
            model_family=limit_config.get_model_family(),
        )

    def refund_capacity(
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
        self._refund_capacity(actual_usage, reservation)

    def refund_capacity_from_response(
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
        if reservation.model_family == _UNLIMITED_FLAG:
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
            total_tokens = (
                usage.total_tokens
                if hasattr(usage, "total_tokens")
                else usage["total_tokens"]
            )
            if total_tokens is None:
                raise ValueError(
                    "total_tokens is None — cannot compute refund. "
                    "Pass actual usage via refund_capacity() instead."
                )
        else:
            if "usage" not in kwargs:
                raise ValueError(
                    "Either 'response' or 'usage' keyword argument is required"
                )
            total_tokens = kwargs["usage"]["total_tokens"]
            if total_tokens is None:
                raise ValueError(
                    "total_tokens is None — cannot compute refund. "
                    "Pass actual usage via refund_capacity() instead."
                )
        actual_usage = {"tokens": total_tokens, "requests": 1}
        validate_refund_usage(actual_usage, set(reservation.usage))
        self._refund_capacity(
            actual_usage,
            reservation,
        )

    def set_max_capacity(
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
        backend.set_max_capacity(metric, per_seconds, value)

    def _refund_capacity(
        self,
        actual_usage: Usage,
        reservation: CapacityReservation,
    ) -> None:
        actual_usage = frozen_usage(actual_usage)
        # Lock-free read is safe: same reasoning as _get_backend (GIL protects
        # dict lookups; values are set once and never mutated/removed).
        backend = self._model_family_to_backend.get(reservation.model_family)
        if backend is None:
            raise ValueError(
                f"Backend not found for model family {reservation.model_family}",
            )
        backend.refund_capacity(reservation.get_usage(), actual_usage)

    def _get_backend(self, cfg: PerModelConfig) -> SyncRateLimiterBackend:
        if not cfg.model_family:
            raise ValueError("cfg.model_family cannot be empty")
        # Lock-free read is safe: GIL protects dict lookups, and the value
        # is set once (never mutated/removed after insertion).
        if cfg.model_family in self._model_family_to_backend:
            return self._model_family_to_backend[cfg.model_family]

        with self._lock:
            # Double-checked locking: re-check after acquiring lock
            if cfg.model_family in self._model_family_to_backend:  # pragma: no cover
                return self._model_family_to_backend[cfg.model_family]

            backend = self._backend.build(cfg, callbacks=self._callbacks)

            self._model_family_to_backend[cfg.model_family] = backend
            return backend
