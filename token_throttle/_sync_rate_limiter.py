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
        self._cfg = cfg
        self._config_getter = lambda model_name: resolve_config(cfg, model_name)
        self._model_family_to_backend: dict[str, SyncRateLimiterBackend] = {}

    def acquire_capacity(self, usage: Usage, model: str) -> CapacityReservation:
        return self._acquire_or_record(usage, model, _block=True)

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
        return self._acquire_capacity(usage, limit_config, _block=_block)

    def _acquire_capacity(
        self,
        usage: FrozenUsage,
        limit_config: PerModelConfig,
        *,
        _block: bool = True,
    ) -> CapacityReservation:
        validate_acquire_usage(usage, limit_config.quotas)

        backend = self._get_backend(limit_config)
        if _block:
            backend.wait_for_capacity(usage)
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

    def _refund_capacity(
        self,
        actual_usage: Usage,
        reservation: CapacityReservation,
    ) -> None:
        actual_usage = frozen_usage(actual_usage)
        backend = self._model_family_to_backend.get(reservation.model_family)
        if backend is None:
            raise ValueError(
                f"Backend not found for model family {reservation.model_family}",
            )
        backend.refund_capacity(reservation.get_usage(), actual_usage)

    def _get_backend(self, cfg: PerModelConfig) -> SyncRateLimiterBackend:
        if not cfg.model_family:
            raise ValueError("cfg.model_family cannot be empty")
        if cfg.model_family in self._model_family_to_backend:
            return self._model_family_to_backend[cfg.model_family]

        with self._lock:
            # Double-checked locking: re-check after acquiring lock
            if cfg.model_family in self._model_family_to_backend:
                return self._model_family_to_backend[cfg.model_family]

            backend = self._backend.build(cfg, callbacks=self._callbacks)

            self._model_family_to_backend[cfg.model_family] = backend
            return backend
