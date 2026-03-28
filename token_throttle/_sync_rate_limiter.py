import threading
import time
import warnings

from frozendict import frozendict

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
    _UNLIMITED_FLAG,
    extract_total_tokens,
    extract_usage_from_response,
    is_unlimited_reservation,
    merge_extra_usage,
    resolve_config,
    validate_acquire_usage,
    validate_extra_usage,
    validate_max_capacity_value,
    validate_metric,
    validate_per_seconds,
    validate_refund_usage,
    validate_timeout,
)


def _quotas_snapshot(cfg: PerModelConfig) -> dict[tuple[str, int], float]:
    """Snapshot of quotas for change detection: {(metric, per_seconds): limit}."""
    return {(q.metric, q.per_seconds): q.limit for q in cfg.quotas}


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
        self._model_family_to_quotas: dict[str, dict[tuple[str, int], float]] = {}

    def acquire_capacity(
        self, usage: Usage, model: str, *, timeout: float | None = None
    ) -> CapacityReservation:
        timeout = validate_timeout(timeout)
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
        timeout = validate_timeout(timeout)
        extra_usage = validate_extra_usage(extra_usage)
        if "model" not in kwargs:
            raise ValueError("'model' parameter is required")
        model = kwargs["model"]

        limit_config = self._config_getter(model)
        if limit_config.is_unlimited:
            if extra_usage:
                raise ValueError("extra_usage must be empty for unlimited capacity")
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
        return self._acquire_capacity(usage, limit_config, timeout=timeout)

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
        if is_unlimited_reservation(reservation):
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
        if is_unlimited_reservation(reservation):
            return
        if response is not None:
            # Pydantic model (OpenAI SDK v1+), raw response dict, or any object
            # with usage data.
            usage = extract_usage_from_response(response)
            total_tokens = extract_total_tokens(usage)
        else:
            if "usage" not in kwargs:
                raise ValueError(
                    "Either 'response' or 'usage' keyword argument is required"
                )
            total_tokens = extract_total_tokens(kwargs["usage"])
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
        metric = validate_metric(metric)
        per_seconds = validate_per_seconds(per_seconds)
        value = validate_max_capacity_value(value)
        limit_config = self._config_getter(model)
        if limit_config.is_unlimited:
            raise ValueError("Cannot set max capacity: model has unlimited quotas")
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
            return self._sync_backend_quotas(cfg)

        with self._lock:
            # Double-checked locking: re-check after acquiring lock
            if cfg.model_family in self._model_family_to_backend:  # pragma: no cover
                return self._sync_backend_quotas(cfg)

            backend = self._backend.build(cfg, callbacks=self._callbacks)
            self._model_family_to_backend[cfg.model_family] = backend
            self._model_family_to_quotas[cfg.model_family] = _quotas_snapshot(cfg)
            return backend

    def _sync_backend_quotas(self, cfg: PerModelConfig) -> SyncRateLimiterBackend:
        """If quotas changed since backend was built, update or rebuild it."""
        model_family = cfg.model_family
        new_snapshot = _quotas_snapshot(cfg)
        old_snapshot = self._model_family_to_quotas[model_family]

        if new_snapshot == old_snapshot:
            return self._model_family_to_backend[model_family]

        if set(new_snapshot) != set(old_snapshot):
            # Metric set changed — must rebuild backend (new metrics need new buckets)
            old_backend = self._model_family_to_backend[model_family]
            surviving_keys = set(new_snapshot) & set(old_snapshot)

            # Snapshot capacities from old backend for metrics that survive the rebuild
            old_capacities = None
            if surviving_keys and hasattr(old_backend, "_get_capacities"):
                with old_backend._condition:  # noqa: SLF001
                    old_capacities, _ = old_backend._get_capacities(time.time())  # noqa: SLF001

            warnings.warn(
                f"Callable config for model family '{model_family}' changed metric set "
                f"(was {sorted(old_snapshot)}, now {sorted(new_snapshot)}). "
                "Rebuilding backend; consumption state for surviving metrics will be transferred.",
                UserWarning,
                stacklevel=2,
            )
            backend = self._backend.build(cfg, callbacks=self._callbacks)

            # Transfer consumption state for metrics present in both old and new backends
            if old_capacities is not None and surviving_keys and hasattr(backend, "_set_capacities"):
                transfer = frozendict({k: old_capacities[k] for k in surviving_keys if k in old_capacities})
                if transfer:
                    with backend._condition:  # noqa: SLF001
                        backend._set_capacities(transfer, time.time())  # noqa: SLF001

            self._model_family_to_backend[model_family] = backend
            self._model_family_to_quotas[model_family] = new_snapshot
            return backend

        # Only limits changed — update in place via set_max_capacity
        backend = self._model_family_to_backend[model_family]
        for bucket_id, new_limit in new_snapshot.items():
            if new_limit != old_snapshot[bucket_id]:
                metric, per_seconds = bucket_id
                backend.set_max_capacity(metric, per_seconds, new_limit)
        self._model_family_to_quotas[model_family] = new_snapshot
        return backend
