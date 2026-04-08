import threading
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
    BucketId,
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
    merge_extra_usage_unrestricted,
    resolve_config,
    resolve_usage_counter_result,
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


def _reservation_bucket_ids(cfg: PerModelConfig) -> frozenset[BucketId]:
    """Bucket ids captured at reservation time for later scoped refunds."""
    return frozenset((q.metric, int(q.per_seconds)) for q in cfg.quotas)


def _resolved_model_family(cfg: PerModelConfig) -> str:
    """
    Stable routing key used to detect unsupported model remaps.

    Unlimited configs still keep their resolved ``model_family`` so a callable
    config can toggle limiting on and off without looking like a backend route
    change.
    """
    return cfg.get_model_family()


def _project_refund_scope(
    reserved_usage: FrozenUsage,
    actual_usage: FrozenUsage,
    reservation_bucket_ids: frozenset[BucketId] | None,
    active_bucket_ids: set[BucketId] | frozenset[BucketId] | None,
) -> tuple[FrozenUsage, FrozenUsage, frozenset[BucketId] | None]:
    """
    Shape refund data to the buckets that still correspond to the reservation.

    Callable configs can rebuild a model-family backend with a different bucket
    set after a reservation was created. Surviving bucket ids keep their
    original refund values, removed bucket ids are dropped, and legacy
    reservations without bucket ids fall back to metric-name projection.
    """
    if active_bucket_ids is None:
        return reserved_usage, actual_usage, reservation_bucket_ids

    active_bucket_ids = frozenset(active_bucket_ids)

    if reservation_bucket_ids is None:
        active_metric_names = frozenset(metric for metric, _ in active_bucket_ids)
        if set(reserved_usage) == set(active_metric_names):
            return reserved_usage, actual_usage, active_bucket_ids
        return (
            frozendict(
                {
                    metric: reserved_usage.get(metric, 0.0)
                    for metric in active_metric_names
                }
            ),
            frozendict(
                {
                    metric: actual_usage.get(metric, 0.0)
                    for metric in active_metric_names
                }
            ),
            active_bucket_ids,
        )

    surviving_bucket_ids = frozenset(
        bucket_id
        for bucket_id in reservation_bucket_ids
        if bucket_id in active_bucket_ids
    )
    if not surviving_bucket_ids:
        return frozendict(), frozendict(), surviving_bucket_ids

    surviving_metric_names = frozenset(metric for metric, _ in surviving_bucket_ids)
    if set(reserved_usage) == set(surviving_metric_names):
        return reserved_usage, actual_usage, surviving_bucket_ids

    return (
        frozendict(
            {
                metric: reserved_usage.get(metric, 0.0)
                for metric in surviving_metric_names
            }
        ),
        frozendict(
            {metric: actual_usage.get(metric, 0.0) for metric in surviving_metric_names}
        ),
        surviving_bucket_ids,
    )


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
        self._model_family_to_model_name: dict[str, str] = {}
        self._model_family_to_quotas: dict[str, dict[tuple[str, int], float]] = {}
        self._model_name_to_model_family: dict[str, str] = {}
        self._model_family_to_runtime_max_capacity: dict[
            str, dict[BucketId, float]
        ] = {}

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
        resolved_model_family = self._validated_model_family(model, limit_config)
        if limit_config.is_unlimited:
            reservation = self._unlimited_reservation(usage, model)
            self._remember_model_family(model, resolved_model_family)
            return reservation
        reservation = self._acquire_capacity(
            model,
            usage, limit_config, _block=_block, timeout=timeout
        )
        self._remember_model_family(model, resolved_model_family)
        return reservation

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
        resolved_model_family = self._validated_model_family(model, limit_config)
        if limit_config.is_unlimited:
            usage = frozendict()
            if limit_config.usage_counter is not None:
                usage = resolve_usage_counter_result(limit_config.usage_counter, **kwargs)
            usage = merge_extra_usage_unrestricted(usage, extra_usage)
            reservation = self._unlimited_reservation(usage, model)
            self._remember_model_family(model, resolved_model_family)
            return reservation
        if limit_config.usage_counter is None:
            raise ValueError("limit_config.usage_counter cannot be None")

        usage = merge_extra_usage(
            resolve_usage_counter_result(limit_config.usage_counter, **kwargs),
            extra_usage,
        )
        reservation = self._acquire_capacity(model, usage, limit_config, timeout=timeout)
        self._remember_model_family(model, resolved_model_family)
        return reservation

    def _acquire_capacity(
        self,
        model: str,
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
        model_family = limit_config.get_model_family()
        self._model_family_to_model_name[model_family] = model
        return CapacityReservation(
            usage=usage,
            model_family=model_family,
            bucket_ids=_reservation_bucket_ids(limit_config),
            model=model,
        )

    def refund_capacity(
        self,
        actual_usage: Usage,
        reservation: CapacityReservation,
    ) -> None:
        if is_unlimited_reservation(reservation):
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
        resolved_model_family = self._validated_model_family(model, limit_config)
        if limit_config.is_unlimited:
            raise ValueError("Cannot set max capacity: model has unlimited quotas")
        model_family = limit_config.get_model_family()
        self._model_family_to_model_name[model_family] = model
        backend = self._model_family_to_backend.get(model_family)
        if backend is None:
            raise ValueError(
                f"No backend for model family '{model_family}'. "
                "Call acquire_capacity or record_usage first."
            )
        backend = self._get_backend(limit_config)
        backend.set_max_capacity(metric, per_seconds, value)
        self._remember_runtime_max_capacity(model_family, metric, per_seconds, value)
        self._remember_model_family(model, resolved_model_family)

    def _refund_capacity(
        self,
        actual_usage: Usage,
        reservation: CapacityReservation,
    ) -> None:
        actual_usage = frozen_usage(actual_usage)
        if reservation.model_family not in self._model_family_to_backend:
            raise ValueError(
                f"Backend not found for model family {reservation.model_family}",
            )
        self._refresh_backend_for_reservation(reservation)
        backend = self._model_family_to_backend[reservation.model_family]
        active_bucket_ids = None
        snapshot = self._model_family_to_quotas.get(reservation.model_family)
        if snapshot is not None:
            active_bucket_ids = frozenset(snapshot)
        reserved_usage, actual_usage, refund_bucket_ids = _project_refund_scope(
            reservation.get_usage(),
            actual_usage,
            reservation.bucket_ids,
            active_bucket_ids,
        )
        if not reserved_usage:
            return
        if refund_bucket_ids is None or refund_bucket_ids == active_bucket_ids:
            backend.refund_capacity(
                reserved_usage,
                actual_usage,
            )
            return
        backend.refund_capacity_for_buckets(
            reserved_usage,
            actual_usage,
            bucket_ids=refund_bucket_ids,
        )

    def _unlimited_reservation(
        self,
        usage: FrozenUsage,
        model: str,
    ) -> CapacityReservation:
        return CapacityReservation(
            usage=usage,
            model_family=_UNLIMITED_FLAG,
            model=model,
            is_unlimited=True,
        )

    def _refresh_backend_for_reservation(
        self,
        reservation: CapacityReservation,
    ) -> None:
        model_name = reservation.model or self._model_family_to_model_name.get(
            reservation.model_family
        )
        if model_name is None:
            return

        limit_config = self._config_getter(model_name)
        if limit_config.is_unlimited:
            return
        if limit_config.get_model_family() != reservation.model_family:
            return
        self._get_backend(limit_config)

    def _get_backend(self, cfg: PerModelConfig) -> SyncRateLimiterBackend:
        if not cfg.model_family:
            raise ValueError("cfg.model_family cannot be empty")
        model_family = cfg.model_family
        new_snapshot = _quotas_snapshot(cfg)

        # Fast path: unchanged configs can reuse the cached backend without
        # taking the limiter lock. Refresh/rebuild work is serialized below so
        # concurrent callers cannot duplicate a rebuild.
        backend = self._model_family_to_backend.get(model_family)
        if (
            backend is not None
            and self._model_family_to_quotas.get(model_family) == new_snapshot
        ):
            return backend

        with self._lock:
            backend = self._model_family_to_backend.get(model_family)
            if backend is not None:
                return self._sync_backend_quotas(cfg)

            backend = self._backend.build(cfg, callbacks=self._callbacks)
            self._model_family_to_backend[model_family] = backend
            self._model_family_to_quotas[model_family] = new_snapshot
            return backend

    def _sync_backend_quotas(self, cfg: PerModelConfig) -> SyncRateLimiterBackend:
        """
        If quotas changed since backend creation, update or rebuild it.

        Caller must hold ``self._lock`` so only one concurrent caller can
        mutate a model-family backend at a time.
        """
        model_family = cfg.model_family
        new_snapshot = _quotas_snapshot(cfg)
        old_snapshot = self._model_family_to_quotas[model_family]

        if new_snapshot == old_snapshot:
            return self._model_family_to_backend[model_family]

        if set(new_snapshot) != set(old_snapshot):
            # Metric set changed — must rebuild backend (new metrics need new buckets)
            old_backend = self._model_family_to_backend[model_family]
            if not old_backend.supports_metric_set_change():
                raise RuntimeError(
                    f"Callable config for model family '{model_family}' changed metric set, "
                    f"but backend {type(old_backend).__name__} does not support "
                    "metric-set changes."
                )

            warnings.warn(
                f"Callable config for model family '{model_family}' changed metric set "
                f"(was {sorted(old_snapshot)}, now {sorted(new_snapshot)}). "
                "Rebuilding backend; consumption state for surviving metrics will be transferred.",
                UserWarning,
                stacklevel=2,
            )
            backend = self._backend.build(cfg, callbacks=self._callbacks)
            backend = old_backend.prepare_reconfigured_backend(backend, cfg)
            self._restore_runtime_max_capacity(
                model_family,
                old_snapshot=old_snapshot,
                new_snapshot=new_snapshot,
                backend=backend,
            )

            self._model_family_to_backend[model_family] = backend
            self._model_family_to_quotas[model_family] = new_snapshot
            return backend

        # Only limits changed — update in place via set_max_capacity
        backend = self._model_family_to_backend[model_family]
        changed_bucket_ids: set[BucketId] = set()
        for bucket_id, new_limit in new_snapshot.items():
            if new_limit != old_snapshot[bucket_id]:
                metric, per_seconds = bucket_id
                backend.set_max_capacity(metric, per_seconds, new_limit)
                changed_bucket_ids.add(bucket_id)
        self._clear_runtime_max_capacity(model_family, changed_bucket_ids)
        self._model_family_to_quotas[model_family] = new_snapshot
        return backend

    def _validated_model_family(
        self,
        model: str,
        limit_config: PerModelConfig,
    ) -> str:
        resolved_model_family = _resolved_model_family(limit_config)
        previous_model_family = self._model_name_to_model_family.get(model)
        if (
            previous_model_family is not None
            and previous_model_family != resolved_model_family
        ):
            raise ValueError(
                f"Config for model '{model}' changed model_family from "
                f"'{previous_model_family}' to '{resolved_model_family}'. "
                "Model routing must stay stable for a limiter instance; "
                "create a new SyncRateLimiter instead."
            )
        return resolved_model_family

    def _remember_model_family(
        self,
        model: str,
        model_family: str,
    ) -> None:
        self._model_name_to_model_family[model] = model_family

    def _remember_runtime_max_capacity(
        self,
        model_family: str,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        overrides = self._model_family_to_runtime_max_capacity.setdefault(
            model_family,
            {},
        )
        overrides[(metric, int(per_seconds))] = value

    def _clear_runtime_max_capacity(
        self,
        model_family: str,
        bucket_ids: set[BucketId],
    ) -> None:
        if not bucket_ids:
            return
        overrides = self._model_family_to_runtime_max_capacity.get(model_family)
        if not overrides:
            return
        for bucket_id in bucket_ids:
            overrides.pop(bucket_id, None)
        if not overrides:
            self._model_family_to_runtime_max_capacity.pop(model_family, None)

    def _restore_runtime_max_capacity(
        self,
        model_family: str,
        *,
        old_snapshot: dict[BucketId, float],
        new_snapshot: dict[BucketId, float],
        backend: SyncRateLimiterBackend,
    ) -> None:
        overrides = self._model_family_to_runtime_max_capacity.get(model_family)
        if not overrides:
            return

        restored_overrides: dict[BucketId, float] = {}
        for bucket_id, value in overrides.items():
            if bucket_id not in new_snapshot:
                continue
            if bucket_id not in old_snapshot:
                continue
            if old_snapshot[bucket_id] != new_snapshot[bucket_id]:
                continue

            metric, per_seconds = bucket_id
            backend.set_max_capacity(metric, per_seconds, value)
            restored_overrides[bucket_id] = value

        if restored_overrides:
            self._model_family_to_runtime_max_capacity[model_family] = (
                restored_overrides
            )
        else:
            self._model_family_to_runtime_max_capacity.pop(model_family, None)
