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
    Quota,
    Usage,
    UsageQuotas,
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


def _config_signature(
    cfg: PerModelConfig,
) -> tuple[bool, tuple[tuple[str, int, float], ...]]:
    """
    Stable family-level config fingerprint.

    ``model_family`` groups models onto the same backend, so every model that
    resolves to the same family must expose identical quota structure and
    unlimited-vs-limited behavior.
    """
    if cfg.is_unlimited:
        return True, ()

    snapshot = tuple(
        sorted(
            (metric, per_seconds, float(limit))
            for (metric, per_seconds), limit in _quotas_snapshot(cfg).items()
        )
    )
    return False, snapshot


def _describe_config_signature(
    signature: tuple[bool, tuple[tuple[str, int, float], ...]],
) -> str:
    is_unlimited, snapshot = signature
    if is_unlimited:
        return "unlimited"
    return ", ".join(
        f"{metric}/{per_seconds}s={limit}" for metric, per_seconds, limit in snapshot
    )


def _cfg_with_preserved_runtime_max_capacity(
    cfg: PerModelConfig,
    *,
    old_snapshot: dict[BucketId, float],
    runtime_overrides: dict[BucketId, float] | None,
) -> PerModelConfig:
    """
    Apply surviving runtime max-capacity overrides to a rebuild config.

    Metric-set rebuilds reconstruct buckets from ``quota.limit``. If a bucket
    still has a live ``set_max_capacity()`` override that should survive the
    rebuild, bake that value into the config used for the rebuild so waiters
    never observe the stale static limit between prepare/install and restore.
    """
    if not runtime_overrides:
        return cfg

    rebuilt_quotas: list[Quota] = []
    updated = False
    for quota in cfg.quotas:
        bucket_id = (quota.metric, int(quota.per_seconds))
        override = runtime_overrides.get(bucket_id)
        if override is None or old_snapshot.get(bucket_id) != float(quota.limit):
            rebuilt_quotas.append(quota)
            continue
        if float(quota.limit) == float(override):
            rebuilt_quotas.append(quota)
            continue
        rebuilt_quotas.append(quota.model_copy(update={"limit": float(override)}))
        updated = True

    if not updated:
        return cfg
    return cfg.model_copy(update={"quotas": UsageQuotas(rebuilt_quotas)})


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


def _warn_refund_refresh_failed(
    *,
    model_name: str,
    model_family: str,
    exc: Exception,
) -> None:
    warnings.warn(
        "Failed to refresh backend during refund for "
        f"model '{model_name}' in model family '{model_family}' "
        f"({type(exc).__name__}: {exc}). Proceeding with cached backend state "
        "to avoid leaking reserved capacity.",
        RuntimeWarning,
        stacklevel=2,
    )


class SyncRateLimiter:
    """
    Synchronous counterpart of ``RateLimiter`` — same architecture and contract.

    Unlike ``RateLimiter`` (which extends ``BaseRateLimiter`` ABC),
    ``SyncRateLimiter`` is a concrete class with no abstract base.
    Adding a ``BaseSyncRateLimiter`` ABC would be a public API change;
    the sync interface is instead documented by its method signatures
    and the ``SyncRateLimiterBackend`` protocol it delegates to.
    """

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
        self._validation_lock = threading.Lock()
        self._callbacks = callbacks
        self._config_getter = lambda model_name: resolve_config(cfg, model_name)
        # Dict mutations below happen both under self._lock and outside it
        # (e.g. _acquire_capacity, set_max_capacity).  Single-key dict
        # assignment is GIL-atomic in CPython, so these lock-free writes are
        # safe on standard interpreters.  Freethreaded Python (PEP 703 /
        # 3.13t) removes this guarantee and may require explicit locking.
        self._model_family_to_backend: dict[str, SyncRateLimiterBackend] = {}
        self._model_family_to_model_name: dict[str, str] = {}
        self._model_family_to_quotas: dict[str, dict[tuple[str, int], float]] = {}
        self._model_name_to_model_family: dict[str, str] = {}
        self._model_family_to_runtime_max_capacity: dict[
            str, dict[BucketId, float]
        ] = {}
        self._model_family_to_validated_signature: dict[
            str, tuple[bool, tuple[tuple[str, int, float], ...]]
        ] = {}
        # Tracks reservation IDs that have already been refunded to prevent
        # double-crediting. Grows monotonically; acceptable for typical
        # lifetimes but may need a bounded structure for long-lived processes
        # with millions of reservations.
        self._refunded_reservation_ids: set[str] = set()

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
        self._validated_model_family(model, limit_config)
        self._validate_shared_model_family_config(model, limit_config)
        if limit_config.is_unlimited:
            return self._unlimited_reservation(usage, model)
        return self._acquire_capacity(
            model, usage, limit_config, _block=_block, timeout=timeout
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
        self._validated_model_family(model, limit_config)
        self._validate_shared_model_family_config(model, limit_config)
        if limit_config.is_unlimited:
            usage = frozendict()
            if limit_config.usage_counter is not None:
                usage = resolve_usage_counter_result(
                    limit_config.usage_counter, **kwargs
                )
            usage = merge_extra_usage_unrestricted(usage, extra_usage)
            return self._unlimited_reservation(usage, model)
        if limit_config.usage_counter is None:
            raise ValueError("limit_config.usage_counter cannot be None")

        usage = merge_extra_usage(
            resolve_usage_counter_result(limit_config.usage_counter, **kwargs),
            extra_usage,
        )
        return self._acquire_capacity(model, usage, limit_config, timeout=timeout)

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
        # Only registered after a successful acquire — failed acquires don't
        # need refund routing, so skipping the cache on failure is intentional.
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
        reservation_metrics = set(reservation.usage)
        expected_metrics = {"tokens", "requests"}
        if reservation_metrics != expected_metrics:
            raise ValueError(
                f"refund_capacity_from_response requires metric names "
                f"{sorted(expected_metrics)} (as set by the create_openai_* "
                f"factories); got reservation with {sorted(reservation_metrics)}. "
                "Use refund_capacity directly for custom metric names."
            )
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
        """
        Dynamically change the max capacity for a specific bucket.

        The override survives subsequent acquires/refunds and config refreshes
        whose quota limits are unchanged. A metric-set change (the callable
        config drops the bucket and later re-adds it) drops the override:
        config-driven reconfiguration wins over runtime overrides, so a
        re-added metric starts from the callable config's static ``quota.limit``
        again. Re-call ``set_max_capacity`` after the re-add to reinstate.
        """
        metric = validate_metric(metric)
        per_seconds = validate_per_seconds(per_seconds)
        value = validate_max_capacity_value(value)
        limit_config = self._config_getter(model)
        self._validated_model_family(model, limit_config)
        self._validate_shared_model_family_config(model, limit_config)
        if limit_config.is_unlimited:
            raise ValueError("Cannot set max capacity: model has unlimited quotas")
        model_family = limit_config.get_model_family()
        if self._model_family_to_backend.get(model_family) is None:
            raise ValueError(
                f"No backend for model family '{model_family}'. "
                "Call acquire_capacity or record_usage first."
            )
        backend = self._get_backend(limit_config)
        backend.set_max_capacity(metric, per_seconds, value)
        self._model_family_to_model_name[model_family] = model
        self._remember_runtime_max_capacity(model_family, metric, per_seconds, value)

    def _refund_capacity(
        self,
        actual_usage: Usage,
        reservation: CapacityReservation,
    ) -> None:
        if reservation.reservation_id in self._refunded_reservation_ids:
            warnings.warn(
                f"Reservation {reservation.reservation_id} has already been "
                "refunded. Ignoring duplicate refund to prevent "
                "double-crediting capacity.",
                UserWarning,
                stacklevel=3,
            )
            return
        actual_usage = frozen_usage(actual_usage)
        if reservation.model_family not in self._model_family_to_backend:
            raise ValueError(
                f"Backend not found for model family {reservation.model_family}",
            )
        self._refresh_backend_for_reservation(reservation)
        backend = self._model_family_to_backend[reservation.model_family]
        # If `_refresh_backend_for_reservation` swallowed an exception (it
        # downgrades refresh failures to RuntimeWarning to keep refunds
        # unblocked), the snapshot below may still describe the pre-refresh
        # bucket set. A reservation made against a now-incompatible bucket
        # set will then surface as a "Refund bucket ids ... not found in
        # backend" ValueError from the backend's validation. That error
        # appears alongside the earlier warning — they together describe
        # the situation.
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
        backend.refund_capacity_for_buckets(
            reserved_usage,
            actual_usage,
            bucket_ids=refund_bucket_ids,
        )
        self._refunded_reservation_ids.add(reservation.reservation_id)

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

        try:
            limit_config = self._config_getter(model_name)
            if limit_config.is_unlimited:
                return
            if limit_config.get_model_family() != reservation.model_family:
                return
            self._get_backend(limit_config)
        except Exception as exc:  # noqa: BLE001
            # Design intent: a refund must never be blocked by a transient
            # failure of the user-supplied config_getter or backend. We fall
            # back to cached backend state and emit a warning. BaseException
            # (KeyboardInterrupt/SystemExit) is intentionally allowed to
            # propagate — those are shutdown signals, not refresh failures.
            _warn_refund_refresh_failed(
                model_name=model_name,
                model_family=reservation.model_family,
                exc=exc,
            )

    def _get_backend(self, cfg: PerModelConfig) -> SyncRateLimiterBackend:
        model_family = cfg.get_model_family()
        new_snapshot = _quotas_snapshot(cfg)

        # Fast path: unchanged configs can reuse the cached backend without
        # taking the limiter lock. The two dict reads are not atomic, but
        # dict.__getitem__ is GIL-atomic in CPython; worst case is a
        # spurious slow-path entry that re-checks under the lock.
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
        model_family = cfg.get_model_family()
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
            rebuild_cfg = _cfg_with_preserved_runtime_max_capacity(
                cfg,
                old_snapshot=old_snapshot,
                runtime_overrides=self._model_family_to_runtime_max_capacity.get(
                    model_family
                ),
            )
            backend = self._backend.build(rebuild_cfg, callbacks=self._callbacks)
            # Invalidate fast-path cache before mutation to close the
            # TOCTOU window where a concurrent reader could match the stale
            # snapshot against an already-mutated backend, tag its reservation
            # with old bucket_ids, and silently leak capacity on refund.
            self._model_family_to_quotas.pop(model_family, None)
            try:
                backend = old_backend.prepare_reconfigured_backend(backend, rebuild_cfg)
                self._restore_runtime_max_capacity(
                    model_family,
                    old_snapshot=old_snapshot,
                    new_snapshot=new_snapshot,
                    backend=backend,
                )
            except BaseException:
                self._model_family_to_quotas[model_family] = old_snapshot
                raise

            self._model_family_to_backend[model_family] = backend
            self._model_family_to_quotas[model_family] = new_snapshot
            return backend

        # Only limits changed — update in place via set_max_capacity.
        # This loop is not atomic across buckets: a concurrent reader may
        # observe some buckets at the old limit and others at the new limit.
        # Each apply_configured_max_capacity is individually atomic, so no
        # bucket is left in an inconsistent state.
        backend = self._model_family_to_backend[model_family]
        changed_bucket_ids: set[BucketId] = set()
        for bucket_id, new_limit in new_snapshot.items():
            if new_limit != old_snapshot[bucket_id]:
                metric, per_seconds = bucket_id
                backend.apply_configured_max_capacity(
                    metric,
                    per_seconds,
                    new_limit,
                )
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

    def _validate_shared_model_family_config(
        self,
        model: str,
        limit_config: PerModelConfig,
    ) -> None:
        # Detects conflicting quotas across models sharing a model_family.
        # Registration in the reverse-lookup map and the validation-signature
        # cache happen inside this method, under _validation_lock, so that
        # validate + register is atomic w.r.t. concurrent threads. A separate
        # lock is used (not self._lock) to avoid deadlock with _get_backend.
        #
        # Steady-state fast path: the cache check runs lock-free. Only the
        # first acquire of a new model (or a signature change) takes the lock.
        #
        # Complexity: O(1) steady-state via cache check. O(M) on first
        # acquire of a new model or after a config-signature change (re-calls
        # config_getter for every sibling). O(M^2) aggregate across first-
        # time acquires of M distinct models in one family.
        model_family = limit_config.get_model_family()
        current_signature = _config_signature(limit_config)

        cached = self._model_family_to_validated_signature.get(model_family)
        if (
            cached is not None
            and cached == current_signature
            and model in self._model_name_to_model_family
        ):
            return

        with self._validation_lock:
            cached = self._model_family_to_validated_signature.get(model_family)
            if (
                cached is not None
                and cached == current_signature
                and model in self._model_name_to_model_family
            ):
                return

            conflicts: list[tuple[str, str]] = []

            # config_getter is called unchecked here: if it raises, the error
            # propagates to the caller. This is intentional — the model was
            # previously registered successfully, so a config_getter failure
            # now indicates a programming error or broken config, not a
            # transient condition that should be papered over.
            for known_model, known_family in sorted(
                self._model_name_to_model_family.items()
            ):
                if known_family != model_family or known_model == model:
                    continue

                known_config = self._config_getter(known_model)
                try:
                    known_resolved_family = self._validated_model_family(
                        known_model, known_config
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"While validating shared-model_family config for "
                        f"'{model}', detected a routing change in sibling "
                        f"'{known_model}': {exc}"
                    ) from exc
                if known_resolved_family != model_family:
                    continue

                known_signature = _config_signature(known_config)
                if known_signature != current_signature:
                    conflicts.append(
                        (known_model, _describe_config_signature(known_signature))
                    )

            if conflicts:
                conflicts_desc = "; ".join(
                    f"{conflict_model} -> {conflict_signature}"
                    for conflict_model, conflict_signature in conflicts
                )
                raise ValueError(
                    f"Config for model_family '{model_family}' is inconsistent across "
                    f"models. Model '{model}' resolves to "
                    f"{_describe_config_signature(current_signature)}, but {conflicts_desc}. "
                    "Models sharing a model_family must return identical quotas and "
                    "unlimited behavior for a limiter instance. Use different "
                    "model_family values for different limits."
                )

            self._model_name_to_model_family[model] = model_family
            self._model_family_to_validated_signature[model_family] = current_signature

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
