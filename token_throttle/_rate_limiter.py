import asyncio
import collections
import logging
import math
import time
import uuid
import warnings

from frozendict import frozendict

from token_throttle._exceptions import CardinalityLimitExceededError
from token_throttle._interfaces._callable_utils import is_async_callable
from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    with_callback_timeout,
)
from token_throttle._interfaces._interfaces import (
    BaseRateLimiter,
    PerModelConfig,
    PerModelConfigGetter,
    RateLimiterBackend,
    RateLimiterBackendBuilderInterface,
    backend_uses_default_prepare_reconfigured_backend,
)
from token_throttle._interfaces._models import (
    MAX_ALIAS_LENGTH,
    MAX_METRIC_LENGTH,
    MAX_MODEL_FAMILY_LENGTH,
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

_logger = logging.getLogger("token_throttle")

DEFAULT_MAX_MODEL_FAMILIES = 10_000
DEFAULT_MAX_METRICS_PER_FAMILY = 100
DEFAULT_MAX_ALIASES = 10_000
DEFAULT_MAX_IN_FLIGHT_RESERVATIONS = 100_000
_REFUND_STATE_PENDING = "pending"
_REFUND_STATE_COMMITTED = "committed"
_REFUND_STATE_FAILED = "failed"
_REFUND_STATE_MISSING = object()


def _refund_state_is_committed(state: object) -> bool:
    return state is None or state == _REFUND_STATE_COMMITTED


def _validate_positive_int_cap(
    value: object,
    *,
    name: str,
    max_value: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an int (got {type(value).__name__})")
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0 (got {value!r})")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value} (got {value!r})")
    return value


def _raise_if_set_max_capacity_args_look_swapped(
    *,
    model: object,
    metric: object,
    config_getter: PerModelConfigGetter,
) -> None:
    if not isinstance(model, str) or not isinstance(metric, str):
        return
    try:
        cfg_for_metric_as_model = config_getter(metric)
    except Exception:  # noqa: BLE001 - best-effort footgun detection only.
        return
    if type(cfg_for_metric_as_model) is not PerModelConfig:
        return
    if cfg_for_metric_as_model.is_unlimited:
        return

    metrics_for_second_arg = {quota.metric for quota in cfg_for_metric_as_model.quotas}
    if model in metrics_for_second_arg and metric not in metrics_for_second_arg:
        raise TypeError(
            "set_max_capacity expects (model, metric, per_seconds, value); "
            "the first two arguments look swapped."
        )


def _missing_model_parameter_error(kwargs: collections.abc.Mapping[str, object]) -> str:
    aliases = ("model_name", "modelName", "model_id", "modelId")
    for alias in aliases:
        if alias in kwargs:
            return f"'model' parameter is required; did you mean 'model' instead of '{alias}'?"
    return "'model' parameter is required"


def _quotas_snapshot(cfg: PerModelConfig) -> dict[tuple[str, int], float]:
    """Snapshot of quotas for change detection: {(metric, per_seconds): limit}."""
    return {(q.metric, q.per_seconds): q.limit for q in cfg.quotas}


def _reservation_bucket_ids(cfg: PerModelConfig) -> frozenset[BucketId]:
    """Bucket ids captured at reservation time for later scoped refunds."""
    return frozenset((q.metric, int(q.per_seconds)) for q in cfg.quotas)


def _zero_actual_usage(reservation: CapacityReservation) -> dict[str, float]:
    return dict.fromkeys(reservation.usage, 0.0)


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
        rebuilt_quotas.append(
            Quota.model_validate(
                {**quota.model_dump(), "limit": float(override)}, strict=True
            )
        )
        updated = True

    if not updated:
        return cfg
    return PerModelConfig.model_validate(
        {**cfg.model_dump(), "quotas": UsageQuotas(rebuilt_quotas)}, strict=True
    )


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
        warnings.warn(
            "Refund dropped: none of the reservation's bucket IDs exist in "
            "the current backend (bucket set was reconfigured after the "
            "reservation was created).",
            RuntimeWarning,
            stacklevel=3,
        )
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


def _raise_limiter_instance_mismatch() -> None:
    raise ValueError(
        "Reservation was issued by a different limiter; refund cross-limiter "
        "is not supported. See L13 N01."
    )


def _raise_legacy_reservation_rejected() -> None:
    raise ValueError(
        "legacy v1.4.x reservations no longer supported in v2.0.0; "
        "drain v1.4.x before upgrade"
    )


def _is_redis_exception(exc: Exception) -> bool:
    return type(exc).__module__.startswith("redis.")


def _raise_backend_external_error(exc: Exception) -> None:
    if _is_redis_exception(exc):
        raise RuntimeError(
            "Rate limiter backend operation failed with a Redis error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    raise exc


def _resolve_usage_counter_result_for_model(
    usage_counter,
    *,
    model_name: str,
    warn_if_sync_counter_blocks_event_loop: bool = False,
    **kwargs,
) -> FrozenUsage:
    try:
        return resolve_usage_counter_result(
            usage_counter,
            warn_if_sync_counter_blocks_event_loop=warn_if_sync_counter_blocks_event_loop,
            **kwargs,
        )
    except KeyError as exc:
        raise ValueError(
            "Rate limiter usage_counter failed with KeyError while counting "
            f"request usage for model {model_name!r}. If this is OpenAIUsageCounter, "
            "token-throttle could not determine the tokenizer for that model; "
            "pass an explicit get_encoding_func."
        ) from exc
    except ValueError as exc:
        if isinstance(exc.__cause__, KeyError):
            raise ValueError(  # noqa: TRY004 - preserving public ValueError contract
                "Rate limiter usage_counter failed with KeyError while counting "
                f"request usage for model {model_name!r}. If this is "
                "OpenAIUsageCounter, token-throttle could not determine the "
                "tokenizer for that model; pass an explicit get_encoding_func."
            ) from exc.__cause__
        raise


async def _resolve_usage_counter_result_for_model_async(
    usage_counter,
    *,
    model_name: str,
    warn_if_sync_counter_blocks_event_loop: bool = False,
    **kwargs,
) -> FrozenUsage:
    async_counter = getattr(usage_counter, "count_request_async", None)
    if callable(async_counter):
        try:
            result = await async_counter(**kwargs)
            return frozen_usage(result)
        except KeyError as exc:
            raise ValueError(
                "Rate limiter usage_counter failed with KeyError while counting "
                f"request usage for model {model_name!r}. If this is "
                "OpenAIUsageCounter, token-throttle could not determine the "
                "tokenizer for that model; pass an explicit get_encoding_func."
            ) from exc
        except ValueError as exc:
            if isinstance(exc.__cause__, KeyError):
                raise ValueError(  # noqa: TRY004 - preserving public ValueError contract
                    "Rate limiter usage_counter failed with KeyError while counting "
                    f"request usage for model {model_name!r}. If this is "
                    "OpenAIUsageCounter, token-throttle could not determine the "
                    "tokenizer for that model; pass an explicit get_encoding_func."
                ) from exc.__cause__
            raise

    return _resolve_usage_counter_result_for_model(
        usage_counter,
        model_name=model_name,
        warn_if_sync_counter_blocks_event_loop=warn_if_sync_counter_blocks_event_loop,
        **kwargs,
    )


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

    ``close_drain_timeout_seconds`` bounds how long ``aclose()`` waits for
    acquire calls that already registered a pending reservation to finish their
    limiter-side state update before the limiter is marked closed.
    """

    def __init__(  # noqa: PLR0913
        self,
        cfg: PerModelConfig | PerModelConfigGetter,
        /,
        backend: RateLimiterBackendBuilderInterface,
        *,
        callbacks: RateLimiterCallbacks | None = None,
        callback_timeout: float | None = 30.0,
        close_drain_timeout_seconds: float = 5.0,
        max_model_families: int = DEFAULT_MAX_MODEL_FAMILIES,
        max_metrics_per_family: int = DEFAULT_MAX_METRICS_PER_FAMILY,
        max_aliases: int = DEFAULT_MAX_ALIASES,
        max_in_flight_reservations: int = DEFAULT_MAX_IN_FLIGHT_RESERVATIONS,
        max_model_family_length: int = MAX_MODEL_FAMILY_LENGTH,
        max_metric_length: int = MAX_METRIC_LENGTH,
        max_alias_length: int = MAX_ALIAS_LENGTH,
    ):
        if callable(cfg) and is_async_callable(cfg):
            raise ValueError("cfg must be a synchronous PerModelConfig getter")
        if callbacks is not None and type(callbacks) is not RateLimiterCallbacks:
            raise TypeError(
                "callbacks must be a RateLimiterCallbacks instance or None "
                f"(got {type(callbacks).__name__})"
            )
        self._backend = backend
        self._lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        callback_timeout = validate_timeout(callback_timeout)
        self._callbacks = callbacks
        self._backend_callbacks = with_callback_timeout(callbacks, callback_timeout)
        self._callback_timeout = callback_timeout
        self._max_model_families = _validate_positive_int_cap(
            max_model_families,
            name="max_model_families",
        )
        self._max_metrics_per_family = _validate_positive_int_cap(
            max_metrics_per_family,
            name="max_metrics_per_family",
        )
        self._max_aliases = _validate_positive_int_cap(
            max_aliases,
            name="max_aliases",
        )
        self._max_in_flight_reservations = _validate_positive_int_cap(
            max_in_flight_reservations,
            name="max_in_flight_reservations",
        )
        self._max_model_family_length = _validate_positive_int_cap(
            max_model_family_length,
            name="max_model_family_length",
            max_value=MAX_MODEL_FAMILY_LENGTH,
        )
        self._max_metric_length = _validate_positive_int_cap(
            max_metric_length,
            name="max_metric_length",
            max_value=MAX_METRIC_LENGTH,
        )
        self._max_alias_length = _validate_positive_int_cap(
            max_alias_length,
            name="max_alias_length",
            max_value=MAX_ALIAS_LENGTH,
        )
        self._config_getter = lambda model_name: resolve_config(
            cfg,
            model_name,
            max_model_family_length=self._max_model_family_length,
            max_alias_length=self._max_alias_length,
        )
        self._model_family_to_backend: dict[str, RateLimiterBackend] = {}
        self._model_family_to_model_name: dict[str, str] = {}
        self._model_family_to_quotas: dict[str, dict[tuple[str, int], float]] = {}
        self._model_name_to_model_family: dict[str, str] = {}
        self._model_family_to_runtime_max_capacity: dict[
            str, dict[BucketId, float]
        ] = {}
        self._model_family_to_validated_signature: dict[
            str, tuple[bool, tuple[tuple[str, int, float], ...]]
        ] = {}
        self._model_name_to_validated_signature: dict[
            str, tuple[bool, tuple[tuple[str, int, float], ...]]
        ] = {}
        self._model_family_signature_counts: dict[
            str, dict[tuple[bool, tuple[tuple[str, int, float], ...]], int]
        ] = {}
        self._model_family_alias_counts: dict[str, int] = {}
        self._refunded_reservation_ids: collections.OrderedDict[str, str | None] = (
            collections.OrderedDict()
        )
        self._refunded_ids_cap = 131_072
        self._refund_state_lock = asyncio.Lock()
        self._refund_locks: dict[str, asyncio.Lock] = {}
        self._refund_lock_refcounts: dict[str, int] = {}
        self._refund_in_progress: set[str] = set()
        self._acquire_guard = asyncio.Lock()
        self._pending_acquire_reservations: set[str] = set()
        self._pending_drained = asyncio.Event()
        self._pending_drained.set()
        self._close_drain_timeout_seconds = validate_timeout(
            close_drain_timeout_seconds
        )
        self._limiter_instance_id = uuid.uuid4().hex
        self._in_flight_reservation_ids: set[str] = set()
        self._closing = False
        self._in_flight_reservation_family: dict[str, str] = {}
        self._model_family_last_touched: dict[str, float] = {}
        self._closed = False

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("RateLimiter is closed")

    def _raise_if_closed_or_closing(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("RateLimiter is closed")

    def _touch_model_family(self, model_family: str) -> None:
        self._model_family_last_touched[model_family] = time.monotonic()

    def _enforce_resolved_config_caps(
        self,
        *,
        model: str,
        model_family: str,
        limit_config: PerModelConfig,
    ) -> None:
        if len(model) > self._max_alias_length:
            raise CardinalityLimitExceededError(
                "max_alias_length exceeded: "
                f"model alias is {len(model)} characters; "
                f"limit is {self._max_alias_length}"
            )
        if len(model_family) > self._max_model_family_length:
            raise CardinalityLimitExceededError(
                "max_model_family_length exceeded: "
                f"model_family is {len(model_family)} characters; "
                f"limit is {self._max_model_family_length}"
            )
        metrics = {quota.metric for quota in limit_config.quotas}
        oversized_metrics = [
            metric for metric in metrics if len(metric) > self._max_metric_length
        ]
        if oversized_metrics:
            metric = oversized_metrics[0]
            raise CardinalityLimitExceededError(
                "max_metric_length exceeded: "
                f"metric is {len(metric)} characters; "
                f"limit is {self._max_metric_length}"
            )
        if len(metrics) > self._max_metrics_per_family:
            raise CardinalityLimitExceededError(
                "max_metrics_per_family exceeded: "
                f"model_family {model_family!r} has {len(metrics)} metrics; "
                f"limit is {self._max_metrics_per_family}"
            )

    def _enforce_new_model_family_cap(self, model_family: str) -> None:
        if (
            model_family not in self._model_family_to_validated_signature
            and len(self._model_family_to_validated_signature)
            >= self._max_model_families
        ):
            raise CardinalityLimitExceededError(
                f"max_model_families exceeded: limit is {self._max_model_families}"
            )

    def _enforce_new_alias_cap(self, model: str) -> None:
        if (
            model not in self._model_name_to_model_family
            and len(self._model_name_to_model_family) >= self._max_aliases
        ):
            raise CardinalityLimitExceededError(
                f"max_aliases exceeded: limit is {self._max_aliases}"
            )

    def _remember_in_flight_reservation(
        self,
        reservation: CapacityReservation,
    ) -> None:
        if (
            reservation.reservation_id not in self._in_flight_reservation_ids
            and reservation.reservation_id not in self._pending_acquire_reservations
            and len(self._in_flight_reservation_ids)
            + len(self._pending_acquire_reservations)
            >= self._max_in_flight_reservations
        ):
            raise CardinalityLimitExceededError(
                "max_in_flight_reservations exceeded: "
                f"limit is {self._max_in_flight_reservations}"
            )
        self._in_flight_reservation_ids.add(reservation.reservation_id)
        self._in_flight_reservation_family[reservation.reservation_id] = (
            reservation.model_family
        )

    def _forget_in_flight_reservation(self, reservation_id: str) -> None:
        self._in_flight_reservation_ids.discard(reservation_id)
        self._pending_acquire_reservations.discard(reservation_id)
        self._in_flight_reservation_family.pop(reservation_id, None)

    def _verify_reservation_limiter_instance(
        self,
        reservation: CapacityReservation,
    ) -> None:
        if reservation.limiter_instance_id is None:
            _logger.warning(
                "Reservation %s has no limiter_instance_id; legacy v1.4.x "
                "reservations are rejected in v2.0.0.",
                reservation.reservation_id,
            )
            _raise_legacy_reservation_rejected()
        if reservation.limiter_instance_id != self._limiter_instance_id:
            _logger.warning(
                "Reservation %s was issued by limiter %s, not this limiter %s",
                reservation.reservation_id,
                reservation.limiter_instance_id,
                self._limiter_instance_id,
            )
            _raise_limiter_instance_mismatch()

    async def aclose(self) -> None:
        """
        Close the limiter and report outstanding reservations.

        Reservations are bound to this limiter instance. After close, new
        acquire/record/refund operations raise ``RuntimeError``; reservations
        that remain unrefunded may no longer be refundable.

        Close is terminal once started: if draining pending acquires or closing
        backend resources fails, the limiter is still marked closed so future
        operations fail cleanly instead of observing a permanent closing state.
        """
        async with self._acquire_guard:
            if self._closed:
                return
            self._closing = True
            if not self._pending_acquire_reservations:
                self._pending_drained.set()

        try:
            await asyncio.wait_for(
                self._pending_drained.wait(),
                timeout=self._close_drain_timeout_seconds,
            )
        except TimeoutError as exc:
            async with self._acquire_guard:
                self._closed = True
                self._closing = False
            raise TimeoutError(
                "Timed out waiting for pending acquire reservations to drain"
            ) from exc

        try:
            async with self._lifecycle_lock:
                await self._backend.aclose()
        except BaseException:
            async with self._acquire_guard:
                self._closed = True
                self._closing = False
            raise

        async with self._acquire_guard:
            self._closed = True
            self._closing = False
        async with self._refund_state_lock:
            refund_locks = list(self._refund_locks.values())
        acquired_locks = []
        try:
            for lock in refund_locks:
                await lock.acquire()
                acquired_locks.append(lock)
            async with self._refund_state_lock:
                in_flight_count = len(self._in_flight_reservation_ids)
        finally:
            for lock in reversed(acquired_locks):
                lock.release()
        self._clear_retained_state_after_close()
        _logger.warning(
            "limiter closed; %d reservations still in flight may not be refundable.",
            in_flight_count,
        )

    def close(self) -> None:
        """
        Synchronous close helper for async ``RateLimiter`` instances.

        Use ``await aclose()`` when calling from an active event loop.
        Outside an event loop, this runs ``aclose()`` to drain pending acquires
        and release owned backend resources.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
            return
        raise RuntimeError(
            "RateLimiter.close() cannot run inside an event loop; use await aclose()"
        )

    def _clear_retained_state_after_close(self) -> None:
        self._model_family_to_backend.clear()
        self._model_family_to_model_name.clear()
        self._model_family_to_quotas.clear()
        self._model_name_to_model_family.clear()
        self._model_family_to_runtime_max_capacity.clear()
        self._model_family_to_validated_signature.clear()
        self._model_name_to_validated_signature.clear()
        self._model_family_signature_counts.clear()
        self._model_family_alias_counts.clear()
        self._refunded_reservation_ids.clear()
        self._refund_locks.clear()
        self._refund_lock_refcounts.clear()
        self._refund_in_progress.clear()
        self._pending_acquire_reservations.clear()
        self._in_flight_reservation_ids.clear()
        self._in_flight_reservation_family.clear()
        self._model_family_last_touched.clear()

    def clear_unused_model_families(self, unused_for_seconds: int) -> int:
        """
        Evict idle in-process model-family state.

        This is the operator-driven cleanup path for long-lived limiters that
        accept many dynamic model aliases. Families with in-flight reservations
        are skipped so refunds can still route to their original backend.

        Redis-backed limiter state is not deleted here; Redis bucket keys have
        their own inactivity TTL and expire independently.
        """
        if type(unused_for_seconds) is not int:
            raise ValueError(
                "unused_for_seconds must be an int "
                f"(got {type(unused_for_seconds).__name__})"
            )
        if unused_for_seconds < 0:
            raise ValueError(
                f"unused_for_seconds must be non-negative (got {unused_for_seconds!r})"
            )

        cutoff = time.monotonic() - unused_for_seconds
        in_flight_families = set(self._in_flight_reservation_family.values())
        expired_families = [
            model_family
            for model_family, last_touched in self._model_family_last_touched.items()
            if model_family not in in_flight_families and last_touched <= cutoff
        ]

        for model_family in expired_families:
            self._model_family_to_backend.pop(model_family, None)
            self._model_family_to_model_name.pop(model_family, None)
            self._model_family_to_quotas.pop(model_family, None)
            self._model_family_to_runtime_max_capacity.pop(model_family, None)
            self._model_family_to_validated_signature.pop(model_family, None)
            self._model_family_last_touched.pop(model_family, None)
            for model_name, known_family in list(
                self._model_name_to_model_family.items()
            ):
                if known_family == model_family:
                    self._model_name_to_model_family.pop(model_name, None)
                    self._model_name_to_validated_signature.pop(model_name, None)
            self._model_family_signature_counts.pop(model_family, None)
            self._model_family_alias_counts.pop(model_family, None)

        return len(expired_families)

    async def acquire_capacity(
        self, usage: Usage, model: str, *, timeout: float | None = None
    ) -> CapacityReservation:
        """
        Wait for capacity, then reserve it.

        ``usage`` is the first positional argument; use keyword arguments when
        readability matters: ``acquire_capacity(usage={...}, model="gpt-4o")``.
        The returned reservation should be refunded by this limiter after the
        external request completes.

        ``timeout`` bounds only the time spent waiting for capacity. It does not
        bound backend operation latency or callback dispatch time; callbacks are
        bounded separately by ``callback_timeout`` configured on the limiter.
        """
        self._raise_if_closed()
        timeout = validate_timeout(timeout)
        return await self._acquire_or_record(usage, model, _block=True, timeout=timeout)

    async def record_usage(self, usage: Usage, model: str) -> CapacityReservation:
        """
        Consume capacity immediately without blocking.

        Use this for post-hoc reporting of usage that has already happened
        externally. Capacity may go negative by design (speedometer pattern);
        the bucket recovers naturally as it refills.
        """
        self._raise_if_closed()
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
        self._validated_model_family(model, limit_config)
        if limit_config.is_unlimited:
            return await self._unlimited_reservation(model, limit_config)
        self._validate_shared_model_family_config(
            model,
            limit_config,
            register=False,
        )
        return await self._acquire_capacity(
            model, usage, limit_config, _block=_block, timeout=timeout
        )

    async def acquire_capacity_for_request(
        self,
        *,
        extra_usage: collections.abc.Mapping[str, int | float] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> CapacityReservation:
        """
        Count request usage, wait for capacity, then reserve it.

        The limiter resolves ``model`` from ``kwargs`` and calls the configured
        synchronous ``usage_counter`` with the request kwargs. For limited
        configs, ``usage_counter`` must be present and must return a usage
        mapping whose keys match the configured quotas after ``extra_usage`` is
        merged. For unlimited configs, the counter may still run for telemetry,
        but the returned reservation carries empty usage and does not consume
        backend capacity.

        ``extra_usage`` is optional; ``None`` and ``{}`` are equivalent. When
        supplied, it must be a mapping of metric names to explicit ``int`` or
        ``float`` values. Values are validated before the counter runs, must be
        finite and non-negative, and are added to the counter output rather than
        replacing it. In limited configs, each ``extra_usage`` key must already
        appear in the counter output; emit zero-valued metrics from the counter
        when a request will top them up via ``extra_usage``. Unlimited configs
        accept additional metric keys, though usage is discarded in the
        unlimited reservation.

        ``timeout`` bounds only the capacity-wait portion. It does not bound
        usage counting, backend operation latency, or callback dispatch time;
        callbacks are bounded separately by ``callback_timeout`` configured on
        the limiter.

        Returns a ``CapacityReservation`` for the counted request. Raises
        ``ValueError`` for invalid timeout, missing or invalid ``model``,
        missing limited-config ``usage_counter``, invalid counter output,
        invalid ``extra_usage``, or backend usage that does not match the
        configured quotas.
        """
        self._raise_if_closed()
        timeout = validate_timeout(timeout)
        extra_usage = validate_extra_usage(extra_usage)
        if "model" not in kwargs:
            raise ValueError(_missing_model_parameter_error(kwargs))
        model = kwargs["model"]

        limit_config = self._config_getter(model)
        self._validated_model_family(model, limit_config)
        if limit_config.is_unlimited:
            self._validate_shared_model_family_config(
                model,
                limit_config,
                register=False,
            )
            # Counter still runs for telemetry consistency (L05 I03);
            # extra_usage shape is still validated; both results are
            # discarded because the unlimited reservation always
            # carries empty usage by construction.
            usage = frozendict()
            if limit_config.usage_counter is not None:
                usage = await _resolve_usage_counter_result_for_model_async(
                    limit_config.usage_counter,
                    model_name=model,
                    warn_if_sync_counter_blocks_event_loop=True,
                    **kwargs,
                )
            merge_extra_usage_unrestricted(usage, extra_usage)
            return await self._unlimited_reservation(model, limit_config)
        if limit_config.usage_counter is None:
            raise ValueError("limit_config.usage_counter cannot be None")
        self._validate_shared_model_family_config(
            model,
            limit_config,
            register=False,
        )

        usage = merge_extra_usage(
            await _resolve_usage_counter_result_for_model_async(
                limit_config.usage_counter,
                model_name=model,
                warn_if_sync_counter_blocks_event_loop=True,
                **kwargs,
            ),
            extra_usage,
        )
        return await self._acquire_capacity(
            model,
            usage,
            limit_config,
            timeout=timeout,
        )

    async def _acquire_capacity(
        self,
        model: str,
        usage: FrozenUsage,
        limit_config: PerModelConfig,
        *,
        _block: bool = True,
        timeout: float | None = None,
    ) -> CapacityReservation:
        validate_acquire_usage(usage, limit_config.quotas)

        model_family = limit_config.get_model_family()
        reservation = CapacityReservation(
            usage=usage,
            model_family=model_family,
            bucket_ids=_reservation_bucket_ids(limit_config),
            model=model,
            limiter_instance_id=self._limiter_instance_id,
        )

        await self._begin_pending_acquire(reservation)
        backend_task: asyncio.Task[None] | None = None
        try:
            # Reserve an in-flight slot before registering family/alias rows.
            # If max_in_flight rejects, validation metadata is never inserted;
            # while the slot is pending, cleanup treats the family as active.
            self._validate_shared_model_family_config(model, limit_config)
            backend = await self._get_backend(limit_config)
            backend_task = asyncio.create_task(
                backend.await_for_capacity(usage, timeout=timeout)
                if _block
                else backend.consume_capacity(usage)
            )
            await backend_task
        except asyncio.CancelledError:
            consumed = (
                backend_task is not None
                and await self._backend_task_succeeded_after_cancel(backend_task)
            )
            if consumed:
                await self._complete_acquire_state_update(
                    self._finalize_pending_acquire(reservation, model)
                )
                if _block:
                    await self._refund_undelivered_acquire(reservation)
            else:
                await self._complete_acquire_state_update(
                    self._rollback_pending_acquire(reservation.reservation_id)
                )
            raise
        except Exception as exc:
            interrupted = await self._complete_acquire_state_update(
                self._rollback_pending_acquire(reservation.reservation_id)
            )
            if interrupted:
                raise asyncio.CancelledError from exc
            _raise_backend_external_error(exc)
        interrupted = await self._complete_acquire_state_update(
            self._finalize_pending_acquire(reservation, model)
        )
        if interrupted:
            if _block:
                await self._refund_undelivered_acquire(reservation)
            raise asyncio.CancelledError
        return reservation

    async def _begin_pending_acquire(self, reservation: CapacityReservation) -> None:
        async with self._acquire_guard:
            self._raise_if_closed_or_closing()
            if not self._pending_acquire_reservations:
                self._pending_drained.clear()
            if (
                reservation.reservation_id not in self._pending_acquire_reservations
                and reservation.reservation_id not in self._in_flight_reservation_ids
                and len(self._in_flight_reservation_ids)
                + len(self._pending_acquire_reservations)
                >= self._max_in_flight_reservations
            ):
                raise CardinalityLimitExceededError(
                    "max_in_flight_reservations exceeded: "
                    f"limit is {self._max_in_flight_reservations}"
                )
            self._pending_acquire_reservations.add(reservation.reservation_id)
            self._in_flight_reservation_family[reservation.reservation_id] = (
                reservation.model_family
            )

    async def _finalize_pending_acquire(
        self,
        reservation: CapacityReservation,
        model: str,
    ) -> None:
        async with self._acquire_guard:
            self._pending_acquire_reservations.discard(reservation.reservation_id)
            self._model_family_to_model_name[reservation.model_family] = model
            self._in_flight_reservation_ids.add(reservation.reservation_id)
            self._in_flight_reservation_family[reservation.reservation_id] = (
                reservation.model_family
            )
            self._touch_model_family(reservation.model_family)
            if not self._pending_acquire_reservations:
                self._pending_drained.set()

    async def _rollback_pending_acquire(self, reservation_id: str) -> None:
        async with self._acquire_guard:
            self._pending_acquire_reservations.discard(reservation_id)
            self._in_flight_reservation_family.pop(reservation_id, None)
            if not self._pending_acquire_reservations:
                self._pending_drained.set()

    async def _refund_undelivered_acquire(
        self,
        reservation: CapacityReservation,
    ) -> None:
        refund_task = asyncio.create_task(
            self.refund_capacity(_zero_actual_usage(reservation), reservation)
        )
        while True:
            try:
                await asyncio.shield(refund_task)
                break
            except asyncio.CancelledError:
                if refund_task.done():
                    break
        refund_task.result()

    async def _complete_acquire_state_update(self, awaitable) -> bool:
        task = asyncio.create_task(awaitable)
        interrupted = False
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                interrupted = True
                if task.done():
                    break
        task.result()
        return interrupted

    async def _backend_task_succeeded_after_cancel(
        self,
        task: asyncio.Task[None],
    ) -> bool:
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.done():
                    break
            except Exception:  # noqa: BLE001
                break
        return task.done() and not task.cancelled() and task.exception() is None

    async def refund_capacity(
        self,
        actual_usage: Usage,
        reservation: CapacityReservation,
    ) -> None:
        """
        Refund unused capacity from a prior reservation.

        ``actual_usage`` must contain the same metric keys as the reservation.
        Unlimited reservations are accepted and ignored because they never
        consumed backend capacity. Reservations are scoped to the bucket ids
        captured at acquire time, so config rebuilds refund only surviving
        buckets.
        """
        self._raise_if_closed()
        if isinstance(actual_usage, CapacityReservation):
            raise TypeError(
                "refund_capacity expects (actual_usage, reservation); "
                "did you mean refund_capacity_from_response?"
            )
        is_unlimited = is_unlimited_reservation(reservation)
        self._verify_reservation_limiter_instance(reservation)
        if is_unlimited:
            self._forget_in_flight_reservation(reservation.reservation_id)
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
        self._raise_if_closed()
        is_unlimited = is_unlimited_reservation(reservation)
        self._verify_reservation_limiter_instance(reservation)
        if is_unlimited:
            self._forget_in_flight_reservation(reservation.reservation_id)
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
        """
        Dynamically change the max capacity for a specific bucket.

        This is a runtime override. To change the static configured quota,
        update the callable config; the limiter will rebuild on the next
        acquire/refund path.

        The override survives subsequent acquires/refunds and config refreshes
        whose quota limits are unchanged. A metric-set change (the callable
        config drops the bucket and later re-adds it) drops the override:
        config-driven reconfiguration wins over runtime overrides, so a
        re-added metric starts from the callable config's static ``quota.limit``
        again. Re-call ``set_max_capacity`` after the re-add to reinstate.
        Cross-process Redis visibility is bounded by the backend's short
        max-capacity cache window.
        """
        self._raise_if_closed_or_closing()
        metric = validate_metric(metric, max_length=self._max_metric_length)
        per_seconds = validate_per_seconds(per_seconds)
        value = validate_max_capacity_value(value)
        _raise_if_set_max_capacity_args_look_swapped(
            model=model,
            metric=metric,
            config_getter=self._config_getter,
        )
        limit_config = self._config_getter(model)
        self._validated_model_family(model, limit_config)
        self._validate_shared_model_family_config(model, limit_config)
        if limit_config.is_unlimited:
            # Audited 2026-05 (R4 L05:I11): set_max_capacity rejects
            # unlimited configs before backend lookup, preserving the
            # semantic "unlimited model" error instead of "no backend".
            raise ValueError("Cannot set max capacity: model has unlimited quotas")
        model_family = limit_config.get_model_family()
        async with self._lifecycle_lock:
            self._raise_if_closed_or_closing()
            async with self._lock:
                if self._model_family_to_backend.get(model_family) is None:
                    raise ValueError(
                        f"No backend for model family '{model_family}'. "
                        "Call acquire_capacity or record_usage first."
                    )
                try:
                    backend = await self._sync_backend_quotas(limit_config)
                except Exception as exc:  # noqa: BLE001 - boundary wrapper preserves cause
                    _raise_backend_external_error(exc)
                await self._set_max_capacity_transactional(
                    backend,
                    model_family=model_family,
                    model=model,
                    metric=metric,
                    per_seconds=per_seconds,
                    value=value,
                )

    def _commit_runtime_max_capacity(
        self,
        model_family: str,
        model: str,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        self._model_family_to_model_name[model_family] = model
        self._remember_runtime_max_capacity(
            model_family,
            metric,
            per_seconds,
            value,
        )

    async def _backend_runtime_max_capacity_matches(
        self,
        backend: RateLimiterBackend,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> bool:
        reader = getattr(backend, "_runtime_max_capacity_for_reconciliation", None)
        if not callable(reader):
            return False
        try:
            actual_value = await reader(metric, per_seconds)
        except Exception:  # noqa: BLE001 - preserves the original backend error.
            return False
        if actual_value is None:
            return False
        try:
            return math.isclose(float(actual_value), value, rel_tol=1e-12)
        except (TypeError, ValueError):
            return False

    async def _reconcile_runtime_max_capacity_after_failed_set(  # noqa: PLR0913
        self,
        backend: RateLimiterBackend,
        *,
        model_family: str,
        model: str,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        if await self._backend_runtime_max_capacity_matches(
            backend,
            metric,
            per_seconds,
            value,
        ):
            self._commit_runtime_max_capacity(
                model_family,
                model,
                metric,
                per_seconds,
                value,
            )

    async def _wait_for_set_max_capacity_task_while_cancelled(
        self,
        task: asyncio.Task[None],
    ) -> None:
        while True:
            try:
                await asyncio.shield(task)
                return
            except asyncio.CancelledError:
                if task.done():
                    return
            except Exception:  # noqa: BLE001 - caller inspects task.exception().
                return

    async def _set_max_capacity_transactional(  # noqa: PLR0913
        self,
        backend: RateLimiterBackend,
        *,
        model_family: str,
        model: str,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        write_task = asyncio.create_task(
            backend.set_max_capacity(metric, per_seconds, value)
        )
        try:
            await asyncio.shield(write_task)
        except asyncio.CancelledError:
            await self._wait_for_set_max_capacity_task_while_cancelled(write_task)
            if write_task.done() and not write_task.cancelled():
                exc = write_task.exception()
                if exc is None:
                    self._commit_runtime_max_capacity(
                        model_family,
                        model,
                        metric,
                        per_seconds,
                        value,
                    )
                elif isinstance(exc, Exception):
                    await self._reconcile_runtime_max_capacity_after_failed_set(
                        backend,
                        model_family=model_family,
                        model=model,
                        metric=metric,
                        per_seconds=per_seconds,
                        value=value,
                    )
            raise
        except Exception as exc:  # noqa: BLE001 - boundary wrapper preserves cause
            await self._reconcile_runtime_max_capacity_after_failed_set(
                backend,
                model_family=model_family,
                model=model,
                metric=metric,
                per_seconds=per_seconds,
                value=value,
            )
            _raise_backend_external_error(exc)
        self._commit_runtime_max_capacity(
            model_family,
            model,
            metric,
            per_seconds,
            value,
        )

    async def _refund_capacity(  # noqa: PLR0915
        self,
        actual_usage: Usage,
        reservation: CapacityReservation,
    ) -> None:
        rid = reservation.reservation_id
        refund_lock = await self._acquire_reservation_refund_lock(rid)
        refund_started = False
        refund_backend: RateLimiterBackend | None = None
        refund_bucket_ids_for_probe: frozenset[BucketId] | None = None
        pre_refund_signature: tuple[tuple[BucketId, object, object], ...] | None = None
        refund_backend_call_started = False
        try:
            async with self._refund_state_lock:
                self._raise_if_closed()
                refund_state = self._refunded_reservation_ids.get(
                    rid,
                    _REFUND_STATE_MISSING,
                )
                if refund_state is not _REFUND_STATE_MISSING and (
                    _refund_state_is_committed(refund_state)
                ):
                    warnings.warn(
                        f"Reservation {rid} has already been "
                        "refunded. Ignoring duplicate refund to prevent "
                        "double-crediting capacity.",
                        UserWarning,
                        stacklevel=3,
                    )
                    return
                if rid in self._refund_in_progress:
                    warnings.warn(
                        f"Reservation {rid} is already being refunded. "
                        "Ignoring duplicate refund to prevent double-crediting capacity.",
                        UserWarning,
                        stacklevel=3,
                    )
                    return
                self._remember_refund_state(rid, _REFUND_STATE_PENDING)
                self._refund_in_progress.add(rid)
                refund_started = True

            try:
                actual_usage = frozen_usage(actual_usage)
                await self._refresh_backend_for_reservation(reservation)
                backend = self._model_family_to_backend.get(reservation.model_family)
                if backend is None:
                    raise ValueError(  # noqa: TRY301
                        f"Backend not found for model family {reservation.model_family}",
                    )
                async with self._refund_state_lock:
                    reservation_in_flight = rid in self._in_flight_reservation_ids
                if (
                    not reservation_in_flight
                    and not backend.supports_durable_refund_dedup()
                ):
                    raise ValueError(  # noqa: TRY301
                        "Reservation is not in flight for this limiter; "
                        "cold-restart refunds require a backend with durable "
                        "refund dedup."
                    )
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
                try:
                    refund_backend = backend
                    refund_bucket_ids_for_probe = refund_bucket_ids
                    pre_refund_signature = self._refund_backend_state_signature(
                        backend,
                        refund_bucket_ids,
                    )
                    refund_backend_call_started = True
                    await backend.refund_capacity_for_buckets(
                        reserved_usage,
                        actual_usage,
                        bucket_ids=refund_bucket_ids,
                        reservation_id=rid,
                    )
                except Exception as exc:  # noqa: BLE001 - boundary wrapper preserves cause
                    if self._refund_backend_state_changed(
                        refund_backend,
                        refund_bucket_ids_for_probe,
                        pre_refund_signature,
                    ):
                        await self._complete_refund_state_update(
                            self._commit_refund_state(rid, reservation.model_family)
                        )
                    _raise_backend_external_error(exc)
                interrupted = await self._complete_refund_state_update(
                    self._commit_refund_state(rid, reservation.model_family)
                )
                if interrupted:
                    raise asyncio.CancelledError  # noqa: TRY301
            except BaseException:
                if self._refund_backend_state_changed(
                    refund_backend,
                    refund_bucket_ids_for_probe,
                    pre_refund_signature,
                ):
                    await self._complete_refund_state_update(
                        self._commit_refund_state(rid, reservation.model_family)
                    )
                elif refund_backend_call_started:
                    await self._mark_refund_state_failed(rid)
                else:
                    await self._clear_refund_state_if_pending(rid)
                raise
            finally:
                if refund_started:
                    async with self._refund_state_lock:
                        self._refund_in_progress.discard(rid)
        finally:
            await self._release_reservation_refund_lock(rid, refund_lock)

    def _remember_refund_state(self, reservation_id: str, state: str) -> None:
        self._refunded_reservation_ids[reservation_id] = state
        self._refunded_reservation_ids.move_to_end(reservation_id)
        while len(self._refunded_reservation_ids) > self._refunded_ids_cap:
            self._refunded_reservation_ids.popitem(last=False)

    async def _commit_refund_state(
        self,
        reservation_id: str,
        model_family: str,
    ) -> None:
        async with self._refund_state_lock:
            self._remember_refund_state(reservation_id, _REFUND_STATE_COMMITTED)
            self._forget_in_flight_reservation(reservation_id)
            self._touch_model_family(model_family)

    async def _mark_refund_state_failed(self, reservation_id: str) -> None:
        async with self._refund_state_lock:
            if self._refunded_reservation_ids.get(reservation_id) == (
                _REFUND_STATE_PENDING
            ):
                self._remember_refund_state(reservation_id, _REFUND_STATE_FAILED)

    async def _clear_refund_state_if_pending(self, reservation_id: str) -> None:
        async with self._refund_state_lock:
            if self._refunded_reservation_ids.get(reservation_id) == (
                _REFUND_STATE_PENDING
            ):
                self._refunded_reservation_ids.pop(reservation_id, None)

    @staticmethod
    def _refund_backend_state_signature(
        backend: RateLimiterBackend,
        bucket_ids: frozenset[BucketId] | None,
    ) -> tuple[tuple[BucketId, object, object], ...] | None:
        # KNOWN UNKNOWN: custom backends do not expose a portable "write landed"
        # probe. The built-in memory backends do, so this preserves the R4
        # post-write-failure idempotency contract without pre-committing failed
        # refunds for opaque backends.
        registry = getattr(backend, "_bucket_registry", None)
        if not isinstance(registry, dict):
            return None
        target_bucket_ids = frozenset(registry) if bucket_ids is None else bucket_ids
        signature: list[tuple[BucketId, object, object]] = []
        for bucket_id in sorted(target_bucket_ids):
            bucket = registry.get(bucket_id)
            if bucket is None:
                return None
            signature.append(
                (
                    bucket_id,
                    getattr(bucket, "capacity", _REFUND_STATE_MISSING),
                    getattr(bucket, "last_checked", _REFUND_STATE_MISSING),
                )
            )
        return tuple(signature)

    def _refund_backend_state_changed(
        self,
        backend: RateLimiterBackend | None,
        bucket_ids: frozenset[BucketId] | None,
        pre_refund_signature: tuple[tuple[BucketId, object, object], ...] | None,
    ) -> bool:
        if backend is None or pre_refund_signature is None:
            return False
        post_refund_signature = self._refund_backend_state_signature(
            backend, bucket_ids
        )
        return post_refund_signature is not None and (
            post_refund_signature != pre_refund_signature
        )

    async def _complete_refund_state_update(self, awaitable) -> bool:
        task = asyncio.create_task(awaitable)
        interrupted = False
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                interrupted = True
                if task.done():
                    break
        task.result()
        return interrupted

    async def _acquire_reservation_refund_lock(
        self,
        reservation_id: str,
    ) -> asyncio.Lock:
        # Lock order: do not wait for a per-reservation lock while holding
        # _refund_state_lock. Refund work holds at most one per-reservation
        # lock, then briefly takes _refund_state_lock for metadata; refresh and
        # backend awaits happen without _refund_state_lock held.
        async with self._refund_state_lock:
            lock = self._refund_locks.get(reservation_id)
            if lock is None:
                lock = asyncio.Lock()
                self._refund_locks[reservation_id] = lock
                self._refund_lock_refcounts[reservation_id] = 0
            self._refund_lock_refcounts[reservation_id] += 1
        try:
            await lock.acquire()
        except BaseException:
            await self._release_reservation_refund_lock_reference(
                reservation_id,
                lock,
            )
            raise
        return lock

    async def _release_reservation_refund_lock(
        self,
        reservation_id: str,
        lock: asyncio.Lock,
    ) -> None:
        lock.release()
        await self._release_reservation_refund_lock_reference(reservation_id, lock)

    async def _release_reservation_refund_lock_reference(
        self,
        reservation_id: str,
        lock: asyncio.Lock,
    ) -> None:
        async with self._refund_state_lock:
            count = self._refund_lock_refcounts.get(reservation_id, 0) - 1
            if count <= 0:
                if self._refund_locks.get(reservation_id) is lock:
                    self._refund_locks.pop(reservation_id, None)
                    self._refund_lock_refcounts.pop(reservation_id, None)
                return
            self._refund_lock_refcounts[reservation_id] = count

    async def _unlimited_reservation(
        self,
        model: str,
        limit_config: PerModelConfig,
    ) -> CapacityReservation:
        # Unlimited reservations bypass metering, so their ``usage`` is
        # never read. The ``CapacityReservation`` field validator
        # requires empty ``usage`` when ``is_unlimited=True``; passing
        # ``frozendict()`` makes the factory the only canonical
        # producer of unlimited reservations and closes V05/V14/I05
        # at construction time.
        reservation = CapacityReservation(
            usage=frozendict(),
            model_family=_UNLIMITED_FLAG,
            model=model,
            is_unlimited=True,
            limiter_instance_id=self._limiter_instance_id,
        )
        async with self._acquire_guard:
            self._raise_if_closed_or_closing()
            self._remember_in_flight_reservation(reservation)
        try:
            self._validate_shared_model_family_config(model, limit_config)
        except BaseException:
            self._forget_in_flight_reservation(reservation.reservation_id)
            raise
        return reservation

    async def _refresh_backend_for_reservation(
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
        except Exception as exc:  # noqa: BLE001
            # Design intent: a refund must never be blocked by a transient
            # failure of the user-supplied config_getter. We fall back to
            # cached backend state and emit a warning. BaseException
            # (KeyboardInterrupt/SystemExit) is intentionally allowed to
            # propagate — those are shutdown signals, not refresh failures.
            _warn_refund_refresh_failed(
                model_name=model_name,
                model_family=reservation.model_family,
                exc=exc,
            )
            return

        if limit_config.is_unlimited:
            raise ValueError(
                "Reservation model family "
                f"{reservation.model_family!r} is now unlimited for model "
                f"{model_name!r}; refund across a limited-to-unlimited "
                "config change is not supported. See L13 N03."
            )
        current_model_family = limit_config.get_model_family()
        if current_model_family != reservation.model_family:
            raise ValueError(
                "Reservation model family "
                f"{reservation.model_family!r} no longer matches current "
                f"config for model {model_name!r} "
                f"({current_model_family!r}); refund across model_family "
                "rerouting is not supported. See L13 N05."
            )

        try:
            await self._get_backend(limit_config)
        except Exception as exc:  # noqa: BLE001
            # Backend refresh failures still fall back to cached state so
            # transient backend errors do not leak reserved capacity.
            _warn_refund_refresh_failed(
                model_name=model_name,
                model_family=reservation.model_family,
                exc=exc,
            )

    async def _get_backend(self, cfg: PerModelConfig) -> RateLimiterBackend:
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

        async with self._lock:
            backend = self._model_family_to_backend.get(model_family)
            if backend is not None:
                return await self._sync_backend_quotas(cfg)

            backend = self._backend.build(cfg, callbacks=self._backend_callbacks)
            self._model_family_to_backend[model_family] = backend
            self._model_family_to_quotas[model_family] = new_snapshot
            return backend

    async def _sync_backend_quotas(self, cfg: PerModelConfig) -> RateLimiterBackend:
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
            if backend_uses_default_prepare_reconfigured_backend(old_backend):
                raise RuntimeError(
                    f"Custom backend {type(old_backend).__name__} claims "
                    "supports_metric_set_change=True but did not override "
                    "prepare_reconfigured_backend — silent state drop would occur. "
                    "Override prepare_reconfigured_backend to handle metric-set "
                    "changes correctly."
                )

            warnings.warn(
                f"Callable config for model family '{model_family}' changed metric set "
                f"(was {sorted(old_snapshot)}, now {sorted(new_snapshot)}). "
                "Rebuilding backend; consumption state for surviving metrics will be "
                "transferred by backends that support it.",
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
            backend = self._backend.build(
                rebuild_cfg, callbacks=self._backend_callbacks
            )
            # Invalidate fast-path cache before mutation to close the
            # TOCTOU window where a concurrent reader could match the stale
            # snapshot against an already-mutated backend, tag its reservation
            # with old bucket_ids, and silently leak capacity on refund.
            self._model_family_to_quotas.pop(model_family, None)
            try:
                backend = await old_backend.prepare_reconfigured_backend(
                    backend, rebuild_cfg
                )
                await self._restore_runtime_max_capacity(
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
                await backend.apply_configured_max_capacity(
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
        self._enforce_resolved_config_caps(
            model=model,
            model_family=resolved_model_family,
            limit_config=limit_config,
        )
        previous_model_family = self._model_name_to_model_family.get(model)
        if (
            previous_model_family is not None
            and previous_model_family != resolved_model_family
        ):
            raise ValueError(
                f"Config for model '{model}' changed model_family from "
                f"'{previous_model_family}' to '{resolved_model_family}'. "
                "Model routing must stay stable for a limiter instance; "
                "create a new RateLimiter instead."
            )
        return resolved_model_family

    def _validate_shared_model_family_config(
        self,
        model: str,
        limit_config: PerModelConfig,
        *,
        register: bool = True,
    ) -> None:
        # Detects conflicting quotas across models sharing a model_family.
        # Registration in the reverse-lookup map (_remember_model_family)
        # happens inside this method, after validation passes, so that
        # validate + register is atomic w.r.t. concurrent callers. Under
        # asyncio this is guaranteed because the method contains no await
        # expressions — the event loop cannot switch tasks mid-execution.
        # See _sync_rate_limiter.py for the threaded variant, which uses
        # an explicit lock.
        #
        # Complexity: O(1) per acquire. Once a family signature has been
        # established, a new alias only checks per-family signature counts.
        # Existing aliases can change signature only when no sibling remains
        # on a different signature.
        model_family = limit_config.get_model_family()
        current_signature = _config_signature(limit_config)

        previous_signature = self._model_name_to_validated_signature.get(model)
        if previous_signature == current_signature:
            self._touch_model_family(model_family)
            return

        current_counts = self._model_family_signature_counts.get(model_family, {})
        next_counts = dict(current_counts)
        if previous_signature is not None:
            previous_count = next_counts.get(previous_signature, 0)
            if previous_count <= 1:
                next_counts.pop(previous_signature, None)
            else:
                next_counts[previous_signature] = previous_count - 1

        conflicting_signatures = [
            signature
            for signature, count in next_counts.items()
            if count > 0 and signature != current_signature
        ]
        reset_counts_to_current = False
        if conflicting_signatures:
            representative = self._model_family_to_model_name.get(model_family)
            if representative is not None and representative != model:
                representative_config = self._config_getter(representative)
                representative_family = self._validated_model_family(
                    representative,
                    representative_config,
                )
                representative_signature = _config_signature(representative_config)
                if (
                    representative_family == model_family
                    and representative_signature == current_signature
                ):
                    next_counts = {
                        current_signature: self._model_family_alias_counts.get(
                            model_family,
                            0,
                        )
                    }
                    conflicting_signatures = []
                    reset_counts_to_current = True

            if conflicting_signatures:
                conflict_signature = conflicting_signatures[0]
                raise ValueError(
                    f"Config for model_family '{model_family}' is inconsistent across "
                    f"models. Model '{model}' resolves to "
                    f"{_describe_config_signature(current_signature)}, but the family "
                    f"is already registered as "
                    f"{_describe_config_signature(conflict_signature)}. "
                    "Models sharing a model_family must return identical quotas and "
                    "unlimited behavior for a limiter instance. Use different "
                    "model_family values for different limits."
                )

        self._enforce_new_model_family_cap(model_family)
        self._enforce_new_alias_cap(model)
        if not register:
            return
        is_new_alias = model not in self._model_name_to_model_family
        already_counted = previous_signature is not None and (
            reset_counts_to_current or previous_signature not in current_counts
        )
        if is_new_alias or not already_counted:
            next_counts[current_signature] = next_counts.get(current_signature, 0) + 1
        self._model_name_to_model_family[model] = model_family
        self._model_name_to_validated_signature[model] = current_signature
        self._model_family_signature_counts[model_family] = next_counts
        if is_new_alias:
            self._model_family_alias_counts[model_family] = (
                self._model_family_alias_counts.get(model_family, 0) + 1
            )
        self._model_family_to_validated_signature[model_family] = current_signature
        self._touch_model_family(model_family)

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
        # Caller must hold self._lock. All writers to _model_family_to_runtime_max_capacity
        # (this method, _restore_runtime_max_capacity, _commit_runtime_max_capacity)
        # serialize on _lock so config rebuilds and runtime overrides stay coherent.
        # Dropping that invariant reopens the ghost-override race (R4 L11 B01/B02).
        if not bucket_ids:
            return
        overrides = self._model_family_to_runtime_max_capacity.get(model_family)
        if not overrides:
            return
        for bucket_id in bucket_ids:
            overrides.pop(bucket_id, None)
        if not overrides:
            self._model_family_to_runtime_max_capacity.pop(model_family, None)

    async def _restore_runtime_max_capacity(
        self,
        model_family: str,
        *,
        old_snapshot: dict[BucketId, float],
        new_snapshot: dict[BucketId, float],
        backend: RateLimiterBackend,
    ) -> None:
        # Caller must hold self._lock; see _clear_runtime_max_capacity.
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
            await backend.set_max_capacity(metric, per_seconds, value)
            restored_overrides[bucket_id] = value

        if restored_overrides:
            self._model_family_to_runtime_max_capacity[model_family] = (
                restored_overrides
            )
        else:
            self._model_family_to_runtime_max_capacity.pop(model_family, None)
