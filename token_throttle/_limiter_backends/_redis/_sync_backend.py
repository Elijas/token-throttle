import contextlib
import hashlib
import inspect
import logging
import math
import threading
import time
import typing
import uuid
import warnings
from contextlib import ExitStack
from typing import ClassVar

try:
    import redis
    import redis.client
    import redis.exceptions
except ImportError as exc:
    raise ImportError(
        'The "redis" package is required for the Redis backend. '
        'Install it with: pip install "token-throttle[redis]"'
    ) from exc
from frozendict import frozendict

from token_throttle._capacity import calculate_capacity
from token_throttle._diagnostic import (
    BackendBucketLimit,
    BackendIntrospectionDiagnostic,
    BucketDiagnostic,
    DiagnosticIssue,
    DiagnosticOverrideSource,
    DiagnosticWaiterState,
    RedisBackendHealthDiagnostic,
    backend_type_for_object,
    make_bucket_diagnostic,
    unavailable_bucket_diagnostic,
    wait_bucket_diagnostics,
    waiter_diagnostic_from_state,
)
from token_throttle._exceptions import (
    DuplicateRefundError,
    UnknownReservationError,
    _mark_unknown_reservation_forget_in_flight,
)
from token_throttle._interfaces._callbacks import (
    BACKEND_CALLBACK_CRITICAL_EXCEPTIONS,
    SyncRateLimiterCallbacks,
    current_limiter_callback_context,
    safe_invoke_sync_callback,
)
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    SyncRateLimiterBackend,
    SyncRateLimiterBackendBuilderInterface,
)
from token_throttle._interfaces._models import Capacities, FrozenUsage
from token_throttle._validation import (
    _revalidate_dto,
    _validate_reservation_id,
    validate_backend_refund_usage_for_bucket_ids,
    validate_backend_usage,
    validate_sleep_interval,
    validate_timeout,
)

from ._keys import (
    DEFAULT_REFUND_DEDUP_TTL_SECONDS,
    redis_acquired_marker_key,
    redis_acquired_marker_value,
    redis_refund_dedup_key,
    validate_redis_key_prefix,
    validate_refund_dedup_ttl_seconds,
)
from ._server_time import sync_server_time
from ._sync_bucket import (
    SyncRedisBucket,
    _bucket_state_missing_keys,
    _bucket_state_present_keys,
    _normalize_bucket_state_pair,
    _raise_pipeline_response_error,
    _safe_redis_value_repr,
    _validate_delete_result,
    _validate_expire_result,
    _validate_pipeline_results,
    _validate_redis_get_result,
    _validate_set_nx_result,
    _validate_set_result,
)
from ._ttl import (
    DEFAULT_BUCKET_TTL_SECONDS,
    resolve_max_reservation_lifetime_seconds_from_ttls,
    validate_redis_ttl_seconds,
    validate_reservation_lifetime_ttl_invariant,
)

_logger = logging.getLogger("token_throttle")
_acquire_logger = logging.getLogger("token_throttle.acquire")
_refund_logger = logging.getLogger("token_throttle.refund")
_lock_logger = logging.getLogger("token_throttle.lock")


class SyncCapacitiesGetterResult(typing.NamedTuple):
    capacities: Capacities
    fresh_start_buckets: list[SyncRedisBucket]


LOCK_TIMEOUT_SECONDS = 30

# Each bucket enqueues 4 pipeline commands in get_capacity()
# (GET last_checked, GET capacity, EXPIRE last_checked, EXPIRE capacity).
# Used to index pipeline results.
_PIPELINE_CMDS_PER_BUCKET = 4

# Each bucket enqueues 1 command for max-capacity override reads (GET override).
# Valid overrides have their TTL refreshed after parsing so legacy/corrupt
# payloads are not kept alive.
_PIPELINE_CMDS_PER_OVERRIDE = 1

DEFAULT_LOCK_BLOCKING_TIMEOUT_SECONDS = 5.0
DEFAULT_LOCK_SLEEP_SECONDS = 0.05
DEFAULT_LOCK_BLOCKING_THREAD_SLEEP_SECONDS = 0.05
_MIN_PRODUCTION_REDIS_POOL_CONNECTIONS = 10
_MISSING = object()
_ACQUIRE_MARKER_SET_SCRIPT = """
local existing = redis.call('GET', KEYS[1])
if existing then
    if existing == ARGV[2] then
        return 'ok'
    end
    return 'duplicate_acquire'
end
local arg_index = 3
for key_index = 2, #KEYS, 2 do
    redis.call('SET', KEYS[key_index], ARGV[arg_index], 'EX', ARGV[arg_index + 2])
    redis.call('SET', KEYS[key_index + 1], ARGV[arg_index + 1], 'EX', ARGV[arg_index + 2])
    arg_index = arg_index + 3
end
local claimed = redis.call('SET', KEYS[1], ARGV[2], 'PX', ARGV[1], 'NX')
if not claimed then
    existing = redis.call('GET', KEYS[1])
    if existing == ARGV[2] then
        return 'ok'
    end
    return 'duplicate_acquire'
end
return 'ok'
"""
_REFUND_WITH_MARKER_SCRIPT = """
local marker = redis.call('GET', KEYS[1])
if not marker then
    if redis.call('EXISTS', KEYS[2]) == 1 then
        return 'duplicate_refund'
    end
    return 'unknown_reservation'
end
if marker ~= ARGV[1] then
    return 'marker_mismatch'
end
local claimed = redis.call('SET', KEYS[2], '1', 'EX', ARGV[2], 'NX')
if not claimed then
    return 'duplicate_refund'
end
local arg_index = 3
for key_index = 3, #KEYS, 2 do
    redis.call('SET', KEYS[key_index], ARGV[arg_index], 'EX', ARGV[arg_index + 2])
    redis.call('SET', KEYS[key_index + 1], ARGV[arg_index + 1], 'EX', ARGV[arg_index + 2])
    arg_index = arg_index + 3
end
redis.call('DEL', KEYS[1])
return 'ok'
"""
_MAX_REDIS_SCRIPT_STATUS_CHARS = 64
_MIN_PRINTABLE_SCRIPT_STATUS_CODE = 32


def _debug_event(
    logger: logging.Logger,
    event_type: str,
    *,
    reservation_id: str | None = None,
    bucket_id: tuple[str, int] | None = None,
    **fields: object,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    event = {
        "event_type": event_type,
        "reservation_id": reservation_id,
        "bucket_id": bucket_id,
        **fields,
    }
    logger.debug(event_type, extra={"token_throttle_event": event})


class RedisScriptResultError(RuntimeError):
    """Redis Lua script returned an unusable status value."""


def _raise_lock_timeout_error() -> typing.NoReturn:
    raise redis.exceptions.LockError("Unable to acquire lock within the time specified")


def _lock_name_hash(lock_name: str | bytes | memoryview) -> str:
    lock_name_bytes = (
        lock_name.encode() if isinstance(lock_name, str) else bytes(lock_name)
    )
    return hashlib.blake2s(lock_name_bytes, digest_size=8).hexdigest()


def _validate_positive_seconds(value: object, *, name: str) -> float:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and greater than 0")
    value_float = float(value)
    if not math.isfinite(value_float) or value_float <= 0:
        raise ValueError(f"{name} must be finite and greater than 0")
    return value_float


def _reservation_lifetime_ttl_ms(value: float | None) -> int:
    if value is None:
        raise ValueError(
            "reservation_lifetime_seconds is required when reservation_id is supplied"
        )
    value_float = _validate_positive_seconds(
        value,
        name="reservation_lifetime_seconds",
    )
    return max(1, math.ceil(value_float * 1000.0))


def _require_acquire_marker_metadata(
    *,
    reservation_model_family: str | None,
    reservation_bucket_ids: set[tuple[str, int]] | frozenset[tuple[str, int]] | None,
    reservation_reserved_usage: FrozenUsage | None,
) -> tuple[str, frozenset[tuple[str, int]], FrozenUsage]:
    if (
        reservation_model_family is None
        or reservation_bucket_ids is None
        or reservation_reserved_usage is None
    ):
        raise ValueError(
            "reservation marker metadata is required for marker-authorized refunds"
        )
    return (
        reservation_model_family,
        frozenset(reservation_bucket_ids),
        reservation_reserved_usage,
    )


def _validate_acquire_marker_refund_scope(
    *,
    reserved_usage: FrozenUsage,
    refund_bucket_ids: frozenset[tuple[str, int]],
    marker_bucket_ids: frozenset[tuple[str, int]],
    marker_reserved_usage: FrozenUsage,
) -> None:
    if not refund_bucket_ids.issubset(marker_bucket_ids):
        raise ValueError("refund bucket_ids do not match the acquire marker")
    refund_metric_names = frozenset(metric for metric, _ in refund_bucket_ids)
    for metric in refund_metric_names:
        if metric not in marker_reserved_usage:
            raise ValueError("refund reserved_usage does not match the acquire marker")
        if float(reserved_usage[metric]) != float(marker_reserved_usage[metric]):
            raise ValueError("refund reserved_usage does not match the acquire marker")


def _decode_redis_script_status(value: object, *, context: str) -> str:
    if isinstance(value, bytes):
        try:
            status = value.decode()
        except UnicodeDecodeError as exc:
            raise RedisScriptResultError(
                f"{context}: Redis script status was not valid UTF-8 "
                f"({_safe_redis_value_repr(value)})"
            ) from exc
    elif isinstance(value, str):
        status = value
    else:
        raise RedisScriptResultError(
            f"{context}: Redis script returned unexpected status "
            f"{_safe_redis_value_repr(value)}"
        )
    if len(status) > _MAX_REDIS_SCRIPT_STATUS_CHARS or any(
        ord(char) < _MIN_PRINTABLE_SCRIPT_STATUS_CODE for char in status
    ):
        raise RedisScriptResultError(
            f"{context}: Redis script returned invalid status "
            f"{_safe_redis_value_repr(status)}"
        )
    return status


def _redis_value_matches(value: object, expected: str) -> bool:
    if isinstance(value, bytes):
        try:
            return value.decode() == expected
        except UnicodeDecodeError:
            return False
    return value == expected


def _warn_if_small_redis_pool(redis_client: object, *, stacklevel: int = 3) -> None:
    pool = getattr(redis_client, "connection_pool", None)
    max_connections = getattr(pool, "max_connections", None)
    if (
        isinstance(max_connections, int)
        and not isinstance(max_connections, bool)
        and max_connections < _MIN_PRODUCTION_REDIS_POOL_CONNECTIONS
    ):
        message = (
            "Redis connection_pool.max_connections is less than 10. "
            "This is likely too small for production token-throttle workloads; "
            "prefer a BlockingConnectionPool sized to at least "
            "max_concurrent_acquires plus Redis command headroom."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=stacklevel)
        _lock_logger.warning(message)


def _private_pool_count(pool: object, attr_name: str) -> int | None:
    value = getattr(pool, attr_name, None)
    if value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return None


def _is_mock_object(value: object) -> bool:
    return type(value).__module__.startswith("unittest.mock")


def _ensure_bucket_matches_backend(
    bucket: SyncRedisBucket,
    *,
    key_prefix: str,
    redis_client: redis.Redis,
) -> None:
    bucket_key_prefix = getattr(bucket, "key_prefix", _MISSING)
    if bucket_key_prefix is not _MISSING and bucket_key_prefix != key_prefix:
        raise ValueError(
            "SyncRedisBucket key_prefix must match SyncRedisBackend key_prefix "
            f"(bucket={bucket_key_prefix!r}, backend={key_prefix!r})"
        )

    bucket_redis = getattr(bucket, "_redis", _MISSING)
    if bucket_redis is _MISSING or bucket_redis is redis_client:
        return

    bucket_pool = getattr(bucket_redis, "connection_pool", None)
    backend_pool = getattr(redis_client, "connection_pool", None)
    if bucket_pool is not None and bucket_pool is backend_pool:
        return
    if (
        bucket_pool is None
        or backend_pool is None
        or _is_mock_object(bucket_pool)
        or _is_mock_object(backend_pool)
    ):
        return

    raise ValueError(
        "SyncRedisBucket redis client must be the same object as SyncRedisBackend redis "
        "or share the same connection_pool"
    )


def _ensure_buckets_match_backend(
    buckets: typing.Iterable[SyncRedisBucket],
    *,
    key_prefix: str,
    redis_client: redis.Redis,
) -> None:
    for bucket in buckets:
        _ensure_bucket_matches_backend(
            bucket,
            key_prefix=key_prefix,
            redis_client=redis_client,
        )


class _SyncRedisLockStack(ExitStack):
    """
    ExitStack that also holds the acquired Lock objects so callers can
    extend their TTLs before long-running writes (see _extend_locks).
    """

    def __init__(self) -> None:
        super().__init__()
        self.locks: list[redis.lock.Lock] = []


class SyncRedisBackendBuilder(SyncRateLimiterBackendBuilderInterface):
    """
    Build synchronous Redis limiter backends.

    ``owns_redis_client`` defaults to ``False`` because Redis clients are often
    shared across limiters. Set it to ``True`` only when this builder is the
    lifecycle owner for ``redis_client``; then limiter ``close()`` cascades to
    ``redis_client.close()``.

    For bounded Redis deployments, prefer ``redis.BlockingConnectionPool`` and
    size ``max_connections`` to at least ``max_concurrent_acquires`` plus
    headroom for lock acquire/release, ``TIME``, and pipeline commands.
    """

    def __init__(  # noqa: PLR0913
        self,
        redis_client: redis.Redis,
        *,
        key_prefix: str,
        sleep_interval: float | None = None,
        bucket_ttl_seconds: int = DEFAULT_BUCKET_TTL_SECONDS,
        override_ttl_seconds: int | None = None,
        refund_dedup_ttl_seconds: int = DEFAULT_REFUND_DEDUP_TTL_SECONDS,
        owns_redis_client: bool = False,
        lock_blocking_timeout_seconds: float = DEFAULT_LOCK_BLOCKING_TIMEOUT_SECONDS,
        lock_sleep_seconds: float = DEFAULT_LOCK_SLEEP_SECONDS,
        lock_blocking_thread_sleep_seconds: float = (
            DEFAULT_LOCK_BLOCKING_THREAD_SLEEP_SECONDS
        ),
    ) -> None:
        super().__init__()
        if not isinstance(redis_client, redis.Redis):
            raise TypeError(
                "redis_client expected redis.Redis, "
                f"got {type(redis_client).__name__}; for async use "
                "redis.asyncio.Redis with RedisBackendBuilder"
            )
        self._redis = redis_client
        self._owns_redis_client = owns_redis_client
        self._key_prefix = validate_redis_key_prefix(key_prefix)
        self._sleep_interval = validate_sleep_interval(sleep_interval)
        self._bucket_ttl_seconds = validate_redis_ttl_seconds(
            bucket_ttl_seconds, name="bucket_ttl_seconds"
        )
        self._override_ttl_seconds = validate_redis_ttl_seconds(
            (
                self._bucket_ttl_seconds
                if override_ttl_seconds is None
                else override_ttl_seconds
            ),
            name="override_ttl_seconds",
        )
        self._refund_dedup_ttl_seconds = validate_refund_dedup_ttl_seconds(
            refund_dedup_ttl_seconds
        )
        self._lock_blocking_timeout_seconds = _validate_positive_seconds(
            lock_blocking_timeout_seconds,
            name="lock_blocking_timeout_seconds",
        )
        self._lock_sleep_seconds = _validate_positive_seconds(
            lock_sleep_seconds,
            name="lock_sleep_seconds",
        )
        self._lock_blocking_thread_sleep_seconds = _validate_positive_seconds(
            lock_blocking_thread_sleep_seconds,
            name="lock_blocking_thread_sleep_seconds",
        )

    def resolve_max_reservation_lifetime_seconds(
        self,
        max_reservation_lifetime_seconds: float | None,
    ) -> float | None:
        """Resolve caller lifetime input against Redis bucket and refund TTL settings."""
        return resolve_max_reservation_lifetime_seconds_from_ttls(
            max_reservation_lifetime_seconds=max_reservation_lifetime_seconds,
            bucket_ttl_seconds=self._bucket_ttl_seconds,
            refund_dedup_ttl_seconds=self._refund_dedup_ttl_seconds,
        )

    def validate_reservation_lifetime_seconds(
        self,
        max_reservation_lifetime_seconds: float | None,
    ) -> None:
        """Validate that Redis TTLs outlive the maximum refundable reservation lifetime."""
        validate_reservation_lifetime_ttl_invariant(
            max_reservation_lifetime_seconds=max_reservation_lifetime_seconds,
            bucket_ttl_seconds=self._bucket_ttl_seconds,
            refund_dedup_ttl_seconds=self._refund_dedup_ttl_seconds,
        )

    def close(self) -> None:
        if not self._owns_redis_client:
            return
        close = getattr(self._redis, "close", None)
        if callable(close):
            result = close()
            if inspect.iscoroutine(result):
                result.close()

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> "SyncRedisBackend":
        cfg = _revalidate_dto(cfg)
        if callbacks is not None:
            _revalidate_dto(callbacks)
        redis_buckets = []
        for quota in cfg.quotas:
            b = SyncRedisBucket(
                quota=quota,
                limit_config=cfg,
                redis_client=self._redis,
                key_prefix=self._key_prefix,
                bucket_ttl_seconds=self._bucket_ttl_seconds,
                override_ttl_seconds=self._override_ttl_seconds,
            )
            redis_buckets.append(b)
        return SyncRedisBackend(
            buckets=redis_buckets,
            redis=self._redis,
            key_prefix=self._key_prefix,
            refund_dedup_ttl_seconds=self._refund_dedup_ttl_seconds,
            sleep_interval=self._sleep_interval,
            lock_blocking_timeout_seconds=self._lock_blocking_timeout_seconds,
            lock_sleep_seconds=self._lock_sleep_seconds,
            lock_blocking_thread_sleep_seconds=self._lock_blocking_thread_sleep_seconds,
            callbacks=callbacks,
            limit_config=cfg,
        )


class SyncRedisBackend(SyncRateLimiterBackend):
    DEFAULT_SLEEP_INTERVAL: ClassVar[float] = 0.1
    MAX_CROSS_WORKER_POLL: ClassVar[float] = 1.0

    def __init__(  # noqa: PLR0913
        self,
        buckets: list[SyncRedisBucket],
        redis: redis.Redis,
        limit_config: PerModelConfig,
        *,
        key_prefix: str,
        refund_dedup_ttl_seconds: int = DEFAULT_REFUND_DEDUP_TTL_SECONDS,
        sleep_interval: float | None = None,
        lock_blocking_timeout_seconds: float = DEFAULT_LOCK_BLOCKING_TIMEOUT_SECONDS,
        lock_sleep_seconds: float = DEFAULT_LOCK_SLEEP_SECONDS,
        lock_blocking_thread_sleep_seconds: float = (
            DEFAULT_LOCK_BLOCKING_THREAD_SLEEP_SECONDS
        ),
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> None:
        super().__init__()
        limit_config = _revalidate_dto(limit_config)
        if callbacks is not None:
            _revalidate_dto(callbacks)
        self._redis = redis
        self._key_prefix = validate_redis_key_prefix(key_prefix)
        _ensure_buckets_match_backend(
            buckets,
            key_prefix=self._key_prefix,
            redis_client=self._redis,
        )
        self.sorted_buckets = sorted(buckets, key=lambda b: b.full_redis_key)
        self._refund_dedup_ttl_seconds = validate_refund_dedup_ttl_seconds(
            refund_dedup_ttl_seconds
        )
        validated_sleep_interval = validate_sleep_interval(sleep_interval)
        self._sleep_interval = (
            self.DEFAULT_SLEEP_INTERVAL
            if validated_sleep_interval is None
            else validated_sleep_interval
        )
        _warn_if_small_redis_pool(redis)
        self._lock_blocking_timeout_seconds = _validate_positive_seconds(
            lock_blocking_timeout_seconds,
            name="lock_blocking_timeout_seconds",
        )
        self._lock_sleep_seconds = _validate_positive_seconds(
            lock_sleep_seconds,
            name="lock_sleep_seconds",
        )
        self._lock_blocking_thread_sleep_seconds = _validate_positive_seconds(
            lock_blocking_thread_sleep_seconds,
            name="lock_blocking_thread_sleep_seconds",
        )
        self._callbacks = callbacks
        self._limit_config = limit_config
        self._usage_metric_names: set[str] = {bucket.usage_metric for bucket in buckets}
        self._local_condition = threading.Condition()
        self._legacy_override_probe_complete = False
        self._diagnostic_waiters: dict[str, DiagnosticWaiterState] = {}

    def add_bucket(self, bucket: SyncRedisBucket) -> None:
        """Add a sync Redis bucket; it must share this backend's client and key prefix."""
        _ensure_bucket_matches_backend(
            bucket,
            key_prefix=self._key_prefix,
            redis_client=self._redis,
        )
        self.sorted_buckets = sorted(
            [*self._snapshot_buckets(), bucket],
            key=lambda existing_bucket: existing_bucket.full_redis_key,
        )
        self._usage_metric_names = self._usage_metric_names_for(self.sorted_buckets)

    def _effective_lock_blocking_timeout(
        self,
        blocking_timeout: float | None,
    ) -> float:
        if blocking_timeout is None:
            return self._lock_blocking_timeout_seconds
        requested_timeout = validate_timeout(blocking_timeout)
        if requested_timeout is None:
            return self._lock_blocking_timeout_seconds
        return min(requested_timeout, self._lock_blocking_timeout_seconds)

    def supports_metric_set_change(self) -> bool:
        # Surviving bucket state lives in Redis under stable keys, so a rebuilt
        # backend can safely point at the same buckets.
        return True

    def supports_durable_refund_dedup(self) -> bool:
        return True

    def supports_acquire_marker_authority(self) -> bool:
        return True

    def introspect(self) -> BackendIntrospectionDiagnostic:
        as_of_monotonic = time.monotonic()
        buckets = self._snapshot_buckets()
        issues: list[DiagnosticIssue] = []
        try:
            bucket_diagnostics = self._introspect_buckets_read_only(
                buckets,
                as_of_monotonic=as_of_monotonic,
                issues=issues,
            )
        except Exception as exc:  # noqa: BLE001
            issues.append(
                DiagnosticIssue(
                    severity="warning",
                    component="redis_backend",
                    message=f"Redis backend introspection failed: {type(exc).__name__}: {exc}",
                    model_family=self._limit_config.get_model_family(),
                )
            )
            quota_limits = {
                (quota.metric, int(quota.per_seconds)): float(quota.limit)
                for quota in self._limit_config.quotas
            }
            bucket_diagnostics = tuple(
                unavailable_bucket_diagnostic(
                    model_family=self._limit_config.get_model_family(),
                    bucket_id=(bucket.usage_metric, int(bucket.per_seconds)),
                    configured_limit=quota_limits.get(
                        (bucket.usage_metric, int(bucket.per_seconds)),
                        bucket.configured_max_capacity,
                    ),
                    local_override=None,
                    backend_type=backend_type_for_object(self),
                    as_of_monotonic=as_of_monotonic,
                )
                for bucket in buckets
            )
        with self._local_condition:
            waiters = tuple(
                waiter_diagnostic_from_state(state, as_of_monotonic=as_of_monotonic)
                for state in sorted(
                    self._diagnostic_waiters.values(),
                    key=lambda item: (item.wait_started_monotonic, item.waiter_id),
                )
            )
        return BackendIntrospectionDiagnostic(
            model_family=self._limit_config.get_model_family(),
            backend_type=backend_type_for_object(self),
            as_of_monotonic=as_of_monotonic,
            buckets=bucket_diagnostics,
            waits=waiters,
            memory_health=None,
            redis_health=self._redis_backend_health(bucket_count=len(buckets)),
            issues=tuple(issues),
        )

    def _introspect_buckets_read_only(
        self,
        buckets: tuple[SyncRedisBucket, ...],
        *,
        as_of_monotonic: float,
        issues: list[DiagnosticIssue],
    ) -> tuple[BucketDiagnostic, ...]:
        if not buckets:
            return ()
        current_time = sync_server_time(self._redis)
        pipeline = self._redis.pipeline()
        for bucket in buckets:
            pipeline.get(bucket._last_checked_key)  # noqa: SLF001
            pipeline.get(bucket._capacity_key)  # noqa: SLF001
            pipeline.get(bucket._max_capacity_key)  # noqa: SLF001
        try:
            results = pipeline.execute()
        except redis.exceptions.ResponseError as exc:
            _raise_pipeline_response_error("SyncRedisBackend.introspect", exc)
        results = _validate_pipeline_results(
            results,
            context="SyncRedisBackend.introspect",
            expected_count=len(buckets) * 3,
        )
        diagnostics: list[BucketDiagnostic] = []
        quota_limits = {
            (quota.metric, int(quota.per_seconds)): float(quota.limit)
            for quota in self._limit_config.quotas
        }
        for index, bucket in enumerate(buckets):
            offset = index * 3
            diagnostics.append(
                self._diagnostic_bucket_from_raw(
                    bucket,
                    last_checked_raw=results[offset],
                    capacity_raw=results[offset + 1],
                    override_raw=results[offset + 2],
                    current_time=current_time,
                    as_of_monotonic=as_of_monotonic,
                    configured_limit=quota_limits.get(
                        (bucket.usage_metric, int(bucket.per_seconds)),
                        bucket.configured_max_capacity,
                    ),
                    issues=issues,
                )
            )
        return tuple(diagnostics)

    def _diagnostic_bucket_from_raw(  # noqa: PLR0913
        self,
        bucket: SyncRedisBucket,
        *,
        last_checked_raw: object,
        capacity_raw: object,
        override_raw: object,
        current_time: float,
        as_of_monotonic: float,
        configured_limit: float,
        issues: list[DiagnosticIssue],
    ) -> BucketDiagnostic:
        metric = bucket.usage_metric
        per_seconds = int(bucket.per_seconds)
        try:
            last_checked = _validate_redis_get_result(
                last_checked_raw,
                context=(
                    f"SyncRedisBackend.introspect({bucket.full_redis_key}) last_checked"
                ),
            )
            stored_capacity = _validate_redis_get_result(
                capacity_raw,
                context=f"SyncRedisBackend.introspect({bucket.full_redis_key}) capacity",
            )
            override_raw = _validate_redis_get_result(
                override_raw,
                context=(
                    f"SyncRedisBackend.introspect({bucket.full_redis_key}) "
                    "max_capacity_override"
                ),
            )
            override = bucket._deserialize_max_capacity_override(override_raw)  # noqa: SLF001
            effective = configured_limit if override is None else override
            source: DiagnosticOverrideSource = "none" if override is None else "backend"
            if (last_checked is None) != (stored_capacity is None):
                issues.append(
                    DiagnosticIssue(
                        severity="warning",
                        component="redis_backend",
                        message="Redis bucket has partial state",
                        model_family=self._limit_config.get_model_family(),
                        metric=metric,
                        per_seconds=per_seconds,
                    )
                )
                return make_bucket_diagnostic(
                    model_family=self._limit_config.get_model_family(),
                    metric=metric,
                    per_seconds=per_seconds,
                    backend_type=backend_type_for_object(self),
                    current_capacity=None,
                    configured_limit=configured_limit,
                    effective_max_capacity=effective,
                    override_source=source,
                    status="partial_missing",
                    as_of_monotonic=as_of_monotonic,
                )
            calculated = calculate_capacity(
                last_checked=(None if last_checked is None else float(last_checked)),
                outdated_capacity=(
                    None if stored_capacity is None else float(stored_capacity)
                ),
                current_time=current_time,
                max_capacity=effective,
                rate_per_sec=effective / per_seconds,
                bucket_id=bucket.full_redis_key,
            )
            return make_bucket_diagnostic(
                model_family=self._limit_config.get_model_family(),
                metric=metric,
                per_seconds=per_seconds,
                backend_type=backend_type_for_object(self),
                current_capacity=calculated.amount,
                configured_limit=configured_limit,
                effective_max_capacity=effective,
                override_source=source,
                status="fresh_start" if calculated.is_fresh_start else "ok",
                as_of_monotonic=as_of_monotonic,
            )
        except Exception as exc:  # noqa: BLE001
            issues.append(
                DiagnosticIssue(
                    severity="warning",
                    component="redis_backend",
                    message=f"Redis bucket introspection failed: {type(exc).__name__}: {exc}",
                    model_family=self._limit_config.get_model_family(),
                    metric=metric,
                    per_seconds=per_seconds,
                )
            )
            return unavailable_bucket_diagnostic(
                model_family=self._limit_config.get_model_family(),
                bucket_id=(metric, per_seconds),
                configured_limit=configured_limit,
                local_override=None,
                backend_type=backend_type_for_object(self),
                as_of_monotonic=as_of_monotonic,
                status="corrupt",
            )

    def _redis_backend_health(
        self, *, bucket_count: int
    ) -> RedisBackendHealthDiagnostic:
        pool = getattr(self._redis, "connection_pool", None)
        max_connections = getattr(pool, "max_connections", None)
        if type(max_connections) is not int:
            max_connections = None
        in_use = _private_pool_count(pool, "_in_use_connections")
        available = _private_pool_count(pool, "_available_connections")
        return RedisBackendHealthDiagnostic(
            model_family_count=1,
            bucket_count=bucket_count,
            connection_pool_class=None if pool is None else type(pool).__name__,
            connection_pool_max_connections=max_connections,
            connection_pool_in_use_connections=in_use,
            connection_pool_available_connections=available,
            pool_counts_observed_with_private_attrs=(
                in_use is not None or available is not None
            ),
            local_marker_count_estimate=0,
            local_refund_dedup_count_estimate=0,
        )

    def _refund_dedup_key(self, reservation_id: str | None) -> str | None:
        if reservation_id is None:
            return None
        reservation_id = _validate_reservation_id(reservation_id)
        return redis_refund_dedup_key(self._key_prefix, reservation_id)

    def _acquired_marker_key(self, reservation_id: str | None) -> str | None:
        if reservation_id is None:
            return None
        reservation_id = _validate_reservation_id(reservation_id)
        return redis_acquired_marker_key(self._key_prefix, reservation_id)

    def _refund_dedup_exists(self, reservation_id: str | None) -> bool:
        key = self._refund_dedup_key(reservation_id)
        if key is None:
            return False
        return bool(self._redis.exists(key))

    def _commit_refund_dedup(self, reservation_id: str | None) -> bool:
        key = self._refund_dedup_key(reservation_id)
        if key is None:
            return True
        claimed = self._redis.set(
            key,
            "1",
            ex=self._refund_dedup_ttl_seconds,
            nx=True,
        )
        _debug_event(
            _refund_logger,
            "redis_refund_dedup_write",
            reservation_id=reservation_id,
            bucket_id=None,
            claimed=bool(claimed),
        )
        if claimed:
            return True
        assert reservation_id is not None  # noqa: S101
        self._warn_refund_dedup_duplicate(reservation_id)
        return False

    def _warn_refund_dedup_duplicate(self, reservation_id: str) -> None:
        message = (
            f"Reservation {reservation_id} has already been refunded according "
            "to Redis refund dedup. Ignoring duplicate refund to prevent "
            "double-crediting capacity."
        )
        warnings.warn(message, UserWarning, stacklevel=3)
        _logger.warning(message)

    def _claim_refund_dedup(self, reservation_id: str | None) -> bool:
        return self._commit_refund_dedup(reservation_id)

    def _acquire_marker_matches(
        self,
        acquired_marker_key: str,
        acquired_marker_value: str,
        *,
        reservation_id: str | None = None,
    ) -> bool:
        try:
            marker = self._redis.get(acquired_marker_key)
        except redis.exceptions.RedisError:
            return False
        matches = _redis_value_matches(marker, acquired_marker_value)
        _debug_event(
            _acquire_logger,
            "redis_acquire_marker_get",
            reservation_id=reservation_id,
            bucket_id=None,
            matched=matches,
        )
        return matches

    def _refund_tombstone_exists(
        self,
        refund_dedup_key: str,
        *,
        reservation_id: str | None = None,
    ) -> bool:
        try:
            exists = bool(self._redis.exists(refund_dedup_key))
        except redis.exceptions.RedisError:
            return False
        _debug_event(
            _refund_logger,
            "redis_refund_dedup_get",
            reservation_id=reservation_id,
            bucket_id=None,
            exists=exists,
        )
        return exists

    def _snapshot_buckets(self) -> tuple[SyncRedisBucket, ...]:
        return tuple(self.sorted_buckets)

    def _probe_legacy_override_keys_once(
        self,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
    ) -> None:
        if self._legacy_override_probe_complete:
            return
        target_buckets = self.sorted_buckets if buckets is None else buckets
        for bucket in target_buckets:
            result = self._redis.get(bucket._legacy_max_capacity_key)  # noqa: SLF001
            bucket.handle_legacy_max_capacity_key(result)
        self._legacy_override_probe_complete = True

    def _runtime_max_capacity_for_reconciliation(
        self,
        metric: str,
        per_seconds: int,
    ) -> float | None:
        bucket = self._find_bucket(self._snapshot_buckets(), metric, per_seconds)
        if bucket is None:
            return None
        return bucket.refresh_max_capacity_from_redis()

    def _validation_metric_names(
        self,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
    ) -> set[str]:
        target_buckets = self._snapshot_buckets() if buckets is None else tuple(buckets)
        if target_buckets:
            return self._usage_metric_names_for(target_buckets)
        return set(self._usage_metric_names)

    @staticmethod
    def _usage_metric_names_for(
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket],
    ) -> set[str]:
        return {bucket.usage_metric for bucket in buckets}

    @staticmethod
    def _ensure_usage_metrics_are_active(
        usage: FrozenUsage,
        active_metric_names: set[str],
    ) -> None:
        missing_metrics = sorted(set(usage) - active_metric_names)
        if missing_metrics:
            raise ValueError(
                "Usage metrics "
                f"{missing_metrics} are no longer active after backend reconfiguration. "
                f"Active metrics are {sorted(active_metric_names)}."
            )

    @staticmethod
    def _find_bucket(
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket],
        metric: str,
        per_seconds: int,
    ) -> SyncRedisBucket | None:
        return next(
            (
                bucket
                for bucket in buckets
                if bucket.usage_metric == metric
                and int(bucket.per_seconds) == int(per_seconds)
            ),
            None,
        )

    @staticmethod
    def _combined_bucket_snapshot(
        *bucket_groups: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket],
    ) -> tuple[SyncRedisBucket, ...]:
        buckets_by_key: dict[str, SyncRedisBucket] = {}
        for bucket_group in bucket_groups:
            for bucket in bucket_group:
                buckets_by_key[bucket.full_redis_key] = bucket
        return tuple(
            sorted(buckets_by_key.values(), key=lambda bucket: bucket.full_redis_key)
        )

    def _lock(
        self,
        *,
        timeout: float,
        blocking_timeout: float | None = None,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
        reservation_id: str | None = None,
    ) -> _SyncRedisLockStack:
        """Acquire locks for a fixed bucket snapshot in a consistent order."""
        stack = _SyncRedisLockStack()
        target_buckets = self.sorted_buckets if buckets is None else buckets
        effective_blocking_timeout = self._effective_lock_blocking_timeout(
            blocking_timeout
        )
        stop_trying_at = time.monotonic() + effective_blocking_timeout

        try:
            for bucket in target_buckets:
                if stack.locks:
                    self._extend_locks(stack, reservation_id=reservation_id)
                remaining = max(0.0, stop_trying_at - time.monotonic())
                lock = bucket.lock(timeout=timeout, sleep=self._lock_sleep_seconds)
                bucket_id = (bucket.usage_metric, int(bucket.per_seconds))
                # Generate the token ourselves so a best-effort CAS
                # release after a KeyboardInterrupt-style cancel can run
                # without relying on lock.local.token (which is only
                # populated AFTER the SET NX succeeds; a cancel in that
                # window would otherwise orphan the lock for its TTL).
                token = uuid.uuid4().hex
                try:
                    acquired = lock.acquire(
                        sleep=self._lock_blocking_thread_sleep_seconds,
                        blocking_timeout=remaining,
                        token=token,
                    )
                except BaseException:
                    # Best-effort CAS release; only deletes the key if
                    # its value still matches our token.
                    with contextlib.suppress(Exception):
                        lua_release = lock.lua_release
                        if lua_release is not None:
                            lua_release(
                                keys=[lock.name], args=[token], client=self._redis
                            )
                    raise
                if not acquired:
                    _raise_lock_timeout_error()
                _debug_event(
                    _lock_logger,
                    "redis_lock_acquire",
                    reservation_id=reservation_id,
                    bucket_id=bucket_id,
                )
                stack.locks.append(lock)
                stack.callback(
                    self._release_lock_logged,
                    lock,
                    bucket_id=bucket_id,
                    reservation_id=reservation_id,
                )
        except BaseException:
            stack.close()
            raise

        return stack

    @staticmethod
    def _release_lock_logged(
        lock: redis.lock.Lock,
        *,
        bucket_id: tuple[str, int],
        reservation_id: str | None,
    ) -> None:
        try:
            lock.release()
        finally:
            _debug_event(
                _lock_logger,
                "redis_lock_release",
                reservation_id=reservation_id,
                bucket_id=bucket_id,
            )

    @staticmethod
    def _extend_locks(
        stack: _SyncRedisLockStack,
        *,
        reservation_id: str | None = None,
    ) -> None:
        """
        Reset each held lock's TTL to LOCK_TIMEOUT_SECONDS; see the
        async backend's ``_extend_locks`` for the rationale.
        """
        for lock in stack.locks:
            try:
                lock.reacquire()
            except redis.exceptions.LockNotOwnedError as exc:
                raise redis.exceptions.LockError(
                    "Lock expired or was stolen mid-operation; aborting write."
                ) from exc
            _debug_event(
                _lock_logger,
                "redis_lock_extension",
                reservation_id=reservation_id,
                bucket_id=None,
                lock_name_hash=_lock_name_hash(lock.name),
            )

    def _get_capacities_unsafe(  # noqa: PLR0915
        self,
        pipeline: redis.client.Pipeline | None = None,
        current_time: float | None = None,
        *,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
    ) -> SyncCapacitiesGetterResult:
        """Get capacities for all buckets."""
        if pipeline is None:
            pipeline = self._redis.pipeline()
        target_buckets = self.sorted_buckets if buckets is None else buckets
        self._probe_legacy_override_keys_once(target_buckets)

        if current_time is None:
            current_time = sync_server_time(self._redis)

        for bucket in target_buckets:
            bucket.get_capacity(pipeline=pipeline, current_time=current_time)

        # Include max_capacity in the pipeline to avoid extra round-trips.
        # Override TTL refresh happens after parsing so invalid legacy payloads
        # are not kept alive.
        for bucket in target_buckets:
            pipeline.get(bucket._max_capacity_key)  # noqa: SLF001

        num_buckets = len(target_buckets)
        expected_results = num_buckets * (
            _PIPELINE_CMDS_PER_BUCKET + _PIPELINE_CMDS_PER_OVERRIDE
        )
        try:
            results = pipeline.execute()
        except redis.exceptions.ResponseError as exc:
            _raise_pipeline_response_error(
                "SyncRedisBackend._get_capacities_unsafe", exc
            )
        results = _validate_pipeline_results(
            results,
            context="SyncRedisBackend._get_capacities_unsafe",
            expected_count=expected_results,
        )

        new_capacities: dict[tuple[str, int], float] = {}
        fresh_start_buckets: list[SyncRedisBucket] = []
        partial_state_buckets: list[SyncRedisBucket] = []
        for i, bucket in enumerate(target_buckets):
            idx = i * _PIPELINE_CMDS_PER_BUCKET
            last_checked_raw = results[
                idx + SyncRedisBucket.PIPELINE_LAST_CHECKED_OFFSET
            ]
            capacity_raw = results[idx + SyncRedisBucket.PIPELINE_CAPACITY_OFFSET]
            missing_keys = _bucket_state_missing_keys(last_checked_raw, capacity_raw)
            present_keys = _bucket_state_present_keys(last_checked_raw, capacity_raw)
            last_checked, capacity = _normalize_bucket_state_pair(
                last_checked_raw,
                capacity_raw,
                context=(
                    f"SyncRedisBackend._get_capacities_unsafe({bucket.full_redis_key})"
                ),
                current_time=current_time,
            )
            _validate_expire_result(
                results[idx + 2],
                context=(
                    "SyncRedisBackend._get_capacities_unsafe"
                    f"({bucket.full_redis_key}) last_checked TTL"
                ),
            )
            _validate_expire_result(
                results[idx + 3],
                context=(
                    "SyncRedisBackend._get_capacities_unsafe"
                    f"({bucket.full_redis_key}) capacity TTL"
                ),
            )
            # max_capacity GETs come after all per-bucket state commands.
            max_capacity_idx = num_buckets * _PIPELINE_CMDS_PER_BUCKET + (
                i * _PIPELINE_CMDS_PER_OVERRIDE
            )
            max_capacity_raw = _validate_redis_get_result(
                results[max_capacity_idx],
                context=(
                    "SyncRedisBackend._get_capacities_unsafe"
                    f"({bucket.full_redis_key}) max_capacity_override"
                ),
            )
            if bucket.update_max_capacity_from_result(max_capacity_raw):
                self._redis.expire(
                    bucket._max_capacity_key,  # noqa: SLF001
                    bucket._override_ttl_seconds,  # noqa: SLF001
                )
            result = bucket.calculate_capacity(
                last_checked,
                capacity,
                current_time,
            )
            result = _revalidate_dto(result)
            if len(missing_keys) == 1:
                bucket._set_missing_consumption_data_context(  # noqa: SLF001
                    reason="partial_state_drained",
                    missing_keys=missing_keys,
                    present_keys=present_keys,
                )
                partial_state_buckets.append(bucket)
                fresh_start_buckets.append(bucket)
            elif result.is_fresh_start:
                bucket._set_missing_consumption_data_context(  # noqa: SLF001
                    reason="fresh_start",
                    missing_keys=missing_keys,
                    present_keys=present_keys,
                )
                fresh_start_buckets.append(bucket)
            else:
                bucket._set_missing_consumption_data_context(  # noqa: SLF001
                    reason=None,
                    missing_keys=(),
                    present_keys=present_keys,
                )
            new_capacities[(bucket.usage_metric, int(bucket.per_seconds))] = (
                result.amount
            )

        if partial_state_buckets:
            repair_pipeline = self._redis.pipeline()
            for bucket in partial_state_buckets:
                bucket.set_capacity(
                    0.0,
                    pipeline=repair_pipeline,
                    current_time=current_time,
                    execute=False,
                )
            try:
                repair_results = repair_pipeline.execute()
            except redis.exceptions.ResponseError as exc:
                _raise_pipeline_response_error(
                    "SyncRedisBackend._get_capacities_unsafe partial-state repair",
                    exc,
                )
            repair_results = _validate_pipeline_results(
                repair_results,
                context="SyncRedisBackend._get_capacities_unsafe partial-state repair",
                expected_count=2 * len(partial_state_buckets),
            )
            for i, bucket in enumerate(partial_state_buckets):
                _validate_set_result(
                    repair_results[i * 2],
                    context=(
                        "SyncRedisBackend._get_capacities_unsafe partial-state repair"
                        f"({bucket.full_redis_key}) last_checked"
                    ),
                )
                _validate_set_result(
                    repair_results[i * 2 + 1],
                    context=(
                        "SyncRedisBackend._get_capacities_unsafe partial-state repair"
                        f"({bucket.full_redis_key}) capacity"
                    ),
                )

        return SyncCapacitiesGetterResult(
            capacities=frozendict(new_capacities),
            fresh_start_buckets=fresh_start_buckets,
        )

    def _set_capacities_unsafe(  # noqa: PLR0913, PLR0915
        self,
        new_capacities: Capacities,
        pipeline: redis.client.Pipeline | None = None,
        current_time: float | None = None,
        *,
        allow_negative: bool = False,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
        refund_dedup_key: str | None = None,
        refund_dedup_reservation_id: str | None = None,
        refund_dedup_ttl_seconds: int | None = None,
        acquired_marker_key: str | None = None,
        acquired_marker_value: str | None = None,
        acquired_marker_ttl_ms: int | None = None,
        delete_acquired_marker_key: str | None = None,
        reservation_id: str | None = None,
    ) -> bool:
        """
        Set capacities for all buckets. Caller must hold the distributed lock.

        allow_negative=True is required for consume_capacity (speedometer)
        and refund_capacity (preserves negative debt for natural refill).
        """
        if pipeline is None:
            pipeline = self._redis.pipeline()
        target_buckets = self.sorted_buckets if buckets is None else buckets

        if current_time is None:
            current_time = sync_server_time(self._redis)

        if acquired_marker_key is not None:
            if acquired_marker_value is None or acquired_marker_ttl_ms is None:
                raise ValueError("acquired marker writes require marker value and TTL")
            for bucket in target_buckets:
                _debug_event(
                    _acquire_logger,
                    "redis_acquire_marker_write",
                    reservation_id=reservation_id,
                    bucket_id=(bucket.usage_metric, int(bucket.per_seconds)),
                )
            keys: list[str] = [acquired_marker_key]
            args: list[str | bytes | int | float] = [
                acquired_marker_ttl_ms,
                acquired_marker_value,
            ]
            for (usage_metric, per_seconds), amount in new_capacities.items():
                matching_bucket = self._find_bucket(
                    target_buckets,
                    usage_metric,
                    per_seconds,
                )
                if matching_bucket is None:
                    raise ValueError(
                        f"Bucket '{usage_metric}/{per_seconds}s' not found"
                    )
                if not math.isfinite(float(amount)):
                    raise ValueError(f"capacity must be finite (got {amount!r})")
                normalized_amount = amount if allow_negative else max(0, amount)
                if normalized_amount == 0.0:
                    normalized_amount = 0.0
                keys.extend(
                    [
                        matching_bucket._last_checked_key,  # noqa: SLF001
                        matching_bucket._capacity_key,  # noqa: SLF001
                    ]
                )
                args.extend(
                    [
                        current_time,
                        normalized_amount,
                        matching_bucket._bucket_ttl_seconds,  # noqa: SLF001
                    ]
                )
            try:
                result = self._redis.eval(
                    _ACQUIRE_MARKER_SET_SCRIPT,
                    len(keys),
                    *keys,
                    *args,
                )
            except redis.exceptions.RedisError:
                if self._acquire_marker_matches(
                    acquired_marker_key,
                    acquired_marker_value,
                    reservation_id=reservation_id,
                ):
                    return True
                raise
            status = _decode_redis_script_status(
                result, context="SyncRedisBackend acquire marker script"
            )
            if status == "ok":
                return True
            if status == "duplicate_acquire":
                if self._acquire_marker_matches(
                    acquired_marker_key,
                    acquired_marker_value,
                    reservation_id=reservation_id,
                ):
                    return True
                raise DuplicateRefundError(
                    "reservation already acquired",
                    reason="duplicate_acquire",
                    reservation_id=reservation_id,
                    model_family=self._limit_config.get_model_family(),
                )
            raise RedisScriptResultError(
                "Redis acquire marker script returned unknown status "
                f"{_safe_redis_value_repr(status)}"
            )

        for (usage_metric, per_seconds), amount in new_capacities.items():
            matching_bucket = self._find_bucket(
                target_buckets,
                usage_metric,
                per_seconds,
            )
            if matching_bucket is None:
                raise ValueError(f"Bucket '{usage_metric}/{per_seconds}s' not found")
            matching_bucket.set_capacity(
                amount,
                pipeline=pipeline,
                current_time=current_time,
                execute=False,
                allow_negative=allow_negative,
            )
        if refund_dedup_key is not None:
            _debug_event(
                _refund_logger,
                "redis_refund_dedup_write",
                reservation_id=refund_dedup_reservation_id or reservation_id,
                bucket_id=None,
            )
            pipeline.set(
                refund_dedup_key,
                "1",
                ex=refund_dedup_ttl_seconds,
                nx=True,
            )
        if delete_acquired_marker_key is not None:
            _debug_event(
                _acquire_logger,
                "redis_acquire_marker_delete",
                reservation_id=reservation_id,
                bucket_id=None,
            )
            pipeline.delete(delete_acquired_marker_key)
        try:
            results = pipeline.execute()
        except redis.exceptions.ResponseError as exc:
            _raise_pipeline_response_error(
                "SyncRedisBackend._set_capacities_unsafe", exc
            )
        results = _validate_pipeline_results(
            results,
            context="SyncRedisBackend._set_capacities_unsafe",
            expected_count=(len(new_capacities) * 2)
            + int(refund_dedup_key is not None)
            + int(delete_acquired_marker_key is not None),
        )
        for slot in range(len(new_capacities) * 2):
            _validate_set_result(
                results[slot],
                context=f"SyncRedisBackend._set_capacities_unsafe capacity slot {slot}",
            )
        if refund_dedup_key is None:
            if delete_acquired_marker_key is not None:
                _validate_delete_result(
                    results[len(new_capacities) * 2],
                    context=(
                        "SyncRedisBackend._set_capacities_unsafe acquired marker DEL"
                    ),
                )
            return True
        dedup_idx = len(new_capacities) * 2
        if delete_acquired_marker_key is not None:
            _validate_delete_result(
                results[dedup_idx + 1],
                context="SyncRedisBackend._set_capacities_unsafe acquired marker DEL",
            )
        if _validate_set_nx_result(
            results[dedup_idx],
            context="SyncRedisBackend._set_capacities_unsafe refund dedup SET NX",
        ):
            return True
        if refund_dedup_reservation_id is not None:
            self._warn_refund_dedup_duplicate(refund_dedup_reservation_id)
        return False

    def _commit_refund_with_acquire_marker_unsafe(  # noqa: PLR0913
        self,
        new_capacities: Capacities,
        *,
        current_time: float,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket],
        acquired_marker_key: str,
        acquired_marker_value: str,
        refund_dedup_key: str,
        reservation_id: str | None = None,
        reservation_model_family: str | None = None,
    ) -> None:
        for bucket in buckets:
            _debug_event(
                _refund_logger,
                "redis_refund_marker_getdel",
                reservation_id=reservation_id,
                bucket_id=(bucket.usage_metric, int(bucket.per_seconds)),
            )
        _debug_event(
            _refund_logger,
            "redis_refund_dedup_write",
            reservation_id=reservation_id,
            bucket_id=None,
        )
        keys: list[str] = [acquired_marker_key, refund_dedup_key]
        args: list[str | bytes | int | float] = [
            acquired_marker_value,
            self._refund_dedup_ttl_seconds,
        ]
        for (usage_metric, per_seconds), amount in new_capacities.items():
            matching_bucket = self._find_bucket(
                buckets,
                usage_metric,
                per_seconds,
            )
            if matching_bucket is None:
                raise ValueError(f"Bucket '{usage_metric}/{per_seconds}s' not found")
            if not math.isfinite(float(amount)):
                raise ValueError(f"capacity must be finite (got {amount!r})")
            normalized_amount = amount
            if normalized_amount == 0.0:
                normalized_amount = 0.0
            keys.extend(
                [
                    matching_bucket._last_checked_key,  # noqa: SLF001
                    matching_bucket._capacity_key,  # noqa: SLF001
                ]
            )
            args.extend(
                [
                    current_time,
                    normalized_amount,
                    matching_bucket._bucket_ttl_seconds,  # noqa: SLF001
                ]
            )
        try:
            result = self._redis.eval(
                _REFUND_WITH_MARKER_SCRIPT,
                len(keys),
                *keys,
                *args,
            )
        except redis.exceptions.RedisError:
            if self._refund_tombstone_exists(
                refund_dedup_key,
                reservation_id=reservation_id,
            ):
                return
            raise
        status = _decode_redis_script_status(
            result, context="SyncRedisBackend refund marker script"
        )
        if status == "ok":
            return
        if status == "duplicate_refund":
            raise DuplicateRefundError(
                "reservation already refunded",
                reason="already_refunded",
                reservation_id=reservation_id,
                model_family=reservation_model_family,
            )
        if status == "unknown_reservation":
            raise _mark_unknown_reservation_forget_in_flight(
                UnknownReservationError(
                    "reservation was never acquired by this backend",
                    reservation_id=reservation_id,
                    model_family=reservation_model_family,
                )
            )
        if status == "marker_mismatch":
            raise UnknownReservationError(
                "reservation was never acquired by this backend",
                reservation_id=reservation_id,
                model_family=reservation_model_family,
            )
        raise RedisScriptResultError(
            "Redis refund marker script returned unknown status "
            f"{_safe_redis_value_repr(status)}"
        )

    def _bucket_ids(
        self,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
    ) -> frozenset[tuple[str, int]]:
        target_buckets = self.sorted_buckets if buckets is None else buckets
        return frozenset(
            (bucket.usage_metric, int(bucket.per_seconds)) for bucket in target_buckets
        )

    def _normalize_check_result(
        self,
        result: tuple[bool, Capacities, Capacities, float | None]
        | tuple[
            bool,
            Capacities,
            Capacities,
            float | None,
            tuple[SyncRedisBucket, ...],
        ]
        | tuple[
            bool,
            Capacities,
            Capacities,
            float | None,
            float | None,
            tuple[SyncRedisBucket, ...],
        ],
    ) -> tuple[
        bool,
        Capacities,
        Capacities,
        float | None,
        float | None,
        tuple[SyncRedisBucket, ...],
    ]:
        if len(result) == 4:  # noqa: PLR2004
            available, preconsumption, postconsumption, consumed_monotonic = result
            return (
                available,
                preconsumption,
                postconsumption,
                consumed_monotonic,
                None,
                self._snapshot_buckets(),
            )
        if len(result) == 5:  # noqa: PLR2004
            available, preconsumption, postconsumption, consumed_monotonic, buckets = (
                result
            )
            return (
                available,
                preconsumption,
                postconsumption,
                consumed_monotonic,
                None,
                tuple(buckets),
            )
        if len(result) == 6:  # noqa: PLR2004
            (
                available,
                preconsumption,
                postconsumption,
                consumed_monotonic,
                consumed_at_seconds,
                buckets,
            ) = result
            return (
                available,
                preconsumption,
                postconsumption,
                consumed_monotonic,
                consumed_at_seconds,
                tuple(buckets),
            )
        raise RuntimeError(
            "_check_and_consume_capacity() must return 4, 5, or 6 values",
        )

    def _compute_sleep_for_wait(
        self,
        usage: FrozenUsage,
        preconsumption: Capacities,
        *,
        buckets: tuple[SyncRedisBucket, ...],
    ) -> float:
        if buckets:
            return self._compute_sleep(usage, preconsumption, buckets=buckets)
        return self._compute_sleep(usage, preconsumption)

    def _check_and_consume_capacity(
        self,
        usage_: FrozenUsage,
        *,
        lock_blocking_timeout: float | None = None,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> tuple[
        bool,
        Capacities,
        Capacities,
        float | None,
        float | None,
        tuple[SyncRedisBucket, ...],
    ]:
        """Check if there's enough capacity and consume it if available."""
        usage: FrozenUsage = frozendict(
            {metric: float(amount) for metric, amount in usage_.items()},
        )
        buckets = self._snapshot_buckets()
        preconsumption_capacities: Capacities = frozendict()
        # Empty on the failure path; callers only read postconsumption on success.
        postconsumption_capacities: Capacities = frozendict()
        consumed_monotonic: float | None = None
        consumed_at_seconds: float | None = None
        current_time: float = 0.0
        fresh_start_buckets: list[SyncRedisBucket] = []
        consumed = False
        try:
            with self._lock(
                timeout=LOCK_TIMEOUT_SECONDS,
                blocking_timeout=lock_blocking_timeout,
                buckets=buckets,
                reservation_id=reservation_id,
            ) as lock_stack:
                # Pipeline is reused: _get_capacities_unsafe executes it (clearing
                # the command buffer), then _set_capacities_unsafe adds new commands
                # and executes again.  Safe because redis-py clears the buffer on execute().
                pipeline = self._redis.pipeline()
                current_time = sync_server_time(self._redis)

                preconsumption_capacities, fresh_start_buckets = (
                    self._get_capacities_unsafe(
                        pipeline=pipeline,
                        current_time=current_time,
                        buckets=buckets,
                    )
                )
                active_metric_names = {
                    metric for metric, _ in preconsumption_capacities
                }
                self._ensure_usage_metrics_are_active(usage, active_metric_names)

                # Fail fast: if usage exceeds any bucket's max_capacity, it can
                # never be satisfied (capacity is capped at max_capacity).
                for usage_metric_name, usage_amount in usage.items():
                    for bucket in buckets:
                        if bucket.usage_metric != usage_metric_name:
                            continue
                        # Uses cached max_capacity — refreshed from the pipeline
                        # result by update_max_capacity_from_result() inside
                        # _get_capacities_unsafe just above.
                        if usage_amount > bucket.max_capacity:
                            raise ValueError(  # noqa: TRY301
                                f"Usage value for {usage_metric_name} ({usage_amount}) "
                                f"exceeds bucket max capacity ({bucket.max_capacity})",
                            )

                for usage_metric_name, usage_amount in usage.items():
                    for (
                        capacity_metric_name,
                        _,
                    ), capacity_amount in preconsumption_capacities.items():
                        if usage_metric_name != capacity_metric_name:
                            continue
                        if usage_amount > capacity_amount:
                            self._fresh_start_buckets_callback(fresh_start_buckets)
                            return (
                                False,
                                preconsumption_capacities,
                                postconsumption_capacities,
                                consumed_monotonic,
                                consumed_at_seconds,
                                buckets,
                            )

                postconsumption_dict = dict(preconsumption_capacities)
                for (
                    capacity_metric_name,
                    per_seconds,
                ), capacity_amount in preconsumption_capacities.items():
                    matching_usage_amount = usage.get(capacity_metric_name)
                    if matching_usage_amount is None:
                        continue
                    postconsumption_dict[(capacity_metric_name, per_seconds)] = (
                        capacity_amount - matching_usage_amount
                    )
                postconsumption_capacities = frozendict(postconsumption_dict)
                consumed_monotonic = time.monotonic()
                # Extend lock TTL before write to guard against GC-pause
                # induced lost-update races; see _extend_locks.
                self._extend_locks(lock_stack, reservation_id=reservation_id)
                # `allow_negative=False` is correct here only because the
                # `usage_amount > capacity_amount` gate above guarantees
                # post = pre - usage >= 0. Keep this branch in sync with that
                # gate; consume_capacity (speedometer) writes with
                # `allow_negative=True` because it has no such guarantee.
                self._set_capacities_unsafe(
                    postconsumption_capacities,
                    pipeline=pipeline,
                    current_time=current_time,
                    allow_negative=False,
                    buckets=buckets,
                    acquired_marker_key=self._acquired_marker_key(reservation_id),
                    acquired_marker_value=(
                        redis_acquired_marker_value(
                            reservation_id=reservation_id,
                            model_family=self._limit_config.get_model_family(),
                            bucket_ids=self._bucket_ids(buckets),
                            usage=usage,
                        )
                        if reservation_id is not None
                        else None
                    ),
                    acquired_marker_ttl_ms=(
                        _reservation_lifetime_ttl_ms(reservation_lifetime_seconds)
                        if reservation_id is not None
                        else None
                    ),
                    reservation_id=reservation_id,
                )
                consumed = True
                consumed_at_seconds = current_time
            self._fresh_start_buckets_callback(fresh_start_buckets)
            if self._callbacks and self._callbacks.on_capacity_consumed:
                self._invoke_callback_safe(
                    self._callbacks.on_capacity_consumed,
                    callback_slot="on_capacity_consumed",
                    model_family=self._limit_config.get_model_family(),
                    preconsumption_capacities=preconsumption_capacities,
                    postconsumption_capacities=postconsumption_capacities,
                    usage=usage,
                    current_time=current_time,
                )
        except BaseException:
            # KI/SystemExit are the sync analogue of asyncio.CancelledError:
            # they can interrupt mid-statement and need the same best-effort
            # refund to avoid leaking capacity. Do not narrow to Exception.
            if consumed:
                try:  # noqa: SIM105
                    self._refund_cancelled_consumption(
                        usage,
                        buckets=buckets,
                        acquired_marker_key=self._acquired_marker_key(reservation_id),
                    )
                except BaseException:  # noqa: BLE001, S110
                    # Best-effort: sync interrupts mirror async cancellation
                    # cleanup. Swallow so the original interrupt propagates.
                    pass
            raise
        return (
            True,
            preconsumption_capacities,
            postconsumption_capacities,
            consumed_monotonic,
            consumed_at_seconds,
            buckets,
        )

    def consume_capacity(
        self,
        usage: FrozenUsage,
        *,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        """
        Consume capacity unconditionally.

        Capacity may go negative by design (speedometer pattern); this tracks
        overshoot rather than blocking.
        """
        buckets = self._snapshot_buckets()
        validate_backend_usage(usage, self._validation_metric_names(buckets))
        usage = frozendict(
            {metric: float(amount) for metric, amount in usage.items()},
        )
        preconsumption_capacities: Capacities = frozendict()
        postconsumption_capacities: Capacities = frozendict()
        current_time: float = 0.0
        fresh_start_buckets: list[SyncRedisBucket] = []
        with self._lock(
            timeout=LOCK_TIMEOUT_SECONDS,
            buckets=buckets,
            reservation_id=reservation_id,
        ) as lock_stack:
            pipeline = self._redis.pipeline()
            current_time = sync_server_time(self._redis)

            preconsumption_capacities, fresh_start_buckets = (
                self._get_capacities_unsafe(
                    pipeline=pipeline,
                    current_time=current_time,
                    buckets=buckets,
                )
            )
            active_metric_names = {metric for metric, _ in preconsumption_capacities}
            self._ensure_usage_metrics_are_active(usage, active_metric_names)

            for usage_metric_name, usage_amount in usage.items():
                for bucket in buckets:
                    if bucket.usage_metric != usage_metric_name:
                        continue
                    if usage_amount > bucket.max_capacity:
                        message = (
                            f"record_usage value for {usage_metric_name} "
                            f"({usage_amount}) exceeds bucket max capacity "
                            f"({bucket.max_capacity}). Capacity will go deeply "
                            "negative."
                        )
                        warnings.warn(message, RuntimeWarning, stacklevel=2)
                        _acquire_logger.warning(message)

            max_cap = {(b.usage_metric, b.per_seconds): b.max_capacity for b in buckets}
            postconsumption_dict = dict(preconsumption_capacities)
            for (
                capacity_metric_name,
                per_seconds,
            ), capacity_amount in preconsumption_capacities.items():
                matching_usage_amount = usage.get(capacity_metric_name)
                if matching_usage_amount is None:
                    continue
                postconsumption_dict[(capacity_metric_name, per_seconds)] = max(
                    capacity_amount - matching_usage_amount,
                    -max_cap[(capacity_metric_name, per_seconds)],
                )
            postconsumption_capacities = frozendict(postconsumption_dict)
            # Extend lock TTL before write; see _extend_locks.
            self._extend_locks(lock_stack, reservation_id=reservation_id)
            self._set_capacities_unsafe(
                postconsumption_capacities,
                pipeline=pipeline,
                current_time=current_time,
                allow_negative=True,
                buckets=buckets,
                acquired_marker_key=self._acquired_marker_key(reservation_id),
                acquired_marker_value=(
                    redis_acquired_marker_value(
                        reservation_id=reservation_id,
                        model_family=self._limit_config.get_model_family(),
                        bucket_ids=self._bucket_ids(buckets),
                        usage=usage,
                    )
                    if reservation_id is not None
                    else None
                ),
                acquired_marker_ttl_ms=(
                    _reservation_lifetime_ttl_ms(reservation_lifetime_seconds)
                    if reservation_id is not None
                    else None
                ),
                reservation_id=reservation_id,
            )
        self._fresh_start_buckets_callback(fresh_start_buckets)
        if self._callbacks and self._callbacks.on_capacity_consumed:
            self._invoke_callback_safe(
                self._callbacks.on_capacity_consumed,
                callback_slot="on_capacity_consumed",
                model_family=self._limit_config.get_model_family(),
                preconsumption_capacities=preconsumption_capacities,
                postconsumption_capacities=postconsumption_capacities,
                usage=usage,
                current_time=current_time,
            )
        return current_time

    def wait_for_capacity(  # noqa: PLR0915
        self,
        usage: FrozenUsage,
        *,
        timeout: float | None = None,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        """Wait until all buckets have the required capacity."""
        validate_backend_usage(usage, self._validation_metric_names())
        timeout = validate_timeout(timeout)
        usage = frozendict({metric: float(amount) for metric, amount in usage.items()})
        deadline = None if timeout is None else time.monotonic() + timeout
        waiter_key = reservation_id or f"redis:{uuid.uuid4().hex}"
        waiter_registered = False
        has_waited = False
        wait_started_at: float | None = None
        wait_start_callback_overhead = 0.0
        try:
            while True:
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                try:
                    (
                        available,
                        preconsumption,
                        postconsumption,
                        consumed_monotonic,
                        consumed_at_seconds,
                        buckets,
                    ) = self._normalize_check_result(
                        self._check_and_consume_capacity(
                            usage,
                            lock_blocking_timeout=remaining,
                            reservation_id=reservation_id,
                            reservation_lifetime_seconds=reservation_lifetime_seconds,
                        )
                    )
                except redis.exceptions.LockError as exc:
                    # When deadline is None (no caller timeout), the configured
                    # Redis lock blocking timeout still bounds lock polling.
                    # Propagating raw LockError preserves that distinction from
                    # a caller-level wait timeout.
                    if deadline is None:  # pragma: no cover
                        raise
                    raise self._capacity_timeout_error(usage, frozendict()) from exc
                if available:
                    try:
                        if has_waited:
                            wait_time_s = max(
                                0.0,
                                (consumed_monotonic or time.monotonic())
                                - (
                                    wait_started_at
                                    or (consumed_monotonic or time.monotonic())
                                )
                                - wait_start_callback_overhead,
                            )
                            if (
                                self._callbacks
                                and self._callbacks.after_wait_end_consumption
                            ):
                                self._invoke_callback_safe(
                                    self._callbacks.after_wait_end_consumption,
                                    callback_slot="after_wait_end_consumption",
                                    model_family=self._limit_config.get_model_family(),
                                    preconsumption_capacities=preconsumption,
                                    postconsumption_capacities=postconsumption,
                                    usage=frozendict(usage),
                                    wait_time_s=wait_time_s,
                                    **current_limiter_callback_context(),
                                )
                    except BaseException:
                        # KI/SystemExit are the sync analogue of asyncio.CancelledError:
                        # they can interrupt mid-statement and need the same best-effort
                        # refund to avoid leaking capacity. Do not narrow to Exception.
                        try:  # noqa: SIM105
                            self._refund_cancelled_consumption(
                                usage,
                                buckets=buckets,
                                acquired_marker_key=self._acquired_marker_key(
                                    reservation_id
                                ),
                            )
                        except BaseException:  # noqa: BLE001, S110
                            # Best-effort: sync interrupts mirror async cancellation
                            # cleanup. Swallow so the original interrupt propagates.
                            pass
                        raise
                    return consumed_at_seconds

                with self._local_condition:
                    self._upsert_diagnostic_waiter_locked(
                        waiter_key,
                        reservation_id=reservation_id,
                        usage=usage,
                        capacities=dict(preconsumption),
                        buckets=buckets,
                        deadline=deadline,
                        wait_started_at=wait_started_at,
                    )
                    waiter_registered = True

                if deadline is not None and time.monotonic() >= deadline:
                    raise self._capacity_timeout_error(
                        usage,
                        preconsumption,
                        buckets=buckets,
                    )

                if not has_waited:
                    has_waited = True
                    wait_started_at = time.monotonic()
                    if self._callbacks and self._callbacks.on_wait_start:
                        callback_started = time.monotonic()
                        if deadline is not None and callback_started >= deadline:
                            raise self._capacity_timeout_error(
                                usage,
                                preconsumption,
                                buckets=buckets,
                            )
                        self._invoke_callback_safe(
                            self._callbacks.on_wait_start,
                            callback_slot="on_wait_start",
                            model_family=self._limit_config.get_model_family(),
                            preconsumption_capacities=preconsumption,
                            usage=usage,
                            **current_limiter_callback_context(),
                        )
                        wait_start_callback_overhead += (
                            time.monotonic() - callback_started
                        )
                        if deadline is not None and time.monotonic() >= deadline:
                            raise self._capacity_timeout_error(
                                usage,
                                preconsumption,
                                buckets=buckets,
                            )

                computed = self._compute_sleep_for_wait(
                    usage,
                    preconsumption,
                    buckets=buckets,
                )
                effective = min(computed, self.MAX_CROSS_WORKER_POLL)
                if deadline is not None:
                    effective = min(effective, max(0, deadline - time.monotonic()))
                with self._local_condition:
                    self._local_condition.wait(timeout=max(0.001, effective))
        finally:
            if waiter_registered:
                with self._local_condition:
                    self._diagnostic_waiters.pop(waiter_key, None)

    def _compute_sleep(
        self,
        usage: FrozenUsage,
        preconsumption: Capacities,
        *,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
    ) -> float:
        """Compute max wait across all buckets based on deficit / rate."""
        max_wait = 0.0
        target_buckets = self.sorted_buckets if buckets is None else buckets
        for (metric, per_seconds), current_cap in preconsumption.items():
            if metric not in usage:
                continue
            needed = float(usage[metric])
            deficit = needed - current_cap
            if deficit <= 0:
                continue
            bucket = self._find_bucket(
                target_buckets,
                metric,
                per_seconds,
            )
            if bucket is None:
                raise ValueError(
                    f"No bucket found for metric='{metric}', per_seconds={per_seconds}"
                )
            rate_per_sec = bucket._rate_per_sec  # noqa: SLF001
            if not math.isfinite(rate_per_sec) or rate_per_sec <= 0:
                raise ValueError(
                    "Bucket rate is non-positive/non-finite — likely a "
                    "misconfigured max_capacity"
                )
            wait = deficit / rate_per_sec
            max_wait = max(max_wait, wait)
        return max_wait if max_wait > 0 else self._sleep_interval

    def _capacity_timeout_error(
        self,
        usage: FrozenUsage,
        preconsumption: Capacities,
        *,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
        computed_sleep: float | None = None,
    ) -> TimeoutError:
        bottleneck_id: tuple[str, int] | None = None
        available: float | None = None
        requested: float | None = None
        largest_deficit = 0.0
        for bucket_id, cap_amount in preconsumption.items():
            metric, _ = bucket_id
            usage_amount = usage.get(metric)
            if usage_amount is None:
                continue
            deficit = float(usage_amount) - float(cap_amount)
            if deficit > largest_deficit:
                largest_deficit = deficit
                bottleneck_id = bucket_id
                available = float(cap_amount)
                requested = float(usage_amount)
        if computed_sleep is None:
            with contextlib.suppress(Exception):
                if buckets is None:
                    computed_sleep = self._compute_sleep(usage, preconsumption)
                else:
                    computed_sleep = self._compute_sleep_for_wait(
                        usage,
                        preconsumption,
                        buckets=tuple(buckets),
                    )
        return TimeoutError(
            "Timed out waiting for capacity "
            f"(bottleneck={bottleneck_id}, available={available}, "
            f"requested={requested}, computed_sleep={computed_sleep})"
        )

    def _upsert_diagnostic_waiter_locked(  # noqa: PLR0913
        self,
        waiter_key: str,
        *,
        reservation_id: str | None,
        usage: FrozenUsage,
        capacities: dict[tuple[str, int], float],
        buckets: tuple[SyncRedisBucket, ...],
        deadline: float | None,
        wait_started_at: float | None,
    ) -> None:
        started = wait_started_at or time.monotonic()
        if not hasattr(self, "_diagnostic_waiters"):
            self._diagnostic_waiters = {}
        limits = {
            (bucket.usage_metric, int(bucket.per_seconds)): BackendBucketLimit(
                effective_max_capacity=bucket.max_capacity,
                refill_rate_per_second=bucket.max_capacity / int(bucket.per_seconds),
            )
            for bucket in buckets
        }
        self._diagnostic_waiters[waiter_key] = DiagnosticWaiterState(
            waiter_id=waiter_key,
            reservation_id=reservation_id,
            model_family=self._limit_config.get_model_family(),
            model=None,
            request_id=None,
            state="waiting_for_capacity",
            usage=usage,
            wait_started_monotonic=started,
            timeout_deadline_monotonic=deadline,
            blocked_buckets=wait_bucket_diagnostics(
                model_family=self._limit_config.get_model_family(),
                usage=usage,
                capacities=capacities,
                limits=limits,
            ),
        )

    def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        self.refund_capacity_for_buckets(
            reserved_usage,
            actual_usage,
            bucket_ids=self._bucket_ids(),
        )

    def refund_capacity_for_buckets(  # noqa: PLR0913, PLR0915
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        *,
        bucket_ids: set[tuple[str, int]] | frozenset[tuple[str, int]] | None = None,
        reservation_id: str | None = None,
        reservation_model_family: str | None = None,
        reservation_bucket_ids: set[tuple[str, int]]
        | frozenset[tuple[str, int]]
        | None = None,
        reservation_reserved_usage: FrozenUsage | None = None,
    ) -> bool:
        """Refund unused capacity back to the rate limiter based on actual usage."""
        buckets = self._snapshot_buckets()
        backend_bucket_ids = self._bucket_ids(buckets)
        refund_bucket_ids = (
            backend_bucket_ids if bucket_ids is None else frozenset(bucket_ids)
        )
        validate_backend_refund_usage_for_bucket_ids(
            reserved_usage,
            actual_usage,
            refund_bucket_ids,
            backend_bucket_ids,
        )
        expected_marker_value: str | None = None
        acquired_marker_key = self._acquired_marker_key(reservation_id)
        refund_dedup_key = self._refund_dedup_key(reservation_id)
        if reservation_id is not None:
            if acquired_marker_key is None or refund_dedup_key is None:
                raise UnknownReservationError(
                    "reservation was never acquired by this backend",
                    reservation_id=reservation_id,
                    model_family=reservation_model_family,
                )
            (
                marker_model_family,
                marker_bucket_ids,
                marker_reserved_usage,
            ) = _require_acquire_marker_metadata(
                reservation_model_family=reservation_model_family,
                reservation_bucket_ids=reservation_bucket_ids,
                reservation_reserved_usage=reservation_reserved_usage,
            )
            _validate_acquire_marker_refund_scope(
                reserved_usage=reserved_usage,
                refund_bucket_ids=refund_bucket_ids,
                marker_bucket_ids=marker_bucket_ids,
                marker_reserved_usage=marker_reserved_usage,
            )
            expected_marker_value = redis_acquired_marker_value(
                reservation_id=reservation_id,
                model_family=marker_model_family,
                bucket_ids=marker_bucket_ids,
                usage=marker_reserved_usage,
            )
        if not refund_bucket_ids:
            if (
                reservation_id is not None
                and acquired_marker_key is not None
                and refund_dedup_key is not None
                and expected_marker_value is not None
            ):
                self._commit_refund_with_acquire_marker_unsafe(
                    frozendict(),
                    current_time=sync_server_time(self._redis),
                    buckets=(),
                    acquired_marker_key=acquired_marker_key,
                    acquired_marker_value=expected_marker_value,
                    refund_dedup_key=refund_dedup_key,
                    reservation_id=reservation_id,
                    reservation_model_family=reservation_model_family,
                )
            return True
        # Calculate how much to refund for each metric
        refund_usage_: dict[str, float] = {}
        for metric, reserved_amount in reserved_usage.items():
            # Key guaranteed to exist: SyncRateLimiter.refund_capacity() calls
            # validate_refund_usage() before reaching the backend.
            actual_amount = actual_usage[metric]
            refund_amount = float(reserved_amount) - float(actual_amount)

            # Check for overuse and log a warning
            if refund_amount < 0:
                message = (
                    f"Actual usage ({actual_amount}) for {metric} exceeds "
                    f"reserved usage ({reserved_amount}). Applying negative refund."
                )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                _refund_logger.warning(message)

            # Include both positive and negative refunds
            refund_usage_[metric] = refund_amount
        refund_usage: frozendict[str, float] = frozendict(refund_usage_)

        fresh_start_buckets: list[SyncRedisBucket] = []
        with self._lock(
            timeout=LOCK_TIMEOUT_SECONDS,
            buckets=buckets,
            reservation_id=reservation_id,
        ) as lock_stack:
            pipeline = self._redis.pipeline()
            current_time = sync_server_time(self._redis)

            # Get current capacities (which already account for time-based refill)
            prerefund_capacities, fresh_start_buckets = self._get_capacities_unsafe(
                pipeline=pipeline,
                current_time=current_time,
                buckets=buckets,
            )

            # Apply refund amounts to current capacity
            updated_capacities_: dict[tuple[str, int], float] = dict(
                prerefund_capacities,
            )
            for (
                capability_usage_metric,
                per_seconds,
            ) in prerefund_capacities:
                bucket_id = (capability_usage_metric, int(per_seconds))
                if bucket_id not in refund_bucket_ids:
                    continue
                matching_refund_amount = refund_usage.get(capability_usage_metric)
                if matching_refund_amount is None:
                    continue
                bucket = self._find_bucket(
                    buckets,
                    capability_usage_metric,
                    per_seconds,
                )
                if bucket is None:
                    raise ValueError(
                        f"Bucket '{capability_usage_metric}/{per_seconds}s' not found",
                    )

                # Apply refund (positive or negative), cap at max_capacity.
                # Negative capacity is preserved so the token-bucket refill
                # handles recovery — clamping to 0 here would erase debt
                # from the record_usage (speedometer) path.
                refund_amount = max(matching_refund_amount, -bucket.max_capacity)
                updated_capacities_[(capability_usage_metric, int(per_seconds))] = max(
                    -bucket.max_capacity,
                    min(
                        updated_capacities_[(capability_usage_metric, int(per_seconds))]
                        + refund_amount,
                        bucket.max_capacity,  # cached — refreshed from pipeline result in _get_capacities_unsafe
                    ),
                )
            updated_capacities = frozendict(updated_capacities_)

            # Extend lock TTL before write; see _extend_locks.
            self._extend_locks(lock_stack, reservation_id=reservation_id)
            # Option B from FIX-42: defer the Redis tombstone until the same
            # pipeline execution as the capacity write. This prevents the
            # permanent lost-refund failure where SET NX succeeds and the later
            # bucket mutation fails; exact concurrent retries are serialized by
            # the bucket locks, while lock expiry/manual writers can still race.
            if reservation_id is None:
                self._set_capacities_unsafe(
                    frozendict(updated_capacities),
                    pipeline=pipeline,
                    current_time=current_time,
                    allow_negative=True,
                    buckets=buckets,
                    reservation_id=reservation_id,
                )
            else:
                assert acquired_marker_key is not None  # noqa: S101
                assert refund_dedup_key is not None  # noqa: S101
                assert expected_marker_value is not None  # noqa: S101
                self._commit_refund_with_acquire_marker_unsafe(
                    frozendict(updated_capacities),
                    current_time=current_time,
                    buckets=buckets,
                    acquired_marker_key=acquired_marker_key,
                    acquired_marker_value=expected_marker_value,
                    refund_dedup_key=refund_dedup_key,
                    reservation_id=reservation_id,
                    reservation_model_family=reservation_model_family,
                )
        with self._local_condition:
            self._local_condition.notify_all()
        self._fresh_start_buckets_callback(fresh_start_buckets)
        if self._callbacks and self._callbacks.on_capacity_refunded:
            self._invoke_callback_safe(
                self._callbacks.on_capacity_refunded,
                callback_slot="on_capacity_refunded",
                model_family=self._limit_config.get_model_family(),
                reserved_usage=reserved_usage,
                actual_usage=actual_usage,
                refunded_usage=refund_usage,
                prerefund_capacities=prerefund_capacities,
                postrefund_capacities=updated_capacities,
            )
        return True

    def _snapshot_bucket_state(self, bucket: SyncRedisBucket) -> None:
        """
        Freeze ``bucket`` in Redis at its accrued capacity under the CURRENT rate.

        Uncapped anchor (stored + elapsed*old_rate) preserves raw values above
        ``max_capacity`` — reads apply ``min(max_capacity, …)`` and a later
        cap raise can re-expose the hidden overflow.
        """
        refresh_max_capacity = getattr(
            bucket,
            "refresh_max_capacity_from_redis",
            None,
        )
        if callable(refresh_max_capacity):
            refresh_max_capacity()
        else:  # test fakes and compatible custom bucket shims
            bucket.get_max_capacity()
        current_time = sync_server_time(self._redis)
        pipeline = self._redis.pipeline()
        pipeline.get(bucket._last_checked_key)  # noqa: SLF001
        pipeline.get(bucket._capacity_key)  # noqa: SLF001
        pipeline.expire(bucket._last_checked_key, bucket._bucket_ttl_seconds)  # noqa: SLF001
        pipeline.expire(bucket._capacity_key, bucket._bucket_ttl_seconds)  # noqa: SLF001
        try:
            results = pipeline.execute()
        except redis.exceptions.ResponseError as exc:
            _raise_pipeline_response_error(
                "SyncRedisBackend._snapshot_bucket_state", exc
            )
        results = _validate_pipeline_results(
            results,
            context=f"SyncRedisBackend._snapshot_bucket_state({bucket.full_redis_key})",
            expected_count=4,
        )
        last_checked_raw, stored_raw = _normalize_bucket_state_pair(
            results[SyncRedisBucket.PIPELINE_LAST_CHECKED_OFFSET],
            results[SyncRedisBucket.PIPELINE_CAPACITY_OFFSET],
            context=f"SyncRedisBackend._snapshot_bucket_state({bucket.full_redis_key})",
            current_time=current_time,
        )
        _validate_expire_result(
            results[2],
            context=(
                f"SyncRedisBackend._snapshot_bucket_state({bucket.full_redis_key}) "
                "last_checked TTL"
            ),
        )
        _validate_expire_result(
            results[3],
            context=(
                f"SyncRedisBackend._snapshot_bucket_state({bucket.full_redis_key}) "
                "capacity TTL"
            ),
        )
        if last_checked_raw is None or stored_raw is None:
            # Partial state (one None) is treated the same as full absence:
            # the bucket will start fresh on the next acquire. This is the
            # correct fallback — anchoring with incomplete data would produce
            # a wrong capacity value.
            _logger.warning(
                "Bucket %s: snapshot skipped due to missing Redis state "
                "(last_checked=%r, capacity=%r).",
                bucket.full_redis_key,
                last_checked_raw,
                stored_raw,
            )
            return
        try:
            last_checked = float(last_checked_raw)
            stored = float(stored_raw)
        except (TypeError, ValueError) as parse_error:
            _logger.warning(
                "Stale Redis bucket state at %r: %r; "
                "snapshot skipped due to unparseable Redis state "
                "(last_checked=%s, capacity=%s).",
                bucket.full_redis_key,
                parse_error,
                _safe_redis_value_repr(last_checked_raw),
                _safe_redis_value_repr(stored_raw),
            )
            return
        if not (math.isfinite(last_checked) and math.isfinite(stored)):
            _logger.warning(
                "Bucket %s: snapshot skipped due to non-finite Redis state "
                "(last_checked=%s, capacity=%s).",
                bucket.full_redis_key,
                _safe_redis_value_repr(last_checked_raw),
                _safe_redis_value_repr(stored_raw),
            )
            return
        time_passed = current_time - last_checked
        if time_passed < 0:
            time_passed = 0.0
        anchored = stored + time_passed * bucket._rate_per_sec  # noqa: SLF001
        bucket.set_capacity(
            anchored,
            current_time=current_time,
            allow_negative=True,
        )

    def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        buckets = self._snapshot_buckets()
        bucket = self._find_bucket(
            buckets,
            metric,
            per_seconds,
        )
        if bucket is None:
            raise ValueError(f"Bucket '{metric}/{per_seconds}s' not found")
        with self._lock(timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets) as lock_stack:
            self._extend_locks(lock_stack)
            self._snapshot_bucket_state(bucket)
            bucket.set_max_capacity(value)
        with self._local_condition:
            self._local_condition.notify_all()

    def apply_configured_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        buckets = self._snapshot_buckets()
        bucket = self._find_bucket(
            buckets,
            metric,
            per_seconds,
        )
        if bucket is None:
            raise ValueError(f"Bucket '{metric}/{per_seconds}s' not found")
        with self._lock(timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets) as lock_stack:
            self._extend_locks(lock_stack)
            self._snapshot_bucket_state(bucket)
            bucket.clear_max_capacity_override()
            bucket.set_configured_max_capacity(value)
        with self._local_condition:
            self._local_condition.notify_all()

    def prepare_reconfigured_backend(
        self,
        new_backend: SyncRateLimiterBackend,
        cfg: PerModelConfig,
    ) -> SyncRateLimiterBackend:
        if not isinstance(new_backend, SyncRedisBackend):
            raise TypeError(
                "SyncRedisBackend can only reconfigure into another SyncRedisBackend"
            )

        current_buckets = self._snapshot_buckets()
        new_buckets = tuple(new_backend.sorted_buckets)
        reconfigure_buckets = self._combined_bucket_snapshot(
            current_buckets, new_buckets
        )
        removed_buckets = tuple(
            bucket
            for bucket in current_buckets
            if self._find_bucket(
                new_buckets,
                bucket.usage_metric,
                int(bucket.per_seconds),
            )
            is None
        )

        with self._lock(
            timeout=LOCK_TIMEOUT_SECONDS, buckets=reconfigure_buckets
        ) as lock_stack:
            for bucket in removed_buckets:
                # Snapshot first so the frozen capacity/last_checked in Redis
                # reflect the bucket's state at the moment of removal. Without
                # this, a later re-add with a different rate would retroactively
                # integrate the new rate across the remove→re-add gap (and any
                # time preceding it). Then clear the runtime override so the
                # re-add starts from the callable config's static quota again.
                self._extend_locks(lock_stack)
                self._snapshot_bucket_state(bucket)
                bucket.clear_max_capacity_override()
            for quota in cfg.quotas:
                matching_bucket = self._find_bucket(
                    new_buckets,
                    quota.metric,
                    int(quota.per_seconds),
                )
                if matching_bucket is None:  # pragma: no cover
                    raise ValueError(
                        f"Bucket '{quota.metric}/{quota.per_seconds}s' not found",
                    )
                current_bucket = self._find_bucket(
                    current_buckets,
                    quota.metric,
                    int(quota.per_seconds),
                )
                # Snapshot before any mutation could change the effective rate.
                # Prefer current_bucket (its _max_capacity_default matches the
                # stored override payload, so the override is accepted and
                # yields the true active rate).
                self._extend_locks(lock_stack)
                self._snapshot_bucket_state(
                    current_bucket if current_bucket is not None else matching_bucket
                )
                if current_bucket is not None and float(
                    current_bucket.configured_max_capacity
                ) != float(quota.limit):
                    matching_bucket.clear_max_capacity_override()
                matching_bucket.set_configured_max_capacity(float(quota.limit))

            self._extend_locks(lock_stack)
            self.install_reconfigured_state(
                buckets=list(new_buckets),
                cfg=cfg,
            )
        with self._local_condition:
            self._local_condition.notify_all()
        return self

    def install_reconfigured_state(
        self,
        *,
        buckets: list[SyncRedisBucket],
        cfg: PerModelConfig,
    ) -> None:
        """Install reconfigured sync Redis buckets after client and key-prefix validation."""
        _ensure_buckets_match_backend(
            buckets,
            key_prefix=self._key_prefix,
            redis_client=self._redis,
        )
        self.sorted_buckets = sorted(buckets, key=lambda bucket: bucket.full_redis_key)
        self._usage_metric_names = {bucket.usage_metric for bucket in buckets}
        self._limit_config = cfg

    def _invoke_callback_safe(
        self,
        callback,
        *,
        callback_slot: str = "callback",
        **kwargs,
    ) -> None:
        """Fire a user callback; delegates to the shared critical-exception dispatcher."""
        safe_invoke_sync_callback(
            callback,
            critical=BACKEND_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter callback",
            callback_slot=callback_slot,
            **kwargs,
        )

    def _refund_cancelled_consumption(
        self,
        usage: FrozenUsage,
        *,
        buckets: tuple[SyncRedisBucket, ...] | list[SyncRedisBucket] | None = None,
        acquired_marker_key: str | None = None,
    ) -> None:
        """
        Refund capacity consumed before a BaseException hit callbacks.

        Fires no callbacks to avoid recursion and another interruption window.
        """
        target_buckets = self._snapshot_buckets() if buckets is None else tuple(buckets)
        with self._lock(
            timeout=LOCK_TIMEOUT_SECONDS,
            buckets=target_buckets,
        ) as lock_stack:
            pipeline = self._redis.pipeline()
            current_time = sync_server_time(self._redis)
            capacities, _ = self._get_capacities_unsafe(
                pipeline=pipeline,
                current_time=current_time,
                buckets=target_buckets,
            )
            refunded: dict[tuple[str, int], float] = dict(capacities)
            for (cap_metric, per_seconds), cap_amount in capacities.items():
                for usage_metric, usage_amount in usage.items():
                    if cap_metric != usage_metric:
                        continue
                    bucket = self._find_bucket(
                        target_buckets,
                        cap_metric,
                        per_seconds,
                    )
                    if bucket is None:  # pragma: no cover
                        raise ValueError(
                            f"Bucket '{cap_metric}/{per_seconds}s' not found",
                        )
                    refunded[(cap_metric, per_seconds)] = min(
                        cap_amount + usage_amount,
                        bucket.max_capacity,
                    )
            self._extend_locks(lock_stack)
            self._set_capacities_unsafe(
                frozendict(refunded),
                pipeline=pipeline,
                current_time=current_time,
                allow_negative=True,
                buckets=target_buckets,
                delete_acquired_marker_key=acquired_marker_key,
            )
        with self._local_condition:
            self._local_condition.notify_all()

    def _fresh_start_buckets_callback(
        self,
        fresh_start_buckets: list[SyncRedisBucket],
    ) -> None:
        if (
            fresh_start_buckets
            and self._callbacks
            and self._callbacks.on_missing_consumption_data
        ):
            for bucket in fresh_start_buckets:
                self._invoke_callback_safe(
                    self._callbacks.on_missing_consumption_data,
                    callback_slot="on_missing_consumption_data",
                    model_family=self._limit_config.get_model_family(),
                    usage_metric=bucket.usage_metric,
                    per_seconds=bucket.per_seconds,
                    missing_state_reason=(
                        bucket._missing_consumption_data_reason  # noqa: SLF001
                    ),
                    missing_state_keys=(
                        bucket._missing_consumption_data_missing_keys  # noqa: SLF001
                    ),
                    present_state_keys=(
                        bucket._missing_consumption_data_present_keys  # noqa: SLF001
                    ),
                )
