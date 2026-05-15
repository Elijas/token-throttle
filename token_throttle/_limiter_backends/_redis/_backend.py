import asyncio
import contextlib
import inspect
import logging
import math
import time
import typing
import uuid
import warnings
from contextlib import AsyncExitStack
from typing import ClassVar

try:
    import redis.asyncio
    import redis.asyncio.client
    import redis.exceptions
except ImportError as exc:
    raise ImportError(
        'The "redis" package is required for the Redis backend. '
        'Install it with: pip install "token-throttle[redis]"'
    ) from exc
from frozendict import frozendict

from token_throttle._exceptions import (
    DuplicateRefundError,
    UnknownReservationError,
    _mark_unknown_reservation_forget_in_flight,
)
from token_throttle._interfaces._callable_utils import (
    suppress_current_task_cancellation,
)
from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    RateLimiterBackend,
    RateLimiterBackendBuilderInterface,
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

from ._bucket import (
    RedisBucket,
    _normalize_bucket_state_pair,
    _raise_pipeline_response_error,
    _validate_pipeline_results,
)
from ._keys import (
    DEFAULT_REFUND_DEDUP_TTL_SECONDS,
    redis_acquired_marker_key,
    redis_acquired_marker_value,
    redis_refund_dedup_key,
    validate_redis_key_prefix,
    validate_refund_dedup_ttl_seconds,
)
from ._server_time import async_server_time
from ._ttl import (
    DEFAULT_BUCKET_TTL_SECONDS,
    resolve_max_reservation_lifetime_seconds_from_ttls,
    validate_redis_ttl_seconds,
    validate_reservation_lifetime_ttl_invariant,
)

_logger = logging.getLogger("token_throttle")

_CRITICAL_CALLBACK_EXCEPTION_TYPES = (
    asyncio.CancelledError,
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
)


def _callback_exception_group_contains_critical(exc: BaseException) -> bool:
    if not isinstance(exc, BaseExceptionGroup):
        return False
    critical, _non_critical = exc.split(_CRITICAL_CALLBACK_EXCEPTION_TYPES)
    return critical is not None


class CapacitiesGetterResult(typing.NamedTuple):
    capacities: Capacities
    fresh_start_buckets: list[RedisBucket]


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


def _raise_lock_timeout_error() -> typing.NoReturn:
    raise redis.exceptions.LockError("Unable to acquire lock within the time specified")


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


def _decode_redis_script_status(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value
    raise RuntimeError(f"Redis script returned unexpected status {value!r}")


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
        warnings.warn(
            "Redis connection_pool.max_connections is less than 10. "
            "This is likely too small for production token-throttle workloads; "
            "prefer a BlockingConnectionPool sized to at least "
            "max_concurrent_acquires plus Redis command headroom.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )


def _is_mock_object(value: object) -> bool:
    return type(value).__module__.startswith("unittest.mock")


def _ensure_bucket_matches_backend(
    bucket: RedisBucket,
    *,
    key_prefix: str,
    redis_client: redis.asyncio.Redis,
) -> None:
    bucket_key_prefix = getattr(bucket, "key_prefix", _MISSING)
    if bucket_key_prefix is not _MISSING and bucket_key_prefix != key_prefix:
        raise ValueError(
            "RedisBucket key_prefix must match RedisBackend key_prefix "
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
        "RedisBucket redis client must be the same object as RedisBackend redis "
        "or share the same connection_pool"
    )


def _ensure_buckets_match_backend(
    buckets: typing.Iterable[RedisBucket],
    *,
    key_prefix: str,
    redis_client: redis.asyncio.Redis,
) -> None:
    for bucket in buckets:
        _ensure_bucket_matches_backend(
            bucket,
            key_prefix=key_prefix,
            redis_client=redis_client,
        )


async def _shielded_lock_release(lock: redis.asyncio.lock.Lock) -> None:
    """Release a Redis lock, shielded so the round-trip finishes even under cancel."""
    await asyncio.shield(lock.release())


class _RedisLockStack(AsyncExitStack):
    """
    AsyncExitStack that also holds the acquired Lock objects.

    The parent ``_lock()`` needs to expose the concrete lock instances so
    callers can call ``extend()`` on them right before long-running writes,
    keeping the lock alive across GC pauses or connection stalls that
    otherwise could let the lock's Redis TTL expire mid-operation and
    allow two workers to both commit writes (lost-update race).
    """

    def __init__(self) -> None:
        super().__init__()
        self.locks: list[redis.asyncio.lock.Lock] = []


class RedisBackendBuilder(RateLimiterBackendBuilderInterface):
    """
    Build async Redis limiter backends.

    ``bucket_ttl_seconds`` controls the expiry refreshed on bucket state.
    ``override_ttl_seconds`` can use a distinct runtime max-capacity override
    expiry; when omitted, it inherits ``bucket_ttl_seconds``. The schema-version
    key is intentionally exempt from expiry because it is a long-lived registry.

    ``refund_dedup_ttl_seconds`` controls how long Redis remembers successful
    refund reservation ids for cross-process idempotency. The default is
    7 days.

    ``owns_redis_client`` defaults to ``False`` because Redis clients are often
    shared across limiters. Set it to ``True`` only when this builder is the
    lifecycle owner for ``redis_client``; then limiter ``aclose()`` cascades to
    ``redis_client.aclose()``.

    For bounded Redis deployments, prefer ``redis.asyncio.BlockingConnectionPool``
    and size ``max_connections`` to at least ``max_concurrent_acquires`` plus
    headroom for lock acquire/release, ``TIME``, and pipeline commands.
    """

    def __init__(  # noqa: PLR0913
        self,
        redis_client: redis.asyncio.Redis,
        *,
        key_prefix: str,
        sleep_interval: float | None = None,
        bucket_ttl_seconds: int = DEFAULT_BUCKET_TTL_SECONDS,
        override_ttl_seconds: int | None = None,
        refund_dedup_ttl_seconds: int = DEFAULT_REFUND_DEDUP_TTL_SECONDS,
        owns_redis_client: bool = False,
        lock_blocking_timeout_seconds: float = DEFAULT_LOCK_BLOCKING_TIMEOUT_SECONDS,
        lock_sleep_seconds: float = DEFAULT_LOCK_SLEEP_SECONDS,
    ) -> None:
        super().__init__()
        client_module = type(redis_client).__module__
        if client_module.startswith("redis.") and not isinstance(
            redis_client, redis.asyncio.Redis
        ):
            raise TypeError(
                "redis_client must be a redis.asyncio.Redis instance "
                f"(got {type(redis_client).__name__})"
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

    def resolve_max_reservation_lifetime_seconds(
        self,
        max_reservation_lifetime_seconds: float | None,
    ) -> float | None:
        return resolve_max_reservation_lifetime_seconds_from_ttls(
            max_reservation_lifetime_seconds=max_reservation_lifetime_seconds,
            bucket_ttl_seconds=self._bucket_ttl_seconds,
            refund_dedup_ttl_seconds=self._refund_dedup_ttl_seconds,
        )

    def validate_reservation_lifetime_seconds(
        self,
        max_reservation_lifetime_seconds: float | None,
    ) -> None:
        validate_reservation_lifetime_ttl_invariant(
            max_reservation_lifetime_seconds=max_reservation_lifetime_seconds,
            bucket_ttl_seconds=self._bucket_ttl_seconds,
            refund_dedup_ttl_seconds=self._refund_dedup_ttl_seconds,
        )

    async def aclose(self) -> None:
        if not self._owns_redis_client:
            return
        aclose = getattr(self._redis, "aclose", None)
        if callable(aclose):
            await aclose()
            return
        close = getattr(self._redis, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    def close(self) -> None:
        if not self._owns_redis_client:
            return
        close = getattr(self._redis, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                result.close()

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> "RateLimiterBackend":
        cfg = _revalidate_dto(cfg)
        if callbacks is not None:
            _revalidate_dto(callbacks)
        redis_buckets = []
        for quota in cfg.quotas:
            b = RedisBucket(
                quota=quota,
                limit_config=cfg,
                redis_client=self._redis,
                key_prefix=self._key_prefix,
                bucket_ttl_seconds=self._bucket_ttl_seconds,
                override_ttl_seconds=self._override_ttl_seconds,
            )
            redis_buckets.append(b)
        return RedisBackend(
            buckets=redis_buckets,
            redis=self._redis,
            key_prefix=self._key_prefix,
            refund_dedup_ttl_seconds=self._refund_dedup_ttl_seconds,
            sleep_interval=self._sleep_interval,
            lock_blocking_timeout_seconds=self._lock_blocking_timeout_seconds,
            lock_sleep_seconds=self._lock_sleep_seconds,
            callbacks=callbacks,
            limit_config=cfg,
        )


class RedisBackend(RateLimiterBackend):
    DEFAULT_SLEEP_INTERVAL: ClassVar[float] = 0.1
    # Cross-worker poll ceiling; intra-process wakeup is instant via _local_condition. Audited 2026-04.
    MAX_CROSS_WORKER_POLL: ClassVar[float] = 1.0

    def __init__(  # noqa: PLR0913
        self,
        buckets: list[RedisBucket],
        redis: redis.asyncio.Redis,
        limit_config: PerModelConfig,
        *,
        key_prefix: str,
        refund_dedup_ttl_seconds: int = DEFAULT_REFUND_DEDUP_TTL_SECONDS,
        sleep_interval: float | None = None,
        lock_blocking_timeout_seconds: float = DEFAULT_LOCK_BLOCKING_TIMEOUT_SECONDS,
        lock_sleep_seconds: float = DEFAULT_LOCK_SLEEP_SECONDS,
        callbacks: RateLimiterCallbacks | None = None,
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
        self._sleep_interval: float = (
            self.DEFAULT_SLEEP_INTERVAL
            if sleep_interval is None
            else validate_sleep_interval(sleep_interval)
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
        self._callbacks = callbacks
        self._limit_config = limit_config
        self._usage_metric_names: set[str] = {bucket.usage_metric for bucket in buckets}
        self._local_condition = asyncio.Condition()
        self._legacy_override_probe_complete = False

    def add_bucket(self, bucket: RedisBucket) -> None:
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

    async def _refund_dedup_exists(self, reservation_id: str | None) -> bool:
        key = self._refund_dedup_key(reservation_id)
        if key is None:
            return False
        return bool(await self._redis.exists(key))

    async def _commit_refund_dedup(self, reservation_id: str | None) -> bool:
        key = self._refund_dedup_key(reservation_id)
        if key is None:
            return True
        claimed = await self._redis.set(
            key,
            "1",
            ex=self._refund_dedup_ttl_seconds,
            nx=True,
        )
        if claimed:
            return True
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

    async def _claim_refund_dedup(self, reservation_id: str | None) -> bool:
        return await self._commit_refund_dedup(reservation_id)

    async def _acquire_marker_matches(
        self,
        acquired_marker_key: str,
        acquired_marker_value: str,
    ) -> bool:
        try:
            marker = await self._redis.get(acquired_marker_key)
        except redis.exceptions.RedisError:
            return False
        return _redis_value_matches(marker, acquired_marker_value)

    async def _refund_tombstone_exists(self, refund_dedup_key: str) -> bool:
        try:
            return bool(await self._redis.exists(refund_dedup_key))
        except redis.exceptions.RedisError:
            return False

    async def _probe_legacy_override_keys_once(
        self,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
    ) -> None:
        if self._legacy_override_probe_complete:
            return
        target_buckets = self.sorted_buckets if buckets is None else buckets
        for bucket in target_buckets:
            result = self._redis.get(bucket._legacy_max_capacity_key)  # noqa: SLF001
            if inspect.isawaitable(result):
                result = await result
            bucket.handle_legacy_max_capacity_key(result)
        self._legacy_override_probe_complete = True

    def _snapshot_buckets(self) -> tuple[RedisBucket, ...]:
        return tuple(getattr(self, "sorted_buckets", ()))

    async def _runtime_max_capacity_for_reconciliation(
        self,
        metric: str,
        per_seconds: int,
    ) -> float | None:
        bucket = self._find_bucket(self._snapshot_buckets(), metric, per_seconds)
        if bucket is None:
            return None
        return await bucket.refresh_max_capacity_from_redis()

    def _validation_metric_names(
        self,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
    ) -> set[str]:
        target_buckets = self._snapshot_buckets() if buckets is None else tuple(buckets)
        if target_buckets:
            return self._usage_metric_names_for(target_buckets)
        return set(getattr(self, "_usage_metric_names", set()))

    @staticmethod
    def _usage_metric_names_for(
        buckets: tuple[RedisBucket, ...] | list[RedisBucket],
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
        buckets: tuple[RedisBucket, ...] | list[RedisBucket],
        metric: str,
        per_seconds: int,
    ) -> RedisBucket | None:
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
        *bucket_groups: tuple[RedisBucket, ...] | list[RedisBucket],
    ) -> tuple[RedisBucket, ...]:
        buckets_by_key: dict[str, RedisBucket] = {}
        for bucket_group in bucket_groups:
            for bucket in bucket_group:
                buckets_by_key[bucket.full_redis_key] = bucket
        return tuple(
            sorted(buckets_by_key.values(), key=lambda bucket: bucket.full_redis_key)
        )

    async def _lock(
        self,
        *,
        timeout: float,
        blocking_timeout: float | None = None,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
    ) -> _RedisLockStack:
        """
        Acquire distributed Redis locks for a fixed bucket snapshot.

        Returns a ``_RedisLockStack`` that holds the acquired locks.  Use as::

            async with await self._lock(timeout=LOCK_TIMEOUT_SECONDS) as stack:
                ...  # all bucket locks held here
                await self._extend_locks(stack)  # before long writes

        If acquiring lock N fails, locks 0..N-1 are released immediately
        (via ``stack.aclose()``) so we never leak partially-acquired locks.
        """
        stack = _RedisLockStack()
        target_buckets = self.sorted_buckets if buckets is None else buckets
        loop = asyncio.get_running_loop()
        effective_blocking_timeout = self._effective_lock_blocking_timeout(
            blocking_timeout
        )
        stop_trying_at = loop.time() + effective_blocking_timeout

        try:
            for bucket in target_buckets:
                if stack.locks:
                    await self._extend_locks(stack)
                remaining = max(0.0, stop_trying_at - loop.time())
                lock = bucket.lock(timeout=timeout, sleep=self._lock_sleep_seconds)
                # Generate the token ourselves and pass it to acquire().
                # This lets us run LUA_RELEASE directly with a known
                # token if cancel arrives mid-acquire: redis-py's
                # lock.release() first checks self.local.token, which is
                # only assigned AFTER the SET NX succeeds. A cancel in
                # that narrow window leaves the Redis key set but
                # local.token=None, so release() raises LockError before
                # ever talking to Redis and the lock leaks for its TTL.
                token = uuid.uuid4().hex.encode()
                try:
                    acquired = await lock.acquire(
                        blocking_timeout=remaining, token=token
                    )
                except BaseException:
                    # Best-effort compensating release with our known
                    # token. The LUA script is CAS: it only deletes if
                    # stored value matches our token, so even if another
                    # acquirer has since taken the lock we cannot release
                    # theirs. asyncio.shield guards against re-cancel
                    # during the release round-trip.
                    with contextlib.suppress(Exception):
                        await asyncio.shield(
                            lock.lua_release(
                                keys=[lock.name],
                                args=[token],
                                client=self._redis,
                            )
                        )
                    raise
                if not acquired:
                    _raise_lock_timeout_error()
                stack.locks.append(lock)
                stack.push_async_callback(_shielded_lock_release, lock)
        except BaseException:
            await stack.aclose()
            raise

        return stack

    @staticmethod
    async def _extend_locks(stack: _RedisLockStack) -> None:
        """
        Reset each held lock's TTL to its configured timeout (LOCK_TIMEOUT_SECONDS).

        Call this immediately before a long-running write (pipeline exec,
        Lua script) so a GC pause / network stall earlier in the critical
        section cannot let the lock's Redis TTL expire mid-operation,
        which would let a second worker acquire the same lock and race
        on the same write.

        If the lock already expired (or was stolen by another worker),
        ``LockNotOwnedError`` surfaces and we re-raise as ``LockError``
        so the caller aborts the write rather than committing on top of
        another worker's state. The caller's surrounding retry loop
        (await_for_capacity, etc.) will observe the error and retry
        cleanly.
        """
        for lock in stack.locks:
            try:
                await lock.reacquire()
            except redis.exceptions.LockNotOwnedError as exc:
                raise redis.exceptions.LockError(
                    "Lock expired or was stolen mid-operation; aborting write."
                ) from exc

    async def _get_capacities_unsafe(
        self,
        pipeline: redis.asyncio.client.Pipeline | None = None,
        current_time: float | None = None,
        *,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
    ) -> CapacitiesGetterResult:
        """Get capacities for all buckets."""
        if pipeline is None:
            pipeline = self._redis.pipeline()
        target_buckets = self.sorted_buckets if buckets is None else buckets
        await self._probe_legacy_override_keys_once(target_buckets)

        if current_time is None:
            current_time = await async_server_time(self._redis)

        for bucket in target_buckets:
            await bucket.get_capacity(pipeline=pipeline, current_time=current_time)

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
            results = await pipeline.execute()
        except redis.exceptions.ResponseError as exc:
            _raise_pipeline_response_error("RedisBackend._get_capacities_unsafe", exc)
        results = _validate_pipeline_results(
            results,
            context="RedisBackend._get_capacities_unsafe",
            expected_count=expected_results,
        )

        # We're using dict instead of Usage because two different application
        # versions might use the same Redis backend that's not cleaned up
        # between deployments, and the new version might have a different
        # Usage class.
        new_capacities: dict[tuple[str, int], float] = {}
        fresh_start_buckets: list[RedisBucket] = []
        for i, bucket in enumerate(target_buckets):
            idx = i * _PIPELINE_CMDS_PER_BUCKET
            last_checked, capacity = _normalize_bucket_state_pair(
                results[idx + RedisBucket.PIPELINE_LAST_CHECKED_OFFSET],
                results[idx + RedisBucket.PIPELINE_CAPACITY_OFFSET],
                context=f"RedisBackend._get_capacities_unsafe({bucket.full_redis_key})",
            )
            # max_capacity GETs come after all per-bucket state commands.
            max_capacity_idx = num_buckets * _PIPELINE_CMDS_PER_BUCKET + (
                i * _PIPELINE_CMDS_PER_OVERRIDE
            )
            if bucket.update_max_capacity_from_result(results[max_capacity_idx]):
                expire_result = self._redis.expire(
                    bucket._max_capacity_key,  # noqa: SLF001
                    bucket._override_ttl_seconds,  # noqa: SLF001
                )
                if inspect.isawaitable(expire_result):
                    await expire_result
            result = bucket.calculate_capacity(
                last_checked,
                capacity,
                current_time,
            )
            result = _revalidate_dto(result)
            if result.is_fresh_start:
                fresh_start_buckets.append(bucket)
            new_capacities[(bucket.usage_metric, int(bucket.per_seconds))] = (
                result.amount
            )

        return CapacitiesGetterResult(
            capacities=frozendict(new_capacities),
            fresh_start_buckets=fresh_start_buckets,
        )

    async def _set_capacities_unsafe(  # noqa: PLR0913
        self,
        new_capacities: Capacities,
        pipeline: redis.asyncio.client.Pipeline | None = None,
        current_time: float | None = None,
        *,
        allow_negative: bool = False,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
        refund_dedup_key: str | None = None,
        refund_dedup_reservation_id: str | None = None,
        refund_dedup_ttl_seconds: int | None = None,
        acquired_marker_key: str | None = None,
        acquired_marker_value: str | None = None,
        acquired_marker_ttl_ms: int | None = None,
        delete_acquired_marker_key: str | None = None,
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
            current_time = await async_server_time(self._redis)

        if acquired_marker_key is not None:
            if acquired_marker_value is None or acquired_marker_ttl_ms is None:
                raise ValueError("acquired marker writes require marker value and TTL")
            keys: list[str] = [acquired_marker_key]
            args: list[object] = [acquired_marker_ttl_ms, acquired_marker_value]
            for (usage_metric, per_seconds), amount in new_capacities.items():
                bucket = self._find_bucket(
                    target_buckets,
                    usage_metric,
                    per_seconds,
                )
                if bucket is None:
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
                        bucket._last_checked_key,  # noqa: SLF001
                        bucket._capacity_key,  # noqa: SLF001
                    ]
                )
                args.extend(
                    [
                        current_time,
                        normalized_amount,
                        bucket._bucket_ttl_seconds,  # noqa: SLF001
                    ]
                )
            try:
                result = await self._redis.eval(
                    _ACQUIRE_MARKER_SET_SCRIPT,
                    len(keys),
                    *keys,
                    *args,
                )
            except redis.exceptions.RedisError:
                if await self._acquire_marker_matches(
                    acquired_marker_key,
                    acquired_marker_value,
                ):
                    return True
                raise
            status = _decode_redis_script_status(result)
            if status == "ok":
                return True
            if status == "duplicate_acquire":
                if await self._acquire_marker_matches(
                    acquired_marker_key,
                    acquired_marker_value,
                ):
                    return True
                raise DuplicateRefundError("reservation already acquired")
            raise RuntimeError(f"Redis acquire marker script failed: {status}")

        for (usage_metric, per_seconds), amount in new_capacities.items():
            bucket = self._find_bucket(
                target_buckets,
                usage_metric,
                per_seconds,
            )
            if bucket is None:
                raise ValueError(f"Bucket '{usage_metric}/{per_seconds}s' not found")
            await bucket.set_capacity(
                amount,
                pipeline=pipeline,
                current_time=current_time,
                execute=False,
                allow_negative=allow_negative,
            )
        if refund_dedup_key is not None:
            pipeline.set(
                refund_dedup_key,
                "1",
                ex=refund_dedup_ttl_seconds,
                nx=True,
            )
        if delete_acquired_marker_key is not None:
            pipeline.delete(delete_acquired_marker_key)
        try:
            results = await pipeline.execute()
        except redis.exceptions.ResponseError as exc:
            _raise_pipeline_response_error("RedisBackend._set_capacities_unsafe", exc)
        if refund_dedup_key is None:
            return True
        results = _validate_pipeline_results(
            results,
            context="RedisBackend._set_capacities_unsafe",
            expected_count=(len(new_capacities) * 2)
            + 1
            + int(delete_acquired_marker_key is not None),
        )
        if results[-1]:
            return True
        if refund_dedup_reservation_id is not None:
            self._warn_refund_dedup_duplicate(refund_dedup_reservation_id)
        return False

    async def _commit_refund_with_acquire_marker_unsafe(  # noqa: PLR0913
        self,
        new_capacities: Capacities,
        *,
        current_time: float,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket],
        acquired_marker_key: str,
        acquired_marker_value: str,
        refund_dedup_key: str,
    ) -> None:
        keys: list[str] = [acquired_marker_key, refund_dedup_key]
        args: list[object] = [
            acquired_marker_value,
            self._refund_dedup_ttl_seconds,
        ]
        for (usage_metric, per_seconds), amount in new_capacities.items():
            bucket = self._find_bucket(
                buckets,
                usage_metric,
                per_seconds,
            )
            if bucket is None:
                raise ValueError(f"Bucket '{usage_metric}/{per_seconds}s' not found")
            if not math.isfinite(float(amount)):
                raise ValueError(f"capacity must be finite (got {amount!r})")
            normalized_amount = amount
            if normalized_amount == 0.0:
                normalized_amount = 0.0
            keys.extend(
                [
                    bucket._last_checked_key,  # noqa: SLF001
                    bucket._capacity_key,  # noqa: SLF001
                ]
            )
            args.extend(
                [
                    current_time,
                    normalized_amount,
                    bucket._bucket_ttl_seconds,  # noqa: SLF001
                ]
            )
        try:
            result = await self._redis.eval(
                _REFUND_WITH_MARKER_SCRIPT,
                len(keys),
                *keys,
                *args,
            )
        except redis.exceptions.RedisError:
            if await self._refund_tombstone_exists(refund_dedup_key):
                return
            raise
        status = _decode_redis_script_status(result)
        if status == "ok":
            return
        if status == "duplicate_refund":
            raise DuplicateRefundError("reservation already refunded")
        if status == "unknown_reservation":
            raise _mark_unknown_reservation_forget_in_flight(
                UnknownReservationError(
                    "reservation was never acquired by this backend"
                )
            )
        if status == "marker_mismatch":
            raise UnknownReservationError(
                "reservation was never acquired by this backend"
            )
        raise RuntimeError(f"Redis refund marker script failed: {status}")

    def _bucket_ids(
        self,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
    ) -> frozenset[tuple[str, int]]:
        target_buckets = self.sorted_buckets if buckets is None else buckets
        return frozenset(
            (bucket.usage_metric, int(bucket.per_seconds)) for bucket in target_buckets
        )

    def _normalize_check_result(
        self,
        result: tuple[bool, Capacities, Capacities, float | None]
        | tuple[bool, Capacities, Capacities, float | None, tuple[RedisBucket, ...]]
        | tuple[
            bool,
            Capacities,
            Capacities,
            float | None,
            float | None,
            tuple[RedisBucket, ...],
        ],
    ) -> tuple[
        bool,
        Capacities,
        Capacities,
        float | None,
        float | None,
        tuple[RedisBucket, ...],
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
        buckets: tuple[RedisBucket, ...],
    ) -> float:
        if buckets:
            return self._compute_sleep(usage, preconsumption, buckets=buckets)
        return self._compute_sleep(usage, preconsumption)

    async def _wait_for_task_outcome_while_cancelled(
        self,
        task: asyncio.Task[None],
    ) -> bool:
        """
        Wait for a shielded write task to settle after outer-task cancellation.

        The caller has already received ``CancelledError``.  We still need the
        write task's final outcome to decide whether capacity was actually
        recorded (and therefore must be refunded or preserved) before we
        propagate cancellation.

        Implementation note — when the outer task is cancelling, each
        ``await asyncio.shield(task)`` re-raises ``CancelledError`` rather than
        blocking to completion. The loop iterates (yielding to the event loop
        each time, so the shielded task makes progress) until ``task.done()``
        becomes True. Total wait is therefore bounded by the inner task's own
        duration — a single Redis pipeline write — not by the loop.
        """
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.done():
                    break
            except Exception:  # noqa: BLE001
                # Task raised a non-cancellation exception (e.g. Redis
                # connection error). No Redis write succeeded, so reporting
                # consumed=False below is correct and the caller will skip
                # refund. SystemExit/KeyboardInterrupt are intentionally *not*
                # caught here: swallowing them could let the shielded write
                # complete in the background while the caller returns
                # consumed=False, producing phantom consumption.
                break
        return task.done() and not task.cancelled() and task.exception() is None

    async def _check_and_consume_capacity(
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
        tuple[RedisBucket, ...],
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
        fresh_start_buckets: list[RedisBucket] = []
        consumed = False
        try:
            async with await self._lock(
                timeout=LOCK_TIMEOUT_SECONDS,
                blocking_timeout=lock_blocking_timeout,
                buckets=buckets,
            ) as lock_stack:
                # Pipeline is reused: _get_capacities_unsafe executes it (clearing
                # the command buffer), then _set_capacities_unsafe adds new commands
                # and executes again.  Safe because redis-py clears the buffer on execute().
                pipeline = self._redis.pipeline()
                current_time = await async_server_time(self._redis)

                (
                    preconsumption_capacities,
                    fresh_start_buckets,
                ) = await self._get_capacities_unsafe(
                    pipeline=pipeline,
                    current_time=current_time,
                    buckets=buckets,
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
                            raise ValueError(
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
                    usage_amount = usage.get(capacity_metric_name)
                    if usage_amount is None:
                        continue
                    postconsumption_dict[(capacity_metric_name, per_seconds)] = (
                        capacity_amount - usage_amount
                    )
                postconsumption_capacities = frozendict(postconsumption_dict)
                # Extend lock TTL immediately before the write so a GC
                # pause during get/check above cannot leave the write
                # racing with another worker after the original TTL
                # lapsed.
                await self._extend_locks(lock_stack)
                # `allow_negative=False` is correct here only because the
                # `usage_amount > capacity_amount` gate above guarantees
                # post = pre - usage >= 0. Keep this branch in sync with that
                # gate; consume_capacity (speedometer) writes with
                # `allow_negative=True` because it has no such guarantee.
                write_task = asyncio.create_task(
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
                    )
                )
                try:
                    await asyncio.shield(write_task)
                except asyncio.CancelledError:
                    consumed = await self._wait_for_task_outcome_while_cancelled(
                        write_task
                    )
                    raise
                consumed = True
                consumed_monotonic = time.monotonic()
                consumed_at_seconds = current_time
            await self._fresh_start_buckets_callback(fresh_start_buckets)
            if self._callbacks and self._callbacks.on_capacity_consumed:
                await self._invoke_callback_safe(
                    self._callbacks.on_capacity_consumed,
                    model_family=self._limit_config.get_model_family(),
                    preconsumption_capacities=preconsumption_capacities,
                    postconsumption_capacities=postconsumption_capacities,
                    usage=usage,
                    current_time=current_time,
                )
        except asyncio.CancelledError:
            if consumed:
                try:  # noqa: SIM105
                    await self._refund_cancelled_consumption(
                        usage,
                        buckets=buckets,
                        acquired_marker_key=self._acquired_marker_key(reservation_id),
                    )
                except BaseException:  # noqa: BLE001, S110
                    # Best-effort: shield ensures background completion.
                    # Swallow so CancelledError always propagates for
                    # structured concurrency (TaskGroups).
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

    async def consume_capacity(
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
        fresh_start_buckets: list[RedisBucket] = []
        async with await self._lock(
            timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets
        ) as lock_stack:
            pipeline = self._redis.pipeline()
            current_time = await async_server_time(self._redis)

            (
                preconsumption_capacities,
                fresh_start_buckets,
            ) = await self._get_capacities_unsafe(
                pipeline=pipeline,
                current_time=current_time,
                buckets=buckets,
            )
            active_metric_names = {metric for metric, _ in preconsumption_capacities}
            self._ensure_usage_metrics_are_active(usage, active_metric_names)

            for usage_metric_name, usage_amount in usage.items():
                for bucket in buckets:
                    if bucket.usage_metric != usage_metric_name:
                        continue
                    if usage_amount > bucket.max_capacity:
                        warnings.warn(
                            f"record_usage value for {usage_metric_name} ({usage_amount}) exceeds "
                            f"bucket max capacity ({bucket.max_capacity}). "
                            f"Capacity will go deeply negative.",
                            RuntimeWarning,
                            stacklevel=2,
                        )

            max_cap = {(b.usage_metric, b.per_seconds): b.max_capacity for b in buckets}
            postconsumption_dict = dict(preconsumption_capacities)
            for (
                capacity_metric_name,
                per_seconds,
            ), capacity_amount in preconsumption_capacities.items():
                usage_amount = usage.get(capacity_metric_name)
                if usage_amount is None:
                    continue
                postconsumption_dict[(capacity_metric_name, per_seconds)] = max(
                    capacity_amount - usage_amount,
                    -max_cap[(capacity_metric_name, per_seconds)],
                )
            postconsumption_capacities = frozendict(postconsumption_dict)
            # Extend lock TTL immediately before the write so a GC pause
            # during get above cannot leave the write racing with another
            # worker after the original TTL lapsed.
            await self._extend_locks(lock_stack)
            write_task = asyncio.create_task(
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
                )
            )
            try:
                await asyncio.shield(write_task)
            except asyncio.CancelledError:
                consumed = await self._wait_for_task_outcome_while_cancelled(write_task)
                if not consumed:
                    raise
                # The shielded Redis write actually landed, so the
                # speedometer reading is already correct. See
                # `suppress_current_task_cancellation` docstring —
                # refunding now would roll back a recorded
                # measurement of real usage.
                #
                # Lock release during the `async with` exit is shielded
                # (via _shielded_lock_release in _lock()), so a re-cancel
                # between suppression and release will not leak the lock.
                #
                # on_capacity_consumed is intentionally NOT fired here:
                # the cancel context is already stripped, so user
                # callbacks would run in a misleading state. See the
                # OnCapacityConsumedCallback docstring for the delivery
                # guarantee contract.
                suppress_current_task_cancellation()
                return None
        # Callbacks fire after the lock is released. Consumption is already
        # durably recorded in Redis, so if CancelledError arrives during
        # callbacks we let it propagate: the caller (e.g. asyncio.timeout)
        # must be informed of the cancel, and Redis state is already
        # correct (speedometer is advanced). Callbacks are best-effort via
        # _invoke_callback_safe.
        await self._fresh_start_buckets_callback(fresh_start_buckets)
        if self._callbacks and self._callbacks.on_capacity_consumed:
            await self._invoke_callback_safe(
                self._callbacks.on_capacity_consumed,
                model_family=self._limit_config.get_model_family(),
                preconsumption_capacities=preconsumption_capacities,
                postconsumption_capacities=postconsumption_capacities,
                usage=usage,
                current_time=current_time,
            )
        return current_time

    async def await_for_capacity(
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
        has_waited = False
        wait_started_at: float | None = None
        wait_start_callback_overhead = 0.0
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
                    await self._check_and_consume_capacity(
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
                raise TimeoutError("Timed out waiting for capacity") from exc
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
                            await self._invoke_callback_safe(
                                self._callbacks.after_wait_end_consumption,
                                model_family=self._limit_config.get_model_family(),
                                preconsumption_capacities=preconsumption,
                                postconsumption_capacities=postconsumption,
                                usage=frozendict(usage),
                                wait_time_s=wait_time_s,
                            )
                except asyncio.CancelledError:
                    try:  # noqa: SIM105
                        await self._refund_cancelled_consumption(
                            usage,
                            buckets=buckets,
                            acquired_marker_key=self._acquired_marker_key(
                                reservation_id
                            ),
                        )
                    except BaseException:  # noqa: BLE001, S110
                        # Best-effort: shield ensures background completion.
                        # Swallow so CancelledError always propagates for
                        # structured concurrency (TaskGroups).
                        pass
                    raise
                return consumed_at_seconds

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for capacity")

            if not has_waited:
                has_waited = True
                wait_started_at = time.monotonic()
                if self._callbacks and self._callbacks.on_wait_start:
                    callback_started = time.monotonic()
                    if deadline is None:
                        await self._invoke_callback_safe(
                            self._callbacks.on_wait_start,
                            model_family=self._limit_config.get_model_family(),
                            preconsumption_capacities=preconsumption,
                            usage=usage,
                        )
                    else:
                        remaining = deadline - callback_started
                        if remaining <= 0:
                            raise TimeoutError("Timed out waiting for capacity")
                        await asyncio.wait_for(
                            self._invoke_callback_safe(
                                self._callbacks.on_wait_start,
                                model_family=self._limit_config.get_model_family(),
                                preconsumption_capacities=preconsumption,
                                usage=usage,
                            ),
                            timeout=remaining,
                        )
                    wait_start_callback_overhead += time.monotonic() - callback_started
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for capacity")

            computed = self._compute_sleep_for_wait(
                usage,
                preconsumption,
                buckets=buckets,
            )
            effective = min(computed, self.MAX_CROSS_WORKER_POLL)
            if deadline is not None:
                effective = min(effective, max(0, deadline - time.monotonic()))
            async with self._local_condition:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._local_condition.wait(),
                        timeout=max(0.001, effective),
                    )

    def _compute_sleep(
        self,
        usage: FrozenUsage,
        preconsumption: Capacities,
        *,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
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

    async def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        await self.refund_capacity_for_buckets(
            reserved_usage,
            actual_usage,
            bucket_ids=self._bucket_ids(),
        )

    async def refund_capacity_for_buckets(  # noqa: PLR0913
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
        """
        Refund unused capacity back to the rate limiter based on actual usage.

        The refund mechanism handles two distinct adjustments:

        1. Token Usage Adjustment:
        - If fewer tokens were actually used than initially reserved
            (e.g., reserved 100 tokens but used only 80), the difference (20)
            is refunded.
        - If more tokens were used than initially reserved
            (e.g., reserved 100 tokens but used 120), the excess (-20)
            is treated as a negative refund, further reducing available capacity.

        2. Consumption Time Adjustment:
        - When capacity is initially acquired, we conservatively assume all
            consumption happens at the START time of the operation.
        - When refunding, we update to assume all consumption happened at the
            END time of the operation.
        - This adjustment occurs EVEN IF no token refund is needed, ensuring
            the system always records the actual end time of consumption.

        This approach provides tight adherence to rate limits without requiring
        knowledge of how resources were consumed between start and end times.
        We don't assume linear consumption or any specific pattern of usage
        during processing.

        Overuse Handling:
        If actual usage exceeds reserved usage for any metric (e.g., reserved 100
        tokens but used 120), this method will:
        1. Log a warning
        2. Apply a negative refund (-20 tokens), reducing available capacity further

        Args:
            reserved_usage: The usage that was originally reserved at the start
                            of the operation
            actual_usage: The actual usage consumed by the end of the operation
                        (may be more or less than reserved_usage)
            bucket_ids: If provided, only refund to buckets matching these
                        (metric, per_seconds) pairs. Used during reconfiguration
                        to refund only to surviving buckets.
            reservation_id: If provided, checked before bucket mutation and
                            committed in Redis only after the refund write is
                            queued successfully.
            reservation_model_family: Model-family identity captured on the
                                      reservation for acquire-marker validation.
            reservation_bucket_ids: Bucket identity set captured on the
                                    reservation for acquire-marker validation.
            reservation_reserved_usage: Reserved usage captured on the
                                        reservation for acquire-marker validation.

        Example:
            TIME N=0: Reserve 100 tokens (assumes all consumed immediately)
            TIME N=10: Operation completes, but only used 80 tokens

            The refund will:
            1. Return 20 unused tokens (100-80)
            2. Update the timestamp to N=10, giving full credit for the elapsed time

            Alternative scenario:
            TIME N=0: Reserve 100 tokens
            TIME N=10: Operation completes, but used 120 tokens

            The refund will:
            1. Apply a negative refund of -20 tokens (100-120)
            2. Update the timestamp to N=10

        """
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
                    "reservation was never acquired by this backend"
                )
            expected_marker_value = redis_acquired_marker_value(
                reservation_id=reservation_id,
                model_family=(
                    reservation_model_family or self._limit_config.get_model_family()
                ),
                bucket_ids=(
                    frozenset(reservation_bucket_ids)
                    if reservation_bucket_ids is not None
                    else backend_bucket_ids
                ),
                usage=reservation_reserved_usage or reserved_usage,
            )
        if not refund_bucket_ids:
            if (
                reservation_id is not None
                and acquired_marker_key is not None
                and refund_dedup_key is not None
                and expected_marker_value is not None
            ):
                await self._commit_refund_with_acquire_marker_unsafe(
                    frozendict(),
                    current_time=await async_server_time(self._redis),
                    buckets=(),
                    acquired_marker_key=acquired_marker_key,
                    acquired_marker_value=expected_marker_value,
                    refund_dedup_key=refund_dedup_key,
                )
            return True
        # Calculate how much to refund for each metric
        refund_usage_: dict[str, float] = {}
        for metric, reserved_amount in reserved_usage.items():
            # Key guaranteed to exist: RateLimiter.refund_capacity() calls
            # validate_refund_usage() before reaching the backend.
            actual_amount = actual_usage[metric]
            refund_amount = float(reserved_amount) - float(actual_amount)

            # Check for overuse and log a warning
            if refund_amount < 0:
                warnings.warn(
                    f"Actual usage ({actual_amount}) for {metric} exceeds "
                    f"reserved usage ({reserved_amount}). Applying negative refund.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Include both positive and negative refunds
            refund_usage_[metric] = refund_amount
        refund_usage: frozendict[str, float] = frozendict(refund_usage_)

        fresh_start_buckets: list[RedisBucket] = []
        async with await self._lock(
            timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets
        ) as lock_stack:
            pipeline = self._redis.pipeline()
            current_time = await async_server_time(self._redis)

            # Get current capacities (which already account for time-based refill)
            (
                prerefund_capacities,
                fresh_start_buckets,
            ) = await self._get_capacities_unsafe(
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
                refund_amount = refund_usage.get(capability_usage_metric)
                if refund_amount is None:
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
                refund_amount = max(refund_amount, -bucket.max_capacity)
                updated_capacities_[(capability_usage_metric, int(per_seconds))] = max(
                    -bucket.max_capacity,
                    min(
                        updated_capacities_[(capability_usage_metric, int(per_seconds))]
                        + refund_amount,
                        bucket.max_capacity,  # cached — refreshed from pipeline result in _get_capacities_unsafe
                    ),
                )
            updated_capacities = frozendict(updated_capacities_)

            # Extend lock TTL before committing the write, see _extend_locks.
            await self._extend_locks(lock_stack)
            # Option B from FIX-42: defer the Redis tombstone until the same
            # pipeline execution as the capacity write. This prevents the
            # permanent lost-refund failure where SET NX succeeds and the later
            # bucket mutation fails; exact concurrent retries are serialized by
            # the bucket locks, while lock expiry/manual writers can still race.
            if reservation_id is None:
                write_task = asyncio.create_task(
                    self._set_capacities_unsafe(
                        frozendict(updated_capacities),
                        pipeline=pipeline,
                        current_time=current_time,
                        allow_negative=True,
                        buckets=buckets,
                    )
                )
            else:
                assert acquired_marker_key is not None  # noqa: S101
                assert refund_dedup_key is not None  # noqa: S101
                assert expected_marker_value is not None  # noqa: S101
                write_task = asyncio.create_task(
                    self._commit_refund_with_acquire_marker_unsafe(
                        frozendict(updated_capacities),
                        current_time=current_time,
                        buckets=buckets,
                        acquired_marker_key=acquired_marker_key,
                        acquired_marker_value=expected_marker_value,
                        refund_dedup_key=refund_dedup_key,
                    )
                )
            try:
                await asyncio.shield(write_task)
            except asyncio.CancelledError:
                refunded = await self._wait_for_task_outcome_while_cancelled(write_task)
                if not refunded:
                    raise
                # Write landed despite cancel — refund is done.
                # Suppress so the caller doesn't retry and double-refund.
                suppress_current_task_cancellation()
                async with self._local_condition:
                    self._local_condition.notify_all()
                write_task.result()
                return True
        async with self._local_condition:
            self._local_condition.notify_all()
        await self._fresh_start_buckets_callback(fresh_start_buckets)
        if self._callbacks and self._callbacks.on_capacity_refunded:
            await self._invoke_callback_safe(
                self._callbacks.on_capacity_refunded,
                model_family=self._limit_config.get_model_family(),
                reserved_usage=reserved_usage,
                actual_usage=actual_usage,
                refunded_usage=refund_usage,
                prerefund_capacities=prerefund_capacities,
                postrefund_capacities=updated_capacities,
            )
        return True

    async def _snapshot_bucket_state(self, bucket: RedisBucket) -> None:
        """
        Freeze ``bucket`` in Redis at its accrued capacity under the CURRENT rate.

        Call before any rate-changing mutation (set_max_capacity,
        clear_max_capacity_override, set_configured_max_capacity when no
        override is present). Caller MUST hold the bucket's distributed lock.

        The anchor is the *uncapped* old-rate integration so that raw stored
        values above ``max_capacity`` are preserved: reads apply
        ``min(max_capacity, …)`` and a later cap raise can re-expose the
        hidden overflow.
        """
        # Force-refresh the override cache so ``bucket._rate_per_sec`` reflects
        # the current effective rate in Redis before we snapshot under it.
        refresh_max_capacity = getattr(
            bucket,
            "refresh_max_capacity_from_redis",
            None,
        )
        if callable(refresh_max_capacity):
            await refresh_max_capacity()
        else:  # test fakes and compatible custom bucket shims
            await bucket.get_max_capacity()
        current_time = await async_server_time(self._redis)
        pipeline = self._redis.pipeline()
        pipeline.get(bucket._last_checked_key)  # noqa: SLF001
        pipeline.get(bucket._capacity_key)  # noqa: SLF001
        pipeline.expire(bucket._last_checked_key, bucket._bucket_ttl_seconds)  # noqa: SLF001
        pipeline.expire(bucket._capacity_key, bucket._bucket_ttl_seconds)  # noqa: SLF001
        try:
            results = await pipeline.execute()
        except redis.exceptions.ResponseError as exc:
            _raise_pipeline_response_error("RedisBackend._snapshot_bucket_state", exc)
        results = _validate_pipeline_results(
            results,
            context=f"RedisBackend._snapshot_bucket_state({bucket.full_redis_key})",
            expected_count=4,
        )
        last_checked_raw, stored_raw = _normalize_bucket_state_pair(
            results[RedisBucket.PIPELINE_LAST_CHECKED_OFFSET],
            results[RedisBucket.PIPELINE_CAPACITY_OFFSET],
            context=f"RedisBackend._snapshot_bucket_state({bucket.full_redis_key})",
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
                "(last_checked=%r, capacity=%r).",
                bucket.full_redis_key,
                parse_error,
                last_checked_raw,
                stored_raw,
            )
            return  # unparseable state — leave as-is; a later write will overwrite.
        if not (math.isfinite(last_checked) and math.isfinite(stored)):
            _logger.warning(
                "Bucket %s: snapshot skipped due to non-finite Redis state "
                "(last_checked=%r, capacity=%r).",
                bucket.full_redis_key,
                last_checked_raw,
                stored_raw,
            )
            return
        time_passed = current_time - last_checked
        if time_passed < 0:
            time_passed = 0.0  # clock skew — same clamp as calculate_capacity.
        anchored = stored + time_passed * bucket._rate_per_sec  # noqa: SLF001
        await bucket.set_capacity(
            anchored,
            current_time=current_time,
            allow_negative=True,
        )

    async def set_max_capacity(
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
        async with await self._lock(
            timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets
        ) as lock_stack:
            await self._extend_locks(lock_stack)
            await self._snapshot_bucket_state(bucket)
            await bucket.set_max_capacity(value)
        async with self._local_condition:
            self._local_condition.notify_all()

    async def apply_configured_max_capacity(
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
        async with await self._lock(
            timeout=LOCK_TIMEOUT_SECONDS, buckets=buckets
        ) as lock_stack:
            await self._extend_locks(lock_stack)
            await self._snapshot_bucket_state(bucket)
            await bucket.clear_max_capacity_override()
            bucket.set_configured_max_capacity(value)
        async with self._local_condition:
            self._local_condition.notify_all()

    async def prepare_reconfigured_backend(
        self,
        new_backend: RateLimiterBackend,
        cfg: PerModelConfig,
    ) -> RateLimiterBackend:
        if not isinstance(new_backend, RedisBackend):
            raise TypeError(
                "RedisBackend can only reconfigure into another RedisBackend"
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

        async with await self._lock(
            timeout=LOCK_TIMEOUT_SECONDS,
            buckets=reconfigure_buckets,
        ) as lock_stack:
            for bucket in removed_buckets:
                # Snapshot first so the frozen capacity/last_checked in Redis
                # reflect the bucket's state at the moment of removal. Without
                # this, a later re-add with a different rate would retroactively
                # integrate the new rate across the remove→re-add gap (and any
                # time preceding it). Then clear the runtime override so the
                # re-add starts from the callable config's static quota again.
                await self._extend_locks(lock_stack)
                await self._snapshot_bucket_state(bucket)
                await bucket.clear_max_capacity_override()
            for quota in cfg.quotas:
                bucket = self._find_bucket(
                    new_buckets,
                    quota.metric,
                    int(quota.per_seconds),
                )
                if bucket is None:  # pragma: no cover
                    raise ValueError(
                        f"Bucket '{quota.metric}/{quota.per_seconds}s' not found",
                    )
                current_bucket = self._find_bucket(
                    current_buckets,
                    quota.metric,
                    int(quota.per_seconds),
                )
                # Snapshot the bucket under its current effective rate before
                # any mutation could change it. Prefer current_bucket (its
                # _max_capacity_default matches the stored override payload,
                # so the override is accepted and yields the true active rate).
                await self._extend_locks(lock_stack)
                await self._snapshot_bucket_state(
                    current_bucket if current_bucket is not None else bucket
                )
                if current_bucket is not None and float(
                    current_bucket.configured_max_capacity
                ) != float(quota.limit):
                    await bucket.clear_max_capacity_override()
                bucket.set_configured_max_capacity(float(quota.limit))

            await self._extend_locks(lock_stack)
            self.install_reconfigured_state(
                buckets=list(new_buckets),
                cfg=cfg,
            )
        async with self._local_condition:
            self._local_condition.notify_all()
        return self

    def install_reconfigured_state(
        self,
        *,
        buckets: list[RedisBucket],
        cfg: PerModelConfig,
    ) -> None:
        _ensure_buckets_match_backend(
            buckets,
            key_prefix=self._key_prefix,
            redis_client=self._redis,
        )
        self.sorted_buckets = sorted(buckets, key=lambda bucket: bucket.full_redis_key)
        self._usage_metric_names = {bucket.usage_metric for bucket in buckets}
        self._limit_config = cfg

    async def _invoke_callback_safe(self, callback, **kwargs) -> None:
        """
        Fire a user callback, suppressing exceptions to prevent capacity leaks.

        Audited 2026-05 (R4 L03): exception ladder verified parity-clean across
        all 4 _invoke_callback_safe implementations.
        Audited 2026-05 (R4 L12:C03): non-group exotic exceptions are swallowed
        or propagated by design; warning filters must not reopen the leak path.
        Audited 2026-05 (R4 L12:C01/C02): BaseExceptionGroup containing
        cancellation or process signals must propagate before best-effort logging.
        """
        try:
            await callback(**kwargs)
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:
            if _callback_exception_group_contains_critical(exc):
                raise
            msg = f"Rate limiter callback raised {type(exc).__name__}: {exc}"
            with contextlib.suppress(Warning):
                warnings.warn(msg, RuntimeWarning, stacklevel=3)
            _logger.warning(msg)

    async def _refund_cancelled_consumption(
        self,
        usage: FrozenUsage,
        *,
        buckets: tuple[RedisBucket, ...] | list[RedisBucket] | None = None,
        acquired_marker_key: str | None = None,
    ) -> None:
        """
        Refund capacity consumed before a CancelledError hit callbacks.

        Uses asyncio.shield() because the refund involves multiple Redis I/O
        await points (lock acquisition, pipeline get, pipeline set).  Shield
        ensures the refund completes even if the task is re-cancelled.
        Fires no callbacks to avoid recursion and another cancellation window.

        Worst case: if the prior lock release was interrupted (cancel arrived
        before the shielded release completed), the lock may still be held in
        Redis.  Re-acquiring here then blocks for up to LOCK_TIMEOUT_SECONDS
        (30 s) until the TTL expires.  This is inherent to the architecture —
        the lock will eventually expire and the refund will proceed.
        """
        target_buckets = self._snapshot_buckets() if buckets is None else tuple(buckets)

        async def _do_refund() -> None:
            async with await self._lock(
                timeout=LOCK_TIMEOUT_SECONDS,
                buckets=target_buckets,
            ) as lock_stack:
                pipeline = self._redis.pipeline()
                current_time = await async_server_time(self._redis)
                capacities, _ = await self._get_capacities_unsafe(
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
                await self._extend_locks(lock_stack)
                await self._set_capacities_unsafe(
                    frozendict(refunded),
                    pipeline=pipeline,
                    current_time=current_time,
                    allow_negative=True,
                    buckets=target_buckets,
                    delete_acquired_marker_key=acquired_marker_key,
                )
            async with self._local_condition:
                self._local_condition.notify_all()

        await asyncio.shield(_do_refund())

    async def _fresh_start_buckets_callback(
        self,
        fresh_start_buckets: list[RedisBucket],
    ) -> None:
        if (
            fresh_start_buckets
            and self._callbacks
            and self._callbacks.on_missing_consumption_data
        ):
            for bucket in fresh_start_buckets:
                await self._invoke_callback_safe(
                    self._callbacks.on_missing_consumption_data,
                    model_family=self._limit_config.get_model_family(),
                    usage_metric=bucket.usage_metric,
                    per_seconds=bucket.per_seconds,
                )
