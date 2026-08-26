from __future__ import annotations

import contextlib
import json
import logging
import math
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from frozendict import frozendict

from token_throttle._capacity import (
    _calculate_rate_per_sec,
    _validate_max_capacity_finite_positive,
    calculate_capacity,
)
from token_throttle._exceptions import (
    BackendLockContentionError,
    DuplicateRefundError,
    UnknownReservationError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from token_throttle._interfaces._models import BucketId, Capacities, FrozenUsage

SCHEMA_VERSION: Final[int] = 1
DEFAULT_BUSY_TIMEOUT_MS: Final[int] = 5000
DEFAULT_PRUNE_BATCH_SIZE: Final[int] = 256

_acquire_logger = logging.getLogger("token_throttle.acquire")
_refund_logger = logging.getLogger("token_throttle.refund")
_logger = logging.getLogger("token_throttle")

_SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS buckets (
    key_prefix TEXT NOT NULL,
    model_family TEXT NOT NULL,
    metric TEXT NOT NULL,
    per_seconds INTEGER NOT NULL,
    capacity REAL,
    last_checked REAL,
    max_capacity REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (key_prefix, model_family, metric, per_seconds)
)""",
    """CREATE TABLE IF NOT EXISTS acquire_markers (
    key_prefix TEXT NOT NULL,
    reservation_id TEXT NOT NULL,
    model_family TEXT NOT NULL,
    bucket_ids_json TEXT NOT NULL,
    reserved_usage_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (key_prefix, reservation_id)
)""",
    """CREATE TABLE IF NOT EXISTS refund_tombstones (
    key_prefix TEXT NOT NULL,
    reservation_id TEXT NOT NULL,
    refunded_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (key_prefix, reservation_id)
)""",
    "CREATE INDEX IF NOT EXISTS idx_buckets_updated_at ON buckets(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_acquire_markers_expires_at "
    "ON acquire_markers(expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_refund_tombstones_expires_at "
    "ON refund_tombstones(expires_at)",
)


@dataclass(frozen=True, slots=True)
class BucketSpec:
    metric: str
    per_seconds: int
    configured_max_capacity: float

    @property
    def bucket_id(self) -> BucketId:
        return (self.metric, self.per_seconds)


@dataclass(frozen=True, slots=True)
class CapacityResult:
    current_time: float
    pre_capacities: Capacities
    post_capacities: Capacities
    max_capacities: Capacities
    fresh_bucket_ids: tuple[BucketId, ...]


@dataclass(frozen=True, slots=True)
class TryConsumeResult:
    available: bool
    result: CapacityResult


@dataclass(frozen=True, slots=True)
class RefundResult:
    current_time: float
    pre_capacities: Capacities
    post_capacities: Capacities
    refunded_usage: FrozenUsage
    fresh_bucket_ids: tuple[BucketId, ...]


@dataclass(frozen=True, slots=True)
class _BucketState:
    spec: BucketSpec
    capacity: float
    max_capacity: float
    is_fresh_start: bool


def _debug_event(
    logger: logging.Logger,
    event_type: str,
    *,
    reservation_id: str | None,
    bucket_id: BucketId | None = None,
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


def _bucket_ids_json(bucket_ids: frozenset[BucketId]) -> str:
    return json.dumps(
        [[metric, int(per_seconds)] for metric, per_seconds in sorted(bucket_ids)],
        separators=(",", ":"),
    )


def _usage_json(usage: Mapping[str, float]) -> str:
    return json.dumps(
        [[metric, float(amount)] for metric, amount in sorted(usage.items())],
        separators=(",", ":"),
    )


def _is_busy_error(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return "locked" in message or "busy" in message


class SqliteEngine:
    """Backend-agnostic SQLite transaction engine for one model family."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        db_path: str,
        key_prefix: str,
        model_family: str,
        buckets: tuple[BucketSpec, ...],
        bucket_ttl_seconds: int,
        refund_dedup_ttl_seconds: int,
        max_reservation_lifetime_seconds: float,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        prune_batch_size: int = DEFAULT_PRUNE_BATCH_SIZE,
    ) -> None:
        self.db_path = db_path
        self.key_prefix = key_prefix
        self.model_family = model_family
        self._buckets = buckets
        self._bucket_by_id = {bucket.bucket_id: bucket for bucket in buckets}
        self._bucket_ttl_seconds = bucket_ttl_seconds
        self._refund_dedup_ttl_seconds = refund_dedup_ttl_seconds
        self._max_reservation_lifetime_seconds = max_reservation_lifetime_seconds
        self._prune_batch_size = prune_batch_size
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(
                db_path,
                timeout=busy_timeout_ms / 1000.0,
                isolation_level=None,
                check_same_thread=False,
            )
            self._connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms:d}")
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._initialize_schema()
        except BaseException:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise

    @property
    def bucket_ids(self) -> frozenset[BucketId]:
        return frozenset(self._bucket_by_id)

    @property
    def metric_names(self) -> set[str]:
        return {bucket.metric for bucket in self._buckets}

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
                self._connection.execute("COMMIT")
            except BaseException as exc:
                with contextlib.suppress(sqlite3.Error):
                    self._connection.execute("ROLLBACK")
                if isinstance(exc, sqlite3.OperationalError) and _is_busy_error(exc):
                    raise BackendLockContentionError(
                        "SQLite database remained busy or locked beyond "
                        "busy_timeout; the transaction made no change and is safe "
                        "to retry."
                    ) from exc
                raise

    def _initialize_schema(self) -> None:
        with self._transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS meta "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = connection.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif row[0] != str(SCHEMA_VERSION):
                raise RuntimeError(
                    "Unsupported token-throttle SQLite schema version "
                    f"{row[0]!r}; this package supports version {SCHEMA_VERSION}. "
                    "Use a compatible token-throttle version or a different database."
                )
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)

    def initialize_buckets(self, current_time: float) -> None:
        with self._transaction() as connection:
            self._prune(connection, current_time)
            for bucket in self._buckets:
                max_capacity = _validate_max_capacity_finite_positive(
                    bucket.configured_max_capacity
                )
                connection.execute(
                    "INSERT OR IGNORE INTO buckets "
                    "(key_prefix, model_family, metric, per_seconds, capacity, "
                    "last_checked, max_capacity, updated_at) "
                    "VALUES (?, ?, ?, ?, NULL, NULL, ?, ?)",
                    (
                        self.key_prefix,
                        self.model_family,
                        bucket.metric,
                        bucket.per_seconds,
                        max_capacity,
                        current_time,
                    ),
                )

    def _prune(self, connection: sqlite3.Connection, current_time: float) -> None:
        connection.execute(
            "DELETE FROM acquire_markers WHERE rowid IN "
            "(SELECT rowid FROM acquire_markers WHERE expires_at <= ? LIMIT ?)",
            (current_time, self._prune_batch_size),
        )
        connection.execute(
            "DELETE FROM refund_tombstones WHERE rowid IN "
            "(SELECT rowid FROM refund_tombstones WHERE expires_at <= ? LIMIT ?)",
            (current_time, self._prune_batch_size),
        )
        connection.execute(
            "DELETE FROM buckets WHERE rowid IN "
            "(SELECT rowid FROM buckets WHERE updated_at <= ? LIMIT ?)",
            (current_time - self._bucket_ttl_seconds, self._prune_batch_size),
        )

    def _bucket_log_id(self, bucket: BucketSpec) -> str:
        return f"sqlite:{self.model_family}:{bucket.metric}:{bucket.per_seconds}"

    def _load_states(
        self,
        connection: sqlite3.Connection,
        current_time: float,
    ) -> tuple[dict[BucketId, _BucketState], tuple[BucketId, ...]]:
        states: dict[BucketId, _BucketState] = {}
        fresh: list[BucketId] = []
        for spec in self._buckets:
            row = connection.execute(
                "SELECT capacity, last_checked, max_capacity FROM buckets "
                "WHERE key_prefix = ? AND model_family = ? AND metric = ? "
                "AND per_seconds = ?",
                (
                    self.key_prefix,
                    self.model_family,
                    spec.metric,
                    spec.per_seconds,
                ),
            ).fetchone()
            if row is None:
                max_capacity = _validate_max_capacity_finite_positive(
                    spec.configured_max_capacity
                )
                connection.execute(
                    "INSERT INTO buckets "
                    "(key_prefix, model_family, metric, per_seconds, capacity, "
                    "last_checked, max_capacity, updated_at) "
                    "VALUES (?, ?, ?, ?, NULL, NULL, ?, ?)",
                    (
                        self.key_prefix,
                        self.model_family,
                        spec.metric,
                        spec.per_seconds,
                        max_capacity,
                        current_time,
                    ),
                )
                capacity_value: float | None = None
                last_checked_value: float | None = None
            else:
                capacity_value = row[0]
                last_checked_value = row[1]
                max_capacity = _validate_max_capacity_finite_positive(row[2])

            if (capacity_value is None) != (last_checked_value is None):
                _logger.warning(
                    "Partial SQLite bucket state detected; draining fail-closed; "
                    "metric=%s model_family=%s bucket_id=%s",
                    spec.metric,
                    self.model_family,
                    self._bucket_log_id(spec),
                )
                capacity_value = 0.0
                last_checked_value = current_time
                connection.execute(
                    "UPDATE buckets SET capacity = 0.0, last_checked = ?, "
                    "updated_at = ? WHERE key_prefix = ? AND model_family = ? "
                    "AND metric = ? AND per_seconds = ?",
                    (
                        current_time,
                        current_time,
                        self.key_prefix,
                        self.model_family,
                        spec.metric,
                        spec.per_seconds,
                    ),
                )

            calculated = calculate_capacity(
                last_checked=last_checked_value,
                outdated_capacity=capacity_value,
                current_time=current_time,
                max_capacity=max_capacity,
                rate_per_sec=_calculate_rate_per_sec(
                    max_capacity,
                    spec.per_seconds,
                ),
                bucket_id=self._bucket_log_id(spec),
            )
            if calculated.is_fresh_start:
                fresh.append(spec.bucket_id)
            states[spec.bucket_id] = _BucketState(
                spec=spec,
                capacity=calculated.amount,
                max_capacity=max_capacity,
                is_fresh_start=calculated.is_fresh_start,
            )
        return states, tuple(fresh)

    @staticmethod
    def _capacities(states: Mapping[BucketId, _BucketState]) -> Capacities:
        return frozendict(
            {bucket_id: state.capacity for bucket_id, state in states.items()}
        )

    def _write_capacities(
        self,
        connection: sqlite3.Connection,
        capacities: Mapping[BucketId, float],
        current_time: float,
    ) -> None:
        for (metric, per_seconds), amount in capacities.items():
            if not math.isfinite(float(amount)):
                raise ValueError(f"capacity must be finite (got {amount!r})")
            normalized = 0.0 if float(amount) == 0.0 else float(amount)
            cursor = connection.execute(
                "UPDATE buckets SET capacity = ?, last_checked = ?, updated_at = ? "
                "WHERE key_prefix = ? AND model_family = ? AND metric = ? "
                "AND per_seconds = ?",
                (
                    normalized,
                    current_time,
                    current_time,
                    self.key_prefix,
                    self.model_family,
                    metric,
                    per_seconds,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"SQLite bucket '{metric}/{per_seconds}s' disappeared during "
                    "a write transaction"
                )

    def _raise_if_duplicate_acquire(
        self,
        connection: sqlite3.Connection,
        reservation_id: str | None,
    ) -> None:
        if reservation_id is None:
            return
        marker = connection.execute(
            "SELECT 1 FROM acquire_markers WHERE key_prefix = ? AND reservation_id = ?",
            (self.key_prefix, reservation_id),
        ).fetchone()
        tombstone = connection.execute(
            "SELECT 1 FROM refund_tombstones WHERE key_prefix = ? "
            "AND reservation_id = ?",
            (self.key_prefix, reservation_id),
        ).fetchone()
        _debug_event(
            _acquire_logger,
            "sqlite_acquire_marker_read",
            reservation_id=reservation_id,
            found=marker is not None,
        )
        if marker is not None or tombstone is not None:
            raise DuplicateRefundError(
                "reservation already acquired",
                reason="duplicate_acquire",
                reservation_id=reservation_id,
                model_family=self.model_family,
            )

    def _insert_marker(
        self,
        connection: sqlite3.Connection,
        *,
        reservation_id: str | None,
        usage: FrozenUsage,
        reservation_lifetime_seconds: float | None,
        current_time: float,
    ) -> None:
        if reservation_id is None:
            return
        if reservation_lifetime_seconds is None:
            raise ValueError(
                "reservation_lifetime_seconds is required when reservation_id "
                "is supplied"
            )
        lifetime = float(reservation_lifetime_seconds)
        if not math.isfinite(lifetime) or lifetime <= 0:
            raise ValueError(
                "reservation_lifetime_seconds must be finite and greater than 0"
            )
        if lifetime > self._max_reservation_lifetime_seconds:
            raise ValueError(
                "reservation_lifetime_seconds exceeds the SQLite backend's "
                "max_reservation_lifetime_seconds"
            )
        _debug_event(
            _acquire_logger,
            "sqlite_acquire_marker_write",
            reservation_id=reservation_id,
        )
        connection.execute(
            "INSERT INTO acquire_markers "
            "(key_prefix, reservation_id, model_family, bucket_ids_json, "
            "reserved_usage_json, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                self.key_prefix,
                reservation_id,
                self.model_family,
                _bucket_ids_json(self.bucket_ids),
                _usage_json(usage),
                current_time,
                current_time + lifetime,
            ),
        )

    def try_consume(
        self,
        usage: FrozenUsage,
        *,
        current_time: float,
        reservation_id: str | None,
        reservation_lifetime_seconds: float | None,
    ) -> TryConsumeResult:
        with self._transaction() as connection:
            self._prune(connection, current_time)
            self._raise_if_duplicate_acquire(connection, reservation_id)
            states, fresh = self._load_states(connection, current_time)
            pre = self._capacities(states)
            max_capacities = frozendict(
                {bucket_id: state.max_capacity for bucket_id, state in states.items()}
            )
            for metric, amount in usage.items():
                for state in states.values():
                    if state.spec.metric == metric and amount > state.max_capacity:
                        raise ValueError(
                            f"Usage value for {metric} ({amount}) exceeds bucket max "
                            f"capacity ({state.max_capacity}) for the "
                            f"{state.spec.per_seconds}s window"
                        )
            available = all(
                float(usage.get(metric, 0.0)) <= state.capacity
                for (metric, _), state in states.items()
            )
            if not available:
                return TryConsumeResult(
                    available=False,
                    result=CapacityResult(
                        current_time=current_time,
                        pre_capacities=pre,
                        post_capacities=frozendict(),
                        max_capacities=max_capacities,
                        fresh_bucket_ids=fresh,
                    ),
                )
            post = frozendict(
                {
                    bucket_id: state.capacity - float(usage.get(state.spec.metric, 0.0))
                    for bucket_id, state in states.items()
                }
            )
            self._write_capacities(connection, post, current_time)
            self._insert_marker(
                connection,
                reservation_id=reservation_id,
                usage=usage,
                reservation_lifetime_seconds=reservation_lifetime_seconds,
                current_time=current_time,
            )
            return TryConsumeResult(
                available=True,
                result=CapacityResult(
                    current_time=current_time,
                    pre_capacities=pre,
                    post_capacities=post,
                    max_capacities=max_capacities,
                    fresh_bucket_ids=fresh,
                ),
            )

    def consume(
        self,
        usage: FrozenUsage,
        *,
        current_time: float,
        reservation_id: str | None,
        reservation_lifetime_seconds: float | None,
    ) -> CapacityResult:
        with self._transaction() as connection:
            self._prune(connection, current_time)
            self._raise_if_duplicate_acquire(connection, reservation_id)
            states, fresh = self._load_states(connection, current_time)
            pre = self._capacities(states)
            max_capacities = frozendict(
                {bucket_id: state.max_capacity for bucket_id, state in states.items()}
            )
            post = frozendict(
                {
                    bucket_id: max(
                        -state.max_capacity,
                        state.capacity - float(usage.get(state.spec.metric, 0.0)),
                    )
                    for bucket_id, state in states.items()
                }
            )
            self._write_capacities(connection, post, current_time)
            self._insert_marker(
                connection,
                reservation_id=reservation_id,
                usage=usage,
                reservation_lifetime_seconds=reservation_lifetime_seconds,
                current_time=current_time,
            )
            return CapacityResult(
                current_time=current_time,
                pre_capacities=pre,
                post_capacities=post,
                max_capacities=max_capacities,
                fresh_bucket_ids=fresh,
            )

    def _verify_marker(
        self,
        connection: sqlite3.Connection,
        *,
        reservation_id: str,
        reservation_model_family: str,
        reservation_bucket_ids: frozenset[BucketId],
        reservation_reserved_usage: FrozenUsage,
    ) -> None:
        tombstone = connection.execute(
            "SELECT 1 FROM refund_tombstones WHERE key_prefix = ? "
            "AND reservation_id = ?",
            (self.key_prefix, reservation_id),
        ).fetchone()
        if tombstone is not None:
            raise DuplicateRefundError(
                "reservation already refunded",
                reason="already_refunded",
                reservation_id=reservation_id,
                model_family=reservation_model_family,
            )
        marker = connection.execute(
            "SELECT model_family, bucket_ids_json, reserved_usage_json "
            "FROM acquire_markers WHERE key_prefix = ? AND reservation_id = ?",
            (self.key_prefix, reservation_id),
        ).fetchone()
        _debug_event(
            _refund_logger,
            "sqlite_refund_marker_read",
            reservation_id=reservation_id,
            found=marker is not None,
        )
        if marker is None:
            raise UnknownReservationError(
                reservation_id=reservation_id,
                model_family=reservation_model_family,
            )
        if (
            marker[0] != reservation_model_family
            or marker[1] != _bucket_ids_json(reservation_bucket_ids)
            or marker[2] != _usage_json(reservation_reserved_usage)
        ):
            raise UnknownReservationError(
                reservation_id=reservation_id,
                model_family=reservation_model_family,
            )

    def refund(  # noqa: PLR0913
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        *,
        refund_bucket_ids: frozenset[BucketId],
        current_time: float,
        reservation_id: str | None,
        reservation_model_family: str | None,
        reservation_bucket_ids: frozenset[BucketId] | None,
        reservation_reserved_usage: FrozenUsage | None,
    ) -> RefundResult:
        refund_usage = frozendict(
            {
                metric: float(amount) - float(actual_usage[metric])
                for metric, amount in reserved_usage.items()
            }
        )
        with self._transaction() as connection:
            self._prune(connection, current_time)
            if reservation_id is not None:
                if (
                    reservation_model_family is None
                    or reservation_bucket_ids is None
                    or reservation_reserved_usage is None
                ):
                    raise ValueError(
                        "reservation marker metadata is required for "
                        "marker-authorized refunds"
                    )
                self._verify_marker(
                    connection,
                    reservation_id=reservation_id,
                    reservation_model_family=reservation_model_family,
                    reservation_bucket_ids=reservation_bucket_ids,
                    reservation_reserved_usage=reservation_reserved_usage,
                )
            states, fresh = self._load_states(connection, current_time)
            pre = self._capacities(states)
            post_dict = dict(pre)
            for bucket_id in refund_bucket_ids:
                state = states[bucket_id]
                refund_amount = max(
                    float(refund_usage[state.spec.metric]),
                    -state.max_capacity,
                )
                post_dict[bucket_id] = max(
                    -state.max_capacity,
                    min(state.capacity + refund_amount, state.max_capacity),
                )
            post = frozendict(post_dict)
            self._write_capacities(
                connection,
                {bucket_id: post[bucket_id] for bucket_id in refund_bucket_ids},
                current_time,
            )
            if reservation_id is not None:
                deleted = connection.execute(
                    "DELETE FROM acquire_markers WHERE key_prefix = ? "
                    "AND reservation_id = ?",
                    (self.key_prefix, reservation_id),
                )
                if deleted.rowcount != 1:
                    raise RuntimeError(
                        "SQLite acquire marker disappeared during refund transaction"
                    )
                _debug_event(
                    _refund_logger,
                    "sqlite_refund_marker_delete",
                    reservation_id=reservation_id,
                )
                connection.execute(
                    "INSERT INTO refund_tombstones "
                    "(key_prefix, reservation_id, refunded_at, expires_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        self.key_prefix,
                        reservation_id,
                        current_time,
                        current_time + self._refund_dedup_ttl_seconds,
                    ),
                )
                _debug_event(
                    _refund_logger,
                    "sqlite_refund_dedup_write",
                    reservation_id=reservation_id,
                )
            return RefundResult(
                current_time=current_time,
                pre_capacities=pre,
                post_capacities=post,
                refunded_usage=refund_usage,
                fresh_bucket_ids=fresh,
            )

    def cleanup_consumption(
        self,
        usage: FrozenUsage,
        *,
        bucket_ids: frozenset[BucketId],
        current_time: float,
        reservation_id: str | None,
    ) -> None:
        with self._transaction() as connection:
            self._prune(connection, current_time)
            if reservation_id is not None:
                marker = connection.execute(
                    "SELECT 1 FROM acquire_markers WHERE key_prefix = ? "
                    "AND reservation_id = ?",
                    (self.key_prefix, reservation_id),
                ).fetchone()
                if marker is None:
                    raise UnknownReservationError(
                        reservation_id=reservation_id,
                        model_family=self.model_family,
                    )
            states, _ = self._load_states(connection, current_time)
            refunded = {
                bucket_id: min(
                    states[bucket_id].capacity
                    + float(usage.get(states[bucket_id].spec.metric, 0.0)),
                    states[bucket_id].max_capacity,
                )
                for bucket_id in bucket_ids
                if bucket_id in states
            }
            self._write_capacities(connection, refunded, current_time)
            if reservation_id is not None:
                connection.execute(
                    "DELETE FROM acquire_markers WHERE key_prefix = ? "
                    "AND reservation_id = ?",
                    (self.key_prefix, reservation_id),
                )
                _debug_event(
                    _acquire_logger,
                    "sqlite_acquire_marker_delete",
                    reservation_id=reservation_id,
                )

    def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
        *,
        current_time: float,
    ) -> None:
        value = _validate_max_capacity_finite_positive(value)
        bucket_id = (metric, per_seconds)
        if bucket_id not in self._bucket_by_id:
            raise ValueError(f"Bucket '{metric}/{per_seconds}s' not found")
        spec = self._bucket_by_id[bucket_id]
        with self._transaction() as connection:
            self._prune(connection, current_time)
            row = connection.execute(
                "SELECT capacity, last_checked, max_capacity FROM buckets "
                "WHERE key_prefix = ? AND model_family = ? AND metric = ? "
                "AND per_seconds = ?",
                (
                    self.key_prefix,
                    self.model_family,
                    metric,
                    per_seconds,
                ),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO buckets "
                    "(key_prefix, model_family, metric, per_seconds, capacity, "
                    "last_checked, max_capacity, updated_at) "
                    "VALUES (?, ?, ?, ?, NULL, NULL, ?, ?)",
                    (
                        self.key_prefix,
                        self.model_family,
                        metric,
                        per_seconds,
                        value,
                        current_time,
                    ),
                )
                return
            capacity, last_checked, old_max = row
            old_max = _validate_max_capacity_finite_positive(old_max)
            if (capacity is None) != (last_checked is None):
                capacity = 0.0
                last_checked = current_time
            if capacity is not None and last_checked is not None:
                calculate_capacity(
                    last_checked=last_checked,
                    outdated_capacity=capacity,
                    current_time=current_time,
                    max_capacity=old_max,
                    rate_per_sec=_calculate_rate_per_sec(
                        old_max,
                        spec.per_seconds,
                    ),
                    bucket_id=self._bucket_log_id(spec),
                )
                elapsed = max(0.0, current_time - float(last_checked))
                anchored = float(capacity) + elapsed * _calculate_rate_per_sec(
                    old_max,
                    spec.per_seconds,
                )
                if not math.isfinite(anchored):
                    anchored = old_max
                capacity = anchored
                last_checked = current_time
            connection.execute(
                "UPDATE buckets SET capacity = ?, last_checked = ?, "
                "max_capacity = ?, updated_at = ? WHERE key_prefix = ? "
                "AND model_family = ? AND metric = ? AND per_seconds = ?",
                (
                    capacity,
                    last_checked,
                    value,
                    current_time,
                    self.key_prefix,
                    self.model_family,
                    metric,
                    per_seconds,
                ),
            )

    def inspect_counts(self) -> dict[str, int]:
        """Return scoped row counts for deterministic backend-specific tests."""
        with self._lock:
            return {
                "buckets": int(
                    self._connection.execute(
                        "SELECT COUNT(*) FROM buckets WHERE key_prefix = ?",
                        (self.key_prefix,),
                    ).fetchone()[0]
                ),
                "acquire_markers": int(
                    self._connection.execute(
                        "SELECT COUNT(*) FROM acquire_markers WHERE key_prefix = ?",
                        (self.key_prefix,),
                    ).fetchone()[0]
                ),
                "refund_tombstones": int(
                    self._connection.execute(
                        "SELECT COUNT(*) FROM refund_tombstones WHERE key_prefix = ?",
                        (self.key_prefix,),
                    ).fetchone()[0]
                ),
            }
