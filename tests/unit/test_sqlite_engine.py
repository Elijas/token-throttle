from __future__ import annotations

import math
import multiprocessing
import os
import sqlite3
import threading
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from token_throttle import (
    DuplicateRefundError,
    PerModelConfig,
    Quota,
    SyncSqliteBackendBuilder,
    UnknownReservationError,
    UsageQuotas,
    frozen_usage,
)
from token_throttle import _capacity as capacity_module
from token_throttle._capacity import calculate_capacity
from token_throttle._limiter_backends._sqlite import _engine as engine_module
from token_throttle._limiter_backends._sqlite._engine import (
    SCHEMA_VERSION,
    BucketSpec,
    SqliteEngine,
)
from token_throttle._limiter_backends._sqlite._ttl import (
    derive_default_max_reservation_lifetime_seconds_from_ttls,
    validate_reservation_lifetime_ttl_invariant,
)


def _config(
    model_family: str = "sqlite-tests",
    *,
    metric: str = "requests",
    limit: float = 10.0,
    per_seconds: int = 10,
) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
        model_family=model_family,
    )


def _engine(  # noqa: PLR0913
    db_path: Path,
    *,
    key_prefix: str = "tests",
    model_family: str = "sqlite-tests",
    limit: float = 10.0,
    per_seconds: int = 10,
    bucket_ttl_seconds: int = 100,
    refund_dedup_ttl_seconds: int = 100,
    max_reservation_lifetime_seconds: float = 20.0,
    override_ttl_seconds: int = 100,
    prune_batch_size: int = 256,
    initialized_at: float = 100.0,
) -> SqliteEngine:
    engine = SqliteEngine(
        db_path=str(db_path),
        key_prefix=key_prefix,
        model_family=model_family,
        buckets=(
            BucketSpec(
                metric="requests",
                per_seconds=per_seconds,
                configured_max_capacity=limit,
            ),
        ),
        bucket_ttl_seconds=bucket_ttl_seconds,
        refund_dedup_ttl_seconds=refund_dedup_ttl_seconds,
        override_ttl_seconds=override_ttl_seconds,
        max_reservation_lifetime_seconds=max_reservation_lifetime_seconds,
        prune_batch_size=prune_batch_size,
    )
    engine.initialize_buckets(initialized_at)
    return engine


def _acquire_in_process(
    db_path: str,
    key_prefix: str,
    reservation_id: str,
    result_queue,
) -> None:
    builder = SyncSqliteBackendBuilder(
        db_path,
        key_prefix=key_prefix,
        busy_timeout_ms=2000,
    )
    try:
        backend = builder.build(_config(limit=2.0))
        backend.wait_for_capacity(
            frozen_usage({"requests": 2}),
            timeout=0,
            reservation_id=reservation_id,
            reservation_lifetime_seconds=20.0,
        )
    except Exception as exc:
        result_queue.put(("error", type(exc).__name__))
    else:
        result_queue.put(("acquired", reservation_id))
    finally:
        builder.close()


def test_sqlite_engine_creates_versioned_wal_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    engine = _engine(db_path)
    try:
        with sqlite3.connect(db_path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            version = connection.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
        assert {"meta", "buckets", "acquire_markers", "refund_tombstones"} <= tables
        assert version == (str(SCHEMA_VERSION),)
        assert journal_mode == ("wal",)
        assert engine._connection.execute("PRAGMA synchronous").fetchone() == (1,)
        assert engine._connection.execute("PRAGMA busy_timeout").fetchone() == (5000,)
        bucket_columns = {
            row[1] for row in engine._connection.execute("PRAGMA table_info(buckets)")
        }
        assert "expires_at" in bucket_columns
    finally:
        engine.close()


def test_sqlite_engine_rejects_unknown_schema_version(tmp_path: Path) -> None:
    db_path = tmp_path / "future.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', '999')"
        )

    with pytest.raises(RuntimeError, match=r"Unsupported.*schema version '999'"):
        _engine(db_path)


def test_sqlite_engine_refill_calls_shared_capacity_math(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(tmp_path / "refill.sqlite3")
    calls = 0
    shared_calculate_capacity = engine_module.calculate_capacity

    def tracked_calculate_capacity(*args, **kwargs):
        nonlocal calls
        calls += 1
        return shared_calculate_capacity(*args, **kwargs)

    monkeypatch.setattr(engine_module, "calculate_capacity", tracked_calculate_capacity)
    try:
        engine.consume(
            frozen_usage({"requests": 10}),
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        result = engine.try_consume(
            frozen_usage({"requests": 5}),
            current_time=105.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert calls >= 2
        assert result.available is True
        assert result.result.pre_capacities[("requests", 10)] == pytest.approx(5.0)
        assert result.result.post_capacities[("requests", 10)] == pytest.approx(0.0)
    finally:
        engine.close()


def test_sqlite_marker_refund_and_tombstone_are_one_lifecycle(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path / "marker.sqlite3")
    reservation_usage = frozen_usage({"requests": 4})
    bucket_ids = frozenset({("requests", 10)})
    try:
        engine.try_consume(
            reservation_usage,
            current_time=100.0,
            reservation_id="reservation-1",
            reservation_lifetime_seconds=20.0,
        )
        marker = engine._connection.execute(
            "SELECT created_at, expires_at FROM acquire_markers"
        ).fetchone()
        assert marker == (100.0, 120.0)

        refunded = engine.refund(
            reservation_usage,
            frozen_usage({"requests": 1}),
            refund_bucket_ids=bucket_ids,
            current_time=101.0,
            reservation_id="reservation-1",
            reservation_model_family="sqlite-tests",
            reservation_bucket_ids=bucket_ids,
            reservation_reserved_usage=reservation_usage,
        )
        assert refunded.post_capacities[("requests", 10)] == pytest.approx(10.0)
        assert engine.inspect_counts() == {
            "buckets": 1,
            "acquire_markers": 0,
            "refund_tombstones": 1,
        }
        tombstone = engine._connection.execute(
            "SELECT refunded_at, expires_at FROM refund_tombstones"
        ).fetchone()
        assert tombstone == (101.0, 201.0)

        with pytest.raises(DuplicateRefundError) as duplicate:
            engine.refund(
                reservation_usage,
                frozen_usage({"requests": 1}),
                refund_bucket_ids=bucket_ids,
                current_time=102.0,
                reservation_id="reservation-1",
                reservation_model_family="sqlite-tests",
                reservation_bucket_ids=bucket_ids,
                reservation_reserved_usage=reservation_usage,
            )
        assert duplicate.value.reason == "already_refunded"
    finally:
        engine.close()


def test_sqlite_unknown_marker_does_not_credit_capacity(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "unknown.sqlite3")
    usage = frozen_usage({"requests": 5})
    bucket_ids = frozenset({("requests", 10)})
    try:
        engine.consume(
            usage,
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        with pytest.raises(UnknownReservationError):
            engine.refund(
                usage,
                frozen_usage({"requests": 0}),
                refund_bucket_ids=bucket_ids,
                current_time=100.0,
                reservation_id="missing",
                reservation_model_family="sqlite-tests",
                reservation_bucket_ids=bucket_ids,
                reservation_reserved_usage=usage,
            )
        remaining = engine.try_consume(
            usage,
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert remaining.available is True
        assert remaining.result.post_capacities[("requests", 10)] == 0.0
    finally:
        engine.close()


def test_sqlite_pruning_is_bounded_per_table(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path / "prune.sqlite3",
        prune_batch_size=1,
        bucket_ttl_seconds=100,
        refund_dedup_ttl_seconds=100,
    )
    try:
        for reservation_id in ("expired-1", "expired-2"):
            engine._connection.execute(
                "INSERT INTO acquire_markers VALUES (?, ?, ?, '[]', '[]', 1, 2)",
                ("tests", reservation_id, "sqlite-tests"),
            )
            engine._connection.execute(
                "INSERT INTO refund_tombstones VALUES (?, ?, 1, 2)",
                ("tests", reservation_id),
            )
        engine.consume(
            frozen_usage({"requests": 0}),
            current_time=10.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert engine.inspect_counts()["acquire_markers"] == 1
        assert engine.inspect_counts()["refund_tombstones"] == 1
    finally:
        engine.close()


def test_sqlite_expired_target_marker_is_rejected_beyond_prune_batch(
    tmp_path: Path,
) -> None:
    engine = _engine(
        tmp_path / "target-marker-expiry.sqlite3",
        prune_batch_size=1,
    )
    usage = frozen_usage({"requests": 4})
    bucket_ids = frozenset({("requests", 10)})
    try:
        engine.consume(
            usage,
            current_time=100.0,
            reservation_id="target-expired-marker",
            reservation_lifetime_seconds=1.0,
        )
        engine._connection.execute(
            "INSERT INTO acquire_markers VALUES (?, ?, ?, '[]', '[]', 0, 1)",
            ("tests", "older-expired-marker", "sqlite-tests"),
        )
        with pytest.raises(UnknownReservationError):
            engine.refund(
                usage,
                frozen_usage({"requests": 0}),
                refund_bucket_ids=bucket_ids,
                current_time=102.0,
                reservation_id="target-expired-marker",
                reservation_model_family="sqlite-tests",
                reservation_bucket_ids=bucket_ids,
                reservation_reserved_usage=usage,
            )
        unavailable = engine.try_consume(
            frozen_usage({"requests": 9}),
            current_time=102.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert unavailable.available is False
    finally:
        engine.close()


def test_sqlite_expired_target_tombstone_allows_id_reuse_beyond_prune_batch(
    tmp_path: Path,
) -> None:
    engine = _engine(
        tmp_path / "target-tombstone-expiry.sqlite3",
        refund_dedup_ttl_seconds=2,
        prune_batch_size=1,
    )
    usage = frozen_usage({"requests": 1})
    bucket_ids = frozenset({("requests", 10)})
    try:
        engine.consume(
            usage,
            current_time=100.0,
            reservation_id="reusable-id",
            reservation_lifetime_seconds=20.0,
        )
        engine.refund(
            usage,
            usage,
            refund_bucket_ids=bucket_ids,
            current_time=100.0,
            reservation_id="reusable-id",
            reservation_model_family="sqlite-tests",
            reservation_bucket_ids=bucket_ids,
            reservation_reserved_usage=usage,
        )
        engine._connection.execute(
            "INSERT INTO refund_tombstones VALUES (?, ?, 0, 1)",
            ("tests", "older-expired-tombstone"),
        )
        reused = engine.try_consume(
            frozen_usage({"requests": 0}),
            current_time=103.0,
            reservation_id="reusable-id",
            reservation_lifetime_seconds=20.0,
        )
        assert reused.available is True
    finally:
        engine.close()


def test_sqlite_bucket_ttl_pruning_restores_only_fresh_capacity(
    tmp_path: Path,
) -> None:
    engine = _engine(
        tmp_path / "bucket-prune.sqlite3",
        bucket_ttl_seconds=10,
    )
    try:
        engine.consume(
            frozen_usage({"requests": 10}),
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        result = engine.consume(
            frozen_usage({"requests": 1}),
            current_time=111.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert result.fresh_bucket_ids == (("requests", 10),)
        assert result.pre_capacities[("requests", 10)] == 10.0
        assert result.post_capacities[("requests", 10)] == 9.0
    finally:
        engine.close()


def test_sqlite_negative_refund_preserves_debt_for_linear_refill(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path / "debt.sqlite3")
    bucket_ids = frozenset({("requests", 10)})
    try:
        engine.consume(
            frozen_usage({"requests": 10}),
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        refunded = engine.refund(
            frozen_usage({"requests": 2}),
            frozen_usage({"requests": 20}),
            refund_bucket_ids=bucket_ids,
            current_time=100.0,
            reservation_id=None,
            reservation_model_family=None,
            reservation_bucket_ids=None,
            reservation_reserved_usage=None,
        )
        assert refunded.post_capacities[("requests", 10)] == -10.0
        halfway = engine.try_consume(
            frozen_usage({"requests": 0}),
            current_time=105.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert halfway.result.pre_capacities[("requests", 10)] == -5.0
    finally:
        engine.close()


def test_sqlite_set_max_capacity_anchors_elapsed_time_at_old_rate(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path / "max.sqlite3")
    try:
        engine.consume(
            frozen_usage({"requests": 10}),
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        engine.set_max_capacity("requests", 10, 20.0, current_time=105.0)
        anchored = engine.try_consume(
            frozen_usage({"requests": 0}),
            current_time=105.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert anchored.result.pre_capacities[("requests", 10)] == 5.0
        refilled = engine.try_consume(
            frozen_usage({"requests": 0}),
            current_time=110.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert refilled.result.pre_capacities[("requests", 10)] == 15.0
    finally:
        engine.close()


def test_sqlite_configured_limits_remain_process_local(tmp_path: Path) -> None:
    db_path = tmp_path / "local-config.sqlite3"
    first = _engine(db_path, limit=10.0)
    second = _engine(db_path, limit=20.0)
    try:
        first_result = first.try_consume(
            frozen_usage({"requests": 0}),
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        second_result = second.try_consume(
            frozen_usage({"requests": 0}),
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        first_again = first.try_consume(
            frozen_usage({"requests": 0}),
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert first_result.result.max_capacities[("requests", 10)] == 10.0
        assert second_result.result.max_capacities[("requests", 10)] == 20.0
        assert first_again.result.max_capacities[("requests", 10)] == 10.0
        with sqlite3.connect(db_path) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(buckets)")
            }
        assert "max_capacity" not in columns
        assert {"override_value", "override_expires_at"} <= columns
    finally:
        first.close()
        second.close()


def test_sqlite_override_is_cross_process_and_expires_to_local_config(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "override.sqlite3"
    first = _engine(db_path, limit=10.0, override_ttl_seconds=2)
    second = _engine(db_path, limit=20.0, override_ttl_seconds=2)
    try:
        first.set_max_capacity("requests", 10, 15.0, current_time=100.0)
        active = second.try_consume(
            frozen_usage({"requests": 0}),
            current_time=101.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        expired = second.try_consume(
            frozen_usage({"requests": 0}),
            current_time=102.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert active.result.max_capacities[("requests", 10)] == 15.0
        assert expired.result.max_capacities[("requests", 10)] == 20.0
        with sqlite3.connect(db_path) as connection:
            override = connection.execute(
                "SELECT override_value, override_expires_at FROM buckets"
            ).fetchone()
        assert override == (None, None)
    finally:
        first.close()
        second.close()


def test_sqlite_config_after_restart_applies_without_rewriting_state(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "restart-config.sqlite3"
    first = _engine(db_path, limit=10.0)
    first.consume(
        frozen_usage({"requests": 5}),
        current_time=100.0,
        reservation_id=None,
        reservation_lifetime_seconds=None,
    )
    first.close()

    restarted = _engine(db_path, limit=20.0, initialized_at=101.0)
    try:
        result = restarted.try_consume(
            frozen_usage({"requests": 0}),
            current_time=101.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert result.result.max_capacities[("requests", 10)] == 20.0
        assert result.result.pre_capacities[("requests", 10)] == pytest.approx(7.0)
    finally:
        restarted.close()


def test_sqlite_apply_configured_max_clears_override_and_anchors_state(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path / "apply-config.sqlite3")
    try:
        engine.consume(
            frozen_usage({"requests": 10}),
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        engine.set_max_capacity("requests", 10, 20.0, current_time=105.0)
        engine.apply_configured_max_capacity("requests", 10, 30.0, current_time=107.0)
        result = engine.try_consume(
            frozen_usage({"requests": 0}),
            current_time=107.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert result.result.pre_capacities[("requests", 10)] == pytest.approx(9.0)
        assert result.result.max_capacities[("requests", 10)] == 30.0
        assert engine._connection.execute(
            "SELECT override_value, override_expires_at FROM buckets"
        ).fetchone() == (None, None)
    finally:
        engine.close()


def test_sqlite_repairs_far_future_last_checked_on_failed_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(tmp_path / "future-clock.sqlite3", bucket_ttl_seconds=10)
    try:
        monkeypatch.setattr(capacity_module, "_backward_clock_warned", False)
        monkeypatch.setattr(capacity_module, "_backward_clock_last_warning_at", None)
        engine._connection.execute(
            "UPDATE buckets SET capacity = 0, last_checked = 1000, "
            "updated_at = 1000, expires_at = 2000"
        )
        with pytest.warns(RuntimeWarning, match="Negative time_passed"):
            failed = engine.try_consume(
                frozen_usage({"requests": 1}),
                current_time=200.0,
                reservation_id=None,
                reservation_lifetime_seconds=None,
            )
        assert failed.available is False
        assert engine._connection.execute(
            "SELECT capacity, last_checked, updated_at FROM buckets"
        ).fetchone() == (0.0, 200.0, 200.0)
        snapshots, _ = engine.inspect_snapshot(current_time=205.0)
        assert snapshots[0].current_capacity == 5.0
        pruned = engine.try_consume(
            frozen_usage({"requests": 0}),
            current_time=211.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert pruned.result.fresh_bucket_ids == (("requests", 10),)
    finally:
        engine.close()


def test_sqlite_operation_sets_deadline_derived_busy_timeout(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "busy-timeout.sqlite3")
    try:
        engine.try_consume(
            frozen_usage({"requests": 0}),
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
            busy_timeout_ms=123,
        )
        assert engine._connection.execute("PRAGMA busy_timeout").fetchone() == (123,)
        engine.consume(
            frozen_usage({"requests": 0}),
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert engine._connection.execute("PRAGMA busy_timeout").fetchone() == (5000,)
    finally:
        engine.close()


def test_sqlite_clock_warning_includes_family_and_metric(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(capacity_module, "_backward_clock_warned", False)
    monkeypatch.setattr(capacity_module, "_backward_clock_last_warning_at", None)
    with pytest.warns(RuntimeWarning, match="Negative time_passed"):
        calculate_capacity(
            last_checked=101.0,
            outdated_capacity=0.0,
            current_time=100.0,
            max_capacity=10.0,
            rate_per_sec=1.0,
            bucket_id="sqlite:clock-family:requests:10",
        )
    record = next(
        record for record in caplog.records if "Negative time_passed" in record.message
    )
    assert record.token_throttle_model_family == "clock-family"  # noqa: S105
    assert record.token_throttle_metric == "requests"  # noqa: S105


def test_sync_sqlite_introspection_reports_live_state_and_durable_counts(
    tmp_path: Path,
) -> None:
    builder = SyncSqliteBackendBuilder(
        tmp_path / "introspection.sqlite3",
        key_prefix="introspection",
    )
    backend = builder.build(_config(limit=10.0))
    try:
        usage = frozen_usage({"requests": 4})
        backend.consume_capacity(
            usage,
            reservation_id="introspection-marker",
            reservation_lifetime_seconds=20.0,
        )
        backend.set_max_capacity("requests", 10, 15.0)
        diagnostic = backend.introspect()
        assert diagnostic.backend_type == "custom"
        assert len(diagnostic.buckets) == 1
        bucket = diagnostic.buckets[0]
        assert bucket.metric == "requests"
        assert bucket.configured_limit == 10.0
        assert bucket.effective_max_capacity == 15.0
        assert bucket.override_source == "backend"
        assert bucket.current_capacity is not None
        assert bucket.current_capacity < 10.0
        assert diagnostic.waits == ()
        assert any("acquire_markers=1" in issue.message for issue in diagnostic.issues)
    finally:
        builder.close()


def test_sync_sqlite_introspection_tracks_and_removes_waiters(tmp_path: Path) -> None:
    builder = SyncSqliteBackendBuilder(
        tmp_path / "waiter-introspection.sqlite3",
        key_prefix="introspection",
        sleep_interval=0.01,
    )
    backend = builder.build(_config(limit=1.0, per_seconds=100))
    usage = frozen_usage({"requests": 1})
    backend.consume_capacity(usage)
    outcome: list[str] = []

    def wait() -> None:
        backend.wait_for_capacity(usage, timeout=2)
        outcome.append("acquired")

    thread = threading.Thread(target=wait)
    thread.start()
    try:
        deadline = time.monotonic() + 1
        diagnostic = backend.introspect()
        while not diagnostic.waits and time.monotonic() < deadline:
            time.sleep(0.01)
            diagnostic = backend.introspect()
        assert len(diagnostic.waits) == 1
        assert diagnostic.waits[0].state == "waiting_for_capacity"
        assert diagnostic.waits[0].blocked_buckets[0].metric == "requests"
        backend.refund_capacity(usage, frozen_usage({"requests": 0}))
        thread.join(timeout=1)
        assert outcome == ["acquired"]
        assert backend.introspect().waits == ()
    finally:
        if thread.is_alive():
            backend.refund_capacity(usage, frozen_usage({"requests": 0}))
            thread.join(timeout=1)
        builder.close()


def test_sync_sqlite_metric_rebuilds_close_and_deregister_old_engines(
    tmp_path: Path,
) -> None:
    builder = SyncSqliteBackendBuilder(
        tmp_path / "sync-rebuild-lifecycle.sqlite3",
        key_prefix="rebuilds",
    )
    current = builder.build(_config(metric="requests"))
    replaced_engines = []
    try:
        for index in range(10):
            cfg = _config(metric="tokens" if index % 2 == 0 else "requests")
            replacement = builder.build(cfg)
            replaced_engines.append(current._engine)
            current = current.prepare_reconfigured_backend(replacement, cfg)
            assert len(builder._engines) == 1
        for engine in replaced_engines:
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                engine.inspect_counts()
    finally:
        builder.close()


def test_sqlite_key_prefixes_isolate_capacity_in_one_database(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "prefixes.sqlite3"
    first = _engine(db_path, key_prefix="first", limit=2.0)
    second = _engine(db_path, key_prefix="second", limit=2.0)
    try:
        first.consume(
            frozen_usage({"requests": 2}),
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        second_result = second.try_consume(
            frozen_usage({"requests": 2}),
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert second_result.available is True
    finally:
        first.close()
        second.close()


def test_sqlite_bucket_pruning_respects_each_rows_absolute_expiry(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "heterogeneous-ttls.sqlite3"
    long_lived = _engine(
        db_path,
        key_prefix="long-lived",
        limit=10.0,
        per_seconds=10,
        bucket_ttl_seconds=100,
    )
    short_lived = _engine(
        db_path,
        key_prefix="short-lived",
        limit=1.0,
        per_seconds=1,
        bucket_ttl_seconds=1,
    )
    try:
        long_lived.consume(
            frozen_usage({"requests": 10}),
            current_time=100.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        short_lived.consume(
            frozen_usage({"requests": 0}),
            current_time=102.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        preserved = long_lived.try_consume(
            frozen_usage({"requests": 0}),
            current_time=102.0,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        )
        assert preserved.result.fresh_bucket_ids == ()
        assert preserved.result.pre_capacities[("requests", 10)] == 2.0
    finally:
        long_lived.close()
        short_lived.close()


def test_sqlite_duplicate_acquire_is_visible_across_connections(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "connections.sqlite3"
    first = _engine(db_path)
    second = _engine(db_path)
    usage = frozen_usage({"requests": 1})
    try:
        first.consume(
            usage,
            current_time=100.0,
            reservation_id="shared-reservation",
            reservation_lifetime_seconds=20.0,
        )
        with pytest.raises(DuplicateRefundError) as duplicate:
            second.consume(
                usage,
                current_time=100.0,
                reservation_id="shared-reservation",
                reservation_lifetime_seconds=20.0,
            )
        assert duplicate.value.reason == "duplicate_acquire"
    finally:
        first.close()
        second.close()


def test_sqlite_acquire_marker_can_be_refunded_by_another_process(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cross-process-refund.sqlite3"
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_acquire_in_process,
        args=(str(db_path), "shared", "child-reservation", result_queue),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 0
    assert result_queue.get(timeout=1) == ("acquired", "child-reservation")

    builder = SyncSqliteBackendBuilder(db_path, key_prefix="shared")
    backend = builder.build(_config(limit=2.0))
    usage = frozen_usage({"requests": 2})
    bucket_ids = frozenset({("requests", 10)})
    try:
        assert backend.refund_capacity_for_buckets(
            usage,
            frozen_usage({"requests": 0}),
            bucket_ids=bucket_ids,
            reservation_id="child-reservation",
            reservation_model_family="sqlite-tests",
            reservation_bucket_ids=bucket_ids,
            reservation_reserved_usage=usage,
        )
        assert backend.wait_for_capacity(usage, timeout=0) is not None
    finally:
        builder.close()
        result_queue.close()


def test_sqlite_cross_process_try_acquire_has_one_atomic_winner(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cross-process-contention.sqlite3"
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_acquire_in_process,
            args=(str(db_path), "shared", f"reservation-{index}", result_queue),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    results = [result_queue.get(timeout=1) for _ in processes]
    assert [status for status, _ in results].count("acquired") == 1
    assert results.count(("error", "TimeoutError")) == 1
    with sqlite3.connect(db_path) as connection:
        marker_count = connection.execute(
            "SELECT COUNT(*) FROM acquire_markers WHERE key_prefix = 'shared'"
        ).fetchone()
    assert marker_count == (1,)
    result_queue.close()


def test_sqlite_builder_resolves_realpath_and_validates_prefix(tmp_path: Path) -> None:
    unresolved = tmp_path / "nested" / ".." / "state.sqlite3"
    builder = SyncSqliteBackendBuilder(unresolved, key_prefix="scope")
    assert builder.db_path == os.path.realpath(unresolved)
    builder.close()

    with pytest.raises(ValueError, match="key_prefix must not be empty"):
        SyncSqliteBackendBuilder(tmp_path / "other.sqlite3", key_prefix="")


def test_sqlite_ttl_invariants_match_durable_reservation_margin() -> None:
    derived = derive_default_max_reservation_lifetime_seconds_from_ttls(
        bucket_ttl_seconds=100,
        refund_dedup_ttl_seconds=80,
    )
    assert derived < 40.0
    assert math.nextafter(40.0, 0.0) == derived

    with pytest.raises(ValueError, match="SQLite TTLs must exceed"):
        validate_reservation_lifetime_ttl_invariant(
            max_reservation_lifetime_seconds=40.0,
            bucket_ttl_seconds=100,
            refund_dedup_ttl_seconds=80,
        )


def test_sqlite_builder_lifetime_knob_is_a_fail_fast_upper_bound(
    tmp_path: Path,
) -> None:
    builder = SyncSqliteBackendBuilder(
        tmp_path / "builder-lifetime.sqlite3",
        key_prefix="scope",
        max_reservation_lifetime_seconds=10.0,
    )
    try:
        assert builder.resolve_max_reservation_lifetime_seconds(None) == 10.0
        assert builder.resolve_max_reservation_lifetime_seconds(5.0) == 5.0
        with pytest.raises(ValueError, match="configured maximum"):
            builder.resolve_max_reservation_lifetime_seconds(11.0)
    finally:
        builder.close()


def test_sqlite_builder_rejects_bucket_ttl_shorter_than_quota_window(
    tmp_path: Path,
) -> None:
    builder = SyncSqliteBackendBuilder(
        tmp_path / "short-ttl.sqlite3",
        key_prefix="scope",
        bucket_ttl_seconds=9,
        refund_dedup_ttl_seconds=100,
        max_reservation_lifetime_seconds=4,
    )
    try:
        with pytest.raises(ValueError, match="bucket_ttl_seconds must be >="):
            builder.build(_config(per_seconds=10))
    finally:
        builder.close()


def test_sqlite_marker_creation_without_lifetime_rolls_back_consumption(
    tmp_path: Path,
) -> None:
    builder = SyncSqliteBackendBuilder(
        tmp_path / "lifetime.sqlite3",
        key_prefix="scope",
    )
    backend = builder.build(_config(limit=2.0))
    usage = frozen_usage({"requests": 2})
    try:
        with pytest.raises(
            ValueError, match="reservation_lifetime_seconds is required"
        ):
            backend.wait_for_capacity(
                usage,
                timeout=0,
                reservation_id="missing-lifetime",
            )
        assert backend.wait_for_capacity(usage, timeout=0) is not None
    finally:
        builder.close()


def test_sqlite_authoritative_refund_requires_all_marker_metadata(
    tmp_path: Path,
) -> None:
    builder = SyncSqliteBackendBuilder(
        tmp_path / "metadata.sqlite3",
        key_prefix="scope",
    )
    backend = builder.build(_config(limit=2.0))
    usage = frozen_usage({"requests": 2})
    bucket_ids = frozenset({("requests", 10)})
    try:
        backend.wait_for_capacity(
            usage,
            reservation_id="metadata-reservation",
            reservation_lifetime_seconds=20.0,
        )
        with pytest.raises(ValueError, match="marker metadata is required"):
            backend.refund_capacity_for_buckets(
                usage,
                frozen_usage({"requests": 0}),
                bucket_ids=bucket_ids,
                reservation_id="metadata-reservation",
            )
        with pytest.raises(TimeoutError):
            backend.wait_for_capacity(usage, timeout=0)
    finally:
        builder.close()
