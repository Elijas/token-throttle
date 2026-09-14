from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

from token_throttle import (
    PerModelConfig,
    Quota,
    SyncSqliteBackendBuilder,
    UsageQuotas,
    frozen_usage,
)
from token_throttle._limiter_backends._sqlite._engine import BucketSpec, SqliteEngine

if TYPE_CHECKING:
    from pathlib import Path


def _engine(db_path: Path, *, key_prefix: str = "retention") -> SqliteEngine:
    return SqliteEngine(
        db_path=str(db_path),
        key_prefix=key_prefix,
        model_family="override-retention",
        buckets=(BucketSpec("requests", 10, 10.0),),
        bucket_ttl_seconds=10,
        override_ttl_seconds=20,
        refund_dedup_ttl_seconds=100,
        max_reservation_lifetime_seconds=4.0,
        prune_batch_size=1,
    )


def _drain_and_override(engine: SqliteEngine, *, usage: float = 10.0) -> None:
    engine.initialize_buckets(clock=lambda: 100.0)
    engine.consume(
        frozen_usage({"requests": usage}),
        reservation_id=None,
        reservation_lifetime_seconds=None,
        clock=lambda: 100.0,
    )
    engine.set_max_capacity("requests", 10, 1.0, clock=lambda: 100.0)


def test_sqlite_builder_accepts_override_ttl_longer_than_bucket_ttl(
    tmp_path: Path,
) -> None:
    builder = SyncSqliteBackendBuilder(
        tmp_path / "builder.sqlite3",
        key_prefix="retention",
        bucket_ttl_seconds=20,
        override_ttl_seconds=40,
        refund_dedup_ttl_seconds=100,
        max_reservation_lifetime_seconds=4.0,
    )
    try:
        builder.build(
            PerModelConfig(
                model_family="override-retention",
                quotas=UsageQuotas(
                    [Quota(metric="requests", limit=10, per_seconds=10)]
                ),
            )
        )
    finally:
        builder.close()


@pytest.mark.parametrize("current_time", [110.0, 111.0, 119.0])
@pytest.mark.parametrize("usage", [10.0, 20.0])
def test_sqlite_snapshot_retains_override_with_fresh_idle_capacity(
    tmp_path: Path, current_time: float, usage: float
) -> None:
    db_path = tmp_path / "snapshot.sqlite3"
    engine = _engine(db_path)
    try:
        _drain_and_override(engine, usage=usage)
        snapshots, _ = engine.inspect_snapshot(clock=lambda: current_time)
        snapshot = snapshots[0]
        assert snapshot.override_active
        assert snapshot.effective_max_capacity == 1.0
        assert snapshot.current_capacity == 1.0
        assert snapshot.is_fresh_start
        with sqlite3.connect(db_path) as connection:
            assert connection.execute(
                "SELECT capacity, override_value, override_expires_at, expires_at "
                "FROM buckets"
            ).fetchone() == (10.0 - usage, 1.0, 120.0, 110.0)
    finally:
        engine.close()


def test_sqlite_write_retains_override_without_extending_its_expiry(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "write.sqlite3"
    engine = _engine(db_path)
    try:
        _drain_and_override(engine)
        result = engine.consume(
            frozen_usage({"requests": 1.0}),
            reservation_id=None,
            reservation_lifetime_seconds=None,
            clock=lambda: 111.0,
        )
        assert result.max_capacities[("requests", 10)] == 1.0
        assert result.pre_capacities[("requests", 10)] == 1.0
        assert result.post_capacities[("requests", 10)] == 0.0
        assert result.fresh_bucket_ids == (("requests", 10),)
        with sqlite3.connect(db_path) as connection:
            assert connection.execute(
                "SELECT override_value, override_expires_at, expires_at FROM buckets"
            ).fetchone() == (1.0, 120.0, 121.0)
        snapshots, _ = engine.inspect_snapshot(clock=lambda: 119.0)
        assert snapshots[0].override_active
        assert snapshots[0].effective_max_capacity == 1.0
        snapshots, _ = engine.inspect_snapshot(clock=lambda: 120.0)
        assert not snapshots[0].override_active
        assert snapshots[0].effective_max_capacity == 10.0
    finally:
        engine.close()


@pytest.mark.parametrize("cleanup_prefix", ["retention", "another-namespace"])
@pytest.mark.parametrize("cleanup_time", [111.0, 120.0])
def test_sqlite_cleanup_from_another_connection_respects_override_expiry(
    tmp_path: Path, cleanup_prefix: str, cleanup_time: float
) -> None:
    db_path = tmp_path / "cleanup.sqlite3"
    engine = _engine(db_path)
    try:
        _drain_and_override(engine)
    finally:
        engine.close()
    start_cleanup = threading.Event()

    def cleanup() -> None:
        assert start_cleanup.wait(timeout=5.0)
        cleaner = _engine(db_path, key_prefix=cleanup_prefix)
        try:
            cleaner.initialize_buckets(clock=lambda: cleanup_time)
        finally:
            cleaner.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        completed = executor.submit(cleanup)
        start_cleanup.set()
        completed.result(timeout=5.0)

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT override_value, override_expires_at FROM buckets "
            "WHERE key_prefix = 'retention'"
        ).fetchone()
    if cleanup_time < 120.0:
        assert row == (1.0, 120.0)
    elif cleanup_prefix == "retention":
        assert row == (None, None)
    else:
        assert row is None
    reopened = _engine(db_path)
    try:
        snapshots, _ = reopened.inspect_snapshot(clock=lambda: cleanup_time)
        snapshot = snapshots[0]
        expected_max = 1.0 if cleanup_time < 120.0 else 10.0
        assert snapshot.effective_max_capacity == expected_max
        assert snapshot.current_capacity == expected_max
        assert snapshot.override_active == (cleanup_time < 120.0)
        assert snapshot.is_fresh_start
    finally:
        reopened.close()


def test_sqlite_replacing_override_after_idle_expiry_discards_old_capacity(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path / "replace.sqlite3")
    try:
        _drain_and_override(engine, usage=20.0)
        engine.set_max_capacity("requests", 10, 2.0, clock=lambda: 111.0)
        snapshots, _ = engine.inspect_snapshot(clock=lambda: 111.0)
        assert snapshots[0].current_capacity == 2.0
        assert snapshots[0].effective_max_capacity == 2.0
        assert snapshots[0].is_fresh_start
    finally:
        engine.close()
