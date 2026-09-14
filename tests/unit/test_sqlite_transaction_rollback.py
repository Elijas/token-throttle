"""Failures after partial SQL writes must preserve retryable accounting."""

from __future__ import annotations

import sqlite3

import pytest

from token_throttle import frozen_usage
from token_throttle._limiter_backends._sqlite._engine import BucketSpec, SqliteEngine


def _engine(path):
    return SqliteEngine(
        db_path=str(path),
        key_prefix="rollback",
        model_family="rollback",
        buckets=(BucketSpec("requests", 10, 10), BucketSpec("tokens", 20, 20)),
        bucket_ttl_seconds=100,
        refund_dedup_ttl_seconds=100,
        override_ttl_seconds=10,
        max_reservation_lifetime_seconds=20,
    )


def _rows(engine):
    return {
        table: engine._connection.execute(
            f"SELECT * FROM {table} ORDER BY rowid"  # noqa: S608
        ).fetchall()
        for table in ("buckets", "acquire_markers", "refund_tombstones")
    }


@pytest.mark.parametrize(
    "operation", ["acquire", "refund", "override", "configured_limit"]
)
def test_partial_transaction_failure_rolls_back_and_retry_survives_reopen(
    tmp_path, operation
):
    path = tmp_path / "rollback.db"
    engine = _engine(path)
    usage = frozen_usage({"requests": 4, "tokens": 6})
    engine.initialize_buckets(clock=lambda: 100)
    engine.consume(
        usage, reservation_id="held", reservation_lifetime_seconds=20, clock=lambda: 100
    )
    engine.set_max_capacity("requests", 10, 8, clock=lambda: 100)
    before = _rows(engine)

    def perform():
        if operation == "acquire":
            return engine.try_consume(
                usage,
                reservation_id="next",
                reservation_lifetime_seconds=20,
                clock=lambda: 101,
            )
        if operation == "refund":
            return engine.refund(
                usage,
                frozen_usage({"requests": 1, "tokens": 2}),
                refund_bucket_ids=engine.bucket_ids,
                reservation_id="held",
                reservation_model_family="rollback",
                reservation_bucket_ids=engine.bucket_ids,
                reservation_reserved_usage=usage,
                clock=lambda: 101,
            )
        method = (
            "set_max_capacity"
            if operation == "override"
            else "apply_configured_max_capacity"
        )
        return getattr(engine, method)("requests", 10, 5, clock=lambda: 101)

    table = "acquire_markers" if operation == "acquire" else "refund_tombstones"
    trigger = (
        f"BEFORE INSERT ON {table}"
        if operation in {"acquire", "refund"}
        else "BEFORE UPDATE OF override_value ON buckets"
    )
    try:
        engine._connection.execute(
            f"CREATE TEMP TRIGGER reject_write {trigger} BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
            perform()
        assert _rows(engine) == before
        assert not engine._connection.in_transaction
        assert engine.configured_max_capacity("requests", 10) == 10
        engine.close()
        engine = _engine(path)
        assert _rows(engine) == before
        perform()
        assert not engine._connection.in_transaction
        if operation == "acquire":
            assert engine.inspect_counts()["acquire_markers"] == 2
        elif operation == "refund":
            assert engine.inspect_counts()["acquire_markers"] == 0
            assert engine.inspect_counts()["refund_tombstones"] == 1
        else:
            snapshots, _ = engine.inspect_snapshot(clock=lambda: 101)
            assert snapshots[0].effective_max_capacity == 5
    finally:
        engine.close()
