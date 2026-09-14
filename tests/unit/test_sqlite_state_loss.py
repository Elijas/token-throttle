"""Owned-store loss injection and committed-observer evidence regressions."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from token_throttle import frozen_usage
from token_throttle._limiter_backends._sqlite._engine import BucketSpec, SqliteEngine


def _engine(path, *, ttl=100, buckets=None):
    return SqliteEngine(
        db_path=str(path),
        key_prefix="loss",
        model_family="loss",
        buckets=(BucketSpec("requests", 10, 10),) if buckets is None else buckets,
        bucket_ttl_seconds=ttl,
        refund_dedup_ttl_seconds=100,
        override_ttl_seconds=100,
        max_reservation_lifetime_seconds=20,
    )


def _consume(engine, now=100):
    return engine.consume(
        frozen_usage({"requests": 1}),
        reservation_id=None,
        reservation_lifetime_seconds=None,
        clock=lambda: now,
    )


def _drop(engine, field):
    if field == "row":
        engine._connection.execute("DELETE FROM buckets")
    else:
        engine._connection.execute(f"UPDATE buckets SET {field} = NULL")  # noqa: S608


@pytest.mark.parametrize("field", ["row", "capacity", "last_checked"])
def test_loss_diagnostic_is_pure_and_blocked_attempt_repairs_durably(tmp_path, field):
    path = tmp_path / "loss.db"
    with closing(_engine(path)) as engine:
        _consume(engine)
        _drop(engine, field)
        before = engine._connection.execute("SELECT * FROM buckets").fetchall()
        for _ in range(2):
            snapshots, _counts = engine.inspect_snapshot(clock=lambda: 100)
            assert snapshots[0].current_capacity == 0
            assert (
                engine._connection.execute("SELECT * FROM buckets").fetchall() == before
            )
        attempt = engine.try_consume(
            frozen_usage({"requests": 1}),
            reservation_id=None,
            reservation_lifetime_seconds=None,
            clock=lambda: 100,
        )
        assert not attempt.available
        (event,) = attempt.result.missing_state_events
        assert event.reason == "state_loss_drained"
        assert set(event.missing_fields) == (
            {"last_checked", "capacity"} if field == "row" else {field}
        )
        with closing(_engine(path)) as reopened:
            assert (
                reopened.inspect_snapshot(clock=lambda: 100)[0][0].current_capacity == 0
            )
        assert engine.inspect_snapshot(clock=lambda: 101)[0][0].current_capacity == 1


@pytest.mark.parametrize("now, expected", [(99, 0), (189.999, 0), (190, 10), (200, 10)])
def test_total_loss_confirmation_boundary_and_backwards_clock(tmp_path, now, expected):
    with closing(_engine(tmp_path / "loss.db")) as engine:
        _consume(engine)
        _drop(engine, "row")
        assert (
            engine.inspect_snapshot(clock=lambda: now)[0][0].current_capacity
            == expected
        )


def test_cold_reader_does_not_manufacture_confirmation(tmp_path):
    path = tmp_path / "loss.db"
    with closing(_engine(path)) as writer, closing(_engine(path)) as reader:
        _consume(writer)
        assert reader.inspect_snapshot(clock=lambda: 100)[0][0].current_capacity == 9
        _drop(writer, "row")
        assert reader.inspect_snapshot(clock=lambda: 100)[0][0].current_capacity == 10
        assert writer.inspect_snapshot(clock=lambda: 100)[0][0].current_capacity == 0


@pytest.mark.parametrize("failure", ["rollback", "commit"])
def test_failed_transaction_does_not_create_or_refresh_proof(tmp_path, failure):
    with closing(_engine(tmp_path / "loss.db")) as engine:
        original = engine._connection

        class FailingConnection:
            def execute(self, sql, *args):
                if sql == "COMMIT":
                    raise sqlite3.OperationalError("injected commit failure")
                return original.execute(sql, *args)

        if failure == "commit":
            engine._connection = FailingConnection()
        else:
            original.execute(
                "CREATE TEMP TRIGGER reject_marker BEFORE INSERT ON acquire_markers BEGIN SELECT RAISE(ABORT, 'injected rollback'); END"
            )
        try:
            with pytest.raises(sqlite3.Error, match="injected"):
                engine.consume(
                    frozen_usage({"requests": 1}),
                    reservation_id="fail",
                    reservation_lifetime_seconds=20,
                    clock=lambda: 100,
                )
        finally:
            engine._connection = original
        assert not original.in_transaction
        engine.initialize_buckets(clock=lambda: 100)
        assert engine.inspect_snapshot(clock=lambda: 100)[0][0].current_capacity == 10
        original.execute("DROP TRIGGER IF EXISTS reject_marker")
        _consume(engine)
        if failure == "commit":
            engine._connection = FailingConnection()
        else:
            original.execute(
                "CREATE TEMP TRIGGER reject_marker BEFORE INSERT ON acquire_markers BEGIN SELECT RAISE(ABORT, 'injected rollback'); END"
            )
        try:
            with pytest.raises(sqlite3.Error, match="injected"):
                engine.consume(
                    frozen_usage({"requests": 1}),
                    reservation_id="fail",
                    reservation_lifetime_seconds=20,
                    clock=lambda: 189,
                )
        finally:
            engine._connection = original
        _drop(engine, "row")
        engine.initialize_buckets(clock=lambda: 190)
        assert engine.inspect_snapshot(clock=lambda: 190)[0][0].current_capacity == 10


def test_warm_diagnostics_and_blocked_reads_do_not_extend_proof(tmp_path):
    with closing(_engine(tmp_path / "loss.db")) as engine:
        _consume(engine)
        # Diagnostic reads at the edge must not renew the original write proof.
        engine.inspect_snapshot(clock=lambda: 189)
        _drop(engine, "row")
        assert engine.inspect_snapshot(clock=lambda: 190)[0][0].current_capacity == 10


def test_failed_loss_repair_base_exception_does_not_leak_to_later_commit(
    tmp_path, monkeypatch
):
    class AbortRepair(BaseException):
        pass

    with closing(_engine(tmp_path / "loss.db")) as engine:
        _consume(engine)
        _drop(engine, "row")
        original_write = engine._write_capacities

        def abort(*args):
            original_write(*args)
            raise AbortRepair

        monkeypatch.setattr(engine, "_write_capacities", abort)
        with pytest.raises(AbortRepair):
            _consume(engine, 189)
        assert not engine._connection.in_transaction
        assert engine._connection.execute("SELECT * FROM buckets").fetchall() == []
        # A later successful empty initialization cannot promote aborted proof.
        engine.initialize_buckets(clock=lambda: 190)
        assert engine.inspect_snapshot(clock=lambda: 190)[0][0].current_capacity == 10


def test_committed_override_expiry_refresh_proves_state_for_cold_engine(tmp_path):
    path = tmp_path / "loss.db"
    with closing(_engine(path)) as writer:
        _consume(writer)
        writer.set_max_capacity("requests", 10, 8, clock=lambda: 100)
        # Keep the bucket live until the fixed override TTL expires.
        _consume(writer, 150)
        with closing(_engine(path)) as observer:
            # A duplicate replay still applies and commits the expired override
            # transition. That complete state/TTL write establishes proof.
            observer._connection.execute(
                "INSERT INTO acquire_markers VALUES ('loss', 'replay', 'loss', '[[\"requests\",10]]', '[[\"requests\",1.0]]', 199, 220)"
            )
            result = observer.consume(
                frozen_usage({"requests": 1}),
                reservation_id="replay",
                reservation_lifetime_seconds=20,
                clock=lambda: 200,
            )
            assert result.replayed
            _drop(observer, "row")
            assert (
                observer.inspect_snapshot(clock=lambda: 200)[0][0].current_capacity == 0
            )


@pytest.mark.parametrize("now, expected", [(189, 0), (190, 10), (200, 10)])
def test_reconfiguration_cannot_extend_source_proof_with_longer_ttl(
    tmp_path, now, expected
):
    path = tmp_path / "loss.db"
    with (
        closing(_engine(path)) as source,
        closing(_engine(path, ttl=1000)) as replacement,
    ):
        _consume(source)
        _drop(source, "row")
        replacement.initialize_buckets(clock=lambda: now)
        replacement.inherit_state_confirmations(source)
        assert (
            replacement.inspect_snapshot(clock=lambda: now)[0][0].current_capacity
            == expected
        )


def test_removed_identity_does_not_resurrect_proof_when_reintroduced(tmp_path):
    path = tmp_path / "loss.db"
    with closing(_engine(path)) as source:
        _consume(source)
        with closing(_engine(path, buckets=(BucketSpec("tokens", 10, 20),))) as middle:
            middle.inherit_state_confirmations(source)
            with closing(_engine(path)) as replacement:
                _drop(source, "row")
                replacement.inherit_state_confirmations(middle)
                assert (
                    replacement.inspect_snapshot(clock=lambda: 100)[0][
                        0
                    ].current_capacity
                    == 10
                )
