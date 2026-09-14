"""Kill an owned writer between bucket and marker writes, then retry."""

from __future__ import annotations

import multiprocessing
import os
import signal

import pytest

from token_throttle import frozen_usage
from token_throttle._limiter_backends._sqlite._engine import BucketSpec, SqliteEngine


def _engine(path):
    return SqliteEngine(
        db_path=str(path),
        key_prefix="crash",
        model_family="crash",
        buckets=(BucketSpec("requests", 10, 10),),
        bucket_ttl_seconds=100,
        refund_dedup_ttl_seconds=100,
        override_ttl_seconds=100,
        max_reservation_lifetime_seconds=20,
    )


def _partial_writer(path, ready, hold):
    engine = _engine(path)
    insert_marker = engine._insert_marker

    def pause_after_capacity_write(*args, **kwargs):
        ready.set()
        if not hold.wait(15):
            raise RuntimeError("parent did not terminate writer")
        return insert_marker(*args, **kwargs)

    engine._insert_marker = pause_after_capacity_write
    try:
        engine.consume(
            frozen_usage({"requests": 4}),
            reservation_id="crashed",
            reservation_lifetime_seconds=20,
            clock=lambda: 100,
        )
    finally:
        engine.close()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX SIGKILL")
def test_killed_sqlite_write_rolls_back_capacity_and_marker(tmp_path):
    path = tmp_path / "crash.db"
    engine = _engine(path)
    engine.initialize_buckets(clock=lambda: 100)
    engine.consume(
        frozen_usage({"requests": 2}),
        reservation_id=None,
        reservation_lifetime_seconds=None,
        clock=lambda: 100,
    )
    context = multiprocessing.get_context("spawn")
    ready, hold = context.Event(), context.Event()
    process = context.Process(target=_partial_writer, args=(path, ready, hold))
    process.start()
    try:
        assert ready.wait(10), "writer never reached partial transaction"
        # WAL readers see the committed state even while a writer has changed it.
        snapshots, counts = engine.inspect_snapshot(clock=lambda: 100)
        assert snapshots[0].current_capacity == 8
        assert counts["acquire_markers"] == 0
        assert process.pid is not None
        os.kill(process.pid, signal.SIGKILL)
        process.join(5)
        assert process.exitcode == -signal.SIGKILL
        engine.close()
        engine = _engine(path)
        result = engine.try_consume(
            frozen_usage({"requests": 8}),
            reservation_id="retry",
            reservation_lifetime_seconds=20,
            clock=lambda: 100,
        )
        assert result.available
        assert engine.inspect_counts()["acquire_markers"] == 1
        assert engine._connection.execute("PRAGMA integrity_check").fetchone() == (
            "ok",
        )
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)
        engine.close()
