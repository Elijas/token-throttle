from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from typing import TYPE_CHECKING

import pytest

from token_throttle import frozen_usage
from token_throttle._limiter_backends._sqlite._engine import BucketSpec, SqliteEngine

if TYPE_CHECKING:
    from pathlib import Path


BUCKET_ID = ("requests", 10)


def _engine(db_path: Path) -> SqliteEngine:
    return SqliteEngine(
        db_path=str(db_path),
        key_prefix="expiry-refill",
        model_family="shared",
        buckets=(BucketSpec("requests", 10, 10.0),),
        bucket_ttl_seconds=1000,
        refund_dedup_ttl_seconds=1000,
        override_ttl_seconds=2,
        max_reservation_lifetime_seconds=20.0,
    )


def _consume(engine: SqliteEngine, amount: float, now: float):
    return engine.consume(
        frozen_usage({"requests": amount}),
        reservation_id=None,
        reservation_lifetime_seconds=None,
        clock=lambda: now,
    )


def _try_consume(engine: SqliteEngine, amount: float, now: float):
    return engine.try_consume(
        frozen_usage({"requests": amount}),
        reservation_id=None,
        reservation_lifetime_seconds=None,
        clock=lambda: now,
    )


def _seed(engine: SqliteEngine, override: float, amount: float = 10.0) -> None:
    engine.initialize_buckets(clock=lambda: 100.0)
    _consume(engine, amount, 100.0)
    engine.set_max_capacity(*BUCKET_ID, override, clock=lambda: 100.0)


@pytest.mark.parametrize("operation", ["consume", "try_consume", "refund"])
@pytest.mark.parametrize(
    ("override", "now", "expected"),
    [
        (1.0, 101.0, 0.1),
        (1.0, 102.0, 0.2),
        (1.0, 105.0, 3.2),
        (1.0, 120.0, 10.0),
        (20.0, 101.0, 2.0),
        (20.0, 102.0, 4.0),
        (20.0, 105.0, 7.0),
        (100.0, 102.0, 10.0),
    ],
)
def test_override_expiry_refill_reads_match_writes(
    tmp_path: Path, operation: str, override: float, now: float, expected: float
) -> None:
    db_path = tmp_path / "expiry-refill.sqlite3"
    with closing(_engine(db_path)) as writer, closing(_engine(db_path)) as reader:
        _seed(writer, override)
        before = reader._connection.execute("SELECT * FROM buckets").fetchall()
        changes = reader._connection.total_changes
        for _ in range(2):
            snapshots, _counts = reader.inspect_snapshot(clock=lambda: now)
            assert snapshots[0].current_capacity == pytest.approx(expected)
            assert snapshots[0].override_active is (now < 102.0)
            assert snapshots[0].effective_max_capacity == (
                override if now < 102.0 else 10.0
            )
            assert snapshots[0].is_fresh_start is False
        assert reader._connection.execute("SELECT * FROM buckets").fetchall() == before
        assert reader._connection.total_changes == changes
        if operation == "consume":
            result = _consume(writer, 0.1, now)
        elif operation == "try_consume":
            attempt = _try_consume(writer, 0.1, now)
            assert attempt.available
            result = attempt.result
        else:
            result = writer.refund(
                frozen_usage({"requests": 0.0}),
                frozen_usage({"requests": 0.1}),
                refund_bucket_ids=frozenset({BUCKET_ID}),
                reservation_id=None,
                reservation_model_family=None,
                reservation_bucket_ids=None,
                reservation_reserved_usage=None,
                clock=lambda: now,
            )
        assert result.pre_capacities[BUCKET_ID] == pytest.approx(expected)
        assert result.post_capacities[BUCKET_ID] == pytest.approx(expected - 0.1)
        snapshots, _counts = reader.inspect_snapshot(clock=lambda: now)
        assert snapshots[0].current_capacity == pytest.approx(expected - 0.1)


def test_override_expiry_failed_attempt_preserves_refill_history(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "failed-expiry-refill.sqlite3"
    with closing(_engine(db_path)) as first, closing(_engine(db_path)) as second:
        _seed(first, 1.0)
        for engine in (first, second, first):
            attempt = _try_consume(engine, 1.0, 102.0)
            assert attempt.available is False
            assert attempt.result.pre_capacities[BUCKET_ID] == pytest.approx(0.2)
        later = _try_consume(second, 1.0, 103.0)
        assert later.available
        assert later.result.pre_capacities[BUCKET_ID] == pytest.approx(1.2)


@pytest.mark.parametrize(("override", "expected"), [(1.0, -4.8), (20.0, -1.0)])
def test_override_expiry_preserves_debt(
    tmp_path: Path, override: float, expected: float
) -> None:
    with closing(_engine(tmp_path / "debt-expiry-refill.sqlite3")) as engine:
        _seed(engine, override, amount=15.0)
        for now, capacity in ((102.0, expected), (103.0, expected + 1.0)):
            snapshots, _counts = engine.inspect_snapshot(clock=lambda now=now: now)
            assert snapshots[0].current_capacity == pytest.approx(capacity)
            attempt = _try_consume(engine, 0.1, now)
            assert attempt.available is False
            assert attempt.result.pre_capacities[BUCKET_ID] == pytest.approx(capacity)


@pytest.mark.parametrize("now", [100.0, 101.0, 102.0, 103.0])
def test_override_expiry_retains_uncapped_lower_then_raise(
    tmp_path: Path, now: float
) -> None:
    with closing(_engine(tmp_path / "overflow-expiry-refill.sqlite3")) as engine:
        _seed(engine, 1.0, amount=0.0)
        snapshots, _counts = engine.inspect_snapshot(clock=lambda: now)
        assert snapshots[0].current_capacity == (1.0 if now < 102.0 else 10.0)
        engine.set_max_capacity(*BUCKET_ID, 20.0, clock=lambda: now)
        result = _try_consume(engine, 0.0, now)
        expected = 10.0 + min(now - 100.0, 2.0) * 0.1 + max(now - 102.0, 0.0)
        assert result.result.pre_capacities[BUCKET_ID] == pytest.approx(expected)


@pytest.mark.parametrize("share_engine", [False, True])
def test_override_expiry_concurrent_consumers_share_refill(
    tmp_path: Path, *, share_engine: bool
) -> None:
    db_path = tmp_path / "concurrent-expiry-refill.sqlite3"
    with closing(_engine(db_path)) as first, closing(_engine(db_path)) as second:
        _seed(first, 1.0)
        start = threading.Barrier(3)

        def attempt(engine: SqliteEngine):
            start.wait(timeout=5)
            return _try_consume(engine, 0.15, 102.0)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(attempt, first),
                pool.submit(attempt, first if share_engine else second),
            ]
            start.wait(timeout=5)
            results = [future.result(timeout=5) for future in futures]
        assert sum(result.available for result in results) == 1
        snapshots, _counts = second.inspect_snapshot(clock=lambda: 102.0)
        assert snapshots[0].current_capacity == pytest.approx(0.05)


@pytest.mark.parametrize(("override", "expected"), [(1.0, 0.2), (20.0, 4.0)])
def test_override_expiry_after_intermediate_capacity_write(
    tmp_path: Path, override: float, expected: float
) -> None:
    with closing(_engine(tmp_path / "intermediate-expiry-refill.sqlite3")) as engine:
        _seed(engine, override)
        _consume(engine, 0.0, 101.0)
        result = _try_consume(engine, 0.0, 102.0)
        assert result.result.pre_capacities[BUCKET_ID] == pytest.approx(expected)


def test_override_expiry_does_not_restore_overflow_after_capacity_write(
    tmp_path: Path,
) -> None:
    with closing(_engine(tmp_path / "capped-expiry-refill.sqlite3")) as engine:
        _seed(engine, 1.0, amount=0.0)
        _consume(engine, 0.0, 101.0)
        engine.set_max_capacity(*BUCKET_ID, 20.0, clock=lambda: 102.0)
        result = _try_consume(engine, 0.0, 102.0)
        assert result.result.pre_capacities[BUCKET_ID] == pytest.approx(1.1)


@pytest.mark.parametrize("override", [1.0, 20.0])
def test_override_expiry_replacement_integrates_both_override_lifetimes(
    tmp_path: Path, override: float
) -> None:
    with closing(_engine(tmp_path / "replacement-expiry-refill.sqlite3")) as engine:
        _seed(engine, override)
        engine.set_max_capacity(*BUCKET_ID, 5.0, clock=lambda: 103.0)
        result = _try_consume(engine, 0.0, 106.0)
        assert result.result.pre_capacities[BUCKET_ID] == pytest.approx(
            2.0 * override / 10.0 + 1.0 + 1.0 + 1.0
        )


def test_override_expiry_failed_attempt_survives_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "reopened-expiry-refill.sqlite3"
    with closing(_engine(db_path)) as first:
        _seed(first, 1.0)
        assert _try_consume(first, 1.0, 102.0).available is False
    with closing(_engine(db_path)) as reopened:
        result = _try_consume(reopened, 1.0, 103.0)
        assert result.available
        assert result.result.pre_capacities[BUCKET_ID] == pytest.approx(1.2)
