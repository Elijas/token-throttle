"""Loss reporting and observer retention across async/sync SQLite rebuilds."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing

import pytest

from token_throttle import (
    PerModelConfig,
    Quota,
    RateLimiter,
    RateLimiterCallbacks,
    SqliteBackendBuilder,
    SyncRateLimiter,
    SyncRateLimiterCallbacks,
    SyncSqliteBackendBuilder,
    UsageQuotas,
    frozen_usage,
)
from token_throttle._limiter_backends._sqlite._engine import SqliteEngine


def _config(**limits):
    return PerModelConfig(
        model_family="loss",
        quotas=UsageQuotas(
            [
                Quota(metric=metric, limit=limit, per_seconds=10)
                for metric, limit in limits.items()
            ]
        ),
    )


@pytest.fixture(params=["sync", "async"])
async def harness(request, tmp_path, monkeypatch):
    events = []

    def callback(**kwargs):
        events.append(kwargs)

    async def async_callback(**kwargs):
        events.append(kwargs)

    def use_clock(original):
        def transaction(engine, **kwargs):
            return original(engine, **{**kwargs, "clock": lambda: 100})

        return transaction

    for name in ("_transaction", "_read_transaction"):
        monkeypatch.setattr(SqliteEngine, name, use_clock(getattr(SqliteEngine, name)))
    configs = [_config(requests=10)]
    path = tmp_path / "loss.db"
    if request.param == "sync":
        builder = SyncSqliteBackendBuilder(path, key_prefix="loss")
        limiter = SyncRateLimiter(
            lambda _model: configs[0],
            backend=builder,
            callbacks=SyncRateLimiterCallbacks(on_missing_consumption_data=callback),
        )
    else:
        builder = SqliteBackendBuilder(path, key_prefix="loss")
        limiter = RateLimiter(
            lambda _model: configs[0],
            backend=builder,
            callbacks=RateLimiterCallbacks(on_missing_consumption_data=async_callback),
        )

    async def call(method, *args, **kwargs):
        operation = getattr(limiter, method)
        if request.param == "sync":
            return await asyncio.to_thread(operation, *args, **kwargs)
        return await operation(*args, **kwargs)

    try:
        yield call, limiter, configs, path, events
    finally:
        await call("aclose" if request.param == "async" else "close")


@pytest.mark.parametrize("field", ["row", "capacity", "last_checked"])
async def test_blocked_loss_callback_has_exact_metadata(harness, field):
    call, _limiter, _configs, path, events = harness
    await call("record_usage", {"requests": 1}, "model")
    assert events[0]["missing_state_reason"] == "fresh_start"
    assert events[0]["missing_state_keys"] == ("last_checked", "capacity")
    events.clear()
    with closing(sqlite3.connect(path, isolation_level=None)) as connection:
        connection.execute(
            "DELETE FROM buckets"
            if field == "row"
            else f"UPDATE buckets SET {field} = NULL"  # noqa: S608
        )
    for _ in range(2):
        with pytest.raises(TimeoutError):
            await call("acquire_capacity", {"requests": 1}, "model", timeout=0)
    assert len(events) == 1
    event = events[0]
    assert event["missing_state_reason"] == "state_loss_drained"
    assert event["missing_state_keys"] == tuple(
        name for name in ("last_checked", "capacity") if field in ("row", name)
    )
    assert event["present_state_keys"] == tuple(
        name for name in ("last_checked", "capacity") if field not in ("row", name)
    )


@pytest.mark.parametrize("wipe_before", [True, False])
async def test_rebuild_keeps_surviving_proof_and_new_buckets_start_full(
    harness, wipe_before
):
    call, limiter, configs, path, _events = harness
    await call("record_usage", {"requests": 1}, "model")
    old_backend = limiter._model_family_to_backend["loss"]

    def wipe():
        with closing(sqlite3.connect(path, isolation_level=None)) as connection:
            connection.execute("DELETE FROM buckets")

    if wipe_before:
        wipe()
    configs[0] = _config(requests=10, tokens=20)
    with pytest.warns(UserWarning, match="changed metric set"):
        await call("record_usage", {"requests": 0, "tokens": 0}, "model")
    new_backend = limiter._model_family_to_backend["loss"]
    assert new_backend is not old_backend
    assert (
        new_backend._engine.inspect_snapshot(clock=lambda: 100)[0][1].current_capacity
        == 20
    )
    if not wipe_before:
        wipe()
    with pytest.raises(TimeoutError):
        await call("acquire_capacity", {"requests": 1, "tokens": 0}, "model", timeout=0)


@pytest.mark.parametrize("field", ["row", "capacity", "last_checked"])
async def test_diagnostics_report_loss_without_repair(harness, field):
    call, limiter, _configs, path, _events = harness
    await call("record_usage", {"requests": 1}, "model")
    with closing(sqlite3.connect(path, isolation_level=None)) as connection:
        connection.execute(
            "DELETE FROM buckets"
            if field == "row"
            else f"UPDATE buckets SET {field} = NULL"  # noqa: S608
        )
        before = connection.execute("SELECT * FROM buckets").fetchall()
        backend = limiter._model_family_to_backend["loss"]
        # Engine snapshot is also the source for the backend diagnostic DTO.
        snapshot = backend._engine.inspect_snapshot(clock=lambda: 100)[0][0]
        assert snapshot.current_capacity == 0
        assert snapshot.missing_state_event.reason == "state_loss_drained"
        diagnostic = backend.introspect()
        if isinstance(limiter, RateLimiter):
            diagnostic = await diagnostic
        assert diagnostic.buckets[0].status == (
            "state_loss" if field == "row" else "partial_missing"
        )
        assert diagnostic.buckets[0].current_capacity == 0
        assert connection.execute("SELECT * FROM buckets").fetchall() == before


@pytest.mark.parametrize("acquisition", [True, False])
async def test_replayed_marker_reports_repair_without_consumption(harness, acquisition):
    call, limiter, _configs, path, events = harness
    await call("record_usage", {"requests": 0}, "model")
    backend = limiter._model_family_to_backend["loss"]

    async def consume():
        method = (
            (
                "await_for_capacity"
                if isinstance(limiter, RateLimiter)
                else "wait_for_capacity"
            )
            if acquisition
            else "consume_capacity"
        )
        result = getattr(backend, method)(
            frozen_usage({"requests": 1}),
            reservation_id="replayed",
            reservation_lifetime_seconds=20,
        )
        if isinstance(limiter, RateLimiter):
            await result

    await consume()
    events.clear()
    with closing(sqlite3.connect(path, isolation_level=None)) as connection:
        connection.execute("DELETE FROM buckets")
        await consume()
        await consume()
        assert connection.execute("SELECT capacity FROM buckets").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM acquire_markers WHERE reservation_id = 'replayed'"
            ).fetchone()[0]
            == 1
        )
    assert len(events) == 1
    assert events[0]["missing_state_reason"] == "state_loss_drained"
