from __future__ import annotations

import asyncio
import concurrent.futures
import sqlite3
import threading
from typing import TYPE_CHECKING

import pytest

from token_throttle import (
    PerModelConfig,
    Quota,
    SqliteBackendBuilder,
    SyncSqliteBackendBuilder,
    UsageQuotas,
    frozen_usage,
)

if TYPE_CHECKING:
    from pathlib import Path


def _config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="requests", limit=10, per_seconds=3600)]),
        model_family="queued-deadlines",
    )


def _observe_write_wait(engine) -> threading.Event:
    entered = threading.Event()

    def trace(statement: str) -> None:
        if statement == "BEGIN IMMEDIATE":
            entered.set()

    engine._connection.set_trace_callback(trace)
    return entered


@pytest.mark.parametrize("timeout", [0, 0.05])
async def test_async_acquire_deadline_includes_executor_queue(
    tmp_path: Path, timeout: float
) -> None:
    database = tmp_path / "executor-queue.sqlite3"
    builder = SqliteBackendBuilder(database, key_prefix="queued-deadlines")
    backend = builder.build(_config())
    entered = _observe_write_wait(backend._engine)
    writer = sqlite3.connect(database, isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")
    blocker = asyncio.create_task(
        backend.consume_capacity(frozen_usage({"requests": 0}))
    )
    acquire = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        acquire = asyncio.create_task(
            backend.await_for_capacity(
                frozen_usage({"requests": 4}),
                timeout=timeout,
                reservation_id="queued",
                reservation_lifetime_seconds=20,
            )
        )
        completed, _ = await asyncio.wait({acquire}, timeout=0.3)
        returned_while_blocked = acquire in completed
    finally:
        writer.execute("ROLLBACK")
        writer.close()
        await blocker
        outcome = (
            (await asyncio.gather(acquire, return_exceptions=True))[0]
            if acquire
            else None
        )
        snapshots, counts = await backend._run_engine(backend._engine.inspect_snapshot)
        await builder.aclose()
    assert isinstance(outcome, TimeoutError), repr(outcome)
    assert returned_while_blocked
    assert snapshots[0].current_capacity == pytest.approx(10)
    assert counts["acquire_markers"] == 0


@pytest.mark.parametrize("timeout", [0, 0.05])
async def test_async_acquire_deadline_includes_engine_lock_queue(
    tmp_path: Path, timeout: float
) -> None:
    builder = SqliteBackendBuilder(
        tmp_path / "async-engine-queue.sqlite3", key_prefix="queued-deadlines"
    )
    backend = builder.build(_config())
    locked = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with backend._engine._lock:
            locked.set()
            if not release.wait(5):
                raise RuntimeError("test did not release engine lock")

    holder = asyncio.create_task(asyncio.to_thread(hold_lock))
    acquire = None
    try:
        assert await asyncio.to_thread(locked.wait, 2)
        acquire = asyncio.create_task(
            backend.await_for_capacity(
                frozen_usage({"requests": 4}),
                timeout=timeout,
                reservation_id="engine-queued",
                reservation_lifetime_seconds=20,
            )
        )
        completed, _ = await asyncio.wait({acquire}, timeout=0.3)
        returned_while_blocked = acquire in completed
    finally:
        release.set()
        await holder
        outcome = (
            (await asyncio.gather(acquire, return_exceptions=True))[0]
            if acquire
            else None
        )
        snapshots, counts = await backend._run_engine(backend._engine.inspect_snapshot)
        await builder.aclose()
    assert isinstance(outcome, TimeoutError), repr(outcome)
    assert returned_while_blocked
    assert snapshots[0].current_capacity == pytest.approx(10)
    assert counts["acquire_markers"] == 0


@pytest.mark.parametrize("pause_after_commit", [False, True])
async def test_async_acquire_deadline_expiry_observes_started_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, pause_after_commit: bool
) -> None:
    builder = SqliteBackendBuilder(
        tmp_path / "started-operation.sqlite3", key_prefix="queued-deadlines"
    )
    backend = builder.build(_config())
    paused = threading.Event()
    release = threading.Event()
    original = backend._engine.try_consume

    def pause_operation(*args, **kwargs):
        result = original(*args, **kwargs) if pause_after_commit else None
        paused.set()
        if not release.wait(5):
            raise RuntimeError("test did not release started operation")
        return result if pause_after_commit else original(*args, **kwargs)

    monkeypatch.setattr(backend._engine, "try_consume", pause_operation)
    acquire = asyncio.create_task(
        backend.await_for_capacity(
            frozen_usage({"requests": 4}),
            timeout=0.1,
            reservation_id="started",
            reservation_lifetime_seconds=20,
        )
    )
    try:
        assert await asyncio.to_thread(paused.wait, 2)
        completed, _ = await asyncio.wait({acquire}, timeout=0.2)
        assert not completed
    finally:
        release.set()
        outcome = (await asyncio.gather(acquire, return_exceptions=True))[0]
        snapshots, counts = await backend._run_engine(backend._engine.inspect_snapshot)
        await builder.aclose()
    assert isinstance(outcome, TimeoutError), repr(outcome)
    assert snapshots[0].current_capacity == pytest.approx(10)
    assert counts["acquire_markers"] == 0


async def test_async_cancelled_executor_queue_acquire_never_consumes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "cancelled-queue.sqlite3"
    builder = SqliteBackendBuilder(database, key_prefix="queued-deadlines")
    backend = builder.build(_config())
    entered = _observe_write_wait(backend._engine)
    writer = sqlite3.connect(database, isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")
    blocker = asyncio.create_task(
        backend.consume_capacity(frozen_usage({"requests": 0}))
    )
    acquire = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        acquire = asyncio.create_task(
            backend.await_for_capacity(
                frozen_usage({"requests": 4}),
                reservation_id="cancelled-queue",
                reservation_lifetime_seconds=20,
            )
        )
        await asyncio.sleep(0)
        acquire.cancel()
        completed, _ = await asyncio.wait({acquire}, timeout=0.3)
        returned_while_blocked = acquire in completed
        probe = asyncio.create_task(
            backend.await_for_capacity(frozen_usage({"requests": 1}), timeout=0)
        )
        probe_completed, _ = await asyncio.wait({probe}, timeout=0.3)
        probe_returned_while_blocked = probe in probe_completed
    finally:
        writer.execute("ROLLBACK")
        writer.close()
        await blocker
        if acquire is not None:
            await asyncio.gather(acquire, return_exceptions=True)
        snapshots, counts = await backend._run_engine(backend._engine.inspect_snapshot)
        await builder.aclose()
    probe_outcome = (await asyncio.gather(probe, return_exceptions=True))[0]
    assert returned_while_blocked
    assert acquire.cancelled()
    assert probe_returned_while_blocked
    assert isinstance(probe_outcome, TimeoutError), repr(probe_outcome)
    assert snapshots[0].current_capacity == pytest.approx(10)
    assert counts["acquire_markers"] == 0


@pytest.mark.parametrize("timeout", [0, 0.05])
def test_sync_acquire_deadline_includes_engine_lock_queue(
    tmp_path: Path, timeout: float
) -> None:
    database = tmp_path / "engine-queue.sqlite3"
    builder = SyncSqliteBackendBuilder(database, key_prefix="queued-deadlines")
    backend = builder.build(_config())
    entered = _observe_write_wait(backend._engine)
    writer = sqlite3.connect(database, isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")
    acquire_started = threading.Event()

    def acquire():
        acquire_started.set()
        try:
            return backend.wait_for_capacity(
                frozen_usage({"requests": 4}),
                timeout=timeout,
                reservation_id="queued",
                reservation_lifetime_seconds=20,
            )
        except Exception as exc:
            return exc

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            blocker = executor.submit(
                backend.consume_capacity, frozen_usage({"requests": 0})
            )
            try:
                assert entered.wait(2)
                attempt = executor.submit(acquire)
                assert acquire_started.wait(2)
                completed, _ = concurrent.futures.wait({attempt}, timeout=0.3)
                returned_while_blocked = attempt in completed
            finally:
                writer.execute("ROLLBACK")
                writer.close()
            blocker.result(timeout=2)
            outcome = attempt.result(timeout=2)
        snapshots, counts = backend._engine.inspect_snapshot()
    finally:
        builder.close()
    assert isinstance(outcome, TimeoutError), repr(outcome)
    assert returned_while_blocked
    assert snapshots[0].current_capacity == pytest.approx(10)
    assert counts["acquire_markers"] == 0
