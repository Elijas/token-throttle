"""Retry terminal close after real SQLite acquisition/refund drains time out."""

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
    RateLimiter,
    SqliteBackendBuilder,
    SyncRateLimiter,
    SyncSqliteBackendBuilder,
    UsageQuotas,
)
from token_throttle._limiter_backends._sqlite._engine import SqliteEngine

if TYPE_CHECKING:
    from pathlib import Path


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="close-drain",
        quotas=UsageQuotas([Quota(metric="requests", limit=10, per_seconds=3600)]),
    )


@pytest.fixture
def gated_engine_call(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    stage = request.param
    entered = threading.Event()
    release = threading.Event()
    engines: list[SqliteEngine] = []
    original = getattr(SqliteEngine, stage)

    def gated(self, *args, **kwargs):
        engines.append(self)
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test did not release SQLite operation")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SqliteEngine, stage, gated)
    try:
        yield stage, entered, release, engines
    finally:
        release.set()


def _assert_retained_state_cleared(limiter: RateLimiter | SyncRateLimiter) -> None:
    assert limiter._closed
    assert not limiter._closing
    assert limiter.snapshot_state()["model_families"] == 0
    assert limiter.snapshot_state()["in_flight_reservations"] == 0
    assert not limiter._reservation_snapshots
    assert not limiter._model_family_to_quotas
    assert not limiter._model_family_to_validated_signature
    assert not limiter._refund_locks


@pytest.mark.parametrize(
    "gated_engine_call", ["initialize_buckets", "refund"], indirect=True
)
async def test_async_close_retry_releases_sqlite_after_drain_timeout(
    tmp_path: Path, gated_engine_call, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, entered, release, engines = gated_engine_call
    builder = SqliteBackendBuilder(tmp_path / "async.sqlite3", key_prefix="retry")
    limiter = RateLimiter(_config(), backend=builder, close_drain_timeout_seconds=0.02)
    if stage == "refund":
        reservation = await limiter.acquire_capacity({"requests": 4}, "model")
        work = asyncio.create_task(
            limiter.refund_capacity({"requests": 0}, reservation)
        )
    else:
        work = asyncio.create_task(limiter.acquire_capacity({"requests": 4}, "model"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        for _ in range(2):
            with pytest.raises(TimeoutError, match="drain"):
                await limiter.aclose()
            with pytest.raises(RuntimeError, match="closed"):
                await limiter.acquire_capacity({"requests": 1}, "model")
            with pytest.raises(RuntimeError, match="closed"):
                await limiter.record_usage({"requests": 1}, "model")
        release.set()
        result = await asyncio.wait_for(work, 2)
        if stage == "initialize_buckets":
            reservation = result
        with pytest.raises(RuntimeError, match="closed"):
            await limiter.refund_capacity({"requests": 0}, reservation)

        backend = builder._backends[0]
        workers = tuple(backend._executor._threads)
        engine = engines[0]
        close_calls = []
        original_close = engine.close

        def counted_close():
            close_calls.append(threading.get_ident())
            original_close()

        monkeypatch.setattr(engine, "close", counted_close)
        await asyncio.gather(limiter.aclose(), limiter.aclose())
        await limiter.aclose()
        assert builder._backends == []
        assert len(close_calls) == 1
        assert engine._closed
        assert workers
        assert all(not worker.is_alive() for worker in workers)
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            engine._connection.execute("SELECT 1")
        _assert_retained_state_cleared(limiter)
    finally:
        release.set()
        await asyncio.gather(work, return_exceptions=True)
        await builder.aclose()


@pytest.mark.parametrize(
    "gated_engine_call", ["initialize_buckets", "refund"], indirect=True
)
def test_sync_close_retry_releases_sqlite_after_drain_timeout(
    tmp_path: Path, gated_engine_call, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, entered, release, engines = gated_engine_call
    builder = SyncSqliteBackendBuilder(tmp_path / "sync.sqlite3", key_prefix="retry")
    limiter = SyncRateLimiter(
        _config(), backend=builder, close_drain_timeout_seconds=0.02
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            if stage == "refund":
                reservation = limiter.acquire_capacity({"requests": 4}, "model")
                work = pool.submit(
                    limiter.refund_capacity, {"requests": 0}, reservation
                )
            else:
                work = pool.submit(limiter.acquire_capacity, {"requests": 4}, "model")
            try:
                assert entered.wait(2)
                for _ in range(2):
                    with pytest.raises(TimeoutError, match="drain"):
                        limiter.close()
                    with pytest.raises(RuntimeError, match="closed"):
                        limiter.acquire_capacity({"requests": 1}, "model")
                    with pytest.raises(RuntimeError, match="closed"):
                        limiter.record_usage({"requests": 1}, "model")
            finally:
                release.set()
            result = work.result(timeout=2)
            if stage == "initialize_buckets":
                reservation = result
            with pytest.raises(RuntimeError, match="closed"):
                limiter.refund_capacity({"requests": 0}, reservation)

            engine = engines[0]
            close_calls = []
            original_close = engine.close

            def counted_close():
                close_calls.append(threading.get_ident())
                original_close()

            monkeypatch.setattr(engine, "close", counted_close)
            closes = [pool.submit(limiter.close) for _ in range(2)]
            for close in closes:
                close.result(timeout=2)
            limiter.close()
            assert builder._backends == []
            assert len(close_calls) == 1
            assert engine._closed
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                engine._connection.execute("SELECT 1")
            _assert_retained_state_cleared(limiter)
    finally:
        release.set()
        builder.close()
