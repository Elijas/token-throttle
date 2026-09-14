from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import threading
import time
from typing import TYPE_CHECKING

import pytest

from token_throttle import (
    BackendLockContentionError,
    PerModelConfig,
    Quota,
    RateLimiter,
    SqliteBackendBuilder,
    UsageQuotas,
)
from token_throttle._limiter_backends._sqlite._backend import SqliteBackend
from token_throttle._limiter_backends._sqlite._engine import SqliteEngine

if TYPE_CHECKING:
    from pathlib import Path


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="cold-initialization",
        quotas=UsageQuotas([Quota(metric="requests", limit=10, per_seconds=60)]),
    )


@pytest.fixture
def locked_database(tmp_path: Path):
    database = tmp_path / "cold.sqlite3"
    initializer = SqliteBackendBuilder(database, key_prefix="cold")
    initializer.build(_config())
    initializer.close()
    locked = threading.Event()
    release = threading.Event()

    def hold_writer() -> None:
        connection = sqlite3.connect(database, isolation_level=None)
        try:
            connection.execute("BEGIN IMMEDIATE")
            locked.set()
            release.wait(5)
        finally:
            connection.close()

    holder = threading.Thread(target=hold_writer, name="cold-init-writer")
    holder.start()
    try:
        assert locked.wait(2)
        yield database, release
    finally:
        release.set()
        holder.join(2)
        assert not holder.is_alive()


async def test_cold_sqlite_initialization_keeps_heartbeat_running(
    locked_database: tuple[Path, threading.Event],
) -> None:
    builder = SqliteBackendBuilder(
        locked_database[0], key_prefix="cold", busy_timeout_ms=250
    )
    limiter = RateLimiter(_config(), backend=builder)
    heartbeat_ran = threading.Event()

    async def heartbeat() -> float:
        started = time.monotonic()
        await asyncio.sleep(0.01)
        heartbeat_ran.set()
        return time.monotonic() - started

    pulse = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    try:
        with contextlib.suppress(Exception):
            await limiter.acquire_capacity({"requests": 1}, "model", timeout=0)
        delay = await pulse
        assert heartbeat_ran.is_set()
        assert delay < 0.15, f"cold initialization blocked heartbeat for {delay:.3f}s"
    finally:
        await limiter.aclose()


@pytest.mark.parametrize("timeout", [None, 0.04])
@pytest.mark.parametrize("fail_after_release", [False, True])
async def test_cold_sqlite_abandoned_initialization_cleans_up_on_owning_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeout: float | None,
    *,
    fail_after_release: bool,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    worker_threads: list[threading.Thread] = []
    close_threads: list[threading.Thread] = []
    original_initialize = SqliteEngine.initialize_buckets
    original_close = SqliteEngine.close

    def gated_initialize(self, *args, **kwargs):
        original_initialize(self, *args, **kwargs)
        worker_threads.append(threading.current_thread())
        entered.set()
        assert release.wait(5)
        if fail_after_release:
            raise ValueError("injected initialization failure")

    def tracked_close(self):
        original_close(self)
        close_threads.append(threading.current_thread())
        closed.set()

    monkeypatch.setattr(SqliteEngine, "initialize_buckets", gated_initialize)
    monkeypatch.setattr(SqliteEngine, "close", tracked_close)
    builder = SqliteBackendBuilder(tmp_path / "abandoned.sqlite3", key_prefix="cold")
    limiter = RateLimiter(_config(), backend=builder)
    acquire = asyncio.create_task(
        limiter.acquire_capacity({"requests": 1}, "model", timeout=timeout)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        if timeout is None:
            acquire.cancel()
        expected_error = asyncio.CancelledError if timeout is None else TimeoutError
        with pytest.raises(expected_error):
            await asyncio.wait_for(acquire, 0.2)
        assert not release.is_set()
        assert not builder._backends
        assert not limiter._pending_acquire_reservations
    finally:
        release.set()
        await asyncio.gather(acquire, return_exceptions=True)
        await limiter.aclose()
        assert await asyncio.to_thread(closed.wait, 2)
        for worker in worker_threads:
            await asyncio.to_thread(worker.join, 2)
            assert not worker.is_alive()
    assert close_threads == worker_threads


@pytest.mark.parametrize(
    "stage", ["_configure_connection", "_initialize_schema", "initialize_buckets"]
)
async def test_cold_sqlite_initialization_error_closes_partial_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    closed = threading.Event()
    workers: list[threading.Thread] = []
    original_connect = sqlite3.connect

    class TrackedConnection(sqlite3.Connection):
        def close(self):
            super().close()
            closed.set()

    def tracked_connect(*args, **kwargs):
        workers.append(threading.current_thread())
        return original_connect(*args, **kwargs, factory=TrackedConnection)

    def fail_initialization(*args, **kwargs):
        raise ValueError("injected initialization failure")

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    monkeypatch.setattr(SqliteEngine, stage, fail_initialization)
    builder = SqliteBackendBuilder(tmp_path / "failed.sqlite3", key_prefix="cold")
    limiter = RateLimiter(_config(), backend=builder)
    try:
        with pytest.raises(ValueError, match="injected initialization failure"):
            await limiter.acquire_capacity({"requests": 1}, "model", timeout=0.5)
        assert closed.is_set()
        assert not builder._backends
    finally:
        await limiter.aclose()
        for worker in workers:
            await asyncio.to_thread(worker.join, 2)
            assert not worker.is_alive()


async def test_cold_sqlite_passes_only_remaining_timeout_to_capacity_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    observed_timeouts: list[float] = []
    original_initialize = SqliteEngine.initialize_buckets
    original_wait = SqliteBackend.await_for_capacity

    def gated_initialize(self, *args, **kwargs):
        original_initialize(self, *args, **kwargs)
        entered.set()
        assert release.wait(5)

    async def tracked_wait(self, *args, timeout=None, **kwargs):
        observed_timeouts.append(timeout)
        return await original_wait(self, *args, timeout=timeout, **kwargs)

    monkeypatch.setattr(SqliteEngine, "initialize_buckets", gated_initialize)
    monkeypatch.setattr(SqliteBackend, "await_for_capacity", tracked_wait)
    builder = SqliteBackendBuilder(tmp_path / "budget.sqlite3", key_prefix="cold")
    limiter = RateLimiter(_config(), backend=builder)
    acquire = asyncio.create_task(
        limiter.acquire_capacity({"requests": 1}, "model", timeout=1)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        await asyncio.sleep(0.05)
        release.set()
        reservation = await acquire
        assert len(observed_timeouts) == 1
        assert 0 < observed_timeouts[0] < 0.96
        await limiter.refund_capacity({"requests": 0}, reservation)
        assert builder._backends[0]._engine.busy_timeout_ms == 5000
    finally:
        release.set()
        await asyncio.gather(acquire, return_exceptions=True)
        await limiter.aclose()


@pytest.mark.parametrize("timeout", [0, 0.04])
async def test_cold_sqlite_waiter_deadline_includes_limiter_initialization_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeout: float,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    original_initialize = SqliteEngine.initialize_buckets

    def gated_initialize(self, *args, **kwargs):
        original_initialize(self, *args, **kwargs)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(SqliteEngine, "initialize_buckets", gated_initialize)
    builder = SqliteBackendBuilder(tmp_path / "shared.sqlite3", key_prefix="cold")
    limiter = RateLimiter(_config(), backend=builder)
    owner = asyncio.create_task(limiter.acquire_capacity({"requests": 1}, "model"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await limiter.acquire_capacity({"requests": 1}, "model", timeout=timeout)
        assert time.monotonic() - started < 0.15
        assert not owner.done()
    finally:
        release.set()
        await asyncio.gather(owner, return_exceptions=True)
        await limiter.aclose()


@pytest.mark.parametrize("timeout", [None, 0, 1])
async def test_cold_sqlite_uncontended_first_acquisition_succeeds(
    tmp_path: Path, timeout: float | None
) -> None:
    builder = SqliteBackendBuilder(tmp_path / "fresh.sqlite3", key_prefix="cold")
    limiter = RateLimiter(_config(), backend=builder)
    try:
        reservation = await limiter.acquire_capacity(
            {"requests": 1}, "model", timeout=timeout
        )
        await limiter.refund_capacity({"requests": 0}, reservation)
    finally:
        await limiter.aclose()


@pytest.mark.parametrize("timeout", [0, 0.04])
async def test_cold_sqlite_initialization_obeys_acquire_timeout(
    locked_database: tuple[Path, threading.Event],
    timeout: float,
) -> None:
    builder = SqliteBackendBuilder(
        locked_database[0], key_prefix="cold", busy_timeout_ms=250
    )
    limiter = RateLimiter(_config(), backend=builder)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            await limiter.acquire_capacity({"requests": 1}, "model", timeout=timeout)
        assert time.monotonic() - started < 0.15
        assert not builder._backends
        assert not limiter._pending_acquire_reservations
    finally:
        await limiter.aclose()


@pytest.mark.parametrize("timeout", [None, 1])
async def test_cold_sqlite_retries_initialization_until_writer_releases(
    locked_database: tuple[Path, threading.Event],
    monkeypatch: pytest.MonkeyPatch,
    timeout: float | None,
) -> None:
    database, release = locked_database
    contended = threading.Event()
    original_initialize = SqliteEngine._initialize_schema

    def tracked_initialize(self, *args, **kwargs):
        try:
            return original_initialize(self, *args, **kwargs)
        except BackendLockContentionError:
            contended.set()
            raise

    monkeypatch.setattr(SqliteEngine, "_initialize_schema", tracked_initialize)
    builder = SqliteBackendBuilder(database, key_prefix="cold")
    limiter = RateLimiter(_config(), backend=builder)
    acquire = asyncio.create_task(
        limiter.acquire_capacity({"requests": 1}, "model", timeout=timeout)
    )
    try:
        assert await asyncio.to_thread(contended.wait, 2)
        release.set()
        reservation = await asyncio.wait_for(acquire, 2)
        await limiter.refund_capacity({"requests": 0}, reservation)
    finally:
        release.set()
        await asyncio.gather(acquire, return_exceptions=True)
        await limiter.aclose()


async def test_cold_sqlite_record_usage_retains_ordinary_contention_error(
    locked_database: tuple[Path, threading.Event],
) -> None:
    builder = SqliteBackendBuilder(
        locked_database[0], key_prefix="cold", busy_timeout_ms=20
    )
    limiter = RateLimiter(_config(), backend=builder)
    try:
        with pytest.raises(BackendLockContentionError):
            await limiter.record_usage({"requests": 1}, "model")
    finally:
        await limiter.aclose()
