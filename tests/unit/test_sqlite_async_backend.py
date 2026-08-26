from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

import pytest

from token_throttle import (
    PerModelConfig,
    Quota,
    SqliteBackendBuilder,
    UsageQuotas,
    frozen_usage,
)

if TYPE_CHECKING:
    from pathlib import Path


def _config(
    *, metric: str = "requests", limit: float = 10.0, per_seconds: int = 10
) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
        model_family="async-sqlite-tests",
    )


async def test_async_sqlite_backend_confines_engine_calls_to_one_worker_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = SqliteBackendBuilder(
        tmp_path / "thread-confinement.sqlite3",
        key_prefix="async-tests",
    )
    backend = builder.build(_config())
    main_thread = threading.get_ident()
    worker_threads: list[int] = []
    original_consume = backend._engine.consume
    original_snapshot = backend._engine.inspect_snapshot

    def tracked_consume(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        return original_consume(*args, **kwargs)

    def tracked_snapshot(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(backend._engine, "consume", tracked_consume)
    monkeypatch.setattr(backend._engine, "inspect_snapshot", tracked_snapshot)
    try:
        await backend.consume_capacity(frozen_usage({"requests": 1}))
        diagnostic = await backend.introspect()
        assert diagnostic.buckets[0].current_capacity is not None
        assert len(set(worker_threads)) == 1
        assert worker_threads[0] != main_thread
    finally:
        await builder.aclose()


async def test_async_sqlite_cancelled_committed_acquire_is_refunded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = SqliteBackendBuilder(
        tmp_path / "cancelled-commit.sqlite3",
        key_prefix="async-tests",
    )
    backend = builder.build(_config(limit=2.0))
    committed = threading.Event()
    release_result = threading.Event()
    original_try_consume = backend._engine.try_consume

    def pause_after_commit(*args, **kwargs):
        result = original_try_consume(*args, **kwargs)
        if result.available:
            committed.set()
            if not release_result.wait(timeout=5):
                raise RuntimeError("test did not release committed SQLite attempt")
        return result

    monkeypatch.setattr(backend._engine, "try_consume", pause_after_commit)
    usage = frozen_usage({"requests": 2})
    task = asyncio.create_task(
        backend.await_for_capacity(
            usage,
            timeout=1,
            reservation_id="cancelled-reservation",
            reservation_lifetime_seconds=20.0,
        )
    )
    try:
        assert await asyncio.to_thread(committed.wait, 2)
        task.cancel()
        release_result.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        diagnostic = await backend.introspect()
        assert any("acquire_markers=0" in issue.message for issue in diagnostic.issues)
        assert await backend.await_for_capacity(usage, timeout=0) is not None
    finally:
        release_result.set()
        await builder.aclose()


async def test_async_sqlite_builder_aclose_is_idempotent(tmp_path: Path) -> None:
    builder = SqliteBackendBuilder(
        tmp_path / "close.sqlite3",
        key_prefix="async-tests",
    )
    builder.build(_config())
    await builder.aclose()
    await builder.aclose()


async def test_async_sqlite_metric_rebuilds_bound_worker_threads_and_registry(
    tmp_path: Path,
) -> None:
    builder = SqliteBackendBuilder(
        tmp_path / "async-rebuild-lifecycle.sqlite3",
        key_prefix="rebuilds",
    )
    current = builder.build(_config(metric="requests"))
    baseline_workers = sum(
        thread.name.startswith("token-throttle-sqlite")
        for thread in threading.enumerate()
    )
    try:
        for index in range(10):
            cfg = _config(metric="tokens" if index % 2 == 0 else "requests")
            replacement = builder.build(cfg)
            current = await current.prepare_reconfigured_backend(replacement, cfg)
            assert len(builder._backends) == 1
            live_workers = sum(
                thread.name.startswith("token-throttle-sqlite")
                for thread in threading.enumerate()
            )
            assert live_workers <= baseline_workers + 1
    finally:
        await builder.aclose()
