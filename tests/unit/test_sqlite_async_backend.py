from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

import pytest

from token_throttle import (
    DuplicateRefundError,
    PerModelConfig,
    Quota,
    RateLimiterCallbacks,
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


async def test_async_sqlite_cancel_during_consumed_callback_cleans_acquire(
    tmp_path: Path,
) -> None:
    callback_entered = asyncio.Event()

    async def block_consumed_callback(**_kwargs) -> None:
        callback_entered.set()
        await asyncio.Event().wait()

    builder = SqliteBackendBuilder(
        tmp_path / "cancelled-callback.sqlite3",
        key_prefix="async-tests",
    )
    backend = builder.build(
        _config(),
        callbacks=RateLimiterCallbacks(on_capacity_consumed=block_consumed_callback),
    )
    usage = frozen_usage({"requests": 4})
    task = asyncio.create_task(
        backend.await_for_capacity(
            usage,
            timeout=1,
            reservation_id="callback-cancel",
            reservation_lifetime_seconds=20.0,
        )
    )
    try:
        await asyncio.wait_for(callback_entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        snapshots, counts = await backend._run_engine(backend._engine.inspect_snapshot)
        assert snapshots[0].current_capacity == pytest.approx(10.0)
        assert counts["acquire_markers"] == 0
    finally:
        task.cancel()
        await builder.aclose()


async def test_async_sqlite_ordinary_callback_error_preserves_acquire(
    tmp_path: Path,
) -> None:
    async def raise_from_callback(**_kwargs) -> None:
        raise RuntimeError("callback failed")

    builder = SqliteBackendBuilder(
        tmp_path / "ordinary-callback-error.sqlite3",
        key_prefix="async-tests",
    )
    backend = builder.build(
        _config(),
        callbacks=RateLimiterCallbacks(on_capacity_consumed=raise_from_callback),
    )
    usage = frozen_usage({"requests": 4})
    try:
        with pytest.warns(RuntimeWarning, match="callback failed"):
            acquired_at = await backend.await_for_capacity(
                usage,
                timeout=1,
                reservation_id="callback-error",
                reservation_lifetime_seconds=20.0,
            )
        assert acquired_at is not None
        snapshots, counts = await backend._run_engine(backend._engine.inspect_snapshot)
        assert snapshots[0].current_capacity == pytest.approx(6.0, abs=0.1)
        assert counts["acquire_markers"] == 1
    finally:
        await builder.aclose()


async def test_async_sqlite_repeated_cancellation_cleans_committed_acquire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = SqliteBackendBuilder(
        tmp_path / "repeated-cancel.sqlite3",
        key_prefix="async-tests",
    )
    backend = builder.build(_config())
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
    usage = frozen_usage({"requests": 4})
    task = asyncio.create_task(
        backend.await_for_capacity(
            usage,
            timeout=1,
            reservation_id="repeated-cancel",
            reservation_lifetime_seconds=20.0,
        )
    )
    try:
        assert await asyncio.to_thread(committed.wait, 2)
        for _ in range(200):
            task.cancel()
            await asyncio.sleep(0)
        release_result.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        snapshots, counts = await backend._run_engine(backend._engine.inspect_snapshot)
        assert snapshots[0].current_capacity == pytest.approx(10.0)
        assert counts["acquire_markers"] == 0
    finally:
        release_result.set()
        task.cancel()
        await builder.aclose()


async def test_async_sqlite_cancel_during_cleanup_still_lands_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback_entered = asyncio.Event()

    async def block_consumed_callback(**_kwargs) -> None:
        callback_entered.set()
        await asyncio.Event().wait()

    builder = SqliteBackendBuilder(
        tmp_path / "cancel-during-cleanup.sqlite3",
        key_prefix="async-tests",
    )
    backend = builder.build(
        _config(),
        callbacks=RateLimiterCallbacks(on_capacity_consumed=block_consumed_callback),
    )
    cleanup_committed = threading.Event()
    release_cleanup = threading.Event()
    original_cleanup = backend._engine.cleanup_consumption

    def pause_after_cleanup(*args, **kwargs):
        result = original_cleanup(*args, **kwargs)
        cleanup_committed.set()
        if not release_cleanup.wait(timeout=5):
            raise RuntimeError("test did not release SQLite cleanup")
        return result

    monkeypatch.setattr(backend._engine, "cleanup_consumption", pause_after_cleanup)
    usage = frozen_usage({"requests": 4})
    task = asyncio.create_task(
        backend.await_for_capacity(
            usage,
            timeout=1,
            reservation_id="cleanup-cancel",
            reservation_lifetime_seconds=20.0,
        )
    )
    try:
        await asyncio.wait_for(callback_entered.wait(), timeout=2)
        task.cancel()
        assert await asyncio.to_thread(cleanup_committed.wait, 2)
        task.cancel()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        snapshots, counts = await backend._run_engine(backend._engine.inspect_snapshot)
        assert snapshots[0].current_capacity == pytest.approx(10.0)
        assert counts["acquire_markers"] == 0
    finally:
        release_cleanup.set()
        task.cancel()
        await builder.aclose()


async def test_async_sqlite_cancel_during_refund_lands_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = SqliteBackendBuilder(
        tmp_path / "cancel-during-refund.sqlite3",
        key_prefix="async-tests",
    )
    backend = builder.build(_config())
    usage = frozen_usage({"requests": 4})
    bucket_ids = frozenset({("requests", 10)})
    await backend.await_for_capacity(
        usage,
        timeout=1,
        reservation_id="refund-cancel",
        reservation_lifetime_seconds=20.0,
    )
    refund_committed = threading.Event()
    release_refund = threading.Event()
    original_refund = backend._engine.refund

    def pause_after_refund(*args, **kwargs):
        result = original_refund(*args, **kwargs)
        refund_committed.set()
        if not release_refund.wait(timeout=5):
            raise RuntimeError("test did not release SQLite refund")
        return result

    monkeypatch.setattr(backend._engine, "refund", pause_after_refund)

    async def refund() -> bool:
        return await backend.refund_capacity_for_buckets(
            usage,
            frozen_usage({"requests": 1}),
            bucket_ids=bucket_ids,
            reservation_id="refund-cancel",
            reservation_model_family="async-sqlite-tests",
            reservation_bucket_ids=bucket_ids,
            reservation_reserved_usage=usage,
        )

    task = asyncio.create_task(refund())
    try:
        assert await asyncio.to_thread(refund_committed.wait, 2)
        task.cancel()
        release_refund.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        _, counts = await backend._run_engine(backend._engine.inspect_snapshot)
        assert counts["acquire_markers"] == 0
        assert counts["refund_tombstones"] == 1
        with pytest.raises(DuplicateRefundError) as duplicate:
            await refund()
        assert duplicate.value.reason == "already_refunded"
    finally:
        release_refund.set()
        task.cancel()
        await builder.aclose()


async def test_async_sqlite_waiter_is_visible_before_wait_callback(
    tmp_path: Path,
) -> None:
    seen_waiters = []
    seen_waiter_keys = []
    backend_holder = []
    usage = frozen_usage({"requests": 1})

    async def on_wait_start(**_kwargs) -> None:
        backend = backend_holder[0]
        seen_waiter_keys.extend(backend._diagnostic_waiters)
        diagnostic = await backend.introspect()
        seen_waiters.extend(diagnostic.waits)
        await backend.refund_capacity(usage, frozen_usage({"requests": 0}))

    builder = SqliteBackendBuilder(
        tmp_path / "waiter-parity.sqlite3",
        key_prefix="async-tests",
    )
    backend = builder.build(
        _config(limit=1.0),
        callbacks=RateLimiterCallbacks(on_wait_start=on_wait_start),
    )
    backend_holder.append(backend)
    try:
        await backend.consume_capacity(usage)
        await backend.await_for_capacity(
            usage,
            timeout=1,
            reservation_id="visible-reservation",
            reservation_lifetime_seconds=20.0,
        )
        assert len(seen_waiters) == 1
        assert seen_waiters[0].reservation_id == "visible-reservation"
        assert seen_waiter_keys != ["visible-reservation"]
    finally:
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
