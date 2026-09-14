"""Public recovery authority survives a failed post-commit cleanup."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import sqlite3
import threading
import time
import uuid
from typing import TYPE_CHECKING

import pytest

from tests._redis_guard import ensure_flush_allowed
from token_throttle import (
    AcquireRefundFailedError,
    BackendLockContentionError,
    DuplicateRefundError,
    MemoryBackendBuilder,
    PerModelConfig,
    Quota,
    RateLimiter,
    RateLimiterCallbacks,
    SqliteBackendBuilder,
    SyncMemoryBackendBuilder,
    SyncRateLimiter,
    SyncRateLimiterCallbacks,
    SyncSqliteBackendBuilder,
    UsageQuotas,
)

if TYPE_CHECKING:
    from pathlib import Path


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="cleanup",
        quotas=UsageQuotas([Quota(metric="requests", limit=10, per_seconds=3600)]),
    )


def _assert_authority(limiter, error, interruption, issued_at) -> None:
    assert error.interrupted_by is interruption
    assert error.__cause__ is error.refund_error
    reservation = error.reservation
    assert reservation.created_at_seconds == issued_at
    assert reservation.bucket_ids == frozenset({("requests", 3600)})
    assert reservation.usage == {"requests": 4}
    assert reservation.model == "alias"
    assert reservation.model_family == "cleanup"
    assert reservation.limiter_instance_id == limiter._limiter_instance_id
    assert limiter.snapshot_state()["in_flight_reservations"] == 1
    assert not limiter._pending_acquire_reservations
    assert (
        limiter._reservation_snapshots[reservation.reservation_id].to_reservation()
        == reservation
    )


def _sqlite_state(path):
    with sqlite3.connect(path) as connection:
        return (
            connection.execute("SELECT capacity FROM buckets").fetchone()[0],
            connection.execute(
                "SELECT reservation_id, created_at, expires_at FROM acquire_markers"
            ).fetchall(),
            connection.execute("SELECT COUNT(*) FROM refund_tombstones").fetchone()[0],
        )


@pytest.mark.parametrize("timeout", [None, 0, 1])
async def test_async_sqlite_writer_contention_preserves_public_recovery(
    tmp_path: Path, timeout: float | None
) -> None:
    path = tmp_path / "async.sqlite3"
    writer = sqlite3.connect(path, isolation_level=None, timeout=0.1)
    interruption = asyncio.CancelledError("interrupted delivery")
    issued_at = None

    async def callback(**kwargs):
        nonlocal issued_at
        assert "reservation_id" not in kwargs
        issued_at = kwargs["current_time"]
        writer.execute("BEGIN IMMEDIATE")
        raise interruption

    limiter = RateLimiter(
        _config(),
        backend=SqliteBackendBuilder(path, key_prefix="cleanup", busy_timeout_ms=5),
        callbacks=RateLimiterCallbacks(on_capacity_consumed=callback),
        callback_timeout=None,
        max_reservation_lifetime_seconds=60,
    )
    try:
        with pytest.raises(AcquireRefundFailedError) as caught:
            await limiter.acquire_capacity({"requests": 4}, "alias", timeout=timeout)
        error = caught.value
        _assert_authority(limiter, error, interruption, issued_at)
        assert isinstance(error.refund_error, BackendLockContentionError)
        capacity, markers, tombstones = _sqlite_state(path)
        assert capacity == 6
        assert markers == [
            (error.reservation.reservation_id, issued_at, issued_at + 60)
        ]
        assert tombstones == 0
        writer.execute("ROLLBACK")
        await limiter.refund_capacity({"requests": 0}, error.reservation)
        assert _sqlite_state(path) == (10, [], 1)
        with pytest.raises(DuplicateRefundError):
            await limiter.refund_capacity(
                {"requests": 0}, error.reservation.model_copy()
            )
        assert _sqlite_state(path) == (10, [], 1)
        assert limiter.snapshot_state()["in_flight_reservations"] == 0
    finally:
        writer.close()
        await limiter.aclose()


@pytest.fixture
async def cleanup_redis(request: pytest.FixtureRequest):
    redis = pytest.importorskip("redis.asyncio")
    client = redis.from_url(
        request.config.getoption("--redis-url"), socket_connect_timeout=0.2
    )
    try:
        await client.ping()
    except redis.ConnectionError:
        await client.aclose()
        pytest.skip("dedicated Redis unavailable")
    ensure_flush_allowed(request.config.getoption("--redis-url"))
    try:
        yield client
    finally:
        await client.aclose()


@pytest.mark.parametrize("background", ["fails", "finishes_after_public_refund"])
async def test_redis_background_cleanup_keeps_recovery_and_cannot_double_credit(  # noqa: PLR0915
    cleanup_redis, monkeypatch: pytest.MonkeyPatch, background: str
) -> None:
    from token_throttle import RedisBackendBuilder  # noqa: PLC0415
    from token_throttle._limiter_backends._redis import (  # noqa: PLC0415
        _backend as redis_module,
    )

    interruption = concurrent.futures.CancelledError("callback interrupted")
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    issued_at = None
    task = None

    async def callback(**kwargs):
        nonlocal issued_at
        issued_at = kwargs["current_time"]
        raise interruption

    limiter = RateLimiter(
        _config(),
        backend=RedisBackendBuilder(
            cleanup_redis, key_prefix=f"cleanup-{uuid.uuid4().hex}"
        ),
        callbacks=RateLimiterCallbacks(on_capacity_consumed=callback),
        callback_timeout=None,
    )
    try:
        backend = await limiter._get_backend(_config())
        original = backend._lock_or_contention
        calls = 0

        async def pause_first_cleanup(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                asyncio.current_task().add_done_callback(lambda _: finished.set())
                entered.set()
                await release.wait()
                if background == "fails":
                    raise ConnectionError("background cleanup failed")
            return await original(**kwargs)

        monkeypatch.setattr(backend, "_lock_or_contention", pause_first_cleanup)
        monkeypatch.setattr(redis_module, "LOCK_CANCEL_REFUND_TIMEOUT_SECONDS", 0.02)
        task = asyncio.create_task(limiter.acquire_capacity({"requests": 4}, "alias"))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel("cancel during cleanup")
        with pytest.raises(AcquireRefundFailedError) as caught:
            await asyncio.wait_for(asyncio.shield(task), 2)
        _assert_authority(limiter, caught.value, interruption, issued_at)
        assert isinstance(caught.value.refund_error, TimeoutError)
        await limiter.refund_capacity({"requests": 0}, caught.value.reservation)
        # Leave headroom so a second cleanup credit would be observable.
        backend._callbacks = None
        other = await limiter.acquire_capacity({"requests": 4}, "alias")
        release.set()
        await asyncio.wait_for(finished.wait(), 2)
        assert (await backend.introspect()).buckets[
            0
        ].current_capacity == pytest.approx(6, abs=0.01)
        with pytest.raises(DuplicateRefundError):
            await limiter.refund_capacity({"requests": 0}, caught.value.reservation)
        await limiter.refund_capacity({"requests": 0}, other)
    finally:
        release.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if entered.is_set():
            await asyncio.wait_for(finished.wait(), 2)
        await limiter.aclose()


@pytest.mark.parametrize("mode", ["async", "sync"])
async def test_redis_after_wait_cleanup_failure_delivers_reservation(
    cleanup_redis,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    mode: str,
) -> None:
    import redis  # noqa: PLC0415
    from frozendict import frozendict  # noqa: PLC0415

    from token_throttle import (  # noqa: PLC0415
        RedisBackendBuilder,
        SyncRedisBackendBuilder,
    )

    asynchronous = mode == "async"
    client = (
        cleanup_redis
        if asynchronous
        else redis.from_url(request.config.getoption("--redis-url"))
    )
    interruption = concurrent.futures.CancelledError("after wait interrupted")
    refund_error = ConnectionError("after wait cleanup failed")

    def callback(**kwargs):
        raise interruption

    async def async_callback(**kwargs):
        callback(**kwargs)

    limiter = (RateLimiter if asynchronous else SyncRateLimiter)(
        _config(),
        backend=(RedisBackendBuilder if asynchronous else SyncRedisBackendBuilder)(
            client, key_prefix=f"cleanup-{uuid.uuid4().hex}"
        ),
        callbacks=(
            RateLimiterCallbacks(after_wait_end_consumption=async_callback)
            if asynchronous
            else SyncRateLimiterCallbacks(after_wait_end_consumption=callback)
        ),
        callback_timeout=None,
    )
    try:
        backend = await _call(limiter._get_backend, _config())
        original = backend._check_and_consume_capacity
        attempts = 0

        def first_attempt():
            nonlocal attempts
            attempts += 1
            return attempts == 1

        empty = frozendict({("requests", 3600): 0.0})
        unavailable = (False, empty, empty, None, None, backend._snapshot_buckets())

        def check(*args, **kwargs):
            return unavailable if first_attempt() else original(*args, **kwargs)

        async def async_check(*args, **kwargs):
            return unavailable if first_attempt() else await original(*args, **kwargs)

        def fail(*args, **kwargs):
            raise refund_error

        async def async_fail(*args, **kwargs):
            fail(*args, **kwargs)

        monkeypatch.setattr(
            backend,
            "_check_and_consume_capacity",
            async_check if asynchronous else check,
        )
        monkeypatch.setattr(
            backend, "_compute_sleep_for_wait", lambda *_args, **_kwargs: 0.001
        )
        monkeypatch.setattr(
            backend,
            "_refund_cancelled_consumption",
            async_fail if asynchronous else fail,
        )
        with pytest.raises(AcquireRefundFailedError) as caught:
            await _call(limiter.acquire_capacity, {"requests": 4}, "alias", timeout=1)
        assert caught.value.interrupted_by is interruption
        assert caught.value.refund_error is refund_error
        assert attempts == 2
        await _call(limiter.refund_capacity, {"requests": 0}, caught.value.reservation)
        assert (await _call(backend.introspect)).buckets[0].current_capacity == 10
    finally:
        await _call(limiter.aclose if asynchronous else limiter.close)
        if not asynchronous:
            client.close()


@pytest.mark.parametrize("timeout", [None, 0, 1])
def test_sync_sqlite_writer_contention_preserves_public_recovery(
    tmp_path: Path, timeout: float | None
) -> None:
    path = tmp_path / "sync.sqlite3"
    writer = sqlite3.connect(path, isolation_level=None, timeout=0.1)
    interruption = concurrent.futures.CancelledError("interrupted delivery")
    issued_at = None

    def callback(**kwargs):
        nonlocal issued_at
        assert "reservation_id" not in kwargs
        issued_at = kwargs["current_time"]
        writer.execute("BEGIN IMMEDIATE")
        raise interruption

    limiter = SyncRateLimiter(
        _config(),
        backend=SyncSqliteBackendBuilder(path, key_prefix="cleanup", busy_timeout_ms=5),
        callbacks=SyncRateLimiterCallbacks(on_capacity_consumed=callback),
        callback_timeout=None,
        max_reservation_lifetime_seconds=60,
    )
    try:
        with pytest.raises(AcquireRefundFailedError) as caught:
            limiter.acquire_capacity({"requests": 4}, "alias", timeout=timeout)
        error = caught.value
        _assert_authority(limiter, error, interruption, issued_at)
        assert isinstance(error.refund_error, BackendLockContentionError)
        capacity, markers, tombstones = _sqlite_state(path)
        assert capacity == 6
        assert markers == [
            (error.reservation.reservation_id, issued_at, issued_at + 60)
        ]
        assert tombstones == 0
        writer.execute("ROLLBACK")
        limiter.refund_capacity({"requests": 0}, error.reservation)
        assert _sqlite_state(path) == (10, [], 1)
        with pytest.raises(DuplicateRefundError):
            limiter.refund_capacity({"requests": 0}, error.reservation.model_copy())
        assert _sqlite_state(path) == (10, [], 1)
        assert limiter.snapshot_state()["in_flight_reservations"] == 0
    finally:
        writer.close()
        limiter.close()


async def _call(function, *args, **kwargs):
    result = function(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("store", ["memory", "sqlite", "redis"])
@pytest.mark.parametrize(
    "cleanup", ["ordinary", "expired", "critical", "group", "success", "record"]
)
async def test_builtin_callback_cleanup_outcomes(  # noqa: PLR0913, PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    mode: str,
    store: str,
    cleanup: str,
) -> None:
    asynchronous = mode == "async"
    client = None
    if store == "redis":
        redis = pytest.importorskip("redis")
        redis_async = pytest.importorskip("redis.asyncio")
        from token_throttle import (  # noqa: PLC0415
            RedisBackendBuilder,
            SyncRedisBackendBuilder,
        )

        client = (redis_async if asynchronous else redis).from_url(
            request.config.getoption("--redis-url"), socket_connect_timeout=0.2
        )
        try:
            await _call(client.ping)
        except redis.ConnectionError:
            await _call(client.aclose if asynchronous else client.close)
            pytest.skip("dedicated Redis unavailable")
        ensure_flush_allowed(request.config.getoption("--redis-url"))
        builder = (RedisBackendBuilder if asynchronous else SyncRedisBackendBuilder)(
            client, key_prefix=f"cleanup-{uuid.uuid4().hex}"
        )
    elif store == "sqlite":
        builder = (SqliteBackendBuilder if asynchronous else SyncSqliteBackendBuilder)(
            tmp_path / "injected.sqlite3", key_prefix="cleanup"
        )
    else:
        builder = (MemoryBackendBuilder if asynchronous else SyncMemoryBackendBuilder)()

    interruption = concurrent.futures.CancelledError("callback interruption")
    refund_error = {
        "ordinary": ConnectionError("cleanup store failed"),
        "expired": ConnectionError("cleanup store failed"),
        "critical": MemoryError("critical cleanup"),
        "group": BaseExceptionGroup("cleanup group", [MemoryError("critical child")]),
        "success": None,
        "record": ConnectionError("must not refund recorded usage"),
    }[cleanup]
    armed = False
    issued_at = None
    cleanup_calls = 0

    def callback(**kwargs):
        nonlocal issued_at
        if armed:
            assert "reservation_id" not in kwargs
            issued_at = kwargs["current_time"]
            raise interruption

    async def async_callback(**kwargs):
        callback(**kwargs)

    limiter = (RateLimiter if asynchronous else SyncRateLimiter)(
        _config(),
        backend=builder,
        callbacks=(
            RateLimiterCallbacks(on_capacity_consumed=async_callback)
            if asynchronous
            else SyncRateLimiterCallbacks(on_capacity_consumed=callback)
        ),
        callback_timeout=None,
        max_reservation_lifetime_seconds=60,
    )
    try:
        warmup = await _call(limiter.acquire_capacity, {"requests": 0}, "alias")
        await _call(limiter.refund_capacity, {"requests": 0}, warmup)
        backend = limiter._model_family_to_backend["cleanup"]
        target = backend._engine if store == "sqlite" else backend
        method = (
            "cleanup_consumption"
            if store == "sqlite"
            else "_refund_cancelled_consumption"
        )
        original = getattr(target, method)

        def fail(*args, **kwargs):
            nonlocal cleanup_calls
            cleanup_calls += 1
            raise refund_error

        async def async_fail(*args, **kwargs):
            fail(*args, **kwargs)

        if refund_error is not None:
            monkeypatch.setattr(
                target,
                method,
                async_fail if asynchronous and store != "sqlite" else fail,
            )
        armed = True
        expected = (
            AcquireRefundFailedError
            if cleanup in {"ordinary", "expired"}
            else type(refund_error)
            if cleanup in {"critical", "group"}
            else type(interruption)
        )
        with pytest.raises(expected) as caught:
            await _call(
                limiter.record_usage
                if cleanup == "record"
                else limiter.acquire_capacity,
                {"requests": 4},
                "alias",
            )
        armed = False
        monkeypatch.setattr(target, method, original)
        diagnostic = await _call(backend.introspect)
        if cleanup in {"ordinary", "expired"}:
            error = caught.value
            _assert_authority(limiter, error, interruption, issued_at)
            assert error.refund_error is refund_error
            assert diagnostic.buckets[0].current_capacity == pytest.approx(6, abs=0.01)
            if cleanup == "expired":
                with monkeypatch.context() as clock:
                    clock.setattr(time, "time", lambda: issued_at + 61)
                    refreshed = error.reservation.model_copy(
                        update={"created_at_seconds": issued_at + 61}
                    )
                    with pytest.raises(
                        ValueError, match="Reservation lifetime exceeded"
                    ):
                        await _call(limiter.refund_capacity, {"requests": 0}, refreshed)
            else:
                await _call(limiter.refund_capacity, {"requests": 0}, error.reservation)
                with pytest.raises(DuplicateRefundError):
                    await _call(
                        limiter.refund_capacity,
                        {"requests": 0},
                        error.reservation.model_copy(),
                    )
                assert (await _call(backend.introspect)).buckets[
                    0
                ].current_capacity == 10
        elif cleanup in {"critical", "group"}:
            assert caught.value is refund_error
        else:
            assert caught.value is interruption
            expected_capacity = 6 if cleanup == "record" else 10
            assert diagnostic.buckets[0].current_capacity == pytest.approx(
                expected_capacity, abs=0.01
            )
            assert cleanup_calls == 0
        assert limiter.snapshot_state()["in_flight_reservations"] == 0
        assert not limiter._pending_acquire_reservations
    finally:
        await _call(limiter.aclose if asynchronous else limiter.close)
        if client is not None:
            await _call(client.aclose if asynchronous else client.close)


@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
@pytest.mark.parametrize("outcome", ["committed", "failed"])
async def test_sqlite_settled_write_cleanup_recovery(  # noqa: PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: str,
    outcome: str,
) -> None:
    path = tmp_path / "worker.sqlite3"
    limiter = RateLimiter(
        _config(),
        backend=SqliteBackendBuilder(path, key_prefix="cleanup", busy_timeout_ms=5),
        max_reservation_lifetime_seconds=60,
    )
    writer = sqlite3.connect(path, isolation_level=None, timeout=0.1)
    settled = threading.Event()
    release = threading.Event()
    observing = asyncio.Event()
    attempt_result = None
    task = None
    try:
        warmup = await limiter.acquire_capacity({"requests": 0}, "alias")
        await limiter.refund_capacity({"requests": 0}, warmup)
        backend = limiter._model_family_to_backend["cleanup"]
        original = backend._engine.try_consume
        original_observer = backend._wait_for_future_while_cancelled

        def pause_result(*args, **kwargs):
            nonlocal attempt_result
            if outcome == "committed":
                attempt_result = original(*args, **kwargs)
            settled.set()
            if not release.wait(3):
                raise RuntimeError("test did not release SQLite result")
            if outcome == "failed":
                raise ConnectionError("unknown or failed write")
            return attempt_result

        async def observe(future):
            observing.set()
            return await original_observer(future)

        monkeypatch.setattr(backend._engine, "try_consume", pause_result)
        monkeypatch.setattr(backend, "_wait_for_future_while_cancelled", observe)
        task = asyncio.create_task(
            limiter.acquire_capacity(
                {"requests": 4},
                "alias",
                timeout=0.1 if interruption == "deadline" else None,
            )
        )
        assert await asyncio.to_thread(settled.wait, 2)
        writer.execute("BEGIN IMMEDIATE")
        if interruption == "cancel":
            task.cancel("caller cancellation")
        await asyncio.wait_for(observing.wait(), 2)
        release.set()
        expected = (
            AcquireRefundFailedError
            if outcome == "committed"
            else (asyncio.CancelledError if interruption == "cancel" else TimeoutError)
        )
        with pytest.raises(expected) as caught:
            await task
        writer.execute("ROLLBACK")
        if outcome == "committed":
            error = caught.value
            assert isinstance(
                error.interrupted_by,
                asyncio.CancelledError if interruption == "cancel" else TimeoutError,
            )
            _assert_authority(
                limiter, error, error.interrupted_by, attempt_result.result.current_time
            )
            assert isinstance(error.refund_error, BackendLockContentionError)
            await limiter.refund_capacity({"requests": 0}, error.reservation)
            assert _sqlite_state(path) == (10, [], 2)
        else:
            assert _sqlite_state(path) == (10, [], 1)
        assert limiter.snapshot_state()["in_flight_reservations"] == 0
    finally:
        release.set()
        writer.close()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await limiter.aclose()


@pytest.mark.parametrize("store", ["memory", "sqlite"])
async def test_repeated_cancellation_drains_cleanup_before_recovery(  # noqa: PLR0915
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: str
) -> None:
    interruption = asyncio.CancelledError("callback cancellation")
    entered = asyncio.Event()
    issued_at = None
    refund_error = ConnectionError("settled cleanup failure")

    async def callback(**kwargs):
        nonlocal issued_at
        issued_at = kwargs["current_time"]
        if store == "memory":
            await backend._condition.acquire()
            entered.set()
        raise interruption

    limiter = RateLimiter(
        _config(),
        backend=(
            MemoryBackendBuilder()
            if store == "memory"
            else SqliteBackendBuilder(tmp_path / "drain.sqlite3", key_prefix="cleanup")
        ),
        callbacks=RateLimiterCallbacks(on_capacity_consumed=callback),
        callback_timeout=None,
    )
    task = None
    try:
        backend = await limiter._get_backend(_config())
        if store == "memory":
            original = backend._get_capacities

            def fail_during_cleanup(*args, **kwargs):
                if issued_at is not None:
                    raise refund_error
                return original(*args, **kwargs)

            monkeypatch.setattr(backend, "_get_capacities", fail_during_cleanup)
        else:
            loop = asyncio.get_running_loop()
            gate = threading.Event()

            def fail_cleanup(*args, **kwargs):
                loop.call_soon_threadsafe(entered.set)
                if not gate.wait(3):
                    raise RuntimeError("cleanup gate was not released")
                raise refund_error

            monkeypatch.setattr(backend._engine, "cleanup_consumption", fail_cleanup)
        task = asyncio.create_task(limiter.acquire_capacity({"requests": 4}, "alias"))
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.sleep(0)
        task.cancel("second cancellation")
        await asyncio.sleep(0)
        assert not task.done()
        if store == "memory":
            backend._condition.release()
        else:
            gate.set()
        with pytest.raises(AcquireRefundFailedError) as caught:
            await task
        _assert_authority(limiter, caught.value, interruption, issued_at)
        assert caught.value.refund_error is refund_error
        monkeypatch.undo()
        await limiter.refund_capacity({"requests": 0}, caught.value.reservation)
    finally:
        if store == "sqlite":
            gate.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await limiter.aclose()
