"""Tests that timeout and cancellation never leak capacity in memory backends.

Regression tests for audit findings:
- Timeout (0 or N) does not consume capacity (Finding 1)
- Cancellation during condition.wait() does not consume capacity (Finding 2)
- Sync thread timeout does not consume capacity (Finding 3)
- CancelledError during async callback leaks capacity (Finding 4 — xfail)

Covers: async MemoryBackend and sync SyncMemoryBackend.
"""

import asyncio
import threading
import time

import pytest
from frozendict import frozendict

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)

# Slow refill so natural recovery is negligible during tests.
# 100 tokens / 3600s = 0.028 tokens/sec — max ~0.3 tokens in a 10s test window.
_SLOW_REFILL_PER_SECONDS = 3600

# Long sleep_interval ensures waiters are in condition.wait(), not between iterations.
_LONG_POLL_INTERVAL = 5.0


def _make_config(
    *, limit: float = 100, per_seconds: int = _SLOW_REFILL_PER_SECONDS, metric: str = "requests"
) -> PerModelConfig:
    return PerModelConfig(
        model_family="test",
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
    )


def _get_bucket_capacity(backend, current_time: float | None = None) -> float:
    """Read the current capacity from the backend's first bucket."""
    if current_time is None:
        current_time = time.time()
    return backend._buckets[0].get_capacity(current_time).amount


# ---------------------------------------------------------------------------
# Group 1: Timeout does not leak capacity (regression — should PASS today)
# ---------------------------------------------------------------------------


class TestAsyncTimeoutNoCapacityLeak:
    """Verify that TimeoutError from await_for_capacity never consumes capacity."""

    async def test_timeout_zero_does_not_consume_capacity(self):
        """timeout=0 rejects immediately without touching bucket state."""
        builder = MemoryBackendBuilder()
        config = _make_config(limit=100)
        backend = builder.build(config)

        # Exhaust capacity
        await backend.await_for_capacity(frozendict({"requests": 100.0}))
        cap_before = _get_bucket_capacity(backend)

        with pytest.raises(TimeoutError):
            await backend.await_for_capacity(frozendict({"requests": 10.0}), timeout=0)

        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)

    async def test_timeout_n_does_not_consume_capacity(self):
        """timeout=N rejects after waiting without consuming any capacity."""
        builder = MemoryBackendBuilder()
        config = _make_config(limit=100)
        backend = builder.build(config)

        await backend.await_for_capacity(frozendict({"requests": 100.0}))
        cap_before = _get_bucket_capacity(backend)

        with pytest.raises(TimeoutError):
            await backend.await_for_capacity(
                frozendict({"requests": 10.0}), timeout=0.3
            )

        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)

    async def test_timeout_preserves_capacity_for_next_caller(self):
        """After a timeout, a subsequent caller with available capacity succeeds."""
        builder = MemoryBackendBuilder()
        # Use fast refill so capacity recovers for the second call
        config = _make_config(limit=100, per_seconds=1)
        backend = builder.build(config)

        await backend.await_for_capacity(frozendict({"requests": 100.0}))

        with pytest.raises(TimeoutError):
            await backend.await_for_capacity(
                frozendict({"requests": 10.0}), timeout=0
            )

        # Wait for refill (~0.2s at 100 tokens/sec gives ~20 tokens)
        await asyncio.sleep(0.2)
        # Should succeed — the timed-out call didn't steal any capacity
        await backend.await_for_capacity(frozendict({"requests": 5.0}), timeout=1.0)


class TestSyncTimeoutNoCapacityLeak:
    """Verify that TimeoutError from wait_for_capacity never consumes capacity."""

    def test_timeout_zero_does_not_consume_capacity(self):
        builder = SyncMemoryBackendBuilder()
        config = _make_config(limit=100)
        backend = builder.build(config)

        backend.wait_for_capacity(frozendict({"requests": 100.0}))
        cap_before = _get_bucket_capacity(backend)

        with pytest.raises(TimeoutError):
            backend.wait_for_capacity(frozendict({"requests": 10.0}), timeout=0)

        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)

    def test_timeout_n_does_not_consume_capacity(self):
        builder = SyncMemoryBackendBuilder()
        config = _make_config(limit=100)
        backend = builder.build(config)

        backend.wait_for_capacity(frozendict({"requests": 100.0}))
        cap_before = _get_bucket_capacity(backend)

        with pytest.raises(TimeoutError):
            backend.wait_for_capacity(frozendict({"requests": 10.0}), timeout=0.3)

        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)

    def test_timeout_preserves_capacity_for_next_caller(self):
        builder = SyncMemoryBackendBuilder()
        config = _make_config(limit=100, per_seconds=1)
        backend = builder.build(config)

        backend.wait_for_capacity(frozendict({"requests": 100.0}))

        with pytest.raises(TimeoutError):
            backend.wait_for_capacity(frozendict({"requests": 10.0}), timeout=0)

        time.sleep(0.2)
        backend.wait_for_capacity(frozendict({"requests": 5.0}), timeout=1.0)


# ---------------------------------------------------------------------------
# Group 2: Cancellation during wait does not leak capacity (regression)
# ---------------------------------------------------------------------------


class TestAsyncCancellationNoCapacityLeak:
    """Verify that cancelling a waiting task never consumes capacity."""

    async def test_cancel_waiting_task_does_not_consume_capacity(self):
        """A cancelled waiter must not have consumed any capacity."""
        builder = MemoryBackendBuilder(sleep_interval=_LONG_POLL_INTERVAL)
        config = _make_config(limit=100)
        backend = builder.build(config)

        await backend.await_for_capacity(frozendict({"requests": 100.0}))
        cap_before = _get_bucket_capacity(backend)

        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 10.0}))
        )
        await asyncio.sleep(0.1)  # let waiter enter condition.wait()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)

    async def test_cancel_does_not_block_subsequent_acquires(self):
        """After cancellation, a new caller can still acquire capacity normally."""
        builder = MemoryBackendBuilder(sleep_interval=_LONG_POLL_INTERVAL)
        config = _make_config(limit=100)
        backend = builder.build(config)

        # Consume 90, leaving 10 available
        await backend.await_for_capacity(frozendict({"requests": 90.0}))

        # Start a waiter that wants more than what's available
        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 50.0}))
        )
        await asyncio.sleep(0.1)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The remaining ~10 tokens should still be available
        await backend.await_for_capacity(frozendict({"requests": 5.0}), timeout=1.0)

    async def test_cancel_multiple_waiters_preserves_capacity(self):
        """Cancelling several waiters simultaneously doesn't leak capacity."""
        builder = MemoryBackendBuilder(sleep_interval=_LONG_POLL_INTERVAL)
        config = _make_config(limit=100)
        backend = builder.build(config)

        await backend.await_for_capacity(frozendict({"requests": 100.0}))
        cap_before = _get_bucket_capacity(backend)

        tasks = [
            asyncio.create_task(
                backend.await_for_capacity(frozendict({"requests": 10.0}))
            )
            for _ in range(5)
        ]
        await asyncio.sleep(0.1)

        for t in tasks:
            t.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(r, asyncio.CancelledError) for r in results)

        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)


# ---------------------------------------------------------------------------
# Group 3: Sync threading timeout does not leak capacity (regression)
# ---------------------------------------------------------------------------


class TestSyncCancellationNoCapacityLeak:
    """Verify that a sync thread timing out never consumes capacity."""

    def test_thread_timeout_does_not_consume_capacity(self):
        """A thread that times out waiting for capacity must not have consumed any."""
        builder = SyncMemoryBackendBuilder(sleep_interval=_LONG_POLL_INTERVAL)
        config = _make_config(limit=100)
        backend = builder.build(config)

        backend.wait_for_capacity(frozendict({"requests": 100.0}))
        cap_before = _get_bucket_capacity(backend)

        error_holder: list[Exception] = []

        def waiter():
            try:
                backend.wait_for_capacity(
                    frozendict({"requests": 10.0}), timeout=0.2
                )
            except TimeoutError as e:
                error_holder.append(e)

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        t.join(timeout=5.0)

        assert len(error_holder) == 1, "Waiter should have raised TimeoutError"
        assert isinstance(error_holder[0], TimeoutError)

        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)


# ---------------------------------------------------------------------------
# Group 4: Callback cancellation leaks capacity (documents known issue)
# ---------------------------------------------------------------------------


class TestAsyncCallbackCancellationLeaksCapacity:
    """
    Finding 4: CancelledError during on_capacity_consumed leaks capacity.

    After _try_consume_locked() succeeds (inside lock), callbacks fire outside the
    lock with real `await` calls. If a CancelledError arrives during a callback,
    capacity is consumed but await_for_capacity() raises CancelledError — the
    caller never gets a CapacityReservation, so no refund is possible.

    Marked xfail(strict=True): the test MUST fail (capacity is leaked).
    """

    @pytest.mark.xfail(
        strict=True,
        reason="Known: CancelledError during on_capacity_consumed callback leaks capacity",
    )
    async def test_cancellation_during_on_capacity_consumed_leaks_capacity(self):
        # Gate ensures the slow path only activates AFTER the initial exhaust.
        # on_capacity_consumed fires for every consumption, including the exhaust.
        gate = asyncio.Event()
        entered_callback = asyncio.Event()

        async def slow_callback(**kwargs):
            if not gate.is_set():
                return  # fast-path for initial consumption
            entered_callback.set()
            await asyncio.sleep(10)  # long enough to guarantee cancellation

        callbacks = RateLimiterCallbacks(on_capacity_consumed=slow_callback)
        builder = MemoryBackendBuilder()
        config = _make_config(limit=100)
        backend = builder.build(config, callbacks=callbacks)

        # Consume 90 tokens, leaving 10 available.
        # The callback fires but returns immediately (gate not set).
        await backend.await_for_capacity(frozendict({"requests": 90.0}))
        cap_before = _get_bucket_capacity(backend)
        assert cap_before == pytest.approx(10.0, abs=1.0)

        # Open the gate — next consumption will enter the slow callback
        gate.set()

        # Start a task that will succeed in consuming 5 tokens, then enter
        # the slow callback (outside the lock)
        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 5.0}))
        )

        # Wait for the callback to be entered — capacity is already consumed
        await asyncio.wait_for(entered_callback.wait(), timeout=2.0)

        # Cancel the task while it's in the callback
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # If capacity were NOT leaked, cap_after would equal cap_before (~10).
        # But capacity WAS consumed (5 tokens), so cap_after should be ~5.
        # The xfail asserts that the capacity was properly preserved (it won't be).
        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)
