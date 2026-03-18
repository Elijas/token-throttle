"""Tests that timeout and cancellation never leak capacity in memory backends.

Regression tests for audit findings:
- Timeout (0 or N) does not consume capacity (Finding 1)
- Cancellation during condition.wait() does not consume capacity (Finding 2)
- Sync thread timeout does not consume capacity (Finding 3)
- CancelledError during async callback refunds capacity (Finding 4)

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
    *,
    limit: float = 100,
    per_seconds: int = _SLOW_REFILL_PER_SECONDS,
    metric: str = "requests",
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
            await backend.await_for_capacity(frozendict({"requests": 10.0}), timeout=0)

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
                backend.wait_for_capacity(frozendict({"requests": 10.0}), timeout=0.2)
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


class TestAsyncCallbackCancellationRefundsCapacity:
    """
    CancelledError during post-consumption callbacks refunds capacity.

    After _try_consume_locked() succeeds (inside lock), callbacks fire outside the
    lock with real `await` calls. If a CancelledError arrives during a callback,
    the backend refunds consumed capacity under the lock and re-raises.
    """

    async def test_cancellation_during_on_capacity_consumed_refunds_capacity(self):
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
        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)

    async def test_cancellation_during_after_wait_end_consumption_refunds_capacity(
        self,
    ):
        """CancelledError during after_wait_end_consumption callback refunds capacity."""
        gate = asyncio.Event()
        entered_callback = asyncio.Event()

        async def slow_wait_end_callback(**kwargs):
            if not gate.is_set():
                return
            entered_callback.set()
            await asyncio.sleep(10)

        callbacks = RateLimiterCallbacks(
            after_wait_end_consumption=slow_wait_end_callback
        )
        builder = MemoryBackendBuilder(sleep_interval=0.01)
        config = _make_config(limit=100, per_seconds=1)  # fast refill
        backend = builder.build(config, callbacks=callbacks)

        # Exhaust capacity so the next call must wait
        await backend.await_for_capacity(frozendict({"requests": 100.0}))
        gate.set()

        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 5.0}))
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=5.0)

        cap_before = _get_bucket_capacity(backend)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Capacity should be restored (5 tokens refunded)
        cap_after = _get_bucket_capacity(backend)
        assert cap_after >= cap_before + 4.0

    async def test_cancellation_during_fresh_start_callback_refunds_capacity(self):
        """CancelledError during on_missing_consumption_data callback refunds capacity."""
        entered_callback = asyncio.Event()

        async def slow_fresh_start_callback(**kwargs):
            entered_callback.set()
            await asyncio.sleep(10)

        callbacks = RateLimiterCallbacks(
            on_missing_consumption_data=slow_fresh_start_callback
        )
        builder = MemoryBackendBuilder()
        # Fresh-start callback fires on the very first call to a bucket
        config = _make_config(limit=100)
        backend = builder.build(config, callbacks=callbacks)

        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 5.0}))
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=2.0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # 5 tokens should be refunded — capacity should be ~100 (fresh bucket)
        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# Group 5: Cancellation refund must preserve negative debt
# ---------------------------------------------------------------------------


class TestCancellationDebtPreservation:
    """
    Regression: _refund_cancelled_consumption called _set_capacities without
    allow_negative=True, so max(0, value) clamped negative debt to 0.

    Scenario:
    1. Task A acquires 50 (capacity → 50)
    2. Task A enters a slow on_capacity_consumed callback
    3. While callback runs, Task B calls consume_capacity(200) → capacity → -150
    4. Task A is cancelled during callback → refund 50 → capacity should be -100
    5. BUG: _set_capacities clamps -100 to 0, erasing 100 units of debt
    """

    async def test_cancellation_refund_preserves_negative_debt(self):
        gate = asyncio.Event()
        entered_callback = asyncio.Event()

        async def slow_callback(**kwargs):
            if not gate.is_set():
                return
            entered_callback.set()
            await asyncio.sleep(10)

        callbacks = RateLimiterCallbacks(on_capacity_consumed=slow_callback)
        builder = MemoryBackendBuilder()
        config = _make_config(limit=100)
        backend = builder.build(config, callbacks=callbacks)

        # Task A: acquire 50 (capacity → 50).  Callback returns fast (gate closed).
        await backend.await_for_capacity(frozendict({"requests": 50.0}))
        cap = _get_bucket_capacity(backend)
        assert cap == pytest.approx(50.0, abs=1.0)

        # Open the gate so the NEXT consumption enters the slow callback
        gate.set()

        # Task A (second call): acquire 10 — will enter slow callback
        task_a = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 10.0}))
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=2.0)

        # Close the gate BEFORE consume_capacity so its callback returns fast
        # (otherwise the main coroutine blocks in slow_callback for 10s)
        gate.clear()

        # While Task A is stuck in callback, Task B drives capacity negative
        # consume_capacity is a "speedometer" op — it uses allow_negative=True
        await backend.consume_capacity(frozendict({"requests": 200.0}))
        cap_after_consume = _get_bucket_capacity(backend)
        # capacity was 40 (50-10 already consumed by task_a) → 40-200 = -160
        assert cap_after_consume < -100, (
            f"Expected deep negative debt, got {cap_after_consume}"
        )

        # Cancel Task A during callback → triggers _refund_cancelled_consumption
        task_a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_a

        # Refund should add back 10 but PRESERVE the negative debt from Task B
        cap_final = _get_bucket_capacity(backend)
        expected = cap_after_consume + 10.0  # debt preserved, only refund added back
        assert cap_final == pytest.approx(expected, abs=1.0), (
            f"Debt erased! Expected {expected}, got {cap_final}. "
            f"If cap_final ≈ 0, the bug is that _refund_cancelled_consumption "
            f"calls _set_capacities without allow_negative=True."
        )


# ---------------------------------------------------------------------------
# Group 6: Double-cancellation / structured concurrency must not leak capacity
# ---------------------------------------------------------------------------


class TestDoubleCancellationNoCapacityLeak:
    """
    Regression: _refund_cancelled_consumption was not wrapped in asyncio.shield(),
    so a second CancelledError (e.g. from TaskGroup cancellation) could interrupt
    the lock acquisition during the refund, permanently leaking capacity.

    The Redis backend was fixed in commit 24d3a95 but the memory backend was not.
    """

    async def test_double_cancellation_does_not_leak_capacity(self):
        """Manual double-cancel with lock contention proves shield is needed.

        Without asyncio.shield(), the second cancel interrupts the lock wait
        inside _refund_cancelled_consumption and capacity is never restored.
        """
        gate = asyncio.Event()
        entered_callback = asyncio.Event()

        async def slow_callback(**kwargs):
            if not gate.is_set():
                return
            entered_callback.set()
            await asyncio.sleep(10)

        callbacks = RateLimiterCallbacks(on_capacity_consumed=slow_callback)
        builder = MemoryBackendBuilder()
        config = _make_config(limit=100)
        backend = builder.build(config, callbacks=callbacks)

        # Consume 90, leaving 10 available (callback fast — gate closed)
        await backend.await_for_capacity(frozendict({"requests": 90.0}))
        cap_before = _get_bucket_capacity(backend)
        assert cap_before == pytest.approx(10.0, abs=1.0)

        # Open gate — next consumption enters the slow callback
        gate.set()

        # Start a task that consumes 5 tokens, then enters slow callback
        # (lock is free here, so consumption succeeds and callback fires)
        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 5.0}))
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=2.0)

        # NOW hold the condition lock from a separate task to force contention
        # when _refund_cancelled_consumption tries to acquire it during refund.
        lock_holder_ready = asyncio.Event()
        lock_holder_release = asyncio.Event()

        async def hold_lock():
            async with backend._condition:
                lock_holder_ready.set()
                await lock_holder_release.wait()

        lock_task = asyncio.create_task(hold_lock())
        await asyncio.wait_for(lock_holder_ready.wait(), timeout=2.0)

        # First cancel — triggers _refund_cancelled_consumption, which tries
        # to acquire the lock (held by lock_task) and blocks at `await`
        task.cancel()
        await asyncio.sleep(0.05)  # let refund attempt start

        # Second cancel — without shield, this kills the lock acquisition
        task.cancel()
        await asyncio.sleep(0.05)

        # Release the held lock — shielded coroutine can now complete
        lock_holder_release.set()
        await lock_task
        await asyncio.sleep(0.1)  # let shielded refund finish

        # Capacity should be restored to ~10, not leaked to ~5
        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0), (
            f"Capacity leaked! Expected ~{cap_before}, got {cap_after}. "
            f"Without asyncio.shield(), double cancellation kills the refund."
        )

    async def test_taskgroup_cancellation_does_not_leak_capacity(self):
        """Structured concurrency scenario: double cancel with full capacity.

        Simulates a TaskGroup-like pattern where a consuming task is cancelled
        while another task holds the condition lock, then cancelled again
        (as happens when an outer scope cancels the TaskGroup itself).
        """
        entered_callback = asyncio.Event()

        async def slow_callback(**kwargs):
            entered_callback.set()
            await asyncio.sleep(10)

        callbacks = RateLimiterCallbacks(on_capacity_consumed=slow_callback)
        builder = MemoryBackendBuilder()
        config = _make_config(limit=100)
        backend = builder.build(config, callbacks=callbacks)

        # Start consuming task — enters slow callback immediately (no gate)
        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 5.0}))
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=2.0)

        # NOW hold the lock (after consumption) to force contention on refund
        lock_holder_ready = asyncio.Event()
        lock_holder_release = asyncio.Event()

        async def hold_lock():
            async with backend._condition:
                lock_holder_ready.set()
                await lock_holder_release.wait()

        lock_task = asyncio.create_task(hold_lock())
        await asyncio.wait_for(lock_holder_ready.wait(), timeout=2.0)

        # Simulate structured concurrency: first cancel (TaskGroup abort),
        # then second cancel (outer scope cleanup)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)

        # Release lock so shielded refund can complete
        lock_holder_release.set()
        await lock_task
        await asyncio.sleep(0.1)

        # Capacity should be restored to ~100, not leaked to ~95
        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(100.0, abs=1.0), (
            f"Capacity leaked! Expected ~100, got {cap_after}. "
            f"Structured concurrency cancellation interrupted the refund."
        )


# ---------------------------------------------------------------------------
# Group 7: consume_capacity CancelledError does NOT refund (intentional)
# ---------------------------------------------------------------------------


class TestConsumeCapacityNoRefundOnCancellation:
    """
    consume_capacity (the record_usage / speedometer path) intentionally does
    NOT refund capacity on CancelledError.  This is the opposite of
    await_for_capacity, which DOES refund.

    The asymmetry exists because consume_capacity records actual usage that
    already occurred — refunding would inflate capacity and violate rate limits.
    """

    async def test_consume_capacity_cancelled_during_callback_does_not_refund(self):
        """CancelledError during consume_capacity callback must NOT restore capacity.

        This documents the intentional asymmetry vs await_for_capacity, which
        DOES refund on CancelledError (Group 4 tests above).
        """
        gate = asyncio.Event()
        entered_callback = asyncio.Event()

        async def slow_callback(**kwargs):
            if not gate.is_set():
                return
            entered_callback.set()
            await asyncio.sleep(10)

        callbacks = RateLimiterCallbacks(on_capacity_consumed=slow_callback)
        builder = MemoryBackendBuilder()
        config = _make_config(limit=100)
        backend = builder.build(config, callbacks=callbacks)

        # Consume 50, leaving 50 (callback fast — gate closed)
        await backend.consume_capacity(frozendict({"requests": 50.0}))
        cap_before = _get_bucket_capacity(backend)
        assert cap_before == pytest.approx(50.0, abs=1.0)

        # Open gate — next consumption enters slow callback
        gate.set()

        task = asyncio.create_task(
            backend.consume_capacity(frozendict({"requests": 20.0}))
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=2.0)

        # Cancel during the callback — capacity is already consumed
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Capacity should be ~30 (50 - 20), NOT refunded back to ~50
        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(30.0, abs=1.0), (
            f"consume_capacity should NOT refund on cancellation! "
            f"Expected ~30, got {cap_after}. "
            f"If ~50, capacity was erroneously refunded."
        )


# ---------------------------------------------------------------------------
# Group 8: CancelledError during condition lock acquisition (pre-check)
# ---------------------------------------------------------------------------


class TestCancellationDuringLockAcquisitionNoLeak:
    """
    CancelledError before any capacity check — while waiting to acquire
    the condition lock in await_for_capacity — must not leak capacity.

    This exercises the path where a task is cancelled at line 257
    (async with self._condition) before _try_consume_locked ever runs.
    """

    async def test_cancel_while_waiting_for_condition_lock(self):
        """A task cancelled while waiting for the condition lock leaks nothing."""
        builder = MemoryBackendBuilder()
        config = _make_config(limit=100)
        backend = builder.build(config)

        # Consume 90, leaving 10
        await backend.await_for_capacity(frozendict({"requests": 90.0}))
        cap_before = _get_bucket_capacity(backend)
        assert cap_before == pytest.approx(10.0, abs=1.0)

        # Hold the condition lock so the next waiter blocks at lock acquisition
        lock_holder_ready = asyncio.Event()
        lock_holder_release = asyncio.Event()

        async def hold_lock():
            async with backend._condition:
                lock_holder_ready.set()
                await lock_holder_release.wait()

        lock_task = asyncio.create_task(hold_lock())
        await asyncio.wait_for(lock_holder_ready.wait(), timeout=2.0)

        # Start a waiter — it will block at `async with self._condition`
        # (never enters the while-loop or _try_consume_locked)
        waiter = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 5.0}))
        )
        await asyncio.sleep(0.05)  # let waiter reach the lock wait

        # Cancel it while it's waiting for the lock
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        # Release the lock holder
        lock_holder_release.set()
        await lock_task

        # Capacity must be unchanged — waiter never consumed anything
        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0), (
            f"Capacity leaked! Expected ~{cap_before}, got {cap_after}. "
            f"CancelledError during lock acquisition consumed capacity."
        )
