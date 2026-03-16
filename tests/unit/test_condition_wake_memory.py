"""Tests that refund_capacity and set_max_capacity wake blocked waiters immediately.

These tests use a long sleep_interval (5.0s) to prove that the waiter is woken
by a condition signal, not by the polling loop.

Covers: async MemoryBackend and sync SyncMemoryBackend.
"""

import asyncio
import contextlib
import threading
import time

import pytest
from frozendict import frozendict

from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)

# Long sleep interval: poll-based approach takes >5s; condition wake takes <1s.
_LONG_POLL_INTERVAL = 5.0


def _make_slow_refill_config(
    *, limit: float = 10, per_seconds: int = 3600, metric: str = "requests"
) -> PerModelConfig:
    """Slow natural refill — capacity won't recover on its own during the test."""
    return PerModelConfig(
        model_family="test",
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
    )


def _make_fast_refill_config(
    *, limit: float = 5, per_seconds: int = 1, metric: str = "requests"
) -> PerModelConfig:
    """Fast refill so set_max_capacity rate increase provides enough capacity."""
    return PerModelConfig(
        model_family="test",
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
    )


# ---------------------------------------------------------------------------
# Async MemoryBackend -- refund wake
# ---------------------------------------------------------------------------


class TestAsyncRefundWake:
    async def test_refund_wakes_blocked_waiter(self):
        """Refunding capacity should wake a blocked waiter immediately."""
        builder = MemoryBackendBuilder(sleep_interval=_LONG_POLL_INTERVAL)
        backend = builder.build(_make_slow_refill_config(limit=10, per_seconds=3600))

        # Exhaust all capacity
        await backend.await_for_capacity(frozendict({"requests": 10.0}))

        # Start a waiter task
        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 5.0}))
        )
        await asyncio.sleep(0.1)  # let waiter enter the wait loop

        # Refund capacity -- should wake the waiter via condition notification
        await backend.refund_capacity(
            reserved_usage=frozendict({"requests": 10.0}),
            actual_usage=frozendict({"requests": 0.0}),
        )

        # Waiter should complete quickly (< 2s), not after 5s poll
        done, pending = await asyncio.wait({task}, timeout=2.0)

        # Cleanup
        for p in pending:
            p.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        assert len(done) == 1, (
            "Waiter should have been woken by refund_capacity, "
            "but it's still blocked (likely waiting for poll interval)"
        )


# ---------------------------------------------------------------------------
# Async MemoryBackend -- set_max_capacity wake
# ---------------------------------------------------------------------------


class TestAsyncSetMaxCapacityWake:
    async def test_set_max_capacity_wakes_blocked_waiter(self):
        """Increasing max_capacity should wake a blocked waiter immediately.

        Uses a fast-refill config (per_seconds=1) so that after set_max_capacity
        increases the rate from 5/s to 100/s, the waiter finds sufficient
        capacity on re-check:
          capacity = min(100, 0 + ~0.2s * 100/s) = 20 >= 3
        """
        builder = MemoryBackendBuilder(sleep_interval=_LONG_POLL_INTERVAL)
        backend = builder.build(_make_fast_refill_config(limit=5, per_seconds=1))

        # Exhaust all capacity
        await backend.await_for_capacity(frozendict({"requests": 5.0}))

        # Start a waiter requesting 3 (blocks because capacity is 0)
        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 3.0}))
        )
        await asyncio.sleep(0.1)  # let waiter enter the wait loop

        # Increase max_capacity: rate goes from 5/s to 100/s
        await backend.set_max_capacity("requests", 1, 100.0)

        done, pending = await asyncio.wait({task}, timeout=2.0)

        for p in pending:
            p.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        assert len(done) == 1, (
            "Waiter should have been woken by set_max_capacity, "
            "but it's still blocked (likely waiting for poll interval)"
        )


# ---------------------------------------------------------------------------
# Sync SyncMemoryBackend -- refund wake
# ---------------------------------------------------------------------------


class TestSyncRefundWake:
    def test_refund_wakes_blocked_waiter(self):
        """Refunding capacity should wake a blocked waiter immediately."""
        builder = SyncMemoryBackendBuilder(sleep_interval=_LONG_POLL_INTERVAL)
        backend = builder.build(_make_slow_refill_config(limit=10, per_seconds=3600))

        # Exhaust all capacity
        backend.wait_for_capacity(frozendict({"requests": 10.0}))

        completed = threading.Event()

        def waiter():
            backend.wait_for_capacity(frozendict({"requests": 5.0}))
            completed.set()

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        time.sleep(0.1)  # let waiter enter the wait loop

        # Refund capacity -- should wake the waiter via condition notification
        backend.refund_capacity(
            reserved_usage=frozendict({"requests": 10.0}),
            actual_usage=frozendict({"requests": 0.0}),
        )

        assert completed.wait(timeout=2.0), (
            "Waiter should have been woken by refund_capacity within 2.0s, "
            "but it's still blocked (likely waiting for poll interval)"
        )


# ---------------------------------------------------------------------------
# Sync SyncMemoryBackend -- set_max_capacity wake
# ---------------------------------------------------------------------------


class TestSyncSetMaxCapacityWake:
    def test_set_max_capacity_wakes_blocked_waiter(self):
        """Increasing max_capacity should wake a blocked waiter immediately."""
        builder = SyncMemoryBackendBuilder(sleep_interval=_LONG_POLL_INTERVAL)
        backend = builder.build(_make_fast_refill_config(limit=5, per_seconds=1))

        # Exhaust all capacity
        backend.wait_for_capacity(frozendict({"requests": 5.0}))

        completed = threading.Event()

        def waiter():
            backend.wait_for_capacity(frozendict({"requests": 3.0}))
            completed.set()

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        time.sleep(0.1)

        # Increase max_capacity: rate goes from 5/s to 100/s
        backend.set_max_capacity("requests", 1, 100.0)

        assert completed.wait(timeout=2.0), (
            "Waiter should have been woken by set_max_capacity within 2.0s, "
            "but it's still blocked (likely waiting for poll interval)"
        )


# ---------------------------------------------------------------------------
# Lost-wakeup regression tests
#
# These tests use on_wait_start as a synchronization point.  In the buggy code,
# on_wait_start fired in the unlocked gap between check and wait, so a refund
# during the callback produced a lost notify_all().  After the fix, on_wait_start
# fires after the wait loop completes (outside the lock).
# ---------------------------------------------------------------------------


class TestSyncLostWakeupRegression:
    def test_refund_during_gap_wakes_waiter(self):
        """Refund between check and wait must not be lost.

        Before the fix: on_wait_start fires in the unlocked gap, the refund's
        notify_all() is lost, and the waiter sleeps for sleep_interval (5s).
        After the fix: check+wait are in the same lock, so refunds are properly
        serialized and the waiter wakes promptly.
        """
        in_callback = threading.Event()

        def on_wait_start(**kw):
            in_callback.set()
            time.sleep(0.3)

        callbacks = SyncRateLimiterCallbacks(on_wait_start=on_wait_start)
        builder = SyncMemoryBackendBuilder(sleep_interval=_LONG_POLL_INTERVAL)
        backend = builder.build(
            _make_slow_refill_config(limit=10, per_seconds=3600),
            callbacks=callbacks,
        )

        # Exhaust all capacity
        backend.wait_for_capacity(frozendict({"requests": 10.0}))

        completed = threading.Event()
        error_holder: list[Exception] = []

        def waiter():
            try:
                backend.wait_for_capacity(frozendict({"requests": 10.0}))
                completed.set()
            except Exception as e:
                error_holder.append(e)
                completed.set()

        t = threading.Thread(target=waiter, daemon=True)
        t.start()

        # Wait for the waiter to either enter the callback (buggy code)
        # or enter condition.wait (fixed code, in_callback won't be set yet)
        in_callback.wait(2.0)

        # Refund all capacity — in buggy code, notify_all() is lost
        backend.refund_capacity(
            frozendict({"requests": 10.0}),
            frozendict({"requests": 0.0}),
        )

        assert completed.wait(2.0), "Lost wakeup: waiter not woken by refund"
        assert not error_holder, f"Waiter raised: {error_holder[0]}"
        t.join(timeout=1.0)


class TestAsyncLostWakeupRegression:
    async def test_refund_during_gap_wakes_waiter(self):
        """Async mirror of the sync lost-wakeup regression test."""
        callback_entered = asyncio.Event()

        async def on_wait_start(**kw):
            callback_entered.set()
            await asyncio.sleep(0.3)

        callbacks = RateLimiterCallbacks(on_wait_start=on_wait_start)
        builder = MemoryBackendBuilder(sleep_interval=_LONG_POLL_INTERVAL)
        backend = builder.build(
            _make_slow_refill_config(limit=10, per_seconds=3600),
            callbacks=callbacks,
        )

        # Exhaust all capacity
        await backend.await_for_capacity(frozendict({"requests": 10.0}))

        async def waiter():
            await backend.await_for_capacity(frozendict({"requests": 10.0}))

        waiter_task = asyncio.create_task(waiter())
        # Give the waiter task a chance to run and enter condition.wait
        await asyncio.sleep(0.1)

        # Refund all capacity
        await backend.refund_capacity(
            frozendict({"requests": 10.0}),
            frozendict({"requests": 0.0}),
        )

        # Waiter should complete promptly
        try:
            await asyncio.wait_for(waiter_task, timeout=2.0)
        except TimeoutError:
            waiter_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter_task
            pytest.fail("Lost wakeup: async waiter not woken by refund")
