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


# ---------------------------------------------------------------------------
# Lost-wakeup regression tests
#
# These tests use on_wait_start as a synchronization point. In the buggy code,
# on_wait_start fired in the unlocked gap between check and wait, so a refund
# during the callback produced a lost notify_all(). After the fix, the backend
# re-checks capacity immediately after the callback before it ever sleeps.
# ---------------------------------------------------------------------------


class TestSyncLostWakeupRegression:
    def test_refund_during_gap_wakes_waiter(self):
        """Refund between check and wait must not be lost.

        Before the fix: on_wait_start fires in the unlocked gap, the refund's
        notify_all() is lost, and the waiter sleeps for sleep_interval (5s).
        After the fix: the waiter re-checks capacity immediately after the
        callback, so refunds are observed before sleeping.
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

        # Wait for the waiter to enter the callback, then refund.
        assert in_callback.wait(2.0), "waiter did not enter on_wait_start"

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
