"""Tests that refund_capacity and set_max_capacity wake blocked waiters immediately.

These tests use a long sleep_interval (5.0s) to prove that the waiter is woken
by a condition signal, not by the polling loop.  With the current poll-based
implementation, these tests will fail (TDD red phase).

Covers: async MemoryBackend and sync SyncMemoryBackend.
"""

import asyncio
import threading
import time

from frozendict import frozendict

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
