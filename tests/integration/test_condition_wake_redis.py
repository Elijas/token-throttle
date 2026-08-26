"""Integration tests for condition-based wake-on-refund and wake-on-set_max_capacity.

Parameterized across all backends (memory + redis) via the backend_builder
and sync_backend_builder fixtures.

These tests use a long sleep_interval (5.0s) to prove that waiters are woken
by condition notification, not by the polling loop.  With poll-based backends,
these tests will fail (TDD red phase).
"""

import asyncio
import threading

from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage

# Long sleep interval: poll-based approach takes >5s; condition wake takes <1s.
_LONG_POLL_INTERVAL = 5.0


def _make_config(
    *, limit: float = 10, per_seconds: int = 3600, metric: str = "requests"
) -> PerModelConfig:
    return PerModelConfig(
        model_family="test",
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
    )


# ---------------------------------------------------------------------------
# Async -- refund wake
# ---------------------------------------------------------------------------


async def test_refund_wakes_blocked_waiter(backend_builder):
    """Refunding capacity should wake a blocked waiter immediately."""
    config = _make_config(limit=10, per_seconds=3600)
    wait_started = asyncio.Event()

    async def on_wait_start(**_kwargs):
        wait_started.set()

    backend = backend_builder.build(
        config,
        callbacks=RateLimiterCallbacks(on_wait_start=on_wait_start),
    )
    backend._sleep_interval = _LONG_POLL_INTERVAL

    # Exhaust all capacity
    await backend.await_for_capacity(frozen_usage({"requests": 10}))

    # Start a waiter task
    task = asyncio.create_task(
        backend.await_for_capacity(frozen_usage({"requests": 5}))
    )
    await asyncio.wait_for(wait_started.wait(), timeout=2.0)

    # Refund capacity -- should wake the waiter via condition notification
    await backend.refund_capacity(
        reserved_usage=frozen_usage({"requests": 10}),
        actual_usage=frozen_usage({"requests": 0}),
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
# Async -- set_max_capacity wake
# ---------------------------------------------------------------------------


async def test_set_max_capacity_wakes_blocked_waiter(backend_builder):
    """Increasing max_capacity should wake a blocked waiter immediately.

    Uses per_seconds=1 so the rate increase from set_max_capacity (5/s -> 100/s)
    produces the requested 3 units within about 0.03 s after notification.
    """
    config = _make_config(limit=5, per_seconds=1)
    wait_started = asyncio.Event()

    async def on_wait_start(**_kwargs):
        wait_started.set()

    backend = backend_builder.build(
        config,
        callbacks=RateLimiterCallbacks(on_wait_start=on_wait_start),
    )
    backend._sleep_interval = _LONG_POLL_INTERVAL

    # Exhaust all capacity
    await backend.await_for_capacity(frozen_usage({"requests": 5}))

    # Waiter for 3 tokens -- blocks because capacity is 0
    task = asyncio.create_task(
        backend.await_for_capacity(frozen_usage({"requests": 3}))
    )
    await asyncio.wait_for(wait_started.wait(), timeout=2.0)

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
# Sync -- refund wake
# ---------------------------------------------------------------------------


def test_sync_refund_wakes_blocked_waiter(sync_backend_builder):
    """Refunding capacity should wake a blocked waiter immediately (sync)."""
    config = _make_config(limit=10, per_seconds=3600)
    wait_started = threading.Event()

    def on_wait_start(**_kwargs):
        wait_started.set()

    backend = sync_backend_builder.build(
        config,
        callbacks=SyncRateLimiterCallbacks(on_wait_start=on_wait_start),
    )
    backend._sleep_interval = _LONG_POLL_INTERVAL

    # Exhaust all capacity
    backend.wait_for_capacity(frozen_usage({"requests": 10}))

    completed = threading.Event()

    def waiter():
        backend.wait_for_capacity(frozen_usage({"requests": 5}))
        completed.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    assert wait_started.wait(timeout=2.0)

    # Refund capacity -- should wake the waiter
    backend.refund_capacity(
        reserved_usage=frozen_usage({"requests": 10}),
        actual_usage=frozen_usage({"requests": 0}),
    )

    assert completed.wait(timeout=2.0), (
        "Waiter should have been woken by refund_capacity within 2.0s"
    )


# ---------------------------------------------------------------------------
# Sync -- set_max_capacity wake
# ---------------------------------------------------------------------------


def test_sync_set_max_capacity_wakes_blocked_waiter(sync_backend_builder):
    """Increasing max_capacity should wake a blocked waiter immediately (sync)."""
    config = _make_config(limit=5, per_seconds=1)
    wait_started = threading.Event()

    def on_wait_start(**_kwargs):
        wait_started.set()

    backend = sync_backend_builder.build(
        config,
        callbacks=SyncRateLimiterCallbacks(on_wait_start=on_wait_start),
    )
    backend._sleep_interval = _LONG_POLL_INTERVAL

    # Exhaust all capacity
    backend.wait_for_capacity(frozen_usage({"requests": 5}))

    completed = threading.Event()

    def waiter():
        backend.wait_for_capacity(frozen_usage({"requests": 3}))
        completed.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    assert wait_started.wait(timeout=2.0)

    # Increase max_capacity: rate goes from 5/s to 100/s
    backend.set_max_capacity("requests", 1, 100.0)

    assert completed.wait(timeout=2.0), (
        "Waiter should have been woken by set_max_capacity within 2.0s"
    )
