"""Tests for simultaneous timeout + cancellation interaction and refund failure handling.

Regression tests for audit findings:
- Cancel during timeout wait propagates CancelledError (not TimeoutError)
- Cancel near timeout boundary still propagates CancelledError
- No capacity leak from repeated cancel+timeout interaction
- Refund failure during cancellation preserves CancelledError for structured concurrency (Finding 3)

Covers: async MemoryBackend.
"""

import asyncio
import logging
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
# Group 1: Cancel during timeout wait — CancelledError must propagate
# ---------------------------------------------------------------------------


class TestCancelDuringTimeoutWait:
    """Verify that task.cancel() during a timeout wait propagates CancelledError.

    Python 3.12+ guarantees this via Task.uncancel(), but these tests lock in
    the guarantee as a regression safety net.
    """

    async def test_cancel_during_timeout_wait_propagates_cancelled_error(self):
        """Exhaust capacity, start task with timeout=10.0, cancel at 0.05s.

        Must raise CancelledError, NOT TimeoutError.
        """
        builder = MemoryBackendBuilder(sleep_interval=_LONG_POLL_INTERVAL)
        config = _make_config(limit=100)
        backend = builder.build(config)

        # Exhaust capacity
        await backend.await_for_capacity(frozendict({"requests": 100.0}))
        cap_before = _get_bucket_capacity(backend)

        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 10.0}), timeout=10.0)
        )
        await asyncio.sleep(0.05)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)

    async def test_cancel_near_timeout_expiry_propagates_cancelled_error(self):
        """Cancel at 0.1s with timeout=0.2s — CancelledError must win.

        Even near the timeout boundary, an explicit cancel takes precedence
        over the natural timeout expiry.
        """
        builder = MemoryBackendBuilder(sleep_interval=_LONG_POLL_INTERVAL)
        config = _make_config(limit=100)
        backend = builder.build(config)

        await backend.await_for_capacity(frozendict({"requests": 100.0}))
        cap_before = _get_bucket_capacity(backend)

        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 10.0}), timeout=0.2)
        )
        await asyncio.sleep(0.1)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)

    async def test_timeout_fires_normally_without_cancel(self):
        """Control: timeout=0.2 without cancel raises TimeoutError. Capacity unchanged."""
        builder = MemoryBackendBuilder(sleep_interval=_LONG_POLL_INTERVAL)
        config = _make_config(limit=100)
        backend = builder.build(config)

        await backend.await_for_capacity(frozendict({"requests": 100.0}))
        cap_before = _get_bucket_capacity(backend)

        with pytest.raises(TimeoutError):
            await backend.await_for_capacity(
                frozendict({"requests": 10.0}), timeout=0.2
            )

        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)

    async def test_cancel_during_timeout_wait_no_capacity_leak(self):
        """Exhaust capacity, start 3 tasks with timeout=10.0, cancel all.

        Verifies no cumulative leak from repeated cancel+timeout interaction.
        """
        builder = MemoryBackendBuilder(sleep_interval=_LONG_POLL_INTERVAL)
        config = _make_config(limit=100)
        backend = builder.build(config)

        await backend.await_for_capacity(frozendict({"requests": 100.0}))
        cap_before = _get_bucket_capacity(backend)

        tasks = [
            asyncio.create_task(
                backend.await_for_capacity(frozendict({"requests": 10.0}), timeout=10.0)
            )
            for _ in range(3)
        ]
        await asyncio.sleep(0.05)

        for t in tasks:
            t.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(r, asyncio.CancelledError) for r in results)

        cap_after = _get_bucket_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)


# ---------------------------------------------------------------------------
# Group 2: Refund failure must preserve CancelledError (Finding 3 fix)
# ---------------------------------------------------------------------------


class TestRefundFailurePreservesCancelledError:
    """Verify that if _refund_cancelled_consumption raises, the original
    CancelledError still propagates.

    Without the fix, a refund failure (Redis down, shield propagates inner error)
    replaces CancelledError with the refund exception. This breaks structured
    concurrency (TaskGroups expect CancelledError).
    """

    async def test_refund_failure_during_cancellation_still_raises_cancelled_error(
        self,
        caplog,
    ):
        """Monkeypatch _refund_cancelled_consumption to raise ConnectionError.

        CancelledError must still propagate, not ConnectionError. A warning
        must be logged about the refund failure so the swallow is observable.
        """
        gate = asyncio.Event()
        entered_callback = asyncio.Event()

        async def slow_callback(**_kwargs):
            if not gate.is_set():
                return
            entered_callback.set()
            await asyncio.sleep(10)

        callbacks = RateLimiterCallbacks(on_capacity_consumed=slow_callback)
        builder = MemoryBackendBuilder()
        config = _make_config(limit=100)
        backend = builder.build(config, callbacks=callbacks)

        # Consume 90 tokens, leaving 10 available.
        await backend.await_for_capacity(frozendict({"requests": 90.0}))

        # Open the gate — next consumption enters the slow callback
        gate.set()

        # Monkeypatch the refund to simulate failure (e.g., Redis down)
        original_refund = backend._refund_cancelled_consumption

        async def failing_refund(_usage, **_kwargs):
            raise ConnectionError("simulated refund failure")

        backend._refund_cancelled_consumption = failing_refund

        # Start a task that consumes 5 tokens, then enters slow callback
        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 5.0}))
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=2.0)

        # Cancel during the callback — triggers CancelledError handler
        task.cancel()

        # The critical assertion: CancelledError must propagate, not ConnectionError
        with (
            caplog.at_level(logging.WARNING, logger="token_throttle"),
            pytest.raises(asyncio.CancelledError),
        ):
            await task

        # The refund failure must not be silently swallowed: a warning naming
        # the failure and the natural-refill fallback must be logged.
        refund_failure_records = [
            r
            for r in caplog.records
            if "cancellation-path refund failed" in r.getMessage()
        ]
        assert len(refund_failure_records) == 1
        assert "natural refill" in refund_failure_records[0].getMessage()
        assert refund_failure_records[0].levelno == logging.WARNING
        assert refund_failure_records[0].exc_info is not None

        # Restore original to avoid side effects
        backend._refund_cancelled_consumption = original_refund

    async def test_refund_cancelled_error_during_cancellation_still_raises_cancelled_error(
        self,
    ):
        """When the refund itself raises CancelledError (from asyncio.shield
        propagating inner task failure), the OUTER CancelledError must still
        propagate correctly.
        """
        gate = asyncio.Event()
        entered_callback = asyncio.Event()

        async def slow_callback(**_kwargs):
            if not gate.is_set():
                return
            entered_callback.set()
            await asyncio.sleep(10)

        callbacks = RateLimiterCallbacks(on_capacity_consumed=slow_callback)
        builder = MemoryBackendBuilder()
        config = _make_config(limit=100)
        backend = builder.build(config, callbacks=callbacks)

        await backend.await_for_capacity(frozendict({"requests": 90.0}))
        gate.set()

        async def refund_raises_cancelled(_usage, **_kwargs):
            raise asyncio.CancelledError

        backend._refund_cancelled_consumption = refund_raises_cancelled

        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 5.0}))
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=2.0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestSyncRefundFailurePreservesOriginalException:
    """Sync mirror of TestRefundFailurePreservesCancelledError.

    Sync code has no CancelledError; a critical callback exception
    (KeyboardInterrupt) plays the same role. If _refund_cancelled_consumption
    raises, the original KeyboardInterrupt must still propagate, and the
    refund failure must be logged rather than silently dropped.
    """

    def test_refund_failure_during_critical_callback_still_raises_original_exception(
        self,
        caplog,
    ):
        def raising_callback(**_kwargs):
            raise KeyboardInterrupt("simulated critical callback failure")

        callbacks = SyncRateLimiterCallbacks(on_capacity_consumed=raising_callback)
        builder = SyncMemoryBackendBuilder()
        config = _make_config(limit=100)
        backend = builder.build(config, callbacks=callbacks)

        # Monkeypatch the refund to simulate failure (e.g., Redis down)
        original_refund = backend._refund_cancelled_consumption

        def failing_refund(_usage, **_kwargs):
            raise ConnectionError("simulated refund failure")

        backend._refund_cancelled_consumption = failing_refund

        # The critical assertion: KeyboardInterrupt must propagate, not ConnectionError
        with (
            caplog.at_level(logging.WARNING, logger="token_throttle"),
            pytest.raises(
                KeyboardInterrupt, match="simulated critical callback failure"
            ),
        ):
            backend.wait_for_capacity(frozendict({"requests": 5.0}))

        # The refund failure must not be silently swallowed.
        refund_failure_records = [
            r
            for r in caplog.records
            if "cancellation-path refund failed" in r.getMessage()
        ]
        assert len(refund_failure_records) == 1
        assert "natural refill" in refund_failure_records[0].getMessage()
        assert refund_failure_records[0].levelno == logging.WARNING
        assert refund_failure_records[0].exc_info is not None

        # Restore original to avoid side effects
        backend._refund_cancelled_consumption = original_refund
