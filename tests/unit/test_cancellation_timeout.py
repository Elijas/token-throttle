"""Regression tests for Bug 1: asyncio.timeout must fire through consume_capacity.

Before the fix, ``suppress_current_task_cancellation`` was called in the
callback-phase CancelledError handler in both memory and Redis backends.
That uncancelled the current task so ``asyncio.Timeout.__aexit__`` never
observed the cancel-in-flight and silently let the ``async with`` block
"expire without firing." The caller was never informed their bound was
violated.

The fix narrows suppression to cancels arriving DURING the shielded write
itself; once the write has landed (mutation is durable), CancelledError
from post-write callbacks propagates so ``asyncio.timeout`` fires
correctly and structured concurrency still works.
"""

import asyncio
import time

import pytest
from frozendict import frozendict

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder


def _make_config(*, limit: float = 100, per_seconds: int = 60) -> PerModelConfig:
    return PerModelConfig(
        model_family="mf",
        quotas=UsageQuotas([Quota(metric="tok", limit=limit, per_seconds=per_seconds)]),
    )


class TestAsyncioTimeoutFiresThroughConsumeCapacity:
    """asyncio.timeout must raise TimeoutError when a post-write callback hangs."""

    async def test_memory_timeout_fires_when_callback_hangs(self):
        async def slow_callback(**_kwargs):
            # Hangs past the outer timeout.
            await asyncio.sleep(10)

        backend = MemoryBackendBuilder().build(
            _make_config(),
            callbacks=RateLimiterCallbacks(on_capacity_consumed=slow_callback),
        )

        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.1):
                await backend.consume_capacity(frozendict({"tok": 1.0}))

    async def test_memory_timeout_preserves_recorded_consumption(self):
        """Even when timeout fires, the mutation before the callback stands."""

        async def slow_callback(**_kwargs):
            await asyncio.sleep(10)

        backend = MemoryBackendBuilder().build(
            _make_config(limit=100),
            callbacks=RateLimiterCallbacks(on_capacity_consumed=slow_callback),
        )

        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.1):
                await backend.consume_capacity(frozendict({"tok": 20.0}))

        cap = backend._buckets[0].get_capacity(time.time()).amount
        assert cap == pytest.approx(80.0, abs=1.0), (
            f"Speedometer invariant violated: expected ~80 (100-20), got {cap}"
        )
