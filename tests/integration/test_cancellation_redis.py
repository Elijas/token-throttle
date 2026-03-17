"""Integration tests: CancelledError during callbacks refunds capacity (Redis backend).

Mirrors the memory backend tests in test_cancellation_memory.py Group 4,
exercising the Redis-specific _refund_cancelled_consumption (with asyncio.shield).
"""

import asyncio
import time

import pytest
from frozendict import frozendict

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder

_SLOW_REFILL_PER_SECONDS = 3600


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
    if current_time is None:
        current_time = time.time()
    bucket = backend.sorted_buckets[0]
    result = bucket.calculate_capacity(
        last_checked=None,
        capacity=None,
        current_time=current_time,
    )
    return result.amount


async def _get_redis_capacity(backend) -> float:
    """Read current capacity from Redis (authoritative source)."""
    pipeline = backend._redis.pipeline()
    current_time = time.time()
    result = await backend._get_capacities_unsafe(pipeline=pipeline, current_time=current_time)
    caps = result.capacities
    # Return the first bucket's capacity
    return next(iter(caps.values()))


@pytest.mark.redis
class TestRedisCallbackCancellationRefundsCapacity:
    """CancelledError during post-consumption callbacks refunds capacity in Redis backend."""

    async def test_cancellation_during_on_capacity_consumed_in_check_and_consume(
        self, redis_client,
    ):
        """CancelledError in _check_and_consume_capacity's on_capacity_consumed refunds."""
        gate = asyncio.Event()
        entered_callback = asyncio.Event()

        async def slow_callback(**kwargs):
            if not gate.is_set():
                return
            entered_callback.set()
            await asyncio.sleep(10)

        callbacks = RateLimiterCallbacks(on_capacity_consumed=slow_callback)
        builder = RedisBackendBuilder(redis_client)
        config = _make_config(limit=100)
        backend = builder.build(config, callbacks=callbacks)

        # Consume 90, leaving 10
        await backend.await_for_capacity(frozendict({"requests": 90.0}))
        cap_before = await _get_redis_capacity(backend)
        assert cap_before == pytest.approx(10.0, abs=1.0)

        gate.set()

        # timeout=0 goes through _check_and_consume_capacity directly
        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 5.0}), timeout=0)
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=5.0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        cap_after = await _get_redis_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)

    async def test_cancellation_during_after_wait_end_consumption(
        self, redis_client,
    ):
        """CancelledError in await_for_capacity's after_wait_end_consumption refunds."""
        gate = asyncio.Event()
        entered_callback = asyncio.Event()

        async def slow_wait_end_callback(**kwargs):
            if not gate.is_set():
                return
            entered_callback.set()
            await asyncio.sleep(10)

        callbacks = RateLimiterCallbacks(after_wait_end_consumption=slow_wait_end_callback)
        builder = RedisBackendBuilder(redis_client, sleep_interval=0.01)
        config = _make_config(limit=100, per_seconds=1)  # fast refill
        backend = builder.build(config, callbacks=callbacks)

        # Exhaust capacity so the next call must wait
        await backend.await_for_capacity(frozendict({"requests": 100.0}))
        gate.set()

        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 5.0}))
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=5.0)

        cap_before = await _get_redis_capacity(backend)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        cap_after = await _get_redis_capacity(backend)
        assert cap_after >= cap_before + 4.0

    async def test_capacity_available_after_cancellation_refund(
        self, redis_client,
    ):
        """After cancellation refund, a subsequent caller can use the restored capacity."""
        gate = asyncio.Event()
        entered_callback = asyncio.Event()

        async def slow_callback(**kwargs):
            if not gate.is_set():
                return
            entered_callback.set()
            await asyncio.sleep(10)

        callbacks = RateLimiterCallbacks(on_capacity_consumed=slow_callback)
        builder = RedisBackendBuilder(redis_client)
        config = _make_config(limit=100)
        backend = builder.build(config, callbacks=callbacks)

        # Consume 95, leaving 5
        await backend.await_for_capacity(frozendict({"requests": 95.0}))
        gate.set()

        # This will consume 5, then get cancelled in the callback
        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 5.0}), timeout=0)
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=5.0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The 5 tokens should be available for this caller
        await backend.await_for_capacity(frozendict({"requests": 5.0}), timeout=1.0)
