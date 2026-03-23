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
    result = await backend._get_capacities_unsafe(
        pipeline=pipeline, current_time=current_time
    )
    caps = result.capacities
    # Return the first bucket's capacity
    return next(iter(caps.values()))


@pytest.mark.redis
class TestRedisCallbackCancellationRefundsCapacity:
    """CancelledError during post-consumption callbacks refunds capacity in Redis backend."""

    async def test_cancellation_during_on_capacity_consumed_in_check_and_consume(
        self,
        redis_client,
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
        self,
        redis_client,
    ):
        """CancelledError in await_for_capacity's after_wait_end_consumption refunds."""
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
        self,
        redis_client,
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


@pytest.mark.redis
class TestRedisLockReleaseCancellationRefundsCapacity:
    """CancelledError during AsyncExitStack lock release refunds capacity.

    Regression: _check_and_consume_capacity's try/except CancelledError was
    outside the async with block. Lock release involves async Redis I/O;
    CancelledError during release bypassed the refund handler.
    """

    async def test_cancellation_during_lock_release_refunds_capacity(
        self,
        redis_client,
    ):
        """CancelledError during lock __aexit__ triggers refund via consumed flag."""
        entered_release = asyncio.Event()

        class _SlowReleaseLock:
            def __init__(self, real_lock):
                self._real_lock = real_lock

            async def acquire(self, **kwargs):
                return await self._real_lock.acquire(**kwargs)

            async def release(self):
                await self._real_lock.release()
                entered_release.set()
                await asyncio.sleep(10)

        builder = RedisBackendBuilder(redis_client)
        config = _make_config(limit=100)
        backend = builder.build(config)

        # Consume 90, leaving 10
        await backend.await_for_capacity(frozendict({"requests": 90.0}))
        cap_before = await _get_redis_capacity(backend)
        assert cap_before == pytest.approx(10.0, abs=1.0)

        # Patch bucket.lock to return slow-release locks
        bucket = backend.sorted_buckets[0]
        original_lock_fn = bucket.lock

        def patched_lock(**kwargs):
            return _SlowReleaseLock(original_lock_fn(**kwargs))

        bucket.lock = patched_lock

        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 5.0}), timeout=0)
        )
        await asyncio.wait_for(entered_release.wait(), timeout=5.0)

        # Cancel during slow release; restore original lock so shielded refund works
        task.cancel()
        bucket.lock = original_lock_fn

        with pytest.raises(asyncio.CancelledError):
            await task

        cap_after = await _get_redis_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0), (
            f"Capacity leaked! Expected ~{cap_before}, got {cap_after}. "
            f"CancelledError during lock release bypassed the refund handler."
        )


@pytest.mark.redis
class TestRedisCancellationDebtPreservation:
    """Cancellation refund preserves negative debt in Redis backend.

    Mirrors test_cancellation_memory.py:TestCancellationDebtPreservation.
    """

    async def test_cancellation_refund_preserves_negative_debt(self, redis_client):
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

        # Task A: acquire 50 (capacity → 50). Callback fast (gate closed).
        await backend.await_for_capacity(frozendict({"requests": 50.0}))
        cap = await _get_redis_capacity(backend)
        assert cap == pytest.approx(50.0, abs=1.0)

        gate.set()

        # Task A (second call): acquire 10 — enters slow callback
        task_a = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 10.0}), timeout=0)
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=5.0)

        # Close gate so consume_capacity's callback returns fast
        gate.clear()

        # Drive capacity negative: was 40 (50-10) → 40-200 = -160
        await backend.consume_capacity(frozendict({"requests": 200.0}))
        cap_after_consume = await _get_redis_capacity(backend)
        assert cap_after_consume < -100, (
            f"Expected deep negative debt, got {cap_after_consume}"
        )

        # Cancel Task A → refund 10
        task_a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_a

        cap_final = await _get_redis_capacity(backend)
        expected = cap_after_consume + 10.0
        assert cap_final == pytest.approx(expected, abs=1.0), (
            f"Debt erased! Expected {expected}, got {cap_final}. "
            f"If cap_final ≈ 0, _refund_cancelled_consumption "
            f"calls _set_capacities without allow_negative=True."
        )


@pytest.mark.redis
class TestRedisConsumeCapacityNoRefundOnCancellation:
    """consume_capacity (record_usage / speedometer) does NOT refund on CancelledError.

    Mirrors test_cancellation_memory.py:TestConsumeCapacityNoRefundOnCancellation.
    This documents the intentional asymmetry: await_for_capacity DOES refund,
    but consume_capacity does NOT — the usage already occurred.
    """

    async def test_consume_capacity_cancelled_during_callback_does_not_refund(
        self,
        redis_client,
    ):
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

        # Consume 50, leaving 50 (callback fast — gate closed)
        await backend.consume_capacity(frozendict({"requests": 50.0}))
        cap_before = await _get_redis_capacity(backend)
        assert cap_before == pytest.approx(50.0, abs=1.0)

        # Open gate — next consumption enters slow callback
        gate.set()

        task = asyncio.create_task(
            backend.consume_capacity(frozendict({"requests": 20.0}))
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=5.0)

        # Cancel during the callback — capacity is already consumed
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Capacity should be ~30 (50 - 20), NOT refunded back to ~50
        cap_after = await _get_redis_capacity(backend)
        assert cap_after == pytest.approx(30.0, abs=1.0), (
            f"consume_capacity should NOT refund on cancellation! "
            f"Expected ~30, got {cap_after}. "
            f"If ~50, capacity was erroneously refunded."
        )


@pytest.mark.redis
class TestRedisDoubleCancellationNoCapacityLeak:
    """Double cancellation (structured concurrency) must not leak capacity.

    Mirrors test_cancellation_memory.py:TestDoubleCancellationNoCapacityLeak.
    asyncio.shield on _refund_cancelled_consumption protects the refund.
    """

    async def test_double_cancellation_does_not_leak_capacity(self, redis_client):
        entered_callback = asyncio.Event()

        async def slow_callback(**kwargs):
            entered_callback.set()
            await asyncio.sleep(10)

        callbacks = RateLimiterCallbacks(on_capacity_consumed=slow_callback)
        builder = RedisBackendBuilder(redis_client)
        config = _make_config(limit=100)
        backend = builder.build(config, callbacks=callbacks)

        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 5.0}), timeout=0)
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=5.0)

        # Hold the Redis distributed lock to force contention during refund
        bucket = backend.sorted_buckets[0]
        held_lock = bucket.lock(timeout=30)
        assert await held_lock.acquire() is True

        # First cancel — refund tries to acquire held lock, blocks in shield
        task.cancel()
        await asyncio.sleep(0.05)

        # Second cancel — shield protects inner refund task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Release lock so shielded refund can complete
        await held_lock.release()
        await asyncio.sleep(0.5)

        cap_after = await _get_redis_capacity(backend)
        assert cap_after == pytest.approx(100.0, abs=1.0), (
            f"Capacity leaked! Expected ~100, got {cap_after}. "
            f"Double cancellation interrupted the refund."
        )


@pytest.mark.redis
class TestRedisPipelineExecuteCancellationRefundsCapacity:
    """CancelledError during pipeline.execute() in _set_capacities_unsafe refunds capacity.

    Regression: consumed=True was set AFTER _set_capacities_unsafe, so a
    CancelledError during the pipeline read-responses phase left capacity
    consumed server-side with no refund (consumed stayed False).
    """

    async def test_cancellation_during_pipeline_execute_refunds_capacity(
        self,
        redis_client,
    ):
        """CancelledError after _set_capacities_unsafe completes triggers refund."""
        builder = RedisBackendBuilder(redis_client)
        config = _make_config(limit=100)
        backend = builder.build(config)

        # Consume 90, leaving 10
        await backend.await_for_capacity(frozendict({"requests": 90.0}))
        cap_before = await _get_redis_capacity(backend)
        assert cap_before == pytest.approx(10.0, abs=1.0)

        # Monkeypatch _set_capacities_unsafe to call the real impl then raise
        # CancelledError — simulates cancellation arriving during
        # pipeline.execute()'s response-read phase.  Fires only once so the
        # shielded refund can use the real method.
        real_set = backend._set_capacities_unsafe
        fire_once = True

        async def cancelling_set(*args, **kwargs):
            nonlocal fire_once
            await real_set(*args, **kwargs)
            if fire_once:
                fire_once = False
                backend._set_capacities_unsafe = real_set
                raise asyncio.CancelledError

        backend._set_capacities_unsafe = cancelling_set

        with pytest.raises(asyncio.CancelledError):
            await backend.await_for_capacity(frozendict({"requests": 5.0}), timeout=0)

        # Give the shielded refund time to complete
        await asyncio.sleep(0.5)

        cap_after = await _get_redis_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0), (
            f"Capacity leaked! Expected ~{cap_before}, got {cap_after}. "
            f"CancelledError during pipeline.execute() bypassed the refund."
        )


# ---------------------------------------------------------------------------
# Multi-metric CancelledError refund (mirrors memory Group 9)
# ---------------------------------------------------------------------------


def _make_multi_metric_config(
    *,
    requests_limit: float = 100,
    tokens_limit: float = 500,
    per_seconds: int = _SLOW_REFILL_PER_SECONDS,
) -> PerModelConfig:
    return PerModelConfig(
        model_family="test",
        quotas=UsageQuotas(
            [
                Quota(metric="requests", limit=requests_limit, per_seconds=per_seconds),
                Quota(metric="tokens", limit=tokens_limit, per_seconds=per_seconds),
            ]
        ),
    )


async def _get_redis_capacities_by_metric(backend) -> dict[str, float]:
    """Read current capacity from Redis for every bucket, keyed by metric name."""
    pipeline = backend._redis.pipeline()
    current_time = time.time()
    result = await backend._get_capacities_unsafe(
        pipeline=pipeline, current_time=current_time
    )
    return {
        metric: amount for (metric, _per_seconds), amount in result.capacities.items()
    }


@pytest.mark.redis
class TestRedisMultiMetricCancellationRefundsAllMetrics:
    """
    All existing cancellation tests use single-metric configs.  The refund loop
    in _refund_cancelled_consumption iterates all (cap_metric, per_seconds) pairs.
    A bug that only refunds the first metric would be invisible to single-metric tests.
    """

    async def test_cancel_during_callback_refunds_both_metrics(self, redis_client):
        """CancelledError during callback refunds BOTH requests AND tokens."""
        gate = asyncio.Event()
        entered_callback = asyncio.Event()

        async def slow_callback(**kwargs):
            if not gate.is_set():
                return
            entered_callback.set()
            await asyncio.sleep(10)

        callbacks = RateLimiterCallbacks(on_capacity_consumed=slow_callback)
        builder = RedisBackendBuilder(redis_client)
        config = _make_multi_metric_config(requests_limit=100, tokens_limit=500)
        backend = builder.build(config, callbacks=callbacks)

        # Consume most capacity (callback fast — gate closed)
        await backend.await_for_capacity(
            frozendict({"requests": 90.0, "tokens": 450.0})
        )
        caps_before = await _get_redis_capacities_by_metric(backend)
        assert caps_before["requests"] == pytest.approx(10.0, abs=1.0)
        assert caps_before["tokens"] == pytest.approx(50.0, abs=1.0)

        gate.set()

        # timeout=0 goes through _check_and_consume_capacity directly
        task = asyncio.create_task(
            backend.await_for_capacity(
                frozendict({"requests": 5.0, "tokens": 20.0}), timeout=0
            )
        )
        await asyncio.wait_for(entered_callback.wait(), timeout=5.0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Give the shielded refund time to complete
        await asyncio.sleep(0.5)

        # BOTH metrics must be refunded
        caps_after = await _get_redis_capacities_by_metric(backend)
        assert caps_after["requests"] == pytest.approx(
            caps_before["requests"], abs=1.0
        ), (
            f"requests not refunded! Before={caps_before['requests']}, "
            f"after={caps_after['requests']}"
        )
        assert caps_after["tokens"] == pytest.approx(caps_before["tokens"], abs=1.0), (
            f"tokens not refunded! Before={caps_before['tokens']}, "
            f"after={caps_after['tokens']}"
        )


# ---------------------------------------------------------------------------
# CancelledError during local condition wait (mirrors memory Group 2)
# ---------------------------------------------------------------------------


@pytest.mark.redis
class TestRedisCancellationDuringLocalConditionWait:
    """
    Memory backend has Group 2 tests for CancelledError during condition.wait().
    This exercises the equivalent path in the Redis backend: _local_condition.wait()
    at _redis/_backend.py:520-523.  Cancellation here must not leak capacity.
    """

    async def test_cancel_during_local_condition_wait_no_capacity_leak(
        self, redis_client
    ):
        """A task cancelled during _local_condition.wait() must not consume capacity."""
        builder = RedisBackendBuilder(redis_client, sleep_interval=5.0)
        config = _make_config(limit=100)
        backend = builder.build(config)

        # Exhaust capacity so the next caller enters the wait loop
        await backend.await_for_capacity(frozendict({"requests": 100.0}))
        cap_before = await _get_redis_capacity(backend)

        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 10.0}))
        )
        # Let the task enter _local_condition.wait() (sleep_interval=5.0 ensures
        # it stays there rather than looping back to check-and-consume)
        await asyncio.sleep(0.2)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        cap_after = await _get_redis_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0), (
            f"Capacity leaked! Expected ~{cap_before}, got {cap_after}. "
            f"CancelledError during _local_condition.wait() consumed capacity."
        )

    async def test_cancel_multiple_local_condition_waiters(self, redis_client):
        """Cancelling several waiters in _local_condition.wait() doesn't leak capacity."""
        builder = RedisBackendBuilder(redis_client, sleep_interval=5.0)
        config = _make_config(limit=100)
        backend = builder.build(config)

        await backend.await_for_capacity(frozendict({"requests": 100.0}))
        cap_before = await _get_redis_capacity(backend)

        tasks = [
            asyncio.create_task(
                backend.await_for_capacity(frozendict({"requests": 10.0}))
            )
            for _ in range(5)
        ]
        await asyncio.sleep(0.2)

        for t in tasks:
            t.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(r, asyncio.CancelledError) for r in results)

        cap_after = await _get_redis_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0), (
            f"Capacity leaked! Expected ~{cap_before}, got {cap_after}. "
            f"Cancelling multiple _local_condition waiters leaked capacity."
        )

    async def test_cancel_does_not_block_subsequent_acquire(self, redis_client):
        """After cancelling a local-condition waiter, a new caller can still acquire."""
        builder = RedisBackendBuilder(redis_client, sleep_interval=5.0)
        config = _make_config(limit=100)
        backend = builder.build(config)

        # Consume 90, leaving 10
        await backend.await_for_capacity(frozendict({"requests": 90.0}))

        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 50.0}))
        )
        await asyncio.sleep(0.2)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The remaining ~10 tokens should still be available
        await backend.await_for_capacity(frozendict({"requests": 5.0}), timeout=2.0)
