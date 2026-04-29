"""Integration tests for shielded-write edge cases in the async Redis backend.

F18.05: _wait_for_task_outcome_while_cancelled except-Exception branch.
F18.06: suppress_current_task_cancellation after successful write.

These two findings are complementary — F18.05 covers the case where the
shielded write FAILS (ConnectionError), and F18.06 covers the case where
it SUCCEEDS despite cancellation. Together they verify both branches of
the "consumed" check after _wait_for_task_outcome_while_cancelled returns.
"""

import asyncio
import time

import pytest
import redis.exceptions
from frozendict import frozendict

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder

_SLOW_REFILL_PER_SECONDS = 3600


def _make_config(
    *,
    limit: float = 100,
    per_seconds: int = _SLOW_REFILL_PER_SECONDS,
    metric: str = "requests",
    model_family: str = "test-shielded",
) -> PerModelConfig:
    return PerModelConfig(
        model_family=model_family,
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
    )


async def _get_redis_capacity(backend) -> float:
    """Read current capacity from Redis (authoritative source)."""
    pipeline = backend._redis.pipeline()
    current_time = time.time()
    result = await backend._get_capacities_unsafe(
        pipeline=pipeline, current_time=current_time
    )
    caps = result.capacities
    return next(iter(caps.values()))


@pytest.mark.redis
class TestWaitTaskOutcomeExceptionBranch:
    """F18.05: except-Exception branch in _wait_for_task_outcome_while_cancelled.

    When the shielded Redis write itself raises (e.g. ConnectionError),
    the method must return consumed=False so the caller propagates
    CancelledError without attempting a refund.
    """

    async def test_connection_error_in_write_reports_not_consumed(self, redis_client):
        """ConnectionError during shielded write → consumed=False → CancelledError propagates."""
        config = _make_config(limit=1000, model_family="test-conn-err")
        backend = RedisBackendBuilder(redis_client).build(config)

        cap_before = await _get_redis_capacity(backend)

        write_entered = asyncio.Event()

        async def failing_set(*args, **kwargs):
            write_entered.set()
            # Yield so the cancel can arrive before we raise
            await asyncio.sleep(0.05)
            raise redis.exceptions.ConnectionError("Simulated write failure")

        backend._set_capacities_unsafe = failing_set

        task = asyncio.create_task(
            backend.consume_capacity(frozendict({"requests": 10.0}))
        )
        await write_entered.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Capacity unchanged: the write never succeeded
        cap_after = await _get_redis_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0), (
            f"Expected capacity unchanged after failed write, "
            f"got {cap_after} (was {cap_before})"
        )

    async def test_connection_error_in_check_and_consume_write(self, redis_client):
        """Same scenario via _check_and_consume_capacity (await_for_capacity path)."""
        config = _make_config(limit=1000, model_family="test-conn-err-check")
        backend = RedisBackendBuilder(redis_client).build(config)

        cap_before = await _get_redis_capacity(backend)

        write_entered = asyncio.Event()

        async def failing_set(*args, **kwargs):
            write_entered.set()
            await asyncio.sleep(0.05)
            raise redis.exceptions.ConnectionError("Simulated write failure")

        backend._set_capacities_unsafe = failing_set

        task = asyncio.create_task(
            backend.await_for_capacity(frozendict({"requests": 10.0}))
        )
        await write_entered.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        cap_after = await _get_redis_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=1.0)


@pytest.mark.redis
class TestSuppressTaskCancellationOnSuccessfulWrite:
    """F18.06: suppress_current_task_cancellation after durable write.

    When consume_capacity's shielded write succeeds despite outer-task
    cancellation, the task must absorb the CancelledError (speedometer
    semantics: the recorded consumption is the correct reading).
    """

    async def test_cancel_after_consume_write_lands_is_absorbed(self, redis_client):
        """Task cancellation absorbed if consume_capacity's write already landed."""
        config = _make_config(limit=1000, model_family="test-suppress-consume")
        backend = RedisBackendBuilder(redis_client).build(config)

        cap_before = await _get_redis_capacity(backend)

        write_entered = asyncio.Event()
        original_set = backend._set_capacities_unsafe

        async def delayed_set(*args, **kwargs):
            write_entered.set()
            await asyncio.sleep(0.05)
            return await original_set(*args, **kwargs)

        backend._set_capacities_unsafe = delayed_set

        task = asyncio.create_task(
            backend.consume_capacity(frozendict({"requests": 10.0}))
        )
        await write_entered.wait()
        task.cancel()

        # The task should NOT raise CancelledError — suppression absorbs it
        results = await asyncio.gather(task, return_exceptions=True)
        assert results[0] is None, (
            f"Expected cancellation absorbed after successful write, got {results[0]!r}"
        )

        # Capacity SHOULD have decreased (write landed)
        cap_after = await _get_redis_capacity(backend)
        assert cap_after < cap_before - 5.0, (
            f"Expected capacity reduced after successful write, "
            f"got {cap_after} (was {cap_before})"
        )

    async def test_cancel_after_refund_write_lands_is_absorbed(self, redis_client):
        """Task cancellation absorbed if refund_capacity's write already landed."""
        config = _make_config(limit=1000, model_family="test-suppress-refund")
        backend = RedisBackendBuilder(redis_client).build(config)

        # Consume capacity first so we can refund
        await backend.consume_capacity(frozendict({"requests": 100.0}))
        cap_before = await _get_redis_capacity(backend)

        write_entered = asyncio.Event()
        original_set = backend._set_capacities_unsafe

        async def delayed_set(*args, **kwargs):
            write_entered.set()
            await asyncio.sleep(0.05)
            return await original_set(*args, **kwargs)

        backend._set_capacities_unsafe = delayed_set

        task = asyncio.create_task(
            backend.refund_capacity(
                reserved_usage=frozendict({"requests": 100.0}),
                actual_usage=frozendict({"requests": 50.0}),
            )
        )
        await write_entered.wait()
        task.cancel()

        results = await asyncio.gather(task, return_exceptions=True)
        assert results[0] is None, (
            f"Expected cancellation absorbed after successful refund write, "
            f"got {results[0]!r}"
        )

        # Capacity should have increased (refund of 50 tokens landed)
        cap_after = await _get_redis_capacity(backend)
        assert cap_after > cap_before + 40.0, (
            f"Expected capacity increased after refund write, "
            f"got {cap_after} (was {cap_before})"
        )
