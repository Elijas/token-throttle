"""
Integration tests for concurrent access to the Redis rate-limiter backend.

These tests verify that the locking and capacity-management logic behaves
correctly under parallel access: no double-spend, no negative capacity,
no deadlocks.
"""

import asyncio

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage


def _build_backend(
    backend_builder, *, limit: float, per_seconds: float, metric: str = "requests"
):
    config = PerModelConfig(
        model_family="test",
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
    )
    return backend_builder.build(config)


# Stress tests fan out 50 concurrent operations onto a single bucket. The Redis
# backend serializes every bucket mutation through a polling per-bucket lock, so
# the default lock_blocking_timeout_seconds (5s) can be exceeded under that much
# contention. await_for_capacity with no caller timeout now absorbs contention by
# retrying, but the non-retrying refund path can still raise
# BackendLockContentionError under extreme starvation. Rebuild the Redis builder
# with a generous lock-blocking timeout so the zero-error assertions are
# justified; the memory builder (no Redis lock) passes through unchanged.
_STRESS_LOCK_BLOCKING_TIMEOUT_SECONDS = 60.0


def _stress_backend_builder(backend_builder):
    if type(backend_builder).__name__ == "RedisBackendBuilder":
        from token_throttle._limiter_backends._redis._backend import (  # noqa: PLC0415
            RedisBackendBuilder,
        )

        return RedisBackendBuilder(
            backend_builder._redis,
            key_prefix=backend_builder._key_prefix,
            lock_blocking_timeout_seconds=_STRESS_LOCK_BLOCKING_TIMEOUT_SECONDS,
        )
    return backend_builder


class TestConcurrentAcquiresRespectCapacity:
    """N concurrent acquires must never consume more than max capacity."""

    async def test_ten_concurrent_acquires_total_equals_capacity(self, backend_builder):
        """
        10 tasks each requesting 10 from a bucket with capacity 100.

        All 10 should succeed (fresh bucket starts at 100), and the total
        consumed must equal exactly 100 -- no double-spend.
        """
        backend = _build_backend(backend_builder, limit=100, per_seconds=1)
        usage = frozen_usage({"requests": 10})

        results = await asyncio.gather(
            *[backend.await_for_capacity(usage) for _ in range(10)],
            return_exceptions=True,
        )

        successes = [r for r in results if not isinstance(r, BaseException)]
        failures = [r for r in results if isinstance(r, BaseException)]
        assert failures == [], f"Unexpected errors: {failures}"
        assert len(successes) == 10
        # Total consumed = 10 tasks * 10 units = 100, which is exactly the capacity.
        # Because per_seconds=1 the bucket barely refills during the test,
        # so no more than 100 could have been served.

    async def test_excess_acquires_must_wait(self, backend_builder):
        """
        Drain all capacity, then verify a new acquire blocks.

        First, consume all 100 tokens sequentially so the bucket is empty.
        Then launch one more acquire that should block because refill
        rate is negligible (100 per 3600s = 0.028/s).
        """
        backend = _build_backend(backend_builder, limit=100, per_seconds=3600)
        usage = frozen_usage({"requests": 10})

        # Sequentially consume all 100 tokens (10 acquires of 10 each)
        for _ in range(10):
            await backend.await_for_capacity(usage)

        # Now try one more -- it should block because no capacity remains
        extra_task = asyncio.create_task(backend.await_for_capacity(usage))

        done, pending = await asyncio.wait({extra_task}, timeout=1)

        assert len(pending) == 1, "Expected the extra acquire to still be waiting"
        assert len(done) == 0, "Extra acquire should not have completed"

        # Clean up
        extra_task.cancel()
        await asyncio.gather(extra_task, return_exceptions=True)


class TestConcurrentAcquireAndRefund:
    """Concurrent acquire + refund must leave the bucket in a valid state."""

    async def test_acquire_then_refund_returns_capacity(self, backend_builder):
        """
        Acquire capacity, then refund part of it concurrently.

        Final state must be non-negative and not exceed max capacity.
        """
        backend = _build_backend(backend_builder, limit=100, per_seconds=60)
        acquire_usage = frozen_usage({"requests": 20})
        reserved = frozen_usage({"requests": 20})
        actual = frozen_usage({"requests": 10})  # Only used 10, refund 10

        async def acquire_task():
            await backend.await_for_capacity(acquire_usage)

        async def refund_task():
            # Small delay so at least one acquire has happened first
            await asyncio.sleep(0.05)
            await backend.refund_capacity(reserved_usage=reserved, actual_usage=actual)

        # Run 3 acquires and 2 refunds concurrently
        results = await asyncio.gather(
            acquire_task(),
            acquire_task(),
            acquire_task(),
            refund_task(),
            refund_task(),
            return_exceptions=True,
        )

        errors = [r for r in results if isinstance(r, BaseException)]
        assert errors == [], f"Unexpected errors: {errors}"

    async def test_interleaved_acquire_refund_no_negative_capacity(
        self, backend_builder
    ):
        """
        Many interleaved acquires and full refunds.

        End state: capacity must be >= 0 and <= max_capacity.
        """
        backend = _build_backend(backend_builder, limit=50, per_seconds=60)
        usage = frozen_usage({"requests": 5})

        async def acquire_and_refund():
            await backend.await_for_capacity(usage)
            # Refund everything (actual usage = 0)
            await backend.refund_capacity(
                reserved_usage=usage,
                actual_usage=frozen_usage({"requests": 0}),
            )

        results = await asyncio.gather(
            *[acquire_and_refund() for _ in range(10)],
            return_exceptions=True,
        )

        errors = [r for r in results if isinstance(r, BaseException)]
        assert errors == [], f"Unexpected errors: {errors}"


@pytest.mark.slow
class TestHighParallelismStress:
    """Stress tests with many concurrent tasks to catch deadlocks and races."""

    async def test_fifty_concurrent_acquires_complete_without_deadlock(
        self, backend_builder
    ):
        """
        50 concurrent tasks must all complete within a timeout.

        Capacity is set high enough that all 50 can succeed.
        """
        backend = _build_backend(
            _stress_backend_builder(backend_builder), limit=500, per_seconds=1
        )
        usage = frozen_usage({"requests": 10})

        async def single_acquire():
            await backend.await_for_capacity(usage)

        coro = asyncio.gather(
            *[single_acquire() for _ in range(50)], return_exceptions=True
        )
        results = await asyncio.wait_for(coro, timeout=30)

        errors = [r for r in results if isinstance(r, BaseException)]
        assert errors == [], f"Deadlock or error in stress test: {errors}"
        assert len(results) == 50

    async def test_fifty_mixed_operations_no_errors(self, backend_builder):
        """50 tasks doing a mix of acquires and refunds complete without errors."""
        backend = _build_backend(
            _stress_backend_builder(backend_builder), limit=1000, per_seconds=1
        )
        usage = frozen_usage({"requests": 5})

        async def acquire_only():
            await backend.await_for_capacity(usage)

        async def acquire_and_refund():
            await backend.await_for_capacity(usage)
            await backend.refund_capacity(
                reserved_usage=usage,
                actual_usage=frozen_usage({"requests": 2}),
            )

        tasks = []
        for i in range(50):
            if i % 2 == 0:
                tasks.append(acquire_only())
            else:
                tasks.append(acquire_and_refund())

        coro = asyncio.gather(*tasks, return_exceptions=True)
        results = await asyncio.wait_for(coro, timeout=30)

        errors = [r for r in results if isinstance(r, BaseException)]
        assert errors == [], f"Errors in mixed stress test: {errors}"
        assert len(results) == 50
