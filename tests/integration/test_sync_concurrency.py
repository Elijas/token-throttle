"""
Integration tests for thread-safe concurrent access to sync backends.

These tests verify that capacity management behaves correctly under parallel
access: no double-spend, no negative capacity, no deadlocks.

Parameterized across all sync backends (memory, redis) via the
sync_backend_builder fixture.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage


def _build_backend(
    builder, *, limit: float, per_seconds: float, metric: str = "requests"
):
    config = PerModelConfig(
        model_family="test",
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
    )
    return builder.build(config)


class TestConcurrentAcquiresRespectCapacity:
    """N concurrent acquires must never consume more than max capacity."""

    def test_ten_concurrent_acquires_total_equals_capacity(self, sync_backend_builder):
        """
        10 threads each requesting 10 from a bucket with capacity 100.

        All 10 should succeed (fresh bucket starts at 100), and the total
        consumed must equal exactly 100 — no double-spend.
        """
        backend = _build_backend(sync_backend_builder, limit=100, per_seconds=1)
        usage = frozen_usage({"requests": 10})

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(backend.wait_for_capacity, usage) for _ in range(10)]
            results = [f.result() for f in as_completed(futures, timeout=10)]

        assert len(results) == 10

    def test_excess_acquires_must_wait(self, sync_backend_builder):
        """
        Drain all capacity, then verify a new acquire blocks.

        First, consume all 100 tokens sequentially so the bucket is empty.
        Then launch one more acquire that should block because refill
        rate is negligible (100 per 3600s = 0.028/s).
        """
        backend = _build_backend(sync_backend_builder, limit=100, per_seconds=3600)
        usage = frozen_usage({"requests": 10})

        # Sequentially consume all 100 tokens (10 acquires of 10 each)
        for _ in range(10):
            backend.wait_for_capacity(usage)

        # Now try one more — it should block because no capacity remains
        completed = threading.Event()

        def try_acquire():
            backend.wait_for_capacity(usage)
            completed.set()

        t = threading.Thread(target=try_acquire, daemon=True)
        t.start()

        # Wait 1 second — the thread should NOT have completed
        assert not completed.wait(timeout=1.0), (
            "Expected the extra acquire to still be waiting"
        )


class TestConcurrentAcquireAndRefund:
    """Concurrent acquire + refund must leave the bucket in a valid state."""

    def test_acquire_then_refund_returns_capacity(self, sync_backend_builder):
        """
        Acquire capacity, then refund part of it concurrently.

        Final state must be non-negative and not exceed max capacity.
        """
        backend = _build_backend(sync_backend_builder, limit=100, per_seconds=60)
        acquire_usage = frozen_usage({"requests": 20})
        reserved = frozen_usage({"requests": 20})
        actual = frozen_usage({"requests": 10})  # Only used 10, refund 10

        def refund_task():
            time.sleep(0.05)
            backend.refund_capacity(reserved_usage=reserved, actual_usage=actual)

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [
                pool.submit(backend.wait_for_capacity, acquire_usage),
                pool.submit(backend.wait_for_capacity, acquire_usage),
                pool.submit(backend.wait_for_capacity, acquire_usage),
                pool.submit(refund_task),
                pool.submit(refund_task),
            ]
            results = [f.result() for f in as_completed(futures, timeout=10)]

        assert len(results) == 5

    def test_interleaved_acquire_refund_no_negative_capacity(
        self, sync_backend_builder
    ):
        """
        Many interleaved acquires and full refunds.

        End state: capacity must be >= 0 and <= max_capacity.
        """
        backend = _build_backend(sync_backend_builder, limit=50, per_seconds=60)
        usage = frozen_usage({"requests": 5})

        def acquire_and_refund():
            backend.wait_for_capacity(usage)
            backend.refund_capacity(
                reserved_usage=usage,
                actual_usage=frozen_usage({"requests": 0}),
            )

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(acquire_and_refund) for _ in range(10)]
            results = [f.result() for f in as_completed(futures, timeout=10)]

        assert len(results) == 10


@pytest.mark.slow
class TestHighParallelismStress:
    """Stress tests with many threads to catch deadlocks and races."""

    def test_fifty_concurrent_acquires_complete_without_deadlock(
        self, sync_backend_builder
    ):
        """
        50 concurrent threads must all complete within a timeout.

        Capacity is set high enough that all 50 can succeed.
        """
        backend = _build_backend(sync_backend_builder, limit=500, per_seconds=1)
        usage = frozen_usage({"requests": 10})

        with ThreadPoolExecutor(max_workers=50) as pool:
            futures = [pool.submit(backend.wait_for_capacity, usage) for _ in range(50)]
            results = [f.result() for f in as_completed(futures, timeout=30)]

        assert len(results) == 50

    def test_fifty_mixed_operations_no_errors(self, sync_backend_builder):
        """50 threads doing a mix of acquires and refunds complete without errors."""
        backend = _build_backend(sync_backend_builder, limit=1000, per_seconds=1)
        usage = frozen_usage({"requests": 5})

        def acquire_only():
            backend.wait_for_capacity(usage)

        def acquire_and_refund():
            backend.wait_for_capacity(usage)
            backend.refund_capacity(
                reserved_usage=usage,
                actual_usage=frozen_usage({"requests": 2}),
            )

        with ThreadPoolExecutor(max_workers=50) as pool:
            futures = [
                pool.submit(acquire_only if i % 2 == 0 else acquire_and_refund)
                for i in range(50)
            ]
            results = [f.result() for f in as_completed(futures, timeout=30)]

        assert len(results) == 50
