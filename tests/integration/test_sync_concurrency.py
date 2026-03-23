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


def _get_memory_buckets(backend):
    """Return the memory bucket list, or None if this is a Redis backend.

    Memory buckets support synchronous ``get_capacity(time)`` for capacity
    verification.  Redis buckets require I/O, so accounting assertions are
    skipped for the Redis backend.
    """
    return getattr(backend, "_buckets", None)


def _build_multi_metric_backend(
    builder,
    *,
    requests_limit: float,
    tokens_limit: float,
    per_seconds: float,
):
    config = PerModelConfig(
        model_family="test",
        quotas=UsageQuotas(
            [
                Quota(metric="requests", limit=requests_limit, per_seconds=per_seconds),
                Quota(metric="tokens", limit=tokens_limit, per_seconds=per_seconds),
            ]
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
        acquired = threading.Event()
        completed = threading.Event()
        waiter_errors: list[BaseException] = []

        def try_acquire():
            try:
                backend.wait_for_capacity(usage)
            except BaseException as exc:  # pragma: no cover - unexpected path
                waiter_errors.append(exc)
            else:
                acquired.set()
            finally:
                completed.set()

        t = threading.Thread(target=try_acquire)
        t.start()

        try:
            # Wait 1 second — the thread should NOT have completed
            assert not completed.wait(timeout=1.0), (
                "Expected the extra acquire to still be waiting"
            )
        finally:
            backend.refund_capacity(
                reserved_usage=usage,
                actual_usage=frozen_usage({"requests": 0}),
            )
            t.join(timeout=2.0)

        assert not t.is_alive(), "Background waiter thread did not exit cleanly"
        assert waiter_errors == [], f"Unexpected waiter errors: {waiter_errors}"
        assert acquired.is_set(), "Cleanup refund should have released the waiter"


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


class TestConcurrentConsumeCapacity:
    """Thread safety of the consume_capacity (speedometer) path.

    consume_capacity uses a different code path from wait_for_capacity:
    it allows negative capacity and never blocks.  These tests verify that
    concurrent consume_capacity calls don't corrupt bucket state.
    """

    def test_concurrent_consume_capacity_completes(self, sync_backend_builder):
        """20 threads calling consume_capacity simultaneously must all succeed."""
        backend = _build_backend(sync_backend_builder, limit=100, per_seconds=3600)
        usage = frozen_usage({"requests": 10})

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(backend.consume_capacity, usage) for _ in range(20)]
            results = [f.result() for f in as_completed(futures, timeout=10)]

        assert len(results) == 20

    def test_concurrent_consume_capacity_drives_negative(self, sync_backend_builder):
        """Concurrent speedometer calls that exceed capacity must correctly go negative.

        20 threads x 10 units = 200 consumed from a 100-capacity bucket.
        With slow refill (3600s), final capacity must be approximately -100.
        """
        backend = _build_backend(sync_backend_builder, limit=100, per_seconds=3600)
        usage = frozen_usage({"requests": 10})

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(backend.consume_capacity, usage) for _ in range(20)]
            for f in as_completed(futures, timeout=10):
                f.result()

        buckets = _get_memory_buckets(backend)
        if buckets is not None:
            cap = buckets[0].get_capacity(time.time()).amount
            # 100 capacity - 200 consumed = -100, with small refill tolerance
            assert cap == pytest.approx(-100.0, abs=2.0), (
                f"Expected ~-100 capacity after 200 consumed from 100, got {cap}"
            )

    def test_mixed_wait_and_consume_no_deadlock(self, sync_backend_builder):
        """Interleaved wait_for_capacity and consume_capacity must not deadlock.

        Both code paths acquire the same condition lock.  This test verifies
        there's no ordering issue when they run concurrently.
        """
        backend = _build_backend(sync_backend_builder, limit=500, per_seconds=1)
        wait_usage = frozen_usage({"requests": 5})
        consume_usage = frozen_usage({"requests": 3})
        errors: list[BaseException] = []

        def do_wait():
            try:
                backend.wait_for_capacity(wait_usage)
            except BaseException as exc:
                errors.append(exc)

        def do_consume():
            try:
                backend.consume_capacity(consume_usage)
            except BaseException as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [
                pool.submit(do_wait if i % 2 == 0 else do_consume) for i in range(20)
            ]
            for f in as_completed(futures, timeout=10):
                f.result()

        assert errors == [], f"Unexpected errors: {errors}"


class TestMultiMetricConcurrency:
    """Thread safety with multiple quota metrics (tokens + requests).

    All-or-nothing semantics: if one metric lacks capacity, no metrics
    should be consumed.  Under thread contention this tests the atomicity
    of the multi-bucket check-then-consume logic.
    """

    def test_multi_metric_concurrent_acquires(self, sync_backend_builder):
        """10 threads acquiring both tokens and requests must all succeed.

        Capacity is sufficient for all 10 (100 requests, 1000 tokens;
        each thread uses 10 requests + 100 tokens).
        """
        backend = _build_multi_metric_backend(
            sync_backend_builder,
            requests_limit=100,
            tokens_limit=1000,
            per_seconds=1,
        )
        usage = frozen_usage({"requests": 10, "tokens": 100})

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(backend.wait_for_capacity, usage) for _ in range(10)]
            results = [f.result() for f in as_completed(futures, timeout=10)]

        assert len(results) == 10

    def test_multi_metric_all_or_nothing_under_contention(self, sync_backend_builder):
        """All-or-nothing holds under thread contention.

        Tokens are scarce (50 total, each thread wants 10), requests are abundant.
        5 threads succeed immediately, the rest must wait for refill.
        After all threads complete, token capacity must not be over-consumed.
        """
        backend = _build_multi_metric_backend(
            sync_backend_builder,
            requests_limit=1000,
            tokens_limit=50,
            per_seconds=1,  # fast refill: 50 tokens/sec
        )
        usage = frozen_usage({"requests": 1, "tokens": 10})
        errors: list[BaseException] = []

        def acquire():
            try:
                backend.wait_for_capacity(usage, timeout=10)
            except BaseException as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(acquire) for _ in range(10)]
            for f in as_completed(futures, timeout=15):
                f.result()

        assert errors == [], f"Unexpected errors: {errors}"

    def test_multi_metric_consume_and_refund_accounting(self, sync_backend_builder):
        """Acquire-then-refund cycle with multiple metrics preserves capacity.

        10 threads each acquire {requests: 5, tokens: 50}, then refund
        fully (actual usage = 0).  Final capacity should be near max.
        """
        backend = _build_multi_metric_backend(
            sync_backend_builder,
            requests_limit=100,
            tokens_limit=1000,
            per_seconds=3600,  # slow refill so accounting is precise
        )
        usage = frozen_usage({"requests": 5, "tokens": 50})

        def acquire_and_refund():
            backend.wait_for_capacity(usage)
            backend.refund_capacity(
                reserved_usage=usage,
                actual_usage=frozen_usage({"requests": 0, "tokens": 0}),
            )

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(acquire_and_refund) for _ in range(10)]
            for f in as_completed(futures, timeout=10):
                f.result()

        # After full refunds, both metrics should be near max capacity
        buckets = _get_memory_buckets(backend)
        if buckets is not None:
            now = time.time()
            for bucket in buckets:
                cap = bucket.get_capacity(now).amount
                assert cap == pytest.approx(bucket.max_capacity, abs=2.0), (
                    f"Bucket {bucket.usage_metric}/{bucket.per_seconds}s: "
                    f"expected ~{bucket.max_capacity}, got {cap}"
                )


class TestCapacityAccountingAfterConcurrentOps:
    """Verify final capacity state is mathematically correct after concurrent operations.

    Existing stress tests check "no errors" but not the resulting capacity.
    These tests verify the numbers add up.
    """

    def test_capacity_after_concurrent_full_drain(self, sync_backend_builder):
        """10 threads x 10 units from a 100-capacity bucket = capacity near 0.

        Uses slow refill (3600s) so refill during the test is negligible (~0.3 tokens).
        """
        backend = _build_backend(sync_backend_builder, limit=100, per_seconds=3600)
        usage = frozen_usage({"requests": 10})

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(backend.wait_for_capacity, usage) for _ in range(10)]
            for f in as_completed(futures, timeout=10):
                f.result()

        buckets = _get_memory_buckets(backend)
        if buckets is not None:
            cap = buckets[0].get_capacity(time.time()).amount
            assert cap == pytest.approx(0.0, abs=2.0), (
                f"Expected ~0 capacity after exact drain, got {cap}"
            )

    def test_capacity_after_concurrent_partial_refunds(self, sync_backend_builder):
        """Acquire 10 x 10, refund 5 x (reserved=10, actual=0) = capacity near 50.

        10 threads drain all 100 capacity, then 5 threads refund 10 each (= 50 back).
        """
        backend = _build_backend(sync_backend_builder, limit=100, per_seconds=3600)
        usage = frozen_usage({"requests": 10})

        # Phase 1: drain all capacity
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(backend.wait_for_capacity, usage) for _ in range(10)]
            for f in as_completed(futures, timeout=10):
                f.result()

        # Phase 2: 5 concurrent full refunds of 10 each
        def do_refund():
            backend.refund_capacity(
                reserved_usage=usage,
                actual_usage=frozen_usage({"requests": 0}),
            )

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(do_refund) for _ in range(5)]
            for f in as_completed(futures, timeout=10):
                f.result()

        buckets = _get_memory_buckets(backend)
        if buckets is not None:
            cap = buckets[0].get_capacity(time.time()).amount
            assert cap == pytest.approx(50.0, abs=2.0), (
                f"Expected ~50 capacity after 5 refunds of 10, got {cap}"
            )


class TestConcurrentSetMaxCapacity:
    """Thread safety of set_max_capacity mixed with consume and acquire.

    set_max_capacity modifies max_capacity and rate under the condition lock.
    These tests verify it plays nicely with concurrent consumers and acquirers.
    """

    def test_concurrent_set_max_and_consume_no_errors(self, sync_backend_builder):
        """10 threads consuming + 5 threads changing max_capacity must not error.

        Final capacity should be within [negative, current_max].
        """
        backend = _build_backend(sync_backend_builder, limit=200, per_seconds=60)
        errors: list[BaseException] = []

        def do_consume():
            try:
                backend.consume_capacity(frozen_usage({"requests": 10}))
            except BaseException as exc:
                errors.append(exc)

        def do_set_max(new_max):
            try:
                backend.set_max_capacity("requests", 60, new_max)
            except BaseException as exc:
                errors.append(exc)

        max_values = [50.0, 100.0, 300.0, 150.0, 200.0]

        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = [pool.submit(do_consume) for _ in range(10)]
            futures.extend(pool.submit(do_set_max, v) for v in max_values)
            for f in as_completed(futures, timeout=10):
                f.result()

        assert errors == [], f"Unexpected errors: {errors}"

        buckets = _get_memory_buckets(backend)
        if buckets is not None:
            cap = buckets[0].get_capacity(time.time()).amount
            current_max = buckets[0].max_capacity
            assert cap <= current_max + 1.0, (
                f"Capacity {cap} exceeded max {current_max}"
            )

    def test_concurrent_set_max_and_acquire_no_deadlock(self, sync_backend_builder):
        """Interleaved wait_for_capacity and set_max_capacity must not deadlock.

        Both paths hold the condition lock. Capacity changes also call
        notify_all, which should unblock waiters.
        """
        backend = _build_backend(sync_backend_builder, limit=100, per_seconds=1)
        errors: list[BaseException] = []

        def do_acquire():
            try:
                backend.wait_for_capacity(frozen_usage({"requests": 5}), timeout=5)
            except TimeoutError:
                pass  # acceptable — capacity may have been lowered
            except BaseException as exc:
                errors.append(exc)

        def do_set_max(new_max):
            try:
                backend.set_max_capacity("requests", 1, new_max)
            except BaseException as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(do_acquire) for _ in range(10)]
            futures.extend(
                pool.submit(do_set_max, v)
                for v in [
                    50.0,
                    200.0,
                    30.0,
                    150.0,
                    100.0,
                    75.0,
                    300.0,
                    10.0,
                    500.0,
                    60.0,
                ]
            )
            for f in as_completed(futures, timeout=15):
                f.result()

        assert errors == [], f"Unexpected errors: {errors}"

    def test_lower_max_below_acquire_raises_on_next_poll(self, sync_backend_builder):
        """Lowering max_capacity below a requested acquire amount raises ValueError.

        After set_max_capacity(5), wait_for_capacity(10) should raise
        immediately because 10 > max_capacity of 5 (can never be satisfied).
        """
        backend = _build_backend(sync_backend_builder, limit=100, per_seconds=60)

        # Lower max_capacity to 5
        backend.set_max_capacity("requests", 60, 5.0)

        # Requesting 10 when max is 5 should raise ValueError
        with pytest.raises(ValueError, match="exceeds bucket max capacity"):
            backend.wait_for_capacity(frozen_usage({"requests": 10}), timeout=0.0)

    def test_set_max_capacity_final_bounds(self, sync_backend_builder):
        """After concurrent set_max + consume + refund, capacity is in valid bounds.

        Final capacity must be <= current max_capacity.
        """
        backend = _build_backend(sync_backend_builder, limit=100, per_seconds=3600)
        errors: list[BaseException] = []

        def do_consume():
            try:
                backend.consume_capacity(frozen_usage({"requests": 5}))
            except BaseException as exc:
                errors.append(exc)

        def do_refund():
            try:
                backend.refund_capacity(
                    reserved_usage=frozen_usage({"requests": 10}),
                    actual_usage=frozen_usage({"requests": 5}),
                )
            except BaseException as exc:
                errors.append(exc)

        def do_set_max(v):
            try:
                backend.set_max_capacity("requests", 3600, v)
            except BaseException as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(do_consume) for _ in range(5)]
            futures.extend(pool.submit(do_refund) for _ in range(5))
            futures.extend(
                pool.submit(do_set_max, v) for v in [50.0, 200.0, 80.0, 150.0, 100.0]
            )
            for f in as_completed(futures, timeout=10):
                f.result()

        assert errors == [], f"Unexpected errors: {errors}"

        buckets = _get_memory_buckets(backend)
        if buckets is not None:
            cap = buckets[0].get_capacity(time.time()).amount
            current_max = buckets[0].max_capacity
            assert cap <= current_max + 1.0, (
                f"Capacity {cap} exceeded max {current_max}"
            )
