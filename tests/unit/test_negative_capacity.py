"""
Tests for negative capacity behavior in in-memory backends.

Verifies that the allow_negative=True change in refund_capacity correctly
preserves negative debt and doesn't introduce regressions.
"""

import asyncio
import time
from unittest.mock import patch

import pytest
from frozendict import frozendict

from token_throttle._capacity import calculate_capacity
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._limiter_backends._memory._backend import MemoryBackend
from token_throttle._limiter_backends._memory._bucket import MemoryBucket
from token_throttle._limiter_backends._memory._sync_backend import SyncMemoryBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_bucket(
    limit: float = 100,
    per_seconds: int = 60,
    metric: str = "tokens",
    model_family: str = "test-model",
) -> MemoryBucket:
    return MemoryBucket(
        metric=metric,
        per_seconds=per_seconds,
        limit=limit,
        model_family=model_family,
    )


def _make_config(
    metric: str = "tokens",
    limit: float = 100,
    per_seconds: int = 60,
) -> PerModelConfig:
    quota = Quota(metric=metric, limit=limit, per_seconds=per_seconds)
    return PerModelConfig(model_family="test", quotas=UsageQuotas([quota]))


def _make_async_backend(
    metric: str = "tokens",
    limit: float = 100,
    per_seconds: int = 60,
    sleep_interval: float = 0.01,
) -> MemoryBackend:
    config = _make_config(metric=metric, limit=limit, per_seconds=per_seconds)
    bucket = make_bucket(metric=metric, limit=limit, per_seconds=per_seconds)
    return MemoryBackend(
        buckets=[bucket], limit_config=config, sleep_interval=sleep_interval
    )


def _make_sync_backend(
    metric: str = "tokens",
    limit: float = 100,
    per_seconds: int = 60,
    sleep_interval: float = 0.01,
) -> SyncMemoryBackend:
    config = _make_config(metric=metric, limit=limit, per_seconds=per_seconds)
    bucket = make_bucket(metric=metric, limit=limit, per_seconds=per_seconds)
    return SyncMemoryBackend(
        buckets=[bucket], limit_config=config, sleep_interval=sleep_interval
    )


# ---------------------------------------------------------------------------
# 1. Refill from negative capacity
# ---------------------------------------------------------------------------


class TestRefillFromNegativeCapacity:
    """When outdated_capacity is negative, refill math should recover naturally."""

    def test_negative_capacity_partial_refill_stays_negative(self):
        """capacity=-100, rate=10/sec, 5s → -100 + 50 = -50"""
        result = calculate_capacity(
            last_checked=1000.0,
            outdated_capacity=-100.0,
            current_time=1005.0,
            max_capacity=600.0,  # rate = 600/60 = 10/sec
            rate_per_sec=10.0,
            bucket_id="test",
        )
        assert result.amount == pytest.approx(-50.0)
        assert result.is_fresh_start is False

    def test_negative_capacity_full_refill_to_zero(self):
        """capacity=-100, rate=10/sec, 10s → -100 + 100 = 0"""
        result = calculate_capacity(
            last_checked=1000.0,
            outdated_capacity=-100.0,
            current_time=1010.0,
            max_capacity=600.0,
            rate_per_sec=10.0,
            bucket_id="test",
        )
        assert result.amount == pytest.approx(0.0)

    def test_negative_capacity_refill_past_zero(self):
        """capacity=-100, rate=10/sec, 15s → -100 + 150 = 50"""
        result = calculate_capacity(
            last_checked=1000.0,
            outdated_capacity=-100.0,
            current_time=1015.0,
            max_capacity=600.0,
            rate_per_sec=10.0,
            bucket_id="test",
        )
        assert result.amount == pytest.approx(50.0)

    def test_negative_capacity_refill_capped_at_max(self):
        """Even from deep negative, capacity can't exceed max_capacity."""
        result = calculate_capacity(
            last_checked=1000.0,
            outdated_capacity=-100.0,
            current_time=2000.0,  # 1000s at 10/sec = 10000 refill
            max_capacity=600.0,
            rate_per_sec=10.0,
            bucket_id="test",
        )
        assert result.amount == pytest.approx(600.0)

    def test_negative_capacity_no_time_stays_negative(self):
        """No time passes → capacity unchanged at -100."""
        result = calculate_capacity(
            last_checked=1000.0,
            outdated_capacity=-100.0,
            current_time=1000.0,
            max_capacity=600.0,
            rate_per_sec=10.0,
            bucket_id="test",
        )
        assert result.amount == pytest.approx(-100.0)

    def test_bucket_get_capacity_with_negative_stored(self):
        """MemoryBucket.get_capacity delegates to calculate_capacity correctly for negatives."""
        bucket = make_bucket(limit=600, per_seconds=60, metric="tokens")
        # rate = 600/60 = 10/sec
        bucket.set_capacity(-100.0, current_time=1000.0, allow_negative=True)
        assert bucket.capacity == -100.0

        result = bucket.get_capacity(current_time=1005.0)
        assert result.amount == pytest.approx(-50.0)
        assert result.is_fresh_start is False


# ---------------------------------------------------------------------------
# 2. acquire_capacity (blocking wait) with negative capacity
# ---------------------------------------------------------------------------


class TestAwaitForCapacityWithNegative:
    """When capacity is negative, await_for_capacity should wait until refilled."""

    async def test_async_waits_until_capacity_positive(self):
        """Simulate negative capacity that refills over time."""
        backend = _make_async_backend(limit=100, per_seconds=1, sleep_interval=0.01)
        # rate = 100/sec; after consume_capacity pushes negative, it should recover fast

        # Force capacity to -20 via consume_capacity
        # First, initialize the bucket by calling consume (starts at max=100, consume 120 → -20)
        await backend.consume_capacity(frozen_usage({"tokens": 120}))

        # Now capacity is -20. Request 5 tokens — should wait until capacity >= 5
        # At 100/sec, need 25 units of refill from -20 to get to 5, so ~0.25 seconds
        start = time.monotonic()
        await backend.await_for_capacity(frozen_usage({"tokens": 5}))
        elapsed = time.monotonic() - start

        # Should have waited some time (capacity was negative)
        assert elapsed > 0.01, f"Expected some wait, but only waited {elapsed:.4f}s"
        # But not too long (100/sec rate should recover quickly)
        assert elapsed < 2.0, f"Waited too long: {elapsed:.4f}s"

    def test_sync_waits_until_capacity_positive(self):
        """Sync mirror of the async test."""
        backend = _make_sync_backend(limit=100, per_seconds=1, sleep_interval=0.01)

        backend.consume_capacity(frozen_usage({"tokens": 120}))

        start = time.monotonic()
        backend.wait_for_capacity(frozen_usage({"tokens": 5}))
        elapsed = time.monotonic() - start

        assert elapsed > 0.01, f"Expected some wait, but only waited {elapsed:.4f}s"
        assert elapsed < 2.0, f"Waited too long: {elapsed:.4f}s"


# ---------------------------------------------------------------------------
# 3. Concurrent record_usage + refund
# ---------------------------------------------------------------------------


class TestConcurrentConsumeAndRefund:
    """Overlapping consume_capacity + refund_capacity must produce accurate accounting."""

    async def test_two_consumes_then_partial_refunds(self):
        """Two consume_capacity calls deplete, partial refunds recover accurately."""
        backend = _make_async_backend(limit=1000, per_seconds=60, sleep_interval=0.01)

        # Initial capacity = 1000 (fresh start)
        # Consume 400, then 300 → should be 1000 - 400 - 300 = 300
        await backend.consume_capacity(frozen_usage({"tokens": 400}))
        await backend.consume_capacity(frozen_usage({"tokens": 300}))

        # Check capacity (roughly 300, ignoring tiny refill)
        bucket = backend._buckets[0]
        cap = bucket.get_capacity(time.time())
        assert cap.amount == pytest.approx(300.0, abs=5.0)

        # Refund: reserved 400, actually used 200 → refund 200
        await backend.refund_capacity(
            frozen_usage({"tokens": 400}),
            frozen_usage({"tokens": 200}),
        )
        cap_after = bucket.get_capacity(time.time())
        # 300 + 200 = 500 (approximately, small refill drift)
        assert cap_after.amount == pytest.approx(500.0, abs=10.0)

        # Refund: reserved 300, actually used 350 → negative refund of -50
        with pytest.warns(RuntimeWarning, match="exceeds reserved usage"):
            await backend.refund_capacity(
                frozen_usage({"tokens": 300}),
                frozen_usage({"tokens": 350}),
            )
        cap_final = bucket.get_capacity(time.time())
        # ~500 - 50 = ~450
        assert cap_final.amount == pytest.approx(450.0, abs=10.0)

    async def test_concurrent_consume_tasks(self):
        """Multiple async consume_capacity tasks don't lose capacity."""
        backend = _make_async_backend(limit=10000, per_seconds=3600, sleep_interval=0.01)
        # rate = 10000/3600 ≈ 2.78/sec, very slow refill → negligible in test

        # Run 10 concurrent consume_capacity(100) tasks
        tasks = [
            asyncio.create_task(
                backend.consume_capacity(frozen_usage({"tokens": 100}))
            )
            for _ in range(10)
        ]
        await asyncio.gather(*tasks)

        bucket = backend._buckets[0]
        cap = bucket.get_capacity(time.time())
        # 10000 - 10*100 = 9000 (plus negligible refill)
        assert cap.amount == pytest.approx(9000.0, abs=50.0)


# ---------------------------------------------------------------------------
# 4. Refund when capacity is already positive (no regression)
# ---------------------------------------------------------------------------


class TestRefundPositiveCapacityNoRegression:
    """allow_negative=True must not break the normal acquire→refund path."""

    async def test_async_positive_refund_stays_positive(self):
        """Normal path: capacity is positive, refund is positive → result stays positive."""
        backend = _make_async_backend(limit=100, per_seconds=60, sleep_interval=0.01)

        # Consume 60 → capacity ≈ 40
        await backend.consume_capacity(frozen_usage({"tokens": 60}))

        # Refund: reserved=60, actual=30 → refund 30
        await backend.refund_capacity(
            frozen_usage({"tokens": 60}),
            frozen_usage({"tokens": 30}),
        )

        bucket = backend._buckets[0]
        cap = bucket.get_capacity(time.time())
        # ~40 + 30 = ~70
        assert cap.amount == pytest.approx(70.0, abs=5.0)
        assert cap.amount > 0, "Capacity should remain positive"

    def test_sync_positive_refund_stays_positive(self):
        """Sync mirror."""
        backend = _make_sync_backend(limit=100, per_seconds=60, sleep_interval=0.01)

        backend.consume_capacity(frozen_usage({"tokens": 60}))

        backend.refund_capacity(
            frozen_usage({"tokens": 60}),
            frozen_usage({"tokens": 30}),
        )

        bucket = backend._buckets[0]
        cap = bucket.get_capacity(time.time())
        assert cap.amount == pytest.approx(70.0, abs=5.0)
        assert cap.amount > 0

    async def test_async_refund_does_not_make_positive_capacity_negative(self):
        """Positive refund on positive capacity can never go negative."""
        backend = _make_async_backend(limit=100, per_seconds=60, sleep_interval=0.01)

        # Consume 50 → capacity ≈ 50
        await backend.consume_capacity(frozen_usage({"tokens": 50}))

        # Refund: reserved=50, actual=10 → refund 40
        await backend.refund_capacity(
            frozen_usage({"tokens": 50}),
            frozen_usage({"tokens": 10}),
        )

        bucket = backend._buckets[0]
        cap = bucket.get_capacity(time.time())
        # 50 + 40 = 90
        assert cap.amount == pytest.approx(90.0, abs=5.0)
        assert cap.amount >= 0


# ---------------------------------------------------------------------------
# 5. Edge: refund_amount exactly equals negative debt
# ---------------------------------------------------------------------------


class TestRefundExactlyEqualsDebt:
    """capacity=-50, refund=+50 → should be 0."""

    async def test_async_exact_debt_cancellation(self):
        backend = _make_async_backend(limit=1000, per_seconds=3600, sleep_interval=0.01)
        # rate ≈ 0.28/sec, negligible refill

        # Consume 1050 → capacity = 1000 - 1050 = -50
        await backend.consume_capacity(frozen_usage({"tokens": 1050}))
        bucket = backend._buckets[0]
        cap_before = bucket.get_capacity(time.time())
        assert cap_before.amount == pytest.approx(-50.0, abs=2.0)

        # Refund exactly 50: reserved=1050, actual=1000 → refund=50
        await backend.refund_capacity(
            frozen_usage({"tokens": 1050}),
            frozen_usage({"tokens": 1000}),
        )
        cap_after = bucket.get_capacity(time.time())
        assert cap_after.amount == pytest.approx(0.0, abs=2.0)

    def test_sync_exact_debt_cancellation(self):
        backend = _make_sync_backend(limit=1000, per_seconds=3600, sleep_interval=0.01)

        backend.consume_capacity(frozen_usage({"tokens": 1050}))
        bucket = backend._buckets[0]
        cap_before = bucket.get_capacity(time.time())
        assert cap_before.amount == pytest.approx(-50.0, abs=2.0)

        backend.refund_capacity(
            frozen_usage({"tokens": 1050}),
            frozen_usage({"tokens": 1000}),
        )
        cap_after = bucket.get_capacity(time.time())
        assert cap_after.amount == pytest.approx(0.0, abs=2.0)


# ---------------------------------------------------------------------------
# 6. Edge: max_capacity cap still works on refund
# ---------------------------------------------------------------------------


class TestMaxCapacityCapOnRefund:
    """Refund must not push capacity above max_capacity."""

    async def test_async_refund_capped_at_max(self):
        backend = _make_async_backend(limit=100, per_seconds=3600, sleep_interval=0.01)
        # rate ≈ 0.028/sec, negligible refill

        # Consume 10 → capacity ≈ 90
        await backend.consume_capacity(frozen_usage({"tokens": 10}))

        # Refund 20: reserved=30, actual=10 → refund=20
        # 90 + 20 = 110, but should be capped at 100
        await backend.refund_capacity(
            frozen_usage({"tokens": 30}),
            frozen_usage({"tokens": 10}),
        )

        bucket = backend._buckets[0]
        cap = bucket.get_capacity(time.time())
        assert cap.amount == pytest.approx(100.0, abs=2.0)
        assert cap.amount <= 100.0 + 1.0  # Tight bound (small refill tolerance)

    def test_sync_refund_capped_at_max(self):
        backend = _make_sync_backend(limit=100, per_seconds=3600, sleep_interval=0.01)

        backend.consume_capacity(frozen_usage({"tokens": 10}))

        backend.refund_capacity(
            frozen_usage({"tokens": 30}),
            frozen_usage({"tokens": 10}),
        )

        bucket = backend._buckets[0]
        cap = bucket.get_capacity(time.time())
        assert cap.amount == pytest.approx(100.0, abs=2.0)
        assert cap.amount <= 100.0 + 1.0

    async def test_async_refund_at_max_stays_at_max(self):
        """Capacity already at max + refund → still max."""
        backend = _make_async_backend(limit=100, per_seconds=3600, sleep_interval=0.01)
        # Don't consume anything, so capacity starts at 100

        # Refund: reserved=50, actual=0 → refund=50
        # 100 + 50 = 150, capped at 100
        await backend.refund_capacity(
            frozen_usage({"tokens": 50}),
            frozen_usage({"tokens": 0}),
        )

        bucket = backend._buckets[0]
        cap = bucket.get_capacity(time.time())
        assert cap.amount == pytest.approx(100.0, abs=2.0)


# ---------------------------------------------------------------------------
# Additional: negative refund (overuse) pushes capacity below zero
# ---------------------------------------------------------------------------


class TestNegativeRefundPushesCapacityNegative:
    """When actual_usage > reserved_usage, refund is negative → capacity can go below 0."""

    async def test_async_negative_refund_creates_debt(self):
        backend = _make_async_backend(limit=100, per_seconds=3600, sleep_interval=0.01)
        # Start at 100, consume 60 → 40

        await backend.consume_capacity(frozen_usage({"tokens": 60}))

        # Negative refund: reserved=60, actual=100 → refund=-40
        # 40 + (-40) = 0
        with pytest.warns(RuntimeWarning, match="exceeds reserved usage"):
            await backend.refund_capacity(
                frozen_usage({"tokens": 60}),
                frozen_usage({"tokens": 100}),
            )

        bucket = backend._buckets[0]
        cap = bucket.get_capacity(time.time())
        assert cap.amount == pytest.approx(0.0, abs=2.0)

    async def test_async_large_negative_refund_goes_deeply_negative(self):
        backend = _make_async_backend(limit=100, per_seconds=3600, sleep_interval=0.01)
        # Start at 100, consume 80 → 20

        await backend.consume_capacity(frozen_usage({"tokens": 80}))

        # Negative refund: reserved=80, actual=200 → refund=-120
        # 20 + (-120) = -100
        with pytest.warns(RuntimeWarning, match="exceeds reserved usage"):
            await backend.refund_capacity(
                frozen_usage({"tokens": 80}),
                frozen_usage({"tokens": 200}),
            )

        bucket = backend._buckets[0]
        cap = bucket.get_capacity(time.time())
        assert cap.amount == pytest.approx(-100.0, abs=2.0)

    def test_sync_negative_refund_goes_negative(self):
        backend = _make_sync_backend(limit=100, per_seconds=3600, sleep_interval=0.01)
        backend.consume_capacity(frozen_usage({"tokens": 80}))

        with pytest.warns(RuntimeWarning, match="exceeds reserved usage"):
            backend.refund_capacity(
                frozen_usage({"tokens": 80}),
                frozen_usage({"tokens": 200}),
            )

        bucket = backend._buckets[0]
        cap = bucket.get_capacity(time.time())
        assert cap.amount == pytest.approx(-100.0, abs=2.0)


# ---------------------------------------------------------------------------
# Regression guard: refund from negative must NOT erase debt
# ---------------------------------------------------------------------------


class TestRefundPreservesNegativeDebt:
    """
    Regression: old code clamped capacity to max(0, ...) on refund,
    which would erase negative debt. The fix uses allow_negative=True.
    """

    async def test_async_refund_from_negative_preserves_debt(self):
        """capacity=-300, refund=+150 → should be -150, NOT 0 or 150."""
        backend = _make_async_backend(limit=1000, per_seconds=3600, sleep_interval=0.01)

        # Push to -300: consume 1300 from initial 1000
        await backend.consume_capacity(frozen_usage({"tokens": 1300}))
        bucket = backend._buckets[0]
        cap = bucket.get_capacity(time.time())
        assert cap.amount == pytest.approx(-300.0, abs=2.0)

        # Refund 150: reserved=200, actual=50 → refund=150
        await backend.refund_capacity(
            frozen_usage({"tokens": 200}),
            frozen_usage({"tokens": 50}),
        )
        cap_after = bucket.get_capacity(time.time())
        # -300 + 150 = -150 (debt preserved, not erased)
        assert cap_after.amount == pytest.approx(-150.0, abs=3.0)
        assert cap_after.amount < 0, "Debt must be preserved, not erased to 0"

    def test_sync_refund_from_negative_preserves_debt(self):
        """Sync mirror of debt preservation test."""
        backend = _make_sync_backend(limit=1000, per_seconds=3600, sleep_interval=0.01)

        backend.consume_capacity(frozen_usage({"tokens": 1300}))
        bucket = backend._buckets[0]
        cap = bucket.get_capacity(time.time())
        assert cap.amount == pytest.approx(-300.0, abs=2.0)

        backend.refund_capacity(
            frozen_usage({"tokens": 200}),
            frozen_usage({"tokens": 50}),
        )
        cap_after = bucket.get_capacity(time.time())
        assert cap_after.amount == pytest.approx(-150.0, abs=3.0)
        assert cap_after.amount < 0
