"""
Integration tests for record_usage (non-blocking metering).

Tests run against all parameterized backends (Redis + memory) via
the `backend_builder` fixture from conftest.py.
"""

import asyncio
import time
from unittest.mock import AsyncMock

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._rate_limiter import RateLimiter


def _make_config(
    *,
    model_family: str = "test",
    metric: str = "requests",
    limit: float = 100,
    per_seconds: float = 60,
    extra_quotas: list[Quota] | None = None,
) -> PerModelConfig:
    quotas = [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
    if extra_quotas:
        quotas.extend(extra_quotas)
    return PerModelConfig(model_family=model_family, quotas=UsageQuotas(quotas))


# ---------------------------------------------------------------------------
# 1. consume_capacity returns immediately (never blocks)
# ---------------------------------------------------------------------------


async def test_consume_capacity_returns_immediately(backend_builder):
    """consume_capacity should complete without waiting, even at zero capacity."""
    config = _make_config(limit=5, per_seconds=1)
    backend = backend_builder.build(config)

    # Exhaust all capacity via blocking path
    await backend.await_for_capacity(frozen_usage({"requests": 5}))

    # consume_capacity should NOT block even though capacity is 0
    start = time.monotonic()
    await backend.consume_capacity(frozen_usage({"requests": 3}))
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"consume_capacity should not block, but took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# 2. consume_capacity allows negative capacity
# ---------------------------------------------------------------------------


async def test_consume_capacity_allows_negative(backend_builder):
    """After consume_capacity, capacity can go below zero."""
    config = _make_config(limit=5, per_seconds=1)
    backend = backend_builder.build(config)

    # Exhaust all capacity
    await backend.await_for_capacity(frozen_usage({"requests": 5}))

    # Record more usage — capacity should go negative
    start = time.monotonic()
    await backend.consume_capacity(frozen_usage({"requests": 5}))
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, "consume_capacity should not block"

    # Now a blocking acquire should have to wait (capacity is at -5)
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1}))
    elapsed = time.monotonic() - start

    assert elapsed >= 0.1, "acquire should wait when capacity is negative"


# ---------------------------------------------------------------------------
# 3. consume_capacity fires on_capacity_consumed callback
# ---------------------------------------------------------------------------


async def test_consume_capacity_fires_callback(backend_builder):
    """consume_capacity should fire the on_capacity_consumed callback."""
    on_consumed = AsyncMock()
    callbacks = RateLimiterCallbacks(on_capacity_consumed=on_consumed)

    config = _make_config(model_family="cb_consume", limit=100, per_seconds=60)
    backend = backend_builder.build(config, callbacks=callbacks)

    await backend.consume_capacity(frozen_usage({"requests": 10}))

    on_consumed.assert_called_once()
    call_kwargs = on_consumed.call_args.kwargs
    assert call_kwargs["model_family"] == "cb_consume"
    assert dict(call_kwargs["usage"]) == {"requests": 10.0}


# ---------------------------------------------------------------------------
# 4. Negative capacity naturally recovers via time-based refill
# ---------------------------------------------------------------------------


async def test_negative_capacity_recovers_via_refill(backend_builder):
    """Capacity that went negative should recover through time-based refill."""
    # 10 units/second — fast enough to observe recovery quickly
    config = _make_config(limit=10, per_seconds=1)
    backend = backend_builder.build(config)

    # Exhaust capacity, then push to -5
    await backend.await_for_capacity(frozen_usage({"requests": 10}))
    await backend.consume_capacity(frozen_usage({"requests": 5}))

    # Wait for capacity to refill from -5 back to positive
    # At 10/s, -5 → 0 takes 0.5s, then +1 more takes 0.1s = ~0.6s total
    await asyncio.sleep(0.8)

    # Should now be able to acquire 1 without blocking (much)
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1}))
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, (
        f"Expected quick acquire after refill, but waited {elapsed:.3f}s"
    )


# ---------------------------------------------------------------------------
# 5. consume_capacity while already negative
# ---------------------------------------------------------------------------


async def test_consume_capacity_while_already_negative(backend_builder):
    """Can keep recording usage even when capacity is already negative."""
    config = _make_config(limit=5, per_seconds=1)
    backend = backend_builder.build(config)

    # Exhaust capacity
    await backend.await_for_capacity(frozen_usage({"requests": 5}))

    # Record three times beyond zero — each should succeed instantly
    for _ in range(3):
        start = time.monotonic()
        await backend.consume_capacity(frozen_usage({"requests": 5}))
        elapsed = time.monotonic() - start
        assert elapsed < 1.0


# ---------------------------------------------------------------------------
# 6. Mixed acquire and record on the same model share capacity
# ---------------------------------------------------------------------------


async def test_mixed_acquire_and_record_same_model(backend_builder):
    """Acquire and consume_capacity share the same bucket pool."""
    config = _make_config(limit=10, per_seconds=1)
    backend = backend_builder.build(config)

    # Consume 8 via blocking acquire
    await backend.await_for_capacity(frozen_usage({"requests": 8}))

    # Record 5 more via non-blocking — pushes to -3
    await backend.consume_capacity(frozen_usage({"requests": 5}))

    # Blocking acquire should now wait
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1}))
    elapsed = time.monotonic() - start

    assert elapsed >= 0.1, "acquire should block after capacity was consumed to negative"


# ---------------------------------------------------------------------------
# 7. record_then_acquire_waits — end-to-end at RateLimiter level
# ---------------------------------------------------------------------------


async def test_record_depletes_then_acquire_waits(backend_builder):
    """Record_usage depleting capacity causes subsequent acquire_capacity to wait."""
    config = _make_config(limit=5, per_seconds=1)
    limiter = RateLimiter(config, backend=backend_builder)

    # Record all capacity away via non-blocking path
    await limiter.record_usage({"requests": 5}, model="test")

    # Blocking acquire should wait for refill
    start = time.monotonic()
    await limiter.acquire_capacity({"requests": 1}, model="test")
    elapsed = time.monotonic() - start

    assert elapsed >= 0.08, f"Expected wait, got {elapsed:.3f}s"
    assert elapsed < 3.0
