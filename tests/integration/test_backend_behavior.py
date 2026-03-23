"""
Backend-agnostic integration tests for RateLimiterBackend contract.

These tests use the parameterized `backend_builder` fixture so they
automatically run against every registered backend (currently Redis,
extensible to future backends).
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# 1. await_for_capacity — immediate success
# ---------------------------------------------------------------------------


async def test_await_for_capacity_immediate_success(backend_builder):
    """Consuming well within the limit should return immediately."""
    config = _make_config(limit=100, per_seconds=60)
    backend = backend_builder.build(config)

    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 10}))
    elapsed = time.monotonic() - start

    # Should complete almost instantly (generous 1 s tolerance for CI)
    assert elapsed < 1.0


# ---------------------------------------------------------------------------
# 1b. Direct backend calls must still validate usage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method_name", "bad_value", "message"),
    [
        ("await_for_capacity", -1.0, "must be non-negative"),
        ("consume_capacity", float("nan"), "must be finite"),
    ],
)
async def test_direct_backend_methods_reject_invalid_usage(
    backend_builder,
    method_name: str,
    bad_value: float,
    message: str,
):
    """Exported backends must reject invalid usage even without RateLimiter."""
    config = _make_config(limit=100, per_seconds=60)
    backend = backend_builder.build(config)
    method = getattr(backend, method_name)

    with pytest.raises(ValueError, match=message):
        await method(frozen_usage({"requests": bad_value}))


async def test_direct_backend_refund_rejects_invalid_actual_usage(backend_builder):
    """Direct refund calls must not accept negative actual usage."""
    config = _make_config(limit=100, per_seconds=60)
    backend = backend_builder.build(config)
    reserved_usage = frozen_usage({"requests": 10.0})

    await backend.await_for_capacity(reserved_usage)

    with pytest.raises(
        ValueError, match="Actual usage value for requests must be non-negative"
    ):
        await backend.refund_capacity(
            reserved_usage=reserved_usage,
            actual_usage=frozen_usage({"requests": -1.0}),
        )


# ---------------------------------------------------------------------------
# 2. await_for_capacity — with wait
# ---------------------------------------------------------------------------


async def test_await_for_capacity_with_wait(backend_builder):
    """When capacity is exhausted, the next call should block until refill."""
    # 5 units per second — consumes fully, then needs to wait for refill.
    config = _make_config(limit=5, per_seconds=1)
    backend = backend_builder.build(config)

    # Exhaust all capacity.
    await backend.await_for_capacity(frozen_usage({"requests": 5}))

    # Next request must wait for refill.
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1}))
    elapsed = time.monotonic() - start

    # The backend polls every 0.1 s (DEFAULT_SLEEP_INTERVAL), so the
    # minimum observable wait is one sleep cycle.
    assert elapsed >= 0.08, f"Expected wait >= 0.08 s, got {elapsed:.3f} s"
    # But not excessively long
    assert elapsed < 2.0


# ---------------------------------------------------------------------------
# 3. All-or-nothing: one metric insufficient -> none consumed
# ---------------------------------------------------------------------------


async def test_all_or_nothing_one_metric_insufficient(backend_builder):
    """If one metric lacks capacity, no metrics should be consumed."""
    config = _make_config(
        limit=10,
        per_seconds=1,
        extra_quotas=[Quota(metric="tokens", limit=10, per_seconds=1)],
    )
    backend = backend_builder.build(config)

    # Exhaust tokens only.
    await backend.await_for_capacity(frozen_usage({"requests": 1, "tokens": 9}))

    # Now request 5 of each — tokens insufficient (only ~1 left), requests fine.
    # This should block until tokens refill.
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1, "tokens": 5}))
    elapsed = time.monotonic() - start

    # Must have waited for token refill (at least some measurable time).
    assert elapsed >= 0.1, f"Expected wait >= 0.1 s, got {elapsed:.3f} s"


# ---------------------------------------------------------------------------
# 4. All-or-nothing: all sufficient -> all consumed immediately
# ---------------------------------------------------------------------------


async def test_all_or_nothing_all_sufficient(backend_builder):
    """When all metrics have capacity, consumption succeeds immediately."""
    config = _make_config(
        limit=100,
        per_seconds=60,
        extra_quotas=[Quota(metric="tokens", limit=10000, per_seconds=60)],
    )
    backend = backend_builder.build(config)

    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1, "tokens": 100}))
    elapsed = time.monotonic() - start

    assert elapsed < 1.0


# ---------------------------------------------------------------------------
# 5. refund_capacity — positive refund
# ---------------------------------------------------------------------------


async def test_refund_capacity_positive_refund(backend_builder):
    """Refunding unused capacity should restore availability."""
    config = _make_config(limit=10, per_seconds=1)
    backend = backend_builder.build(config)

    # Exhaust all capacity.
    await backend.await_for_capacity(frozen_usage({"requests": 10}))

    # Refund: reserved 10, actually used 2 -> refund 8.
    await backend.refund_capacity(
        reserved_usage=frozen_usage({"requests": 10}),
        actual_usage=frozen_usage({"requests": 2}),
    )

    # After refund, we should be able to consume 8 immediately.
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 7}))
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, "Expected immediate capacity after refund"


# ---------------------------------------------------------------------------
# 6. refund_capacity — negative refund (overuse)
# ---------------------------------------------------------------------------


async def test_refund_capacity_negative_refund_warns(backend_builder):
    """Overuse (actual > reserved) should emit a RuntimeWarning."""
    config = _make_config(limit=100, per_seconds=60)
    backend = backend_builder.build(config)

    await backend.await_for_capacity(frozen_usage({"requests": 20}))

    with pytest.warns(RuntimeWarning, match="exceeds reserved usage"):
        await backend.refund_capacity(
            reserved_usage=frozen_usage({"requests": 20}),
            actual_usage=frozen_usage({"requests": 50}),
        )


# ---------------------------------------------------------------------------
# 7. Refund capped at max_capacity
# ---------------------------------------------------------------------------


async def test_refund_capped_at_max_capacity(backend_builder):
    """Refunding more than max_capacity should cap at max_capacity."""
    config = _make_config(limit=10, per_seconds=1)
    backend = backend_builder.build(config)

    # Consume 5, then refund as if we reserved 5 but used 0 -> +5 refund.
    await backend.await_for_capacity(frozen_usage({"requests": 5}))
    await backend.refund_capacity(
        reserved_usage=frozen_usage({"requests": 5}),
        actual_usage=frozen_usage({"requests": 0}),
    )

    # Even after refund, total capacity should not exceed max (10).
    # Exhaust exactly 10 — should succeed immediately (capacity is at most 10).
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 10}))
    elapsed = time.monotonic() - start
    assert elapsed < 1.0

    # But requesting 1 more should block because we're at 0.
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1}))
    elapsed = time.monotonic() - start
    assert elapsed >= 0.05, "Expected wait after exhaustion"


# ---------------------------------------------------------------------------
# 8. Refund updates timestamp
# ---------------------------------------------------------------------------


async def test_refund_updates_timestamp(backend_builder):
    """After refund, the time credit should be adjusted (timestamp updated)."""
    config = _make_config(limit=10, per_seconds=1)
    backend = backend_builder.build(config)

    # Consume all capacity.
    await backend.await_for_capacity(frozen_usage({"requests": 10}))

    # Wait a moment so time passes.
    await asyncio.sleep(0.3)

    # Refund everything (reserved 10, used 0).
    await backend.refund_capacity(
        reserved_usage=frozen_usage({"requests": 10}),
        actual_usage=frozen_usage({"requests": 0}),
    )

    # The refund should set the timestamp to *now*, meaning subsequent capacity
    # calculation starts from the refund time, not the original consumption time.
    # If the timestamp were NOT updated, ~3 units of natural refill (0.3 s * 10/s)
    # would be counted on top of the 10 refunded, possibly exceeding max.
    # With updated timestamp, capacity is exactly 10 (capped at max).
    # Consuming 10 should succeed immediately.
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 10}))
    elapsed = time.monotonic() - start
    assert elapsed < 1.0


# ---------------------------------------------------------------------------
# 9. Capacity refill over time
# ---------------------------------------------------------------------------


async def test_capacity_refill_over_time(backend_builder):
    """Capacity should refill at the expected rate after consumption."""
    # 10 units per second -> 1 unit every 0.1 s.
    config = _make_config(limit=10, per_seconds=1)
    backend = backend_builder.build(config)

    # Exhaust all capacity.
    await backend.await_for_capacity(frozen_usage({"requests": 10}))

    # Wait 0.5 s -> ~5 units should refill.
    await asyncio.sleep(0.5)

    # Requesting 4 should succeed immediately (5 refilled, leaving some margin).
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 4}))
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, f"Expected immediate success, got {elapsed:.3f} s wait"


# ---------------------------------------------------------------------------
# 10. Refill capped at max_capacity
# ---------------------------------------------------------------------------


async def test_refill_capped_at_max_capacity(backend_builder):
    """Even after waiting longer than needed, capacity stays at max."""
    config = _make_config(limit=5, per_seconds=1)
    backend = backend_builder.build(config)

    # Consume 3 of 5.
    await backend.await_for_capacity(frozen_usage({"requests": 3}))

    # Wait 3 s — far more than needed to refill 3 units at 5/s.
    await asyncio.sleep(1.5)

    # Capacity should be at most 5 (max). Consuming 5 should work immediately.
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 5}))
    elapsed = time.monotonic() - start
    assert elapsed < 1.0

    # But 1 more should require waiting (we just consumed 5 of 5).
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1}))
    elapsed = time.monotonic() - start
    assert elapsed >= 0.1


# ---------------------------------------------------------------------------
# 11. on_missing_consumption_data callback fires on fresh start
# ---------------------------------------------------------------------------


async def test_on_missing_consumption_data_fires_on_fresh_start(backend_builder):
    """The on_missing_consumption_data callback fires on first access (no prior data)."""
    on_missing = AsyncMock()
    callbacks = RateLimiterCallbacks(on_missing_consumption_data=on_missing)

    config = _make_config(model_family="fresh_test", limit=100, per_seconds=60)
    backend = backend_builder.build(config, callbacks=callbacks)

    await backend.await_for_capacity(frozen_usage({"requests": 1}))

    on_missing.assert_called_once()
    call_kwargs = on_missing.call_args.kwargs
    assert call_kwargs["model_family"] == "fresh_test"
    assert call_kwargs["usage_metric"] == "requests"


# ---------------------------------------------------------------------------
# 12. All 5 callbacks fire at correct points
# ---------------------------------------------------------------------------


async def test_all_five_callbacks_fire(backend_builder):
    """Run a full acquire-wait-refund cycle and verify all 5 callbacks fire."""
    on_wait_start = AsyncMock()
    after_wait_end = AsyncMock()
    on_consumed = AsyncMock()
    on_refunded = AsyncMock()
    on_missing = AsyncMock()

    callbacks = RateLimiterCallbacks(
        on_wait_start=on_wait_start,
        after_wait_end_consumption=after_wait_end,
        on_capacity_consumed=on_consumed,
        on_capacity_refunded=on_refunded,
        on_missing_consumption_data=on_missing,
    )

    # Tight limit so we can force a wait.
    config = _make_config(model_family="cb_test", limit=5, per_seconds=1)
    backend = backend_builder.build(config, callbacks=callbacks)

    # 1) First call — triggers on_missing_consumption_data + on_capacity_consumed.
    await backend.await_for_capacity(frozen_usage({"requests": 5}))
    assert on_missing.call_count >= 1, "on_missing_consumption_data should have fired"
    assert on_consumed.call_count == 1, "on_capacity_consumed should have fired"

    # 2) Second call — capacity exhausted, triggers on_wait_start, then
    #    after refill triggers after_wait_end_consumption + on_capacity_consumed.
    await backend.await_for_capacity(frozen_usage({"requests": 1}))
    assert on_wait_start.call_count >= 1, "on_wait_start should have fired"
    assert after_wait_end.call_count >= 1, (
        "after_wait_end_consumption should have fired"
    )
    assert on_consumed.call_count >= 2, "on_capacity_consumed should fire again"

    # 3) Refund — triggers on_capacity_refunded.
    await backend.refund_capacity(
        reserved_usage=frozen_usage({"requests": 1}),
        actual_usage=frozen_usage({"requests": 0}),
    )
    assert on_refunded.call_count == 1, "on_capacity_refunded should have fired"


# ---------------------------------------------------------------------------
# 13. consume_capacity — allows negative capacity
# ---------------------------------------------------------------------------


async def test_dynamic_max_capacity_via_backend_api(backend_builder):
    """Changing max_capacity via the backend API affects refill cap."""
    config = _make_config(limit=10, per_seconds=1)
    backend = backend_builder.build(config)

    # Consume all capacity.
    await backend.await_for_capacity(frozen_usage({"requests": 10}))

    # Reduce max_capacity from 10 to 3.
    await backend.set_max_capacity("requests", 1, 3.0)

    # Wait for refill — rate is still 10/s, but cap is now 3.
    await asyncio.sleep(0.5)

    # Requesting 3 should succeed immediately (refill capped at 3).
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 3}))
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, "3 requests should succeed with new max_capacity=3"

    # Requesting 1 more should block (we just consumed all 3).
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1}))
    elapsed = time.monotonic() - start
    assert elapsed >= 0.08, (
        f"Expected wait after exhausting dynamic capacity, got {elapsed:.3f} s"
    )


async def test_consume_capacity_allows_negative(backend_builder):
    """consume_capacity should allow capacity to go below zero."""
    config = _make_config(limit=5, per_seconds=1)
    backend = backend_builder.build(config)

    # Exhaust all capacity via blocking path
    await backend.await_for_capacity(frozen_usage({"requests": 5}))

    # consume_capacity should not block even though capacity is 0
    start = time.monotonic()
    await backend.consume_capacity(frozen_usage({"requests": 5}))
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, "consume_capacity should not block"

    # Subsequent blocking acquire should have to wait (capacity is -5)
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1}))
    elapsed = time.monotonic() - start

    assert elapsed >= 0.1, "acquire should wait when capacity is negative"
