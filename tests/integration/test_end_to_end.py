"""
End-to-end integration tests using the high-level RateLimiter API.

These tests exercise the full lifecycle (acquire -> use -> refund) against
a real backend, using the parameterized `backend_builder` fixture.
"""

import asyncio
import time

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import (
    Quota,
    UsageQuotas,
    frozen_usage,
)
from token_throttle._rate_limiter import RateLimiter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    model_family: str = "test",
    requests_limit: float = 100,
    tokens_limit: float = 10000,
    per_seconds: float = 60,
) -> PerModelConfig:
    return PerModelConfig(
        model_family=model_family,
        quotas=UsageQuotas(
            [
                Quota(metric="requests", limit=requests_limit, per_seconds=per_seconds),
                Quota(metric="tokens", limit=tokens_limit, per_seconds=per_seconds),
            ],
        ),
    )


# ---------------------------------------------------------------------------
# 1. Full acquire -> use -> refund cycle
# ---------------------------------------------------------------------------


async def test_full_acquire_use_refund_cycle(backend_builder):
    """Complete lifecycle: acquire capacity, simulate work, then refund."""
    config = _make_config(model_family="lifecycle")
    limiter = RateLimiter(config, backend=backend_builder)

    reservation = await limiter.acquire_capacity(
        usage={"requests": 1, "tokens": 500},
        model="lifecycle",
    )

    assert reservation.model_family == "lifecycle"
    assert reservation.usage["requests"] == 1
    assert reservation.usage["tokens"] == 500

    # Simulate work (actual used fewer tokens).
    await limiter.refund_capacity(
        actual_usage={"requests": 1, "tokens": 200},
        reservation=reservation,
    )


# ---------------------------------------------------------------------------
# 2. Multiple model families get independent quotas
# ---------------------------------------------------------------------------


async def test_multiple_model_families_independent_quotas(backend_builder):
    """Different model families should have independent capacity pools."""

    def config_getter(model_name: str) -> PerModelConfig:
        return _make_config(
            model_family=model_name,
            requests_limit=5,
            tokens_limit=100,
            per_seconds=1,
        )

    limiter = RateLimiter(config_getter, backend=backend_builder)

    # Exhaust capacity for "gpt-4".
    await limiter.acquire_capacity(
        usage={"requests": 5, "tokens": 100},
        model="gpt-4",
    )

    # "gpt-3.5" should still have full capacity — no blocking.
    start = time.monotonic()
    await limiter.acquire_capacity(
        usage={"requests": 5, "tokens": 100},
        model="gpt-3.5",
    )
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, "gpt-3.5 should not be blocked by gpt-4 exhaustion"


# ---------------------------------------------------------------------------
# 3. Same model family, different model names share backend
# ---------------------------------------------------------------------------


async def test_same_family_different_names_share_capacity(backend_builder):
    """Models mapped to the same family should compete for the same capacity."""

    def config_getter(_model_name: str) -> PerModelConfig:
        # Both "gpt-4-0314" and "gpt-4-0613" map to family "gpt-4".
        return _make_config(
            model_family="gpt-4",
            requests_limit=5,
            tokens_limit=100,
            per_seconds=1,
        )

    limiter = RateLimiter(config_getter, backend=backend_builder)

    # Consume most capacity via "gpt-4-0314".
    await limiter.acquire_capacity(
        usage={"requests": 4, "tokens": 80},
        model="gpt-4-0314",
    )

    # "gpt-4-0613" should share that capacity, so requesting 4 more should block.
    start = time.monotonic()
    await limiter.acquire_capacity(
        usage={"requests": 4, "tokens": 80},
        model="gpt-4-0613",
    )
    elapsed = time.monotonic() - start

    # Should have waited because capacity was shared.
    assert elapsed >= 0.1, f"Expected wait due to shared capacity, got {elapsed:.3f} s"


# ---------------------------------------------------------------------------
# 4. Unlimited config skips backend entirely
# ---------------------------------------------------------------------------


async def test_unlimited_config_skips_backend(backend_builder):
    """Unlimited quotas should bypass the backend entirely."""
    config = PerModelConfig(
        model_family="unlimited",
        quotas=UsageQuotas.unlimited(),
    )
    limiter = RateLimiter(config, backend=backend_builder)

    reservation = await limiter.acquire_capacity(
        usage={},
        model="unlimited",
    )

    assert reservation.model_family == "__rate_limiting_disabled__"
    assert reservation.usage == {}

    # Refund should also work with empty usage.
    await limiter.refund_capacity(
        actual_usage={},
        reservation=reservation,
    )


# ---------------------------------------------------------------------------
# 5. Sequential requests exhaust capacity then block until refill
# ---------------------------------------------------------------------------


async def test_sequential_requests_exhaust_then_block(backend_builder):
    """Making requests until exhaustion should cause the next one to block."""
    config = _make_config(
        model_family="exhaust",
        requests_limit=3,
        tokens_limit=300,
        per_seconds=1,
    )
    limiter = RateLimiter(config, backend=backend_builder)

    # Consume all 3 request slots.
    for _ in range(3):
        await limiter.acquire_capacity(
            usage={"requests": 1, "tokens": 10},
            model="exhaust",
        )

    # 4th request should block until refill.
    start = time.monotonic()
    await limiter.acquire_capacity(
        usage={"requests": 1, "tokens": 10},
        model="exhaust",
    )
    elapsed = time.monotonic() - start

    assert elapsed >= 0.15, f"Expected wait after exhaustion, got {elapsed:.3f} s"


# ---------------------------------------------------------------------------
# 6. Dynamic max_capacity change mid-session
# ---------------------------------------------------------------------------


async def test_dynamic_max_capacity_change(request, backend_builder, redis_client):
    """
    Changing max_capacity via Redis should take effect after cache TTL.

    We reduce the max_capacity mid-session and verify that the refill is
    capped at the new (lower) value, causing a subsequent request to block.
    """
    if request.node.callspec.params.get("backend_builder") != "redis":
        pytest.skip("Dynamic max_capacity via Redis keys is Redis-specific")
    config = _make_config(
        model_family="dynamic",
        requests_limit=10,
        tokens_limit=1000,
        per_seconds=1,
    )
    limiter = RateLimiter(config, backend=backend_builder)

    # Initial acquire to trigger backend creation and populate cache.
    await limiter.acquire_capacity(
        usage={"requests": 1, "tokens": 10},
        model="dynamic",
    )

    # Reduce max_capacity for requests from 10 to 3 via Redis directly.
    await redis_client.set(
        "rate_limiting:dynamic:requests:1:max_capacity",
        3,
    )

    # Wait for cache TTL to expire (1 second cache + margin) so the new
    # max_capacity is picked up.  Natural refill also occurs during this time,
    # but it should now be capped at 3 instead of 10.
    await asyncio.sleep(1.2)

    # Request exactly 3 requests — should succeed immediately since refill
    # capped at new max (3).
    start = time.monotonic()
    await limiter.acquire_capacity(
        usage={"requests": 3, "tokens": 10},
        model="dynamic",
    )
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, "3 requests should succeed with new max_capacity=3"

    # Now requesting even 1 more should block (we just consumed all 3).
    start = time.monotonic()
    await limiter.acquire_capacity(
        usage={"requests": 1, "tokens": 10},
        model="dynamic",
    )
    elapsed = time.monotonic() - start
    assert elapsed >= 0.08, (
        f"Expected wait after exhausting dynamic capacity, got {elapsed:.3f} s"
    )


# ---------------------------------------------------------------------------
# 7. OpenAI-style request: acquire_capacity_for_request -> refund_from_response
# ---------------------------------------------------------------------------


async def test_openai_style_acquire_and_refund(backend_builder):
    """Exercise acquire_capacity_for_request and refund_capacity_from_response."""

    def mock_usage_counter(**kwargs) -> dict[str, float]:
        # Simplified counter: 1 request, count characters as "tokens".
        messages = kwargs.get("messages", [])
        tokens = sum(len(m.get("content", "")) for m in messages)
        return frozen_usage({"requests": 1, "tokens": float(tokens)})

    config = PerModelConfig(
        model_family="openai-test",
        quotas=UsageQuotas(
            [
                Quota(metric="requests", limit=100, per_seconds=60),
                Quota(metric="tokens", limit=10000, per_seconds=60),
            ],
        ),
        usage_counter=mock_usage_counter,
    )
    limiter = RateLimiter(config, backend=backend_builder)

    # Acquire capacity for a request.
    reservation = await limiter.acquire_capacity_for_request(
        extra_usage=None,
        model="openai-test",
        messages=[
            {"role": "user", "content": "Hello, world!"},
        ],
    )

    assert reservation.model_family == "openai-test"
    assert reservation.usage["requests"] == 1
    assert reservation.usage["tokens"] > 0

    # Simulate an OpenAI response with usage data.
    await limiter.refund_capacity_from_response(
        reservation,
        usage={"total_tokens": 25},
    )
