"""End-to-end tests against the real OpenAI API.

Requires OPENAI_API_KEY (env var or tests/manual/.env file).
NOT run in CI/CD — see tests/manual/README.md.
"""

import os
from pathlib import Path

import pytest
from frozendict import frozendict
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

# Load .env from this directory if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from token_throttle import (  # noqa: E402
    MemoryBackendBuilder,
    PerModelConfig,
    Quota,
    RateLimiter,
    UsageQuotas,
    create_logging_callbacks,
)


def _make_limiter() -> RateLimiter:
    return RateLimiter(
        lambda model: PerModelConfig(  # noqa: ARG005
            quotas=UsageQuotas([
                Quota(metric="requests", limit=100, per_seconds=60),
                Quota(metric="tokens", limit=100_000, per_seconds=60),
            ]),
        ),
        backend=MemoryBackendBuilder(sleep_interval=0),
        callbacks=create_logging_callbacks(),
    )


def _make_synthetic_response(total_tokens: int = 15) -> ChatCompletion:
    """Build a ChatCompletion identical in type to what the SDK returns."""
    return ChatCompletion(
        id="chatcmpl-synthetic",
        object="chat.completion",
        created=1234567890,
        model="gpt-5-nano",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=10,
            completion_tokens=total_tokens - 10,
            total_tokens=total_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Synthetic: refund_capacity_from_response with real SDK types (no API call)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refund_from_response_synthetic():
    """Pass a real ChatCompletion object (constructed locally) to refund."""
    limiter = _make_limiter()
    reservation = await limiter.acquire_capacity(
        frozendict({"requests": 1, "tokens": 50}),
        model="gpt-5-nano",
    )

    response = _make_synthetic_response(total_tokens=15)

    assert isinstance(response.usage, CompletionUsage)
    assert not isinstance(response.usage, dict)

    await limiter.refund_capacity_from_response(reservation, response)


# ---------------------------------------------------------------------------
# 2. Synthetic: legacy dict-kwargs path still works
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refund_from_response_legacy_dict():
    """The legacy **kwargs path: usage passed as a dict keyword argument."""
    limiter = _make_limiter()
    reservation = await limiter.acquire_capacity(
        frozendict({"requests": 1, "tokens": 50}),
        model="gpt-5-nano",
    )

    await limiter.refund_capacity_from_response(
        reservation,
        usage={"total_tokens": 30},
    )


# ---------------------------------------------------------------------------
# 3. Live API: acquire + refund with manual usage (any-provider pattern)
# ---------------------------------------------------------------------------

_needs_api_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@_needs_api_key
@pytest.mark.asyncio
async def test_live_manual_acquire_and_refund():
    """The 'any provider' README pattern: acquire_capacity + refund_capacity."""
    from openai import AsyncOpenAI  # noqa: PLC0415

    client = AsyncOpenAI()
    limiter = _make_limiter()

    usage = frozendict({"requests": 1, "tokens": 50})
    reservation = await limiter.acquire_capacity(usage, model="gpt-5-nano")

    response = await client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        max_completion_tokens=10,
    )

    actual_tokens = response.usage.total_tokens
    assert actual_tokens > 0, "Expected non-zero token usage"

    await limiter.refund_capacity(
        {"requests": 1, "tokens": actual_tokens},
        reservation,
    )


# ---------------------------------------------------------------------------
# 4. Live API: refund_capacity_from_response with real ChatCompletion
# ---------------------------------------------------------------------------


@_needs_api_key
@pytest.mark.asyncio
async def test_live_refund_capacity_from_response():
    """The OpenAI quickstart README pattern: response object passed directly."""
    from openai import AsyncOpenAI  # noqa: PLC0415

    client = AsyncOpenAI()
    limiter = _make_limiter()

    usage = frozendict({"requests": 1, "tokens": 50})
    reservation = await limiter.acquire_capacity(usage, model="gpt-5-nano")

    response = await client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Say 'hi' and nothing else."}],
        max_completion_tokens=10,
    )

    assert response.usage is not None, "Expected usage in response"
    assert response.usage.total_tokens > 0

    # This is the exact pattern from the README
    await limiter.refund_capacity_from_response(reservation, response)
