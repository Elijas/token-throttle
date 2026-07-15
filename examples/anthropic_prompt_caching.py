"""
Rate-limit Anthropic Messages with cache-aware ITPM and actual-output OTPM.

Run from the repository root after installing the example dependencies:

    pip install "token-throttle>=10.0.0,<11.0.0" "anthropic>=0.116.0"
    export ANTHROPIC_API_KEY=...
    export ANTHROPIC_RPM=...
    export ANTHROPIC_ITPM=...
    export ANTHROPIC_OTPM=...
    python examples/anthropic_prompt_caching.py

Copy the three limits from Claude Console (Settings > Limits), or read them
with Anthropic's Rate Limits API. They are deliberately required instead of
using a hard-coded usage tier.

Anthropic meters input and output tokens independently. For current models,
ITPM counts uncached input plus cache writes, while cache reads do not count.
OTPM counts actual output as it is produced; ``max_tokens`` is only a response
safety ceiling and never enters the reservation calculation below.

The example reserves OTPM at request launch and refunds at completion. The
server meters OTPM continuously, so this point charge is conservative but can
under-use the window. A streaming application can use ``messages.stream()``,
obtain final usage with ``await stream.get_final_message()``, and perform the
same refund after the stream completes.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from anthropic import AsyncAnthropic

from token_throttle import (
    MemoryBackendBuilder,
    PerModelConfig,
    Quota,
    RateLimiter,
    UsageQuotas,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anthropic.types import Message

MODEL = "claude-haiku-4-5-20251001"
MODEL_FAMILY = "anthropic-claude-haiku-4-5"

# Replace this demo value with the observed p99 of actual output for your task.
# It is intentionally independent from MAX_TOKENS, the response safety ceiling.
OBSERVED_P99_OUTPUT_TOKENS = 128
MAX_TOKENS = 512

# Haiku 4.5 requires at least 4,096 cacheable tokens. Repeating a compact,
# deterministic paragraph keeps this example self-contained while producing a
# large enough shared prefix for a visible cache-read refund.
_CACHE_PARAGRAPH = (
    "Token buckets refill continuously. Reservations prevent oversubscription, "
    "and refunds return unused capacity after actual usage is known. Distributed "
    "workers must share quota state when they use the same provider limits.\n"
)
CACHED_CONTEXT = _CACHE_PARAGRAPH * 320

_LOGGER = logging.getLogger("anthropic_prompt_caching")
_INPUT_REMAINING = "anthropic-ratelimit-input-tokens-remaining"
_OUTPUT_REMAINING = "anthropic-ratelimit-output-tokens-remaining"


def anthropic_usage_counter(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    count_cache_reads_for_itpm: bool = False,
    **_unused: object,
) -> dict[str, int]:
    """
    Translate Anthropic token fields into token-throttle's three metrics.

    Before a request, ``input_tokens`` is the total returned by
    ``messages.count_tokens`` and ``output_tokens`` is the workload's observed
    p99. After a response, pass ``response.usage.model_dump()`` so the same
    function maps the provider's actual charge.

    This example targets current Claude API models. Legacy Claude Haiku 3.5
    counted cache reads toward ITPM; integrations that still target it on a
    platform where it is available must set ``count_cache_reads_for_itpm``.
    """
    cache_creation = cache_creation_input_tokens or 0
    cache_read = cache_read_input_tokens or 0
    itpm = input_tokens + cache_creation
    if count_cache_reads_for_itpm:
        itpm += cache_read
    return {
        "requests": 1,
        "input_tokens": itpm,
        "output_tokens": output_tokens,
    }


def remaining_token_headers(headers: Mapping[str, str]) -> dict[str, str | None]:
    """Return Anthropic's rounded input/output remaining-token headers."""
    return {
        "input_tokens": headers.get(_INPUT_REMAINING),
        "output_tokens": headers.get(_OUTPUT_REMAINING),
    }


def _positive_limit_from_env(name: str) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        raise RuntimeError(
            f"{name} is required; copy the real value from Claude Console "
            "(Settings > Limits) or the Rate Limits API"
        )
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _build_limiter() -> RateLimiter:
    rpm = _positive_limit_from_env("ANTHROPIC_RPM")
    itpm = _positive_limit_from_env("ANTHROPIC_ITPM")
    otpm = _positive_limit_from_env("ANTHROPIC_OTPM")
    return RateLimiter(
        PerModelConfig(
            model_family=MODEL_FAMILY,
            quotas=UsageQuotas(
                [
                    Quota(metric="requests", limit=rpm, per_seconds=60),
                    Quota(metric="input_tokens", limit=itpm, per_seconds=60),
                    Quota(metric="output_tokens", limit=otpm, per_seconds=60),
                ]
            ),
        ),
        backend=MemoryBackendBuilder(),
    )


async def _limited_message(  # noqa: PLR0913
    *,
    client: AsyncAnthropic,
    limiter: RateLimiter,
    system: list[dict[str, object]],
    messages: list[dict[str, object]],
    max_tokens: int,
    reserved_output_tokens: int,
    label: str,
) -> Message:
    # Token counting has its own independent rate limit, so it does not consume
    # Messages API RPM/ITPM/OTPM capacity and needs no bucket here.
    token_count = await client.messages.count_tokens(
        model=MODEL,
        system=system,
        messages=messages,
    )
    reserved_usage = anthropic_usage_counter(
        input_tokens=token_count.input_tokens,
        output_tokens=reserved_output_tokens,
    )
    reservation = await limiter.acquire_capacity(reserved_usage, model=MODEL)

    # If the SDK raises before returning response usage, close conservatively:
    # the reservation is not leaked, but unknown usage is treated as consumed.
    actual_usage = reserved_usage
    try:
        raw_response = await client.messages.with_raw_response.create(
            model=MODEL,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        remaining = remaining_token_headers(raw_response.headers)
        _LOGGER.info(
            "%s remaining (rounded to nearest 1,000): ITPM=%s OTPM=%s",
            label,
            remaining["input_tokens"],
            remaining["output_tokens"],
        )
        message = raw_response.parse()
        actual_usage = anthropic_usage_counter(**message.usage.model_dump())

        reserved_input = reserved_usage["input_tokens"]
        actual_input = actual_usage["input_tokens"]
        refunded_input = reserved_input - actual_input
        refunded_percent = 100 * refunded_input / reserved_input
        _LOGGER.info(
            "%s ITPM reserved=%d actual=%d refunded=%d (%.2f%%)",
            label,
            reserved_input,
            actual_input,
            refunded_input,
            refunded_percent,
        )
        _LOGGER.info(
            "%s OTPM reserved=%d actual=%d refunded=%d",
            label,
            reserved_usage["output_tokens"],
            actual_usage["output_tokens"],
            reserved_usage["output_tokens"] - actual_usage["output_tokens"],
        )
        return message
    finally:
        await limiter.refund_capacity(actual_usage, reservation)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    limiter = _build_limiter()
    system = [
        {
            "type": "text",
            "text": CACHED_CONTEXT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    # The SDK retries residual 429s twice by default and honors retry-after.
    # The limiter prevents ordinary quota overshoot; no second retry loop is
    # added here to compete with the SDK's backoff.
    async with AsyncAnthropic(max_retries=2) as client:
        try:
            prewarm = await _limited_message(
                client=client,
                limiter=limiter,
                system=system,
                messages=[{"role": "user", "content": "warm the shared cache"}],
                max_tokens=0,
                reserved_output_tokens=0,
                label="prewarm",
            )
            prewarm_cache_tokens = (prewarm.usage.cache_creation_input_tokens or 0) + (
                prewarm.usage.cache_read_input_tokens or 0
            )
            if not prewarm_cache_tokens:
                raise RuntimeError(
                    "Anthropic neither created nor reused a prompt cache entry; "
                    "ensure the cached prefix meets the model's minimum length"
                )

            response = await _limited_message(
                client=client,
                limiter=limiter,
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": "In one sentence, why are refunds useful?",
                    }
                ],
                max_tokens=MAX_TOKENS,
                reserved_output_tokens=OBSERVED_P99_OUTPUT_TOKENS,
                label="cache-hit",
            )
            if not (response.usage.cache_read_input_tokens or 0):
                raise RuntimeError(
                    "Expected the second request to read the pre-warmed cache"
                )

            # A refusal is an HTTP success with usage. _limited_message captured
            # that usage and refunded before control reaches this branch.
            if response.stop_reason == "refusal":
                _LOGGER.warning("Claude refused the request; capacity was reconciled")
                return

            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            _LOGGER.info("Claude: %s", text)
        finally:
            await limiter.aclose()


if __name__ == "__main__":
    asyncio.run(main())
