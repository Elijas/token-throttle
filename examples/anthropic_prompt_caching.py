"""
Rate-limit Anthropic Messages with cache-aware ITPM and actual-output OTPM.

Run from the repository root after installing the example dependencies:

    pip install "token-throttle>=10.0.0,<11.0.0" "anthropic>=0.51.0"
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

The example reserves OTPM at request launch and refunds at completion. To make
that recovery measurable in a short A/B run, it validates the real OTPM limit
from the environment but deliberately uses a smaller, faster local output-token
window for the concurrent demo. Its equivalent per-minute rate never exceeds
the configured provider OTPM. The server meters OTPM continuously, so this
point charge is conservative but can under-use the window. A streaming application can use
``messages.stream()``, obtain final usage with
``await stream.get_final_message()``, and perform the same refund after the
stream completes.
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
    RateLimiterCallbacks,
    UsageQuotas,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from anthropic.types import Message, MessageParam, TextBlockParam

    from token_throttle import Capacities, FrozenUsage

MODEL = "claude-haiku-4-5-20251001"
MODEL_FAMILY = "anthropic-claude-haiku-4-5"

# Replace this demo value with the observed p99 of actual output for your task.
# It is intentionally independent from MAX_TOKENS, the response safety ceiling.
OBSERVED_P99_OUTPUT_TOKENS = 128
MAX_TOKENS = 512
PREWARM_MAX_TOKENS = 16
PREWARM_RESERVED_OUTPUT_TOKENS = 16
CONCURRENT_CACHE_HITS = 4

# These define a demo-only local window, not a provider tier default. The real
# ANTHROPIC_OTPM value is still required and validated before this shorter
# window is used so the measured A/B finishes quickly without outpacing the
# provider's configured refill rate.
DEMO_OUTPUT_TOKEN_LIMIT = PREWARM_RESERVED_OUTPUT_TOKENS + OBSERVED_P99_OUTPUT_TOKENS
DEMO_OUTPUT_WINDOW_SECONDS = 6

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


def _build_limiter(*, callbacks: RateLimiterCallbacks | None = None) -> RateLimiter:
    rpm = _positive_limit_from_env("ANTHROPIC_RPM")
    itpm = _positive_limit_from_env("ANTHROPIC_ITPM")
    provider_otpm = _positive_limit_from_env("ANTHROPIC_OTPM")
    equivalent_demo_otpm = (
        DEMO_OUTPUT_TOKEN_LIMIT * 60 + DEMO_OUTPUT_WINDOW_SECONDS - 1
    ) // DEMO_OUTPUT_WINDOW_SECONDS
    if provider_otpm < equivalent_demo_otpm:
        raise RuntimeError(
            f"ANTHROPIC_OTPM must be at least {equivalent_demo_otpm} for this "
            "short-window concurrency demonstration"
        )
    _LOGGER.info(
        "Provider OTPM=%d; using demo-only local output window=%d tokens/%ds "
        "(equivalent OTPM=%d) to force reservation contention",
        provider_otpm,
        DEMO_OUTPUT_TOKEN_LIMIT,
        DEMO_OUTPUT_WINDOW_SECONDS,
        equivalent_demo_otpm,
    )
    return RateLimiter(
        PerModelConfig(
            model_family=MODEL_FAMILY,
            quotas=UsageQuotas(
                [
                    Quota(metric="requests", limit=rpm, per_seconds=60),
                    Quota(metric="input_tokens", limit=itpm, per_seconds=60),
                    Quota(
                        metric="output_tokens",
                        limit=DEMO_OUTPUT_TOKEN_LIMIT,
                        per_seconds=DEMO_OUTPUT_WINDOW_SECONDS,
                    ),
                ]
            ),
        ),
        backend=MemoryBackendBuilder(),
        callbacks=callbacks,
    )


async def _count_input_tokens(
    *,
    client: AsyncAnthropic,
    system: list[TextBlockParam],
    messages: list[MessageParam],
) -> int:
    # Token counting has its own independent rate limit, so it does not consume
    # Messages API RPM/ITPM/OTPM capacity and needs no bucket here.
    token_count = await client.messages.count_tokens(
        model=MODEL,
        system=system,
        messages=messages,
    )
    return token_count.input_tokens


async def _limited_message(  # noqa: PLR0913
    *,
    client: AsyncAnthropic,
    limiter: RateLimiter,
    system: list[TextBlockParam],
    messages: list[MessageParam],
    counted_input_tokens: int,
    max_tokens: int,
    reserved_output_tokens: int,
    label: str,
    refund_unused: bool = True,
) -> Message:
    reserved_usage = anthropic_usage_counter(
        input_tokens=counted_input_tokens,
        output_tokens=reserved_output_tokens,
    )

    # Deliberately conservative: Anthropic may consume input before the SDK
    # raises, so unknown error usage is treated as the full reservation. This
    # explicit value is behaviorally the same as reserve()'s current default;
    # it stays here to document the example's intentional error policy.
    async with limiter.reserve(
        reserved_usage,
        model=MODEL,
        usage_on_error=reserved_usage,
    ) as handle:
        raw_response = await client.messages.with_raw_response.create(
            model=MODEL,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        message = raw_response.parse()
        actual_usage = anthropic_usage_counter(**message.usage.model_dump())
        # The A/B baseline reports the reservation as consumed, suppressing the
        # refund while preserving the real provider usage for the logs below.
        reconciled_usage = actual_usage if refund_unused else reserved_usage
        handle.set_actual_usage(reconciled_usage)

        remaining = remaining_token_headers(raw_response.headers)
        _LOGGER.info(
            "%s remaining (rounded to nearest 1,000): ITPM=%s OTPM=%s",
            label,
            remaining["input_tokens"],
            remaining["output_tokens"],
        )

        reserved_input = reserved_usage["input_tokens"]
        actual_input = actual_usage["input_tokens"]
        refunded_input = reserved_input - reconciled_usage["input_tokens"]
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
            reserved_usage["output_tokens"] - reconciled_usage["output_tokens"],
        )
        return message


def _validate_cache_hits(*, label: str, responses: Sequence[Message]) -> None:
    for task_number, response in enumerate(responses, start=1):
        if not (response.usage.cache_read_input_tokens or 0):
            raise RuntimeError(
                f"Expected {label}-{task_number} to read the pre-warmed cache"
            )

        # A refusal is an HTTP success with usage. _limited_message captured it
        # before the reservation scope exited.
        if response.stop_reason == "refusal":
            _LOGGER.warning(
                "%s-%d was refused; capacity was reconciled",
                label,
                task_number,
            )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    workload_label = "prewarm"
    wait_starts = wait_ends = 0
    total_wait_seconds = 0.0

    async def on_wait_start(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
    ) -> None:
        del model_family, preconsumption_capacities
        nonlocal wait_starts
        wait_starts += 1
        _LOGGER.info(
            "%s wait-start #%d: output reservation=%d",
            workload_label,
            wait_starts,
            usage["output_tokens"],
        )

    async def after_wait_end_consumption(
        *,
        model_family: str,
        usage: FrozenUsage,
        preconsumption_capacities: Capacities,
        postconsumption_capacities: Capacities,
        wait_time_s: float,
    ) -> None:
        del model_family, usage, preconsumption_capacities, postconsumption_capacities
        nonlocal wait_ends, total_wait_seconds
        wait_ends += 1
        total_wait_seconds += wait_time_s
        _LOGGER.info(
            "%s wait-end #%d: waited %.3fs",
            workload_label,
            wait_ends,
            wait_time_s,
        )

    callbacks = RateLimiterCallbacks(
        on_wait_start=on_wait_start,
        after_wait_end_consumption=after_wait_end_consumption,
    )
    limiter = _build_limiter(callbacks=callbacks)
    system: list[TextBlockParam] = [
        {
            "type": "text",
            "text": CACHED_CONTEXT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    async def run_concurrent_workload(
        *,
        client: AsyncAnthropic,
        messages: list[MessageParam],
        counted_input_tokens: int,
        label: str,
        refund_unused: bool,
    ) -> tuple[list[Message], float]:
        nonlocal workload_label, wait_starts, wait_ends, total_wait_seconds
        workload_label = label
        wait_starts = 0
        wait_ends = 0
        total_wait_seconds = 0.0
        started_at = asyncio.get_running_loop().time()
        responses = await asyncio.gather(
            *(
                _limited_message(
                    client=client,
                    limiter=limiter,
                    system=system,
                    messages=messages,
                    counted_input_tokens=counted_input_tokens,
                    max_tokens=MAX_TOKENS,
                    reserved_output_tokens=OBSERVED_P99_OUTPUT_TOKENS,
                    label=f"{label}-{task_number}",
                    refund_unused=refund_unused,
                )
                for task_number in range(1, CONCURRENT_CACHE_HITS + 1)
            )
        )
        wall_seconds = asyncio.get_running_loop().time() - started_at
        _LOGGER.info(
            "%s summary: completed=%d/%d wait-start=%d wait-end=%d "
            "total-wait=%.3fs wall=%.3fs",
            label,
            len(responses),
            CONCURRENT_CACHE_HITS,
            wait_starts,
            wait_ends,
            total_wait_seconds,
            wall_seconds,
        )
        return responses, wall_seconds

    # The SDK retries residual 429s twice by default and honors retry-after.
    # The limiter prevents ordinary quota overshoot; no second retry loop is
    # added here to compete with the SDK's backoff.
    async with AsyncAnthropic(max_retries=2) as client:
        try:
            prewarm_messages: list[MessageParam] = [
                {"role": "user", "content": "warm the shared cache"}
            ]
            prewarm_input_tokens = await _count_input_tokens(
                client=client,
                system=system,
                messages=prewarm_messages,
            )
            prewarm = await _limited_message(
                client=client,
                limiter=limiter,
                system=system,
                messages=prewarm_messages,
                counted_input_tokens=prewarm_input_tokens,
                max_tokens=PREWARM_MAX_TOKENS,
                reserved_output_tokens=PREWARM_RESERVED_OUTPUT_TOKENS,
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

            # Start both sides of the A/B from a full local output window.
            await asyncio.sleep(DEMO_OUTPUT_WINDOW_SECONDS)

            cache_hit_messages: list[MessageParam] = [
                {
                    "role": "user",
                    "content": "In one short sentence, why are refunds useful?",
                }
            ]
            # Every task below sends the same request shape, so one provider
            # count is an honest reservation basis for all four tasks. Counting
            # before gather also makes the limiter—not count-token latency—the
            # concurrency boundary being demonstrated.
            cache_hit_input_tokens = await _count_input_tokens(
                client=client,
                system=system,
                messages=cache_hit_messages,
            )
            _LOGGER.info(
                "Launching %d concurrent cache-hit tasks with refunds",
                CONCURRENT_CACHE_HITS,
            )
            with_refund_responses, with_refund_wall = await run_concurrent_workload(
                client=client,
                messages=cache_hit_messages,
                counted_input_tokens=cache_hit_input_tokens,
                label="with-refunds",
                refund_unused=True,
            )

            await asyncio.sleep(DEMO_OUTPUT_WINDOW_SECONDS)
            _LOGGER.info(
                "Launching the same %d tasks with successful reservations "
                "treated as fully consumed",
                CONCURRENT_CACHE_HITS,
            )
            (
                without_refund_responses,
                without_refund_wall,
            ) = await run_concurrent_workload(
                client=client,
                messages=cache_hit_messages,
                counted_input_tokens=cache_hit_input_tokens,
                label="without-refunds",
                refund_unused=False,
            )

            _validate_cache_hits(
                label="with-refunds",
                responses=with_refund_responses,
            )
            _validate_cache_hits(
                label="without-refunds",
                responses=without_refund_responses,
            )

            _LOGGER.info(
                "Measured A/B: with-refunds=%.3fs without-refunds=%.3fs "
                "recovered=%.3fs speedup=%.2fx",
                with_refund_wall,
                without_refund_wall,
                without_refund_wall - with_refund_wall,
                without_refund_wall / with_refund_wall,
            )
        finally:
            await limiter.aclose()


if __name__ == "__main__":
    asyncio.run(main())
