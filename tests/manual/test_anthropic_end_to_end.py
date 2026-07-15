"""Offline Anthropic example contracts plus an explicitly enabled live smoke test.

The offline tests use real Anthropic SDK response models and run in normal CI.
The live test requires both ``ANTHROPIC_API_KEY`` and
``TOKEN_THROTTLE_RUN_LIVE_ANTHROPIC=1`` in addition to the three real limit
environment variables consumed by the example.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
from anthropic.types import Message, Usage

import examples.anthropic_prompt_caching as example
from token_throttle import (
    MemoryBackendBuilder,
    PerModelConfig,
    Quota,
    RateLimiter,
    UsageQuotas,
)

_EXAMPLE_PATH = (
    Path(__file__).parent.parent.parent / "examples" / "anthropic_prompt_caching.py"
)


def _usage(
    *,
    input_tokens: int = 50,
    output_tokens: int = 12,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
) -> Usage:
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )


def _message(*, usage: Usage, stop_reason: str = "end_turn") -> Message:
    return Message(
        id="msg_test",
        content=[],
        model=example.MODEL,
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=usage,
    )


def _limiter() -> RateLimiter:
    return RateLimiter(
        PerModelConfig(
            model_family=example.MODEL_FAMILY,
            quotas=UsageQuotas(
                [
                    Quota(metric="requests", limit=100, per_seconds=60),
                    Quota(metric="input_tokens", limit=100_000, per_seconds=60),
                    Quota(metric="output_tokens", limit=10_000, per_seconds=60),
                ]
            ),
        ),
        backend=MemoryBackendBuilder(),
    )


class _TokenCount:
    input_tokens = 5_050


class _RawResponse:
    def __init__(self, message: Message) -> None:
        self._message = message
        self.headers = {
            "anthropic-ratelimit-input-tokens-remaining": "95000",
            "anthropic-ratelimit-output-tokens-remaining": "9000",
        }

    def parse(self) -> Message:
        return self._message


class _RawMessages:
    def __init__(
        self, *, message: Message | None = None, error: Exception | None = None
    ):
        self._message = message
        self._error = error

    async def create(self, **_kwargs) -> _RawResponse:
        if self._error is not None:
            raise self._error
        assert self._message is not None
        return _RawResponse(self._message)


class _Messages:
    def __init__(
        self, *, message: Message | None = None, error: Exception | None = None
    ):
        self.with_raw_response = _RawMessages(message=message, error=error)

    async def count_tokens(self, **_kwargs) -> _TokenCount:
        return _TokenCount()


class _Client:
    def __init__(
        self, *, message: Message | None = None, error: Exception | None = None
    ):
        self.messages = _Messages(message=message, error=error)


def test_usage_counter_maps_cache_creation_and_ignores_current_model_cache_reads():
    cache_miss = example.anthropic_usage_counter(
        **_usage(cache_creation_input_tokens=5_000).model_dump()
    )
    cache_hit = example.anthropic_usage_counter(
        **_usage(cache_read_input_tokens=5_000).model_dump()
    )

    assert cache_miss == {
        "requests": 1,
        "input_tokens": 5_050,
        "output_tokens": 12,
    }
    assert cache_hit == {
        "requests": 1,
        "input_tokens": 50,
        "output_tokens": 12,
    }


def test_usage_counter_can_model_legacy_haiku_35_cache_reads():
    usage = _usage(cache_read_input_tokens=5_000).model_dump()

    assert (
        example.anthropic_usage_counter(
            **usage,
            count_cache_reads_for_itpm=True,
        )["input_tokens"]
        == 5_050
    )


def test_remaining_token_headers_are_kept_separate():
    assert example.remaining_token_headers(
        {
            "anthropic-ratelimit-input-tokens-remaining": "12000",
            "anthropic-ratelimit-output-tokens-remaining": "3000",
        }
    ) == {"input_tokens": "12000", "output_tokens": "3000"}


@pytest.mark.asyncio
async def test_refusal_response_reconciles_actual_usage_and_closes_reservation():
    limiter = _limiter()
    client = _Client(
        message=_message(
            usage=_usage(cache_read_input_tokens=5_000, output_tokens=0),
            stop_reason="refusal",
        )
    )
    try:
        response = await example._limited_message(
            client=client,
            limiter=limiter,
            system=[],
            messages=[],
            max_tokens=512,
            reserved_output_tokens=128,
            label="refusal-test",
        )
        assert response.stop_reason == "refusal"
        assert limiter.snapshot_state()["in_flight_reservations"] == 0
    finally:
        await limiter.aclose()


@pytest.mark.asyncio
async def test_sdk_error_closes_reservation_conservatively():
    limiter = _limiter()
    client = _Client(error=RuntimeError("sdk failed"))
    try:
        with pytest.raises(RuntimeError, match="sdk failed"):
            await example._limited_message(
                client=client,
                limiter=limiter,
                system=[],
                messages=[],
                max_tokens=512,
                reserved_output_tokens=128,
                label="error-test",
            )
        assert limiter.snapshot_state()["in_flight_reservations"] == 0
    finally:
        await limiter.aclose()


def test_example_structure_is_anthropic_native():
    source = _EXAMPLE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "tik" + "token" not in source
    assert ".messages.count_tokens(" in source
    assert ".messages.with_raw_response.create(" in source
    assert "anthropic-ratelimit-input-tokens-remaining" in source
    assert "anthropic-ratelimit-output-tokens-remaining" in source
    assert any(isinstance(node, ast.Try) and node.finalbody for node in ast.walk(tree))
    assert not any(isinstance(node, ast.While) for node in ast.walk(tree))

    quota_metrics = {
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Quota"
        for keyword in node.keywords
        if keyword.arg == "metric" and isinstance(keyword.value, ast.Constant)
    }
    assert quota_metrics == {"requests", "input_tokens", "output_tokens"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "anthropic_usage_counter"
        ):
            assert not any(
                isinstance(name, ast.Name) and name.id == "MAX_TOKENS"
                for keyword in node.keywords
                for name in ast.walk(keyword.value)
            )


_live_enabled = bool(
    os.environ.get("ANTHROPIC_API_KEY")
    and os.environ.get("TOKEN_THROTTLE_RUN_LIVE_ANTHROPIC") == "1"
)


@pytest.mark.skipif(
    not _live_enabled,
    reason="set ANTHROPIC_API_KEY and TOKEN_THROTTLE_RUN_LIVE_ANTHROPIC=1",
)
@pytest.mark.asyncio
async def test_live_prompt_cache_reservation_and_refund():
    """Run the complete example; it asserts cache creation and the second hit."""
    await example.main()
