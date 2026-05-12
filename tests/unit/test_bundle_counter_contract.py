"""Regression tests for FIX-11 usage_counter and OpenAI counter hardening."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from token_throttle._factories._openai._token_counter import OpenAIUsageCounter
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._rate_limiter import RateLimiter
from token_throttle._validation import resolve_usage_counter_result


def _make_config(usage_counter=None) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric="tokens", limit=1000),
                Quota(metric="requests", limit=10),
            ]
        ),
        model_family="test",
        usage_counter=usage_counter,
    )


def _make_backend_builder():
    backend = AsyncMock()
    backend.await_for_capacity.return_value = None
    builder = MagicMock()
    builder.build.return_value = backend
    return builder


def _make_counter() -> OpenAIUsageCounter:
    encoding = MagicMock()
    encoding.encode.side_effect = lambda text: list(range(len(text)))
    return OpenAIUsageCounter(get_encoding_func=lambda _model: encoding)


def test_u01_async_generator_usage_counter_rejected_at_config_time():
    async def counter(**_request):
        yield {"tokens": 1, "requests": 1}

    with pytest.raises(ValidationError, match="synchronous callable"):
        _make_config(usage_counter=counter)


def test_u02_fixed_signature_counter_warns_before_filtering_request_kwargs():
    seen = {}

    def counter(model):
        seen["model"] = model
        seen["temperature"] = "not passed"
        return {"tokens": 1, "requests": 1}

    with pytest.warns(UserWarning, match="without \\*\\*kwargs"):
        usage = resolve_usage_counter_result(
            counter,
            model="gpt-4",
            temperature=0.7,
        )

    assert usage["tokens"] == 1
    assert seen == {"model": "gpt-4", "temperature": "not passed"}


def test_u03_counter_exception_is_wrapped_with_cause():
    def counter(**_request):
        raise KeyError("tokens")

    with pytest.raises(ValueError, match="usage_counter raised: KeyError") as exc_info:
        resolve_usage_counter_result(counter, model="gpt-4")

    assert isinstance(exc_info.value.__cause__, KeyError)


async def test_u04_async_dispatch_warns_for_inline_sync_counter():
    def counter(**_request):
        return {"tokens": 1, "requests": 1}

    limiter = RateLimiter(
        _make_config(usage_counter=counter), backend=_make_backend_builder()
    )

    with pytest.warns(UserWarning, match="asyncio event loop"):
        await limiter.acquire_capacity_for_request(model="gpt-4")


def test_u07_openai_counter_rejects_non_string_model():
    counter = _make_counter()

    with pytest.raises(TypeError, match="model must be a non-empty string"):
        counter(42, input="hi")


def test_u08_openai_counter_rejects_non_string_message_content():
    counter = _make_counter()

    with pytest.raises(ValueError, match="All keys and values in messages"):
        counter("gpt-4", messages=[{"role": "user", "content": 123}])


def test_f01_openai_counter_multiplies_output_budget_by_n_and_best_of():
    counter = _make_counter()

    n_result = counter("gpt-4", input="hi", max_tokens=10, n=3)
    best_of_result = counter("gpt-4", input="hi", max_tokens=10, best_of=4)

    assert n_result["tokens"] == 32
    assert best_of_result["tokens"] == 42
    with pytest.raises(ValueError, match="'n' must be"):
        counter("gpt-4", input="hi", max_tokens=10, n=True)
    with pytest.raises(ValueError, match="'best_of' must be"):
        counter("gpt-4", input="hi", max_tokens=10, best_of=-1)


def test_f38_openai_counter_rejects_unknown_max_budget_kwargs():
    counter = _make_counter()

    with pytest.raises(ValueError, match="Unknown OpenAI max_\\* token budget"):
        counter("gpt-4", input="hi", max_token=10)
