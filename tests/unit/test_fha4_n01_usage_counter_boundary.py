"""Regression tests for FHA4-N01 usage-counter boundary hardening."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping

import pytest

from token_throttle._factories._openai._token_counter import OpenAIUsageCounter
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter


def _config(usage_counter) -> PerModelConfig:
    return PerModelConfig(
        model_family="family/gpt",
        quotas=UsageQuotas([Quota(metric="requests", limit=100)]),
        usage_counter=usage_counter,
    )


class _FixedSignatureCounter:
    def __init__(self) -> None:
        self.sync_calls: list[str] = []
        self.async_calls = 0

    def __call__(self, model: str):
        self.sync_calls.append(model)
        return {"requests": 1}

    async def count_request_async(self, model: str):
        self.async_calls += 1
        return {"requests": 99}


class _RaisingCounter:
    def __call__(self, **_kwargs):
        raise RuntimeError("sync boom")

    async def count_request_async(self, **_kwargs):
        raise RuntimeError("async boom")


class _ItemsBomb(Mapping[str, int]):
    def __iter__(self) -> Iterator[str]:
        return iter(["requests"])

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> int:
        if key == "requests":
            return 1
        raise KeyError(key)

    def items(self):
        raise RuntimeError("items boom")


class _IterBomb(Mapping[str, int]):
    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("iter boom")

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> int:
        if key == "requests":
            return 1
        raise KeyError(key)


class _DuplicateItems(Mapping[str, int]):
    def __iter__(self) -> Iterator[str]:
        return iter(["requests"])

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> int:
        if key == "requests":
            return 1
        raise KeyError(key)

    def items(self):
        return [("requests", 1), ("requests", 7)]


class _ExplodingFloat(float):
    def __float__(self):
        raise RuntimeError("float boom")


class _FakeEncoding:
    def encode(self, text: str, **_kwargs: object) -> list[int]:
        return list(range(len(text)))


async def test_async_exact_openai_counter_uses_internal_async_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_calls = 0
    async_calls = 0

    def fake_sync_call(self, model: str, **request):
        nonlocal sync_calls
        sync_calls += 1
        return {"requests": 99}

    async def fake_async_call(self, model: str, **request):
        nonlocal async_calls
        async_calls += 1
        return {"requests": 1}

    monkeypatch.setattr(OpenAIUsageCounter, "__call__", fake_sync_call)
    monkeypatch.setattr(OpenAIUsageCounter, "count_request_async", fake_async_call)
    counter = OpenAIUsageCounter(get_encoding_func=lambda _model: _FakeEncoding())
    limiter = RateLimiter(
        _config(counter),
        backend=MemoryBackendBuilder(),
    )

    try:
        with warnings.catch_warnings(record=True) as warning_record:
            warnings.simplefilter("always")
            reservation = await limiter.acquire_capacity_for_request(
                model="gpt",
                input="hi",
            )
    finally:
        await limiter.aclose()

    assert reservation.usage == {"requests": 1.0}
    assert sync_calls == 0
    assert async_calls == 1
    assert not any(
        "asyncio event loop" in str(warning.message) for warning in warning_record
    )


def test_sync_custom_count_request_async_uses_sync_signature_resolver() -> None:
    counter = _FixedSignatureCounter()
    limiter = SyncRateLimiter(
        _config(counter),
        backend=SyncMemoryBackendBuilder(),
    )

    with pytest.warns(UserWarning, match="without \\*\\*kwargs"):
        reservation = limiter.acquire_capacity_for_request(model="gpt", input="hi")

    assert reservation.usage == {"requests": 1.0}
    assert counter.sync_calls == ["gpt"]
    assert counter.async_calls == 0


async def test_async_custom_count_request_async_uses_sync_signature_resolver() -> None:
    counter = _FixedSignatureCounter()
    limiter = RateLimiter(
        _config(counter),
        backend=MemoryBackendBuilder(),
    )

    try:
        with warnings.catch_warnings(record=True) as warning_record:
            warnings.simplefilter("always")
            reservation = await limiter.acquire_capacity_for_request(
                model="gpt",
                input="hi",
            )
    finally:
        await limiter.aclose()

    warning_messages = [str(warning.message) for warning in warning_record]
    assert any("without **kwargs" in message for message in warning_messages)
    assert any("asyncio event loop" in message for message in warning_messages)
    assert reservation.usage == {"requests": 1.0}
    assert counter.sync_calls == ["gpt"]
    assert counter.async_calls == 0


def test_sync_custom_counter_exceptions_are_wrapped() -> None:
    limiter = SyncRateLimiter(
        _config(_RaisingCounter()),
        backend=SyncMemoryBackendBuilder(),
    )

    with pytest.raises(
        ValueError,
        match="usage_counter raised: RuntimeError: sync boom",
    ) as exc_info:
        limiter.acquire_capacity_for_request(model="gpt")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


async def test_async_custom_counter_exceptions_are_wrapped_by_sync_resolver() -> None:
    limiter = RateLimiter(
        _config(_RaisingCounter()),
        backend=MemoryBackendBuilder(),
    )

    try:
        with (
            pytest.warns(UserWarning, match="asyncio event loop"),
            pytest.raises(
                ValueError,
                match="usage_counter raised: RuntimeError: sync boom",
            ) as exc_info,
        ):
            await limiter.acquire_capacity_for_request(model="gpt")
    finally:
        await limiter.aclose()

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.parametrize("mapping_factory", [_ItemsBomb, _IterBomb])
def test_sync_counter_mapping_protocol_failures_are_value_errors(
    mapping_factory,
) -> None:
    def counter(**_kwargs):
        return mapping_factory()

    limiter = SyncRateLimiter(
        _config(counter),
        backend=SyncMemoryBackendBuilder(),
    )

    with pytest.raises(ValueError, match="usage must yield consistent"):
        limiter.acquire_capacity_for_request(model="gpt")


@pytest.mark.parametrize("mapping_factory", [_ItemsBomb, _IterBomb])
async def test_async_counter_mapping_protocol_failures_are_value_errors(
    mapping_factory,
) -> None:
    def counter(**_kwargs):
        return mapping_factory()

    limiter = RateLimiter(
        _config(counter),
        backend=MemoryBackendBuilder(),
    )

    try:
        with (
            pytest.warns(UserWarning, match="asyncio event loop"),
            pytest.raises(ValueError, match="usage must yield consistent"),
        ):
            await limiter.acquire_capacity_for_request(model="gpt")
    finally:
        await limiter.aclose()


def test_counter_output_rejects_duplicate_mapping_items() -> None:
    with pytest.raises(ValueError, match="duplicate metric keys"):
        frozen_usage(_DuplicateItems())


def test_counter_output_numeric_coercion_exceptions_are_value_errors() -> None:
    with pytest.raises(
        ValueError, match="Usage value for requests must be finite"
    ) as exc_info:
        frozen_usage({"requests": _ExplodingFloat(1.0)})

    assert isinstance(exc_info.value.__cause__, RuntimeError)
