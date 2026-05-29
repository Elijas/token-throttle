"""Regression tests for FHA5-N01 usage validation consistency."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter
from token_throttle._validation import validate_extra_usage


class _NormalizedDuplicateExtraUsage(Mapping[str, int]):
    raw_a = "e\u0301"
    raw_b = "\u00e9"

    def __iter__(self) -> Iterator[str]:
        return iter([self.raw_a, self.raw_b])

    def __len__(self) -> int:
        return 2

    def __getitem__(self, key: str) -> int:
        if key == self.raw_a:
            return 1
        if key == self.raw_b:
            return 2
        raise KeyError(key)

    def items(self):
        return [(self.raw_a, 1), (self.raw_b, 2)]


def _unlimited_config(*, usage_counter=None) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas.unlimited(),
        usage_counter=usage_counter,
    )


def _async_backend_builder():
    return MemoryBackendBuilder()


def _sync_backend_builder():
    return SyncMemoryBackendBuilder()


def test_validate_extra_usage_rejects_normalized_duplicate_metric_keys() -> None:
    with pytest.raises(ValueError, match="duplicate metric keys"):
        validate_extra_usage(_NormalizedDuplicateExtraUsage())


def test_sync_unlimited_extra_usage_rejects_normalized_duplicate_metric_keys() -> None:
    limiter = SyncRateLimiter(_unlimited_config(), backend=_sync_backend_builder())

    with pytest.raises(ValueError, match="duplicate metric keys"):
        limiter.acquire_capacity_for_request(
            model="gpt-4",
            extra_usage=_NormalizedDuplicateExtraUsage(),
        )


async def test_async_unlimited_extra_usage_rejects_normalized_duplicates() -> None:
    limiter = RateLimiter(_unlimited_config(), backend=_async_backend_builder())

    try:
        with pytest.raises(ValueError, match="duplicate metric keys"):
            await limiter.acquire_capacity_for_request(
                model="gpt-4",
                extra_usage=_NormalizedDuplicateExtraUsage(),
            )
    finally:
        await limiter.aclose()


@pytest.mark.parametrize("method_name", ["acquire_capacity", "record_usage"])
def test_sync_unlimited_direct_usage_rejects_negative_values(method_name: str) -> None:
    limiter = SyncRateLimiter(_unlimited_config(), backend=_sync_backend_builder())

    with pytest.raises(ValueError, match="Usage value for tokens must be non-negative"):
        getattr(limiter, method_name)({"tokens": -1}, model="gpt-4")


@pytest.mark.parametrize("method_name", ["acquire_capacity", "record_usage"])
async def test_async_unlimited_direct_usage_rejects_negative_values(
    method_name: str,
) -> None:
    limiter = RateLimiter(_unlimited_config(), backend=_async_backend_builder())

    try:
        with pytest.raises(
            ValueError,
            match="Usage value for tokens must be non-negative",
        ):
            await getattr(limiter, method_name)({"tokens": -1}, model="gpt-4")
    finally:
        await limiter.aclose()


def test_sync_unlimited_request_counter_output_rejects_negative_values() -> None:
    def counter(**_kwargs):
        return {"requests": -1}

    limiter = SyncRateLimiter(
        _unlimited_config(usage_counter=counter),
        backend=_sync_backend_builder(),
    )

    with pytest.raises(
        ValueError, match="Usage value for requests must be non-negative"
    ):
        limiter.acquire_capacity_for_request(model="gpt-4")


async def test_async_unlimited_request_counter_output_rejects_negative_values() -> None:
    def counter(**_kwargs):
        return {"requests": -1}

    limiter = RateLimiter(
        _unlimited_config(usage_counter=counter),
        backend=_async_backend_builder(),
    )

    try:
        with warnings.catch_warnings(record=True) as warning_record:
            warnings.simplefilter("always")
            with pytest.raises(
                ValueError,
                match="Usage value for requests must be non-negative",
            ):
                await limiter.acquire_capacity_for_request(model="gpt-4")
    finally:
        await limiter.aclose()

    assert any(
        "asyncio event loop" in str(warning.message) for warning in warning_record
    )
