"""Regression tests for FIX-18 extra_usage validation hardening."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter
from token_throttle._validation import validate_extra_usage


def _limited_config(*, usage_counter=None) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric="tokens", limit=1000),
                Quota(metric="requests", limit=10),
            ]
        ),
        usage_counter=usage_counter,
    )


def _async_backend_builder():
    backend = AsyncMock()
    backend.await_for_capacity.return_value = None
    builder = MagicMock()
    builder.build.return_value = backend
    return builder


def _sync_backend_builder():
    backend = MagicMock()
    backend.wait_for_capacity.return_value = None
    builder = MagicMock()
    builder.build.return_value = backend
    return builder


class _DuckFloat:
    def __float__(self) -> float:
        return 1.0


class _DuplicateKeyMapping(Mapping[str, int]):
    def __iter__(self) -> Iterator[str]:
        return iter(["tokens", "tokens"])

    def __len__(self) -> int:
        return 2

    def __getitem__(self, key: str) -> int:
        if key == "tokens":
            return 1
        raise KeyError(key)


class _DuplicateItemsMapping(Mapping[str, int]):
    def __iter__(self) -> Iterator[str]:
        return iter(["tokens"])

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> int:
        if key == "tokens":
            return 1
        raise KeyError(key)

    def items(self):
        return [("tokens", 1), ("tokens", 9)]


@pytest.mark.parametrize(
    "raw_amount",
    [
        "5",
        b"5",
        bytearray(b"5"),
        Decimal(5),
        _DuckFloat(),
    ],
)
def test_extra_usage_rejects_float_coercion_inputs(raw_amount):
    with pytest.raises(ValueError, match="int or float"):
        validate_extra_usage({"tokens": raw_amount})


def test_counter_output_rejects_float_coercion_inputs_from_shared_chokepoint():
    with pytest.raises(ValueError, match="int or float"):
        frozen_usage({"tokens": "5"})


def test_validate_extra_usage_rejects_duplicate_mapping_keys():
    with pytest.raises(ValueError, match="duplicate keys"):
        validate_extra_usage(_DuplicateKeyMapping())


def test_validate_extra_usage_rejects_duplicate_mapping_items():
    with pytest.raises(ValueError, match="duplicate metric keys"):
        validate_extra_usage(_DuplicateItemsMapping())


def test_extra_usage_unknown_key_message_names_extra_usage_key():
    def counter(**_kwargs):
        return {"tokens": 100.0, "requests": 1.0}

    limiter = SyncRateLimiter(
        _limited_config(usage_counter=counter),
        backend=_sync_backend_builder(),
    )

    with pytest.raises(
        ValueError,
        match="extra_usage key 'image_tokens' is not in counter output",
    ) as exc_info:
        limiter.acquire_capacity_for_request(
            model="gpt-4",
            extra_usage={"image_tokens": 3},
        )

    assert str(exc_info.value) == (
        "extra_usage key 'image_tokens' is not in counter output - "
        "to add custom metrics, ensure the counter emits the key first."
    )


@pytest.mark.parametrize(
    ("bad_extra_usage", "match"),
    [({"tokens": "5"}, "int or float"), ({"tokens": -1}, "non-negative")],
)
async def test_async_extra_usage_values_are_validated_before_counter(
    bad_extra_usage, match
):
    calls = 0

    def counter(**_kwargs):
        nonlocal calls
        calls += 1
        return {"tokens": 100.0, "requests": 1.0}

    limiter = RateLimiter(
        _limited_config(usage_counter=counter),
        backend=_async_backend_builder(),
    )

    with pytest.raises(ValueError, match=match):
        await limiter.acquire_capacity_for_request(
            model="gpt-4",
            extra_usage=bad_extra_usage,
        )

    assert calls == 0


@pytest.mark.parametrize(
    ("bad_extra_usage", "match"),
    [({"tokens": "5"}, "int or float"), ({"tokens": -1}, "non-negative")],
)
def test_sync_extra_usage_values_are_validated_before_counter(bad_extra_usage, match):
    calls = 0

    def counter(**_kwargs):
        nonlocal calls
        calls += 1
        return {"tokens": 100.0, "requests": 1.0}

    limiter = SyncRateLimiter(
        _limited_config(usage_counter=counter),
        backend=_sync_backend_builder(),
    )

    with pytest.raises(ValueError, match=match):
        limiter.acquire_capacity_for_request(
            model="gpt-4",
            extra_usage=bad_extra_usage,
        )

    assert calls == 0


def test_extra_usage_huge_integer_raises_clear_value_error():
    with pytest.raises(ValueError, match="too large to fit in IEEE 754 double"):
        validate_extra_usage({"tokens": 10**400})


def test_acquire_capacity_for_request_docstring_documents_extra_usage_contract():
    docstring = RateLimiter.acquire_capacity_for_request.__doc__

    assert docstring is not None
    assert "usage_counter" in docstring
    assert "extra_usage" in docstring
    assert "added to the counter output" in docstring
    assert "Raises" in docstring
