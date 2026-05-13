"""Regression tests for residual footgun fixes in FIX-19."""

import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

from token_throttle._factories._openai._model_family import openai_model_family_getter
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter
from token_throttle._validation import extract_total_tokens


def _limited_config(*, family: str = "openai-family") -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric="tokens", limit=1000),
                Quota(metric="requests", limit=10),
            ]
        ),
        model_family=family,
    )


def _async_builder():
    backend = AsyncMock()
    backend.await_for_capacity.return_value = None
    backend.refund_capacity.return_value = None
    backend.set_max_capacity.return_value = None
    builder = MagicMock()
    builder.build.return_value = backend
    return builder


def _sync_builder():
    backend = MagicMock()
    backend.wait_for_capacity.return_value = None
    backend.refund_capacity.return_value = None
    backend.set_max_capacity.return_value = None
    builder = MagicMock()
    builder.build.return_value = backend
    return builder


def _reservation() -> CapacityReservation:
    return CapacityReservation(
        usage={"tokens": 100, "requests": 1},
        model_family="openai-family",
        bucket_ids=frozenset({("tokens", 60), ("requests", 60)}),
        model="gpt-4o",
        limiter_instance_id="limiter",
    )


async def test_async_refund_capacity_detects_reversed_argument_order():
    limiter = RateLimiter(_limited_config(), backend=_async_builder())
    reservation = _reservation()

    with pytest.raises(
        TypeError,
        match=(
            r"refund_capacity expects \(actual_usage, reservation\); "
            "did you mean refund_capacity_from_response"
        ),
    ):
        await limiter.refund_capacity(reservation, {"tokens": 10, "requests": 1})


def test_sync_refund_capacity_detects_reversed_argument_order():
    limiter = SyncRateLimiter(_limited_config(), backend=_sync_builder())
    reservation = _reservation()

    with pytest.raises(
        TypeError,
        match=(
            r"refund_capacity expects \(actual_usage, reservation\); "
            "did you mean refund_capacity_from_response"
        ),
    ):
        limiter.refund_capacity(reservation, {"tokens": 10, "requests": 1})


async def test_async_set_max_capacity_detects_model_metric_swap():
    limiter = RateLimiter(_limited_config(), backend=_async_builder())
    await limiter.acquire_capacity({"tokens": 1, "requests": 1}, model="gpt-4o")

    with pytest.raises(
        TypeError,
        match=r"set_max_capacity expects \(model, metric, per_seconds, value\)",
    ):
        await limiter.set_max_capacity("tokens", "gpt-4o", 60, 5000)


def test_sync_set_max_capacity_detects_model_metric_swap():
    limiter = SyncRateLimiter(_limited_config(), backend=_sync_builder())
    limiter.acquire_capacity({"tokens": 1, "requests": 1}, model="gpt-4o")

    with pytest.raises(
        TypeError,
        match=r"set_max_capacity expects \(model, metric, per_seconds, value\)",
    ):
        limiter.set_max_capacity("tokens", "gpt-4o", 60, 5000)


def test_extract_total_tokens_recommends_manual_sum_for_partial_usage_object():
    class PartialUsage:
        prompt_tokens = 30
        completion_tokens = 20

    with pytest.raises(
        ValueError,
        match=(
            "usage object has prompt_tokens/completion_tokens but no total_tokens; "
            r"sum them manually and use refund_capacity\(\)"
        ),
    ):
        extract_total_tokens(PartialUsage())


@pytest.mark.parametrize(
    "model",
    ["ft::malformed", "ft:", "ft:gpt-4o:", "ft:gpt-4o::org", "openai/ft::malformed"],
)
def test_openai_model_family_getter_rejects_malformed_fine_tune_ids(model):
    with pytest.raises(ValueError, match="Malformed fine-tuned model id"):
        openai_model_family_getter(model)


def test_openai_model_family_getter_keeps_valid_fine_tune_base_family():
    assert openai_model_family_getter("ft:gpt-4o-2024-08-06:job") == "gpt-4o"


def test_metric_set_change_warning_names_backend_support_precondition():
    use_expanded = False

    def config_getter(_model_name: str) -> PerModelConfig:
        quotas = [Quota(metric="tokens", limit=100, per_seconds=60)]
        if use_expanded:
            quotas.append(Quota(metric="requests", limit=10, per_seconds=60))
        return PerModelConfig(
            quotas=UsageQuotas(quotas),
            model_family="openai-family",
        )

    limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())
    limiter.acquire_capacity({"tokens": 1}, model="gpt-4o")

    use_expanded = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        limiter.acquire_capacity({"tokens": 1, "requests": 1}, model="gpt-4o")

    assert any(
        "consumption state for surviving metrics will be transferred by "
        "backends that support it" in str(warning.message)
        for warning in caught
    )
