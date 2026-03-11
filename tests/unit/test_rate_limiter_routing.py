"""Tests for backend caching and model resolution in RateLimiter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._rate_limiter import RateLimiter


def make_mock_backend_builder():
    """Create a mock backend builder that returns a mock backend."""
    mock_backend = AsyncMock()
    mock_backend.await_for_capacity.return_value = None
    mock_backend.refund_capacity.return_value = None

    mock_builder = MagicMock()
    mock_builder.build.return_value = mock_backend
    return mock_builder, mock_backend


def make_limited_config(
    *,
    model_family: str | None = None,
    usage_counter=None,
) -> PerModelConfig:
    """Create a PerModelConfig with tokens and requests quotas."""
    quotas = UsageQuotas(
        [
            Quota(metric="tokens", limit=1000),
            Quota(metric="requests", limit=10),
        ]
    )
    return PerModelConfig(
        quotas=quotas,
        model_family=model_family,
        usage_counter=usage_counter,
    )


class TestStaticVsCallableConfig:
    """Tests for static PerModelConfig and callable PerModelConfigGetter."""

    async def test_static_per_model_config_works(self):
        builder, mock_backend = make_mock_backend_builder()
        config = make_limited_config()
        limiter = RateLimiter(config, backend=builder)

        reservation = await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        assert reservation.model_family == "gpt-4"
        mock_backend.await_for_capacity.assert_awaited_once()

    async def test_callable_per_model_config_getter_works(self):
        builder, mock_backend = make_mock_backend_builder()

        def config_getter(model_name: str) -> PerModelConfig:
            return make_limited_config(model_family=f"family-{model_name}")

        limiter = RateLimiter(config_getter, backend=builder)

        reservation = await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        assert reservation.model_family == "family-gpt-4"
        mock_backend.await_for_capacity.assert_awaited_once()


class TestModelFamilyResolution:
    """Tests for model_family defaulting and preservation."""

    async def test_model_family_defaults_to_model_name_when_none(self):
        builder, _ = make_mock_backend_builder()
        config = make_limited_config(model_family=None)
        limiter = RateLimiter(config, backend=builder)

        reservation = await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4o",
        )

        assert reservation.model_family == "gpt-4o"

    async def test_model_family_preserved_when_explicitly_set(self):
        builder, _ = make_mock_backend_builder()
        config = make_limited_config(model_family="openai-tier")
        limiter = RateLimiter(config, backend=builder)

        reservation = await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4o",
        )

        assert reservation.model_family == "openai-tier"


class TestBackendCaching:
    """Tests for backend caching: same model_family reuses, different creates new."""

    async def test_same_model_family_reuses_cached_backend(self):
        builder, mock_backend = make_mock_backend_builder()
        config = make_limited_config(model_family="shared-family")
        limiter = RateLimiter(config, backend=builder)

        await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )
        await limiter.acquire_capacity(
            {"tokens": 200, "requests": 1},
            model="gpt-4",
        )

        builder.build.assert_called_once()
        assert mock_backend.await_for_capacity.await_count == 2

    async def test_different_model_family_creates_separate_backend(self):
        builder = MagicMock()
        backend_a = AsyncMock()
        backend_a.await_for_capacity.return_value = None
        backend_b = AsyncMock()
        backend_b.await_for_capacity.return_value = None
        builder.build.side_effect = [backend_a, backend_b]

        def config_getter(model_name: str) -> PerModelConfig:
            return make_limited_config(model_family=model_name)

        limiter = RateLimiter(config_getter, backend=builder)

        reservation_a = await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )
        reservation_b = await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="claude-3",
        )

        assert builder.build.call_count == 2
        assert reservation_a.model_family == "gpt-4"
        assert reservation_b.model_family == "claude-3"
        backend_a.await_for_capacity.assert_awaited_once()
        backend_b.await_for_capacity.assert_awaited_once()


class TestAcquireCapacityForRequestMerge:
    """Tests for acquire_capacity_for_request merging usage_counter + extra_usage."""

    async def test_merges_usage_counter_and_extra_usage(self):
        builder, mock_backend = make_mock_backend_builder()

        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        config = make_limited_config(usage_counter=fake_counter)
        limiter = RateLimiter(config, backend=builder)

        reservation = await limiter.acquire_capacity_for_request(
            extra_usage={"tokens": 50, "requests": 2},
            model="gpt-4",
        )

        assert reservation.model_family == "gpt-4"
        # The merged usage should be tokens=150, requests=3
        called_usage = mock_backend.await_for_capacity.call_args[0][0]
        assert float(called_usage["tokens"]) == pytest.approx(150.0)
        assert float(called_usage["requests"]) == pytest.approx(3.0)

    async def test_no_extra_usage_uses_counter_only(self):
        builder, mock_backend = make_mock_backend_builder()

        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        config = make_limited_config(usage_counter=fake_counter)
        limiter = RateLimiter(config, backend=builder)

        reservation = await limiter.acquire_capacity_for_request(model="gpt-4")

        assert reservation.model_family == "gpt-4"
        called_usage = mock_backend.await_for_capacity.call_args[0][0]
        assert float(called_usage["tokens"]) == pytest.approx(100.0)
        assert float(called_usage["requests"]) == pytest.approx(1.0)
