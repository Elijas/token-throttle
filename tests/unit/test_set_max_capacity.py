"""Tests for RateLimiter.set_max_capacity."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._rate_limiter import RateLimiter


def make_mock_backend_builder():
    """Create a mock backend builder that returns a mock backend."""
    mock_backend = AsyncMock()
    mock_backend.await_for_capacity.return_value = None
    mock_backend.refund_capacity.return_value = None
    mock_backend.set_max_capacity.return_value = None

    mock_builder = MagicMock()
    mock_builder.build.return_value = mock_backend
    return mock_builder, mock_backend


def make_limited_config(
    *,
    model_family: str | None = None,
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
    )


class TestSetMaxCapacity:
    """Tests for RateLimiter.set_max_capacity."""

    async def test_raises_before_any_traffic(self):
        builder, _ = make_mock_backend_builder()
        config = make_limited_config(model_family="gpt-4o")
        limiter = RateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="No backend for model family"):
            await limiter.set_max_capacity(
                model="gpt-4o",
                metric="tokens",
                per_seconds=60,
                value=5000,
            )

    async def test_delegates_to_backend_after_acquire(self):
        builder, mock_backend = make_mock_backend_builder()
        config = make_limited_config(model_family="gpt-4o")
        limiter = RateLimiter(config, backend=builder)

        await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4o",
        )

        await limiter.set_max_capacity(
            model="gpt-4o",
            metric="tokens",
            per_seconds=60,
            value=5000,
        )

        mock_backend.set_max_capacity.assert_awaited_once_with("tokens", 60, 5000)

    async def test_delegates_to_backend_after_record_usage(self):
        builder, mock_backend = make_mock_backend_builder()
        config = make_limited_config(model_family="gpt-4o")
        limiter = RateLimiter(config, backend=builder)

        await limiter.record_usage(
            {"tokens": 100, "requests": 1},
            model="gpt-4o",
        )

        await limiter.set_max_capacity(
            model="gpt-4o",
            metric="requests",
            per_seconds=60,
            value=200,
        )

        mock_backend.set_max_capacity.assert_awaited_once_with("requests", 60, 200)

    async def test_uses_model_family_from_config(self):
        builder, mock_backend = make_mock_backend_builder()

        def config_getter(_model_name: str) -> PerModelConfig:
            return make_limited_config(model_family="openai-tier")

        limiter = RateLimiter(config_getter, backend=builder)

        # Traffic under "gpt-4o" maps to family "openai-tier"
        await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4o",
        )

        # set_max_capacity for a different model that maps to the same family
        await limiter.set_max_capacity(
            model="gpt-4o-mini",
            metric="tokens",
            per_seconds=60,
            value=9000,
        )

        mock_backend.set_max_capacity.assert_awaited_once_with("tokens", 60, 9000)


class TestSetMaxCapacityValidation:
    """set_max_capacity should validate `value` at the public API boundary."""

    async def _make_limiter_with_backend(self):
        builder, _mock_backend = make_mock_backend_builder()
        config = make_limited_config(model_family="gpt-4o")
        limiter = RateLimiter(config, backend=builder)
        await limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4o")
        return limiter

    async def test_boolean_value_raises(self):
        limiter = await self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="max_capacity must not be a boolean"):
            await limiter.set_max_capacity("gpt-4o", "tokens", 60, True)  # noqa: FBT003

    async def test_nan_value_raises(self):
        limiter = await self._make_limiter_with_backend()
        with pytest.raises(
            ValueError, match="max_capacity must be finite and greater than 0"
        ):
            await limiter.set_max_capacity("gpt-4o", "tokens", 60, float("nan"))

    async def test_inf_value_raises(self):
        limiter = await self._make_limiter_with_backend()
        with pytest.raises(
            ValueError, match="max_capacity must be finite and greater than 0"
        ):
            await limiter.set_max_capacity("gpt-4o", "tokens", 60, float("inf"))

    async def test_negative_value_raises(self):
        limiter = await self._make_limiter_with_backend()
        with pytest.raises(
            ValueError, match="max_capacity must be finite and greater than 0"
        ):
            await limiter.set_max_capacity("gpt-4o", "tokens", 60, -5.0)

    async def test_zero_value_raises(self):
        limiter = await self._make_limiter_with_backend()
        with pytest.raises(
            ValueError, match="max_capacity must be finite and greater than 0"
        ):
            await limiter.set_max_capacity("gpt-4o", "tokens", 60, 0.0)

    @pytest.mark.parametrize("value", ["100", None, []])
    async def test_wrong_type_value_raises_type_specific_message(self, value):
        limiter = await self._make_limiter_with_backend()
        with pytest.raises(
            ValueError,
            match=rf"max_capacity must be an int or float \(got {type(value).__name__}\)",
        ):
            await limiter.set_max_capacity("gpt-4o", "tokens", 60, value)


class TestSetMaxCapacityMetricValidation:
    """set_max_capacity should validate `metric` at the public API boundary."""

    async def _make_limiter_with_backend(self):
        builder, _mock_backend = make_mock_backend_builder()
        config = make_limited_config(model_family="gpt-4o")
        limiter = RateLimiter(config, backend=builder)
        await limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4o")
        return limiter

    async def test_boolean_metric_raises(self):
        limiter = await self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="metric must be a str"):
            await limiter.set_max_capacity("gpt-4o", True, 60, 5000)  # noqa: FBT003

    async def test_non_string_metric_raises(self):
        limiter = await self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="metric must be a str"):
            await limiter.set_max_capacity("gpt-4o", 42, 60, 5000)

    async def test_empty_metric_raises(self):
        limiter = await self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="metric must not be empty"):
            await limiter.set_max_capacity("gpt-4o", "", 60, 5000)


class TestSetMaxCapacityPerSecondsValidation:
    """set_max_capacity should validate `per_seconds` at the public API boundary."""

    async def _make_limiter_with_backend(self):
        builder, _mock_backend = make_mock_backend_builder()
        config = make_limited_config(model_family="gpt-4o")
        limiter = RateLimiter(config, backend=builder)
        await limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4o")
        return limiter

    async def test_boolean_per_seconds_raises(self):
        limiter = await self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="per_seconds must not be a boolean"):
            await limiter.set_max_capacity("gpt-4o", "tokens", True, 5000)  # noqa: FBT003

    async def test_float_per_seconds_raises(self):
        limiter = await self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            await limiter.set_max_capacity("gpt-4o", "tokens", 60.5, 5000)

    async def test_string_per_seconds_raises(self):
        limiter = await self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            await limiter.set_max_capacity("gpt-4o", "tokens", "60", 5000)

    async def test_zero_per_seconds_raises(self):
        limiter = await self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            await limiter.set_max_capacity("gpt-4o", "tokens", 0, 5000)

    async def test_negative_per_seconds_raises(self):
        limiter = await self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            await limiter.set_max_capacity("gpt-4o", "tokens", -1, 5000)


class TestSetMaxCapacityCoercion:
    """Coerce validated values to float before passing to backend."""

    async def test_decimal_value_is_rejected(self):
        """Max-capacity validation accepts only builtin int/float values."""
        config = make_limited_config(model_family="test-model")
        limiter = RateLimiter(config, backend=MemoryBackendBuilder())

        await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1}, model="test-model"
        )

        with pytest.raises(ValueError, match="max_capacity must be an int or float"):
            await limiter.set_max_capacity("test-model", "tokens", 60, Decimal(5000))

    async def test_int_value_is_coerced_to_float(self):
        """Backend receives a float, not the raw int."""
        builder, mock_backend = make_mock_backend_builder()
        config = make_limited_config(model_family="gpt-4o")
        limiter = RateLimiter(config, backend=builder)

        await limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4o")

        await limiter.set_max_capacity("gpt-4o", "tokens", 60, 5000)

        # The backend should receive a float, not an int
        args = mock_backend.set_max_capacity.call_args[0]
        assert isinstance(args[2], float)
