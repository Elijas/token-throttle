"""Tests for all ValueError paths in RateLimiter.acquire_capacity, acquire_capacity_for_request, and refund_capacity."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
from token_throttle._rate_limiter import RateLimiter

_UNLIMITED_FLAG = "__rate_limiting_disabled__"


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


def make_unlimited_config() -> PerModelConfig:
    return PerModelConfig(quotas=UsageQuotas.unlimited())


class TestAcquireCapacityValidation:
    """Tests for ValueError paths in acquire_capacity."""

    async def test_empty_model_name_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="model_name cannot be empty"):
            await limiter.acquire_capacity({"tokens": 1, "requests": 1}, model="")

    async def test_unlimited_config_with_nonempty_usage_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(
            ValueError, match="Usage must be empty for unlimited capacity"
        ):
            await limiter.acquire_capacity({"tokens": 5}, model="gpt-4")

    async def test_unlimited_config_with_empty_usage_returns_unlimited_reservation(
        self,
    ):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        reservation = await limiter.acquire_capacity({}, model="gpt-4")

        assert reservation.model_family == _UNLIMITED_FLAG
        assert dict(reservation.usage) == {}

    async def test_mismatched_usage_keys_vs_quota_keys_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="do not match quota keys"):
            await limiter.acquire_capacity({"tokens": 1}, model="gpt-4")

    async def test_negative_usage_value_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be non-negative"):
            await limiter.acquire_capacity({"tokens": -1, "requests": 1}, model="gpt-4")


class TestAcquireCapacityForRequestValidation:
    """Tests for ValueError paths in acquire_capacity_for_request."""

    async def test_missing_model_param_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="'model' parameter is required"):
            await limiter.acquire_capacity_for_request(extra_usage=None)

    async def test_none_usage_counter_raises(self):
        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=None)
        limiter = RateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="usage_counter cannot be None"):
            await limiter.acquire_capacity_for_request(
                extra_usage=None,
                model="gpt-4",
            )

    async def test_extra_usage_with_unknown_key_raises(self):
        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = RateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="Usage key 'unknown_metric' not found"):
            await limiter.acquire_capacity_for_request(
                extra_usage={"unknown_metric": 5},
                model="gpt-4",
            )


class TestRefundCapacityValidation:
    """Tests for ValueError paths in refund_capacity."""

    async def test_unlimited_reservation_with_nonempty_usage_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        reservation = CapacityReservation(usage={}, model_family=_UNLIMITED_FLAG)

        with pytest.raises(
            ValueError,
            match="Usage must be empty for unlimited capacity reservations",
        ):
            await limiter.refund_capacity({"tokens": 5}, reservation)

    async def test_unlimited_reservation_with_empty_usage_is_noop(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        reservation = CapacityReservation(usage={}, model_family=_UNLIMITED_FLAG)

        result = await limiter.refund_capacity({}, reservation)

        assert result is None

    async def test_mismatched_usage_keys_raises(self):
        builder, _mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        # First acquire to populate the backend cache
        await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        with pytest.raises(ValueError, match="do not match reservation usage keys"):
            await limiter.refund_capacity({"tokens": 50}, reservation)

    async def test_unrecognized_model_family_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="nonexistent-family",
        )

        with pytest.raises(ValueError, match="Backend not found for model family"):
            await limiter.refund_capacity(
                {"tokens": 50, "requests": 1},
                reservation,
            )


class TestRefundCapacityFromResponseValidation:
    """Tests for refund_capacity_from_response value paths."""

    async def test_response_with_none_usage_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        class FakeResponse:
            usage = None

        with pytest.raises(ValueError, match=r"response\.usage is None"):
            await limiter.refund_capacity_from_response(
                reservation, response=FakeResponse()
            )
