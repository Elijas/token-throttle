"""Tests for all ValueError paths in SyncRateLimiter.acquire_capacity and refund_capacity."""

from unittest.mock import MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
from token_throttle._sync_rate_limiter import SyncRateLimiter

_UNLIMITED_FLAG = "__rate_limiting_disabled__"


def make_mock_backend_builder():
    """Create a mock backend builder that returns a mock backend."""
    mock_backend = MagicMock()
    mock_backend.wait_for_capacity.return_value = None
    mock_backend.refund_capacity.return_value = None

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


def make_unlimited_config() -> PerModelConfig:
    return PerModelConfig(quotas=UsageQuotas.unlimited())


class TestAcquireCapacityValidation:
    """Tests for ValueError paths in acquire_capacity."""

    def test_empty_model_name_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="model_name cannot be empty"):
            limiter.acquire_capacity({"tokens": 1, "requests": 1}, model="")

    def test_unlimited_config_with_nonempty_usage_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(
            ValueError, match="Usage must be empty for unlimited capacity"
        ):
            limiter.acquire_capacity({"tokens": 5}, model="gpt-4")

    def test_unlimited_config_with_empty_usage_returns_unlimited_reservation(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        reservation = limiter.acquire_capacity({}, model="gpt-4")

        assert reservation.model_family == _UNLIMITED_FLAG
        assert dict(reservation.usage) == {}

    def test_mismatched_usage_keys_vs_quota_keys_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="do not match quota keys"):
            limiter.acquire_capacity({"tokens": 1}, model="gpt-4")

    def test_negative_usage_value_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be non-negative"):
            limiter.acquire_capacity({"tokens": -1, "requests": 1}, model="gpt-4")

    def test_usage_exceeding_quota_limit_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="exceeds the limit"):
            limiter.acquire_capacity({"tokens": 9999, "requests": 1}, model="gpt-4")


class TestRefundCapacityValidation:
    """Tests for ValueError paths in refund_capacity."""

    def test_unlimited_reservation_with_nonempty_usage_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        reservation = CapacityReservation(usage={}, model_family=_UNLIMITED_FLAG)

        with pytest.raises(
            ValueError,
            match="Usage must be empty for unlimited capacity reservations",
        ):
            limiter.refund_capacity({"tokens": 5}, reservation)

    def test_unlimited_reservation_with_empty_usage_is_noop(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        reservation = CapacityReservation(usage={}, model_family=_UNLIMITED_FLAG)

        result = limiter.refund_capacity({}, reservation)

        assert result is None

    def test_mismatched_usage_keys_raises(self):
        builder, _mock_backend = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        # First acquire to populate the backend cache
        limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        with pytest.raises(ValueError, match="do not match reservation usage keys"):
            limiter.refund_capacity({"tokens": 50}, reservation)

    def test_unrecognized_model_family_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="nonexistent-family",
        )

        with pytest.raises(ValueError, match="Backend not found for model family"):
            limiter.refund_capacity(
                {"tokens": 50, "requests": 1},
                reservation,
            )
