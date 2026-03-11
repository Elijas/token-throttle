"""Tests for SyncRateLimiter.record_usage — sync non-blocking metering."""

from unittest.mock import MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._sync_rate_limiter import SyncRateLimiter

_UNLIMITED_FLAG = "__rate_limiting_disabled__"


def make_mock_backend_builder():
    """Create a mock backend builder that returns a mock backend."""
    mock_backend = MagicMock()
    mock_backend.wait_for_capacity.return_value = None
    mock_backend.consume_capacity.return_value = None
    mock_backend.refund_capacity.return_value = None

    mock_builder = MagicMock()
    mock_builder.build.return_value = mock_backend
    return mock_builder, mock_backend


def make_limited_config(
    *,
    model_family: str | None = None,
) -> PerModelConfig:
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


class TestRecordUsageCallsConsumeCapacity:
    """record_usage delegates to backend.consume_capacity (not wait_for_capacity)."""

    def test_record_usage_calls_consume_capacity(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        reservation = limiter.record_usage(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        mock_backend.consume_capacity.assert_called_once()
        mock_backend.wait_for_capacity.assert_not_called()
        assert reservation.model_family == "gpt-4"

    def test_record_usage_returns_reservation_with_correct_usage(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        reservation = limiter.record_usage(
            {"tokens": 500, "requests": 3},
            model="gpt-4",
        )

        assert float(reservation.usage["tokens"]) == 500.0
        assert float(reservation.usage["requests"]) == 3.0


class TestRecordUsageValidation:
    """record_usage applies the same validation as acquire_capacity."""

    def test_empty_model_name_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="model_name cannot be empty"):
            limiter.record_usage({"tokens": 1, "requests": 1}, model="")

    def test_mismatched_usage_keys_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="do not match quota keys"):
            limiter.record_usage({"tokens": 1}, model="gpt-4")

    def test_negative_usage_value_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be non-negative"):
            limiter.record_usage({"tokens": -1, "requests": 1}, model="gpt-4")


class TestRecordUsageUnlimited:
    """record_usage with unlimited config."""

    def test_unlimited_with_empty_usage_returns_unlimited_reservation(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        reservation = limiter.record_usage({}, model="gpt-4")

        assert reservation.model_family == _UNLIMITED_FLAG
        assert dict(reservation.usage) == {}

    def test_unlimited_with_nonempty_usage_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(
            ValueError,
            match="Usage must be empty for unlimited capacity",
        ):
            limiter.record_usage({"tokens": 5}, model="gpt-4")
