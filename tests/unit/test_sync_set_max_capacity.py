"""Tests for SyncRateLimiter.set_max_capacity."""

from unittest.mock import MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._sync_rate_limiter import SyncRateLimiter


def make_mock_backend_builder():
    """Create a mock backend builder that returns a mock backend."""
    mock_backend = MagicMock()
    mock_backend.wait_for_capacity.return_value = None
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


class TestSyncSetMaxCapacity:
    """Tests for SyncRateLimiter.set_max_capacity."""

    def test_raises_before_any_traffic(self):
        builder, _ = make_mock_backend_builder()
        config = make_limited_config(model_family="gpt-4o")
        limiter = SyncRateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="No backend for model family"):
            limiter.set_max_capacity(
                model="gpt-4o",
                metric="tokens",
                per_seconds=60,
                value=5000,
            )

    def test_delegates_to_backend_after_acquire(self):
        builder, mock_backend = make_mock_backend_builder()
        config = make_limited_config(model_family="gpt-4o")
        limiter = SyncRateLimiter(config, backend=builder)

        limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4o",
        )

        limiter.set_max_capacity(
            model="gpt-4o",
            metric="tokens",
            per_seconds=60,
            value=5000,
        )

        mock_backend.set_max_capacity.assert_called_once_with("tokens", 60, 5000)

    def test_delegates_to_backend_after_record_usage(self):
        builder, mock_backend = make_mock_backend_builder()
        config = make_limited_config(model_family="gpt-4o")
        limiter = SyncRateLimiter(config, backend=builder)

        limiter.record_usage(
            {"tokens": 100, "requests": 1},
            model="gpt-4o",
        )

        limiter.set_max_capacity(
            model="gpt-4o",
            metric="requests",
            per_seconds=60,
            value=200,
        )

        mock_backend.set_max_capacity.assert_called_once_with("requests", 60, 200)

    def test_uses_model_family_from_config(self):
        builder, mock_backend = make_mock_backend_builder()

        def config_getter(_model_name: str) -> PerModelConfig:
            return make_limited_config(model_family="openai-tier")

        limiter = SyncRateLimiter(config_getter, backend=builder)

        # Traffic under "gpt-4o" maps to family "openai-tier"
        limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4o",
        )

        # set_max_capacity for a different model that maps to the same family
        limiter.set_max_capacity(
            model="gpt-4o-mini",
            metric="tokens",
            per_seconds=60,
            value=9000,
        )

        mock_backend.set_max_capacity.assert_called_once_with("tokens", 60, 9000)
