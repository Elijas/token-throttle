"""Tests for SyncRateLimiter.set_max_capacity."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
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


class TestSyncSetMaxCapacityValidation:
    """set_max_capacity should validate `value` at the public API boundary."""

    def _make_limiter_with_backend(self):
        builder, _mock_backend = make_mock_backend_builder()
        config = make_limited_config(model_family="gpt-4o")
        limiter = SyncRateLimiter(config, backend=builder)
        limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4o")
        return limiter

    def test_boolean_value_raises(self):
        limiter = self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="max_capacity must not be a boolean"):
            limiter.set_max_capacity("gpt-4o", "tokens", 60, True)  # noqa: FBT003

    def test_nan_value_raises(self):
        limiter = self._make_limiter_with_backend()
        with pytest.raises(
            ValueError, match="max_capacity must be finite and greater than 0"
        ):
            limiter.set_max_capacity("gpt-4o", "tokens", 60, float("nan"))

    def test_inf_value_raises(self):
        limiter = self._make_limiter_with_backend()
        with pytest.raises(
            ValueError, match="max_capacity must be finite and greater than 0"
        ):
            limiter.set_max_capacity("gpt-4o", "tokens", 60, float("inf"))

    def test_negative_value_raises(self):
        limiter = self._make_limiter_with_backend()
        with pytest.raises(
            ValueError, match="max_capacity must be finite and greater than 0"
        ):
            limiter.set_max_capacity("gpt-4o", "tokens", 60, -5.0)

    def test_zero_value_raises(self):
        limiter = self._make_limiter_with_backend()
        with pytest.raises(
            ValueError, match="max_capacity must be finite and greater than 0"
        ):
            limiter.set_max_capacity("gpt-4o", "tokens", 60, 0.0)


class TestSyncSetMaxCapacityMetricValidation:
    """set_max_capacity should validate `metric` at the public API boundary."""

    def _make_limiter_with_backend(self):
        builder, _mock_backend = make_mock_backend_builder()
        config = make_limited_config(model_family="gpt-4o")
        limiter = SyncRateLimiter(config, backend=builder)
        limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4o")
        return limiter

    def test_boolean_metric_raises(self):
        limiter = self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="metric must be a str"):
            limiter.set_max_capacity("gpt-4o", True, 60, 5000)  # noqa: FBT003

    def test_non_string_metric_raises(self):
        limiter = self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="metric must be a str"):
            limiter.set_max_capacity("gpt-4o", 42, 60, 5000)

    def test_empty_metric_raises(self):
        limiter = self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="metric must not be empty"):
            limiter.set_max_capacity("gpt-4o", "", 60, 5000)


class TestSyncSetMaxCapacityPerSecondsValidation:
    """set_max_capacity should validate `per_seconds` at the public API boundary."""

    def _make_limiter_with_backend(self):
        builder, _mock_backend = make_mock_backend_builder()
        config = make_limited_config(model_family="gpt-4o")
        limiter = SyncRateLimiter(config, backend=builder)
        limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4o")
        return limiter

    def test_boolean_per_seconds_raises(self):
        limiter = self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="per_seconds must not be a boolean"):
            limiter.set_max_capacity("gpt-4o", "tokens", True, 5000)  # noqa: FBT003

    def test_float_per_seconds_raises(self):
        limiter = self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            limiter.set_max_capacity("gpt-4o", "tokens", 60.5, 5000)

    def test_string_per_seconds_raises(self):
        limiter = self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            limiter.set_max_capacity("gpt-4o", "tokens", "60", 5000)

    def test_zero_per_seconds_raises(self):
        limiter = self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            limiter.set_max_capacity("gpt-4o", "tokens", 0, 5000)

    def test_negative_per_seconds_raises(self):
        limiter = self._make_limiter_with_backend()
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            limiter.set_max_capacity("gpt-4o", "tokens", -1, 5000)


class TestSyncSetMaxCapacityCoercion:
    """Coerce validated values to float before passing to backend."""

    def test_decimal_value_is_rejected(self):
        """Max-capacity validation accepts only builtin int/float values."""
        config = make_limited_config(model_family="test-model")
        limiter = SyncRateLimiter(config, backend=SyncMemoryBackendBuilder())

        limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="test-model")

        with pytest.raises(ValueError, match="max_capacity must be finite"):
            limiter.set_max_capacity("test-model", "tokens", 60, Decimal(5000))

    def test_int_value_is_coerced_to_float(self):
        """Backend receives a float, not the raw int."""
        builder, mock_backend = make_mock_backend_builder()
        config = make_limited_config(model_family="gpt-4o")
        limiter = SyncRateLimiter(config, backend=builder)

        limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4o")

        limiter.set_max_capacity("gpt-4o", "tokens", 60, 5000)

        # The backend should receive a float, not an int
        args = mock_backend.set_max_capacity.call_args[0]
        assert isinstance(args[2], float)
