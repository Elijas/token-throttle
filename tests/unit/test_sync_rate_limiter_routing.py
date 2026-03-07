"""Tests for backend caching and model resolution in SyncRateLimiter."""

from unittest.mock import MagicMock

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._sync_rate_limiter import SyncRateLimiter


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


class TestStaticVsCallableConfig:
    """Tests for static PerModelConfig and callable PerModelConfigGetter."""

    def test_static_per_model_config_works(self):
        builder, mock_backend = make_mock_backend_builder()
        config = make_limited_config()
        limiter = SyncRateLimiter(config, backend=builder)

        reservation = limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        assert reservation.model_family == "gpt-4"
        mock_backend.wait_for_capacity.assert_called_once()

    def test_callable_per_model_config_getter_works(self):
        builder, mock_backend = make_mock_backend_builder()

        def config_getter(model_name: str) -> PerModelConfig:
            return make_limited_config(model_family=f"family-{model_name}")

        limiter = SyncRateLimiter(config_getter, backend=builder)

        reservation = limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        assert reservation.model_family == "family-gpt-4"
        mock_backend.wait_for_capacity.assert_called_once()


class TestModelFamilyResolution:
    """Tests for model_family defaulting and preservation."""

    def test_model_family_defaults_to_model_name_when_none(self):
        builder, _ = make_mock_backend_builder()
        config = make_limited_config(model_family=None)
        limiter = SyncRateLimiter(config, backend=builder)

        reservation = limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4o",
        )

        assert reservation.model_family == "gpt-4o"

    def test_model_family_preserved_when_explicitly_set(self):
        builder, _ = make_mock_backend_builder()
        config = make_limited_config(model_family="openai-tier")
        limiter = SyncRateLimiter(config, backend=builder)

        reservation = limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4o",
        )

        assert reservation.model_family == "openai-tier"


class TestBackendCaching:
    """Tests for backend caching: same model_family reuses, different creates new."""

    def test_same_model_family_reuses_cached_backend(self):
        builder, mock_backend = make_mock_backend_builder()
        config = make_limited_config(model_family="shared-family")
        limiter = SyncRateLimiter(config, backend=builder)

        limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )
        limiter.acquire_capacity(
            {"tokens": 200, "requests": 1},
            model="gpt-4",
        )

        builder.build.assert_called_once()
        assert mock_backend.wait_for_capacity.call_count == 2

    def test_different_model_family_creates_separate_backend(self):
        builder = MagicMock()
        backend_a = MagicMock()
        backend_a.wait_for_capacity.return_value = None
        backend_b = MagicMock()
        backend_b.wait_for_capacity.return_value = None
        builder.build.side_effect = [backend_a, backend_b]

        def config_getter(model_name: str) -> PerModelConfig:
            return make_limited_config(model_family=model_name)

        limiter = SyncRateLimiter(config_getter, backend=builder)

        reservation_a = limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )
        reservation_b = limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="claude-3",
        )

        assert builder.build.call_count == 2
        assert reservation_a.model_family == "gpt-4"
        assert reservation_b.model_family == "claude-3"
        backend_a.wait_for_capacity.assert_called_once()
        backend_b.wait_for_capacity.assert_called_once()
