"""Tests for backend caching and model resolution in SyncRateLimiter."""

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
    mock_backend.apply_configured_max_capacity.return_value = None

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

    def test_same_model_family_with_different_quotas_raises(self):
        builder, mock_backend = make_mock_backend_builder()

        def config_getter(model_name: str) -> PerModelConfig:
            limit = 100 if model_name == "large" else 1
            return PerModelConfig(
                quotas=UsageQuotas(
                    [Quota(metric="requests", limit=limit, per_seconds=60)]
                ),
                model_family="shared-family",
            )

        limiter = SyncRateLimiter(config_getter, backend=builder)

        limiter.acquire_capacity({"requests": 1}, model="large")

        with pytest.raises(ValueError, match="inconsistent across models"):
            limiter.acquire_capacity({"requests": 1}, model="small")

        builder.build.assert_called_once()
        assert mock_backend.wait_for_capacity.call_count == 1

    def test_same_model_family_allows_global_quota_refresh_when_models_agree(self):
        builder, mock_backend = make_mock_backend_builder()
        current_limit = 10

        def config_getter(model_name: str) -> PerModelConfig:
            return PerModelConfig(
                quotas=UsageQuotas(
                    [Quota(metric="requests", limit=current_limit, per_seconds=60)]
                ),
                model_family="shared-family",
            )

        limiter = SyncRateLimiter(config_getter, backend=builder)

        limiter.acquire_capacity({"requests": 1}, model="a")

        current_limit = 20
        limiter.acquire_capacity({"requests": 1}, model="b")

        mock_backend.apply_configured_max_capacity.assert_called_once_with(
            "requests", 60, 20
        )


class TestAcquireCapacityForRequestMerge:
    """Tests for acquire_capacity_for_request merging usage_counter + extra_usage."""

    def test_merges_usage_counter_and_extra_usage(self):
        builder, mock_backend = make_mock_backend_builder()

        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        config = make_limited_config(usage_counter=fake_counter)
        limiter = SyncRateLimiter(config, backend=builder)

        reservation = limiter.acquire_capacity_for_request(
            extra_usage={"tokens": 50, "requests": 2},
            model="gpt-4",
        )

        assert reservation.model_family == "gpt-4"
        # The merged usage should be tokens=150, requests=3
        called_usage = mock_backend.wait_for_capacity.call_args[0][0]
        assert float(called_usage["tokens"]) == pytest.approx(150.0)
        assert float(called_usage["requests"]) == pytest.approx(3.0)

    def test_no_extra_usage_uses_counter_only(self):
        builder, mock_backend = make_mock_backend_builder()

        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        config = make_limited_config(usage_counter=fake_counter)
        limiter = SyncRateLimiter(config, backend=builder)

        reservation = limiter.acquire_capacity_for_request(model="gpt-4")

        assert reservation.model_family == "gpt-4"
        called_usage = mock_backend.wait_for_capacity.call_args[0][0]
        assert float(called_usage["tokens"]) == pytest.approx(100.0)
        assert float(called_usage["requests"]) == pytest.approx(1.0)
