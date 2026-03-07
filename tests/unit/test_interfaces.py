"""Tests for interfaces in token_throttle._interfaces._interfaces."""

import pytest
from frozendict import frozendict

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas


class TestPerModelConfig:
    def test_construction_with_quotas_and_model_family(self):
        quotas = UsageQuotas([Quota(metric="requests", limit=100.0)])
        config = PerModelConfig(
            quotas=quotas,
            model_family="gpt-4o",
        )
        assert config.quotas is quotas
        assert config.model_family == "gpt-4o"

    def test_get_model_family_returns_family_when_set(self):
        quotas = UsageQuotas([Quota(metric="requests", limit=100.0)])
        config = PerModelConfig(
            quotas=quotas,
            model_family="claude-3",
        )
        assert config.get_model_family() == "claude-3"

    def test_get_model_family_raises_when_none(self):
        quotas = UsageQuotas([Quota(metric="requests", limit=100.0)])
        config = PerModelConfig(quotas=quotas)
        with pytest.raises(ValueError, match="model_family must be defined"):
            config.get_model_family()

    def test_is_unlimited_delegates_to_usage_quotas(self):
        unlimited_quotas = UsageQuotas.unlimited()
        config_unlimited = PerModelConfig(
            quotas=unlimited_quotas,
            model_family="test",
        )
        assert config_unlimited.is_unlimited is True

        limited_quotas = UsageQuotas([Quota(metric="requests", limit=100.0)])
        config_limited = PerModelConfig(
            quotas=limited_quotas,
            model_family="test",
        )
        assert config_limited.is_unlimited is False

    def test_usage_counter_defaults_to_none(self):
        quotas = UsageQuotas([Quota(metric="requests", limit=100.0)])
        config = PerModelConfig(quotas=quotas, model_family="test")
        assert config.usage_counter is None

    def test_usage_counter_accepts_callable(self):
        def my_counter(**_request) -> frozendict:
            return frozendict({"requests": 1.0})

        quotas = UsageQuotas([Quota(metric="requests", limit=100.0)])
        config = PerModelConfig(
            quotas=quotas,
            model_family="test",
            usage_counter=my_counter,
        )
        assert config.usage_counter is my_counter
