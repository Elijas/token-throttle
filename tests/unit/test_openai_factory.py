"""
Tests for OpenAI factory and model family regex.

Source: token_throttle/_factories/_openai/_openai_rate_limiter.py
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._factories._openai._openai_rate_limiter import (
    create_openai_redis_rate_limiter,
    openai_model_family_getter,
)
from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._rate_limiter import RateLimiter


class TestModelFamilyGetter:
    """Tests for openai_model_family_getter regex behavior.

    The regex strips date/snapshot suffixes in two formats:
    - MMDD: exactly 4 digits (e.g. -0613, -0125, -1106)
    - ISO: YYYY-MM-DD (e.g. -2024-04-09, -2025-08-07)

    Single/triple-digit version numbers (-1, -2, -002) are preserved.
    """

    # --- MMDD snapshot suffixes (4 digits) are stripped ---

    def test_strips_mmdd_suffix_0314(self):
        assert openai_model_family_getter("gpt-4-0314") == "gpt-4"

    def test_strips_mmdd_suffix_0613(self):
        assert openai_model_family_getter("gpt-4-0613") == "gpt-4"

    def test_strips_mmdd_suffix_0125(self):
        assert openai_model_family_getter("gpt-3.5-turbo-0125") == "gpt-3.5-turbo"

    def test_strips_mmdd_suffix_0914(self):
        assert openai_model_family_getter("gpt-3.5-turbo-instruct-0914") == "gpt-3.5-turbo-instruct"

    def test_strips_mmdd_suffix_1106(self):
        assert openai_model_family_getter("tts-1-1106") == "tts-1"

    # --- ISO date suffixes (YYYY-MM-DD) are stripped ---

    def test_strips_iso_date(self):
        assert openai_model_family_getter("gpt-4-turbo-2024-04-09") == "gpt-4-turbo"

    def test_strips_iso_date_gpt4o(self):
        assert openai_model_family_getter("gpt-4o-2024-08-06") == "gpt-4o"

    def test_strips_iso_date_gpt4o_mini(self):
        assert openai_model_family_getter("gpt-4o-mini-2024-07-18") == "gpt-4o-mini"

    def test_strips_iso_date_o1(self):
        assert openai_model_family_getter("o1-2024-12-17") == "o1"

    def test_strips_iso_date_o3_mini(self):
        assert openai_model_family_getter("o3-mini-2025-01-31") == "o3-mini"

    def test_strips_iso_date_deep_research(self):
        assert openai_model_family_getter("o4-mini-deep-research-2025-06-26") == "o4-mini-deep-research"

    def test_strips_iso_date_dotted_version(self):
        assert openai_model_family_getter("gpt-4.1-mini-2025-04-14") == "gpt-4.1-mini"

    # --- Version numbers (1-3 digits) are preserved ---

    def test_preserves_single_digit_gpt4(self):
        assert openai_model_family_getter("gpt-4") == "gpt-4"

    def test_preserves_single_digit_gpt5(self):
        assert openai_model_family_getter("gpt-5") == "gpt-5"

    def test_preserves_single_digit_tts1(self):
        assert openai_model_family_getter("tts-1") == "tts-1"

    def test_preserves_single_digit_whisper1(self):
        assert openai_model_family_getter("whisper-1") == "whisper-1"

    def test_preserves_single_digit_dalle(self):
        assert openai_model_family_getter("dall-e-2") == "dall-e-2"

    def test_preserves_triple_digit_002(self):
        assert openai_model_family_getter("babbage-002") == "babbage-002"

    def test_preserves_triple_digit_ada(self):
        assert openai_model_family_getter("text-embedding-ada-002") == "text-embedding-ada-002"

    # --- Models without date suffixes are unchanged ---

    def test_no_suffix_gpt4o(self):
        assert openai_model_family_getter("gpt-4o") == "gpt-4o"

    def test_no_suffix_gpt4o_mini(self):
        assert openai_model_family_getter("gpt-4o-mini") == "gpt-4o-mini"

    def test_no_suffix_gpt4_turbo(self):
        assert openai_model_family_getter("gpt-4-turbo") == "gpt-4-turbo"

    def test_no_suffix_no_dashes(self):
        assert openai_model_family_getter("davinci") == "davinci"

    def test_no_suffix_16k(self):
        assert openai_model_family_getter("gpt-3.5-turbo-16k") == "gpt-3.5-turbo-16k"

    # --- Date + text suffixes (e.g. -preview) are stripped ---

    def test_strips_mmdd_preview_suffix(self):
        assert openai_model_family_getter("gpt-4-0125-preview") == "gpt-4"

    def test_strips_mmdd_preview_suffix_turbo(self):
        assert openai_model_family_getter("gpt-4-1106-preview") == "gpt-4"

    def test_strips_iso_date_preview_suffix(self):
        assert openai_model_family_getter("gpt-4-turbo-2024-04-09-preview") == "gpt-4-turbo"

    # --- Standalone -preview suffix is stripped ---

    def test_strips_standalone_preview_o1(self):
        assert openai_model_family_getter("o1-preview") == "o1"

    def test_strips_standalone_preview_gpt4_turbo(self):
        assert openai_model_family_getter("gpt-4-turbo-preview") == "gpt-4-turbo"

    # --- Provider prefix stripping ---

    def test_strips_openai_prefix(self):
        assert openai_model_family_getter("openai/gpt-4-0613") == "gpt-4"

    def test_strips_openai_prefix_no_date(self):
        assert openai_model_family_getter("openai/gpt-4o") == "gpt-4o"

    def test_empty_string(self):
        assert openai_model_family_getter("") == ""


class TestCreateOpenAIRedisRateLimiter:
    """Tests for create_openai_redis_rate_limiter factory."""

    def test_returns_rate_limiter_instance(self):
        mock_redis = MagicMock()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            rpm=100,
            tpm=10000,
        )
        assert isinstance(limiter, RateLimiter)

    def test_config_getter_produces_correct_quotas(self):
        mock_redis = MagicMock()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            rpm=60,
            tpm=5000,
        )
        # Access the internal config getter to verify quota setup
        config = limiter._config_getter("gpt-4")
        quota_names = set(config.quotas.names)
        assert quota_names == {"requests", "tokens"}

    def test_config_getter_uses_model_family(self):
        mock_redis = MagicMock()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            rpm=100,
            tpm=10000,
        )
        config = limiter._config_getter("gpt-4-0314")
        # openai_model_family_getter("gpt-4-0314") -> "gpt-4"
        assert config.model_family == "gpt-4"

    def test_config_getter_has_usage_counter(self):
        mock_redis = MagicMock()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            rpm=100,
            tpm=10000,
        )
        config = limiter._config_getter("gpt-4")
        assert config.usage_counter is not None

    def test_custom_callbacks_are_passed(self):
        mock_redis = MagicMock()
        custom_callbacks = RateLimiterCallbacks()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            rpm=100,
            tpm=10000,
            callbacks=custom_callbacks,
        )
        assert limiter._callbacks is custom_callbacks

    def test_default_callbacks_when_none_provided(self):
        mock_redis = MagicMock()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            rpm=100,
            tpm=10000,
        )
        # Default callbacks are created via create_loguru_callbacks
        assert limiter._callbacks is not None

    def test_quota_limits_match_parameters(self):
        mock_redis = MagicMock()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            rpm=200,
            tpm=40000,
        )
        config = limiter._config_getter("gpt-4")
        request_quotas = config.quotas.get_quotas("requests")
        token_quotas = config.quotas.get_quotas("tokens")
        assert len(request_quotas) == 1
        assert request_quotas[0].limit == 200
        assert request_quotas[0].per_seconds == 60
        assert len(token_quotas) == 1
        assert token_quotas[0].limit == 40000
        assert token_quotas[0].per_seconds == 60
