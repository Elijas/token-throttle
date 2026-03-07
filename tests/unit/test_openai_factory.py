"""
Tests for OpenAI factory and model family regex.

Source: token_throttle/_factories/_openai/_openai_rate_limiter.py
"""

from unittest.mock import MagicMock

from token_throttle._factories._openai._openai_rate_limiter import (
    create_openai_redis_rate_limiter,
    openai_model_family_getter,
)
from token_throttle._rate_limiter import RateLimiter


class TestModelFamilyGetter:
    """Tests for openai_model_family_getter regex behavior."""

    def test_strips_trailing_date_digits(self):
        """gpt-4-0314 -> strips '-0314' -> 'gpt-4'"""
        assert openai_model_family_getter("gpt-4-0314") == "gpt-4"

    def test_strips_trailing_date_digits_0613(self):
        """gpt-4-0613 -> strips '-0613' -> 'gpt-4'"""
        assert openai_model_family_getter("gpt-4-0613") == "gpt-4"

    def test_no_trailing_digits_unchanged(self):
        """gpt-4o has no trailing dash+digits, stays unchanged."""
        assert openai_model_family_getter("gpt-4o") == "gpt-4o"

    def test_mini_variant_unchanged(self):
        """gpt-4o-mini has no trailing dash+digits, stays unchanged."""
        assert openai_model_family_getter("gpt-4o-mini") == "gpt-4o-mini"

    def test_compound_date_only_strips_last_segment(self):
        """gpt-4-turbo-2024-04-09: regex strips only last '-09' (trailing digits)."""
        # re.sub(r"-\d+$", "", "gpt-4-turbo-2024-04-09") strips "-09"
        assert (
            openai_model_family_getter("gpt-4-turbo-2024-04-09")
            == "gpt-4-turbo-2024-04"
        )

    def test_model_with_single_trailing_digit(self):
        """A model ending with -3 has the suffix stripped."""
        assert openai_model_family_getter("some-model-3") == "some-model"

    def test_model_with_no_dashes(self):
        """A model with no dashes at all remains unchanged."""
        assert openai_model_family_getter("davinci") == "davinci"

    def test_model_ending_with_letters(self):
        """Trailing letters are not stripped."""
        assert openai_model_family_getter("gpt-4-turbo") == "gpt-4-turbo"

    def test_model_with_version_number_in_middle(self):
        """Only the trailing digit segment is affected."""
        assert openai_model_family_getter("gpt-3.5-turbo-0125") == "gpt-3.5-turbo"

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
        from token_throttle._interfaces._callbacks import RateLimiterCallbacks

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
