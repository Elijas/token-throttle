"""
Tests for OpenAI factory and model family regex.

Source: token_throttle/_factories/_openai/_openai_rate_limiter.py
"""

import pytest

pytest.importorskip("redis", reason="redis package not installed")

import redis as _sync_redis
import redis.asyncio as _async_redis
from pydantic import ValidationError

from token_throttle._factories._openai._openai_rate_limiter import (
    create_openai_redis_rate_limiter,
    openai_model_family_getter,
)
from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._rate_limiter import RateLimiter


def _async_redis_mock() -> _async_redis.Redis:
    return _async_redis.Redis()


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
        assert (
            openai_model_family_getter("gpt-3.5-turbo-instruct-0914")
            == "gpt-3.5-turbo-instruct"
        )

    def test_strips_mmdd_suffix_1106(self):
        assert openai_model_family_getter("tts-1-1106") == "tts-1"

    # --- ISO date suffixes (YYYY-MM-DD) are stripped ---

    def test_strips_iso_date(self):
        assert openai_model_family_getter("gpt-4-turbo-2024-04-09") == "gpt-4-turbo"

    def test_strips_iso_date_gpt4o(self):
        assert openai_model_family_getter("gpt-4o-2024-08-06") == "gpt-4o"

    def test_strips_iso_date_gpt4o_mini(self):
        assert openai_model_family_getter("gpt-4o-mini-2024-07-18") == "gpt-4o-mini"

    def test_strips_yyyymmdd_date_gpt4o(self):
        assert openai_model_family_getter("gpt-4o-20241203") == "gpt-4o"

    def test_strips_iso_date_o1(self):
        assert openai_model_family_getter("o1-2024-12-17") == "o1"

    def test_strips_iso_date_o3_mini(self):
        assert openai_model_family_getter("o3-mini-2025-01-31") == "o3-mini"

    def test_strips_iso_date_deep_research(self):
        assert (
            openai_model_family_getter("o4-mini-deep-research-2025-06-26")
            == "o4-mini-deep-research"
        )

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
        assert (
            openai_model_family_getter("text-embedding-ada-002")
            == "text-embedding-ada-002"
        )

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
        assert (
            openai_model_family_getter("gpt-4-turbo-2024-04-09-preview")
            == "gpt-4-turbo"
        )

    def test_strips_yyyymmdd_preview_suffix(self):
        assert openai_model_family_getter("gpt-4o-20241203-preview") == "gpt-4o"

    # --- -preview before date suffix (e.g. o1-preview-2024-09-12) ---

    def test_strips_preview_before_iso_date_o1(self):
        assert openai_model_family_getter("o1-preview-2024-09-12") == "o1"

    def test_strips_preview_before_iso_date_realtime(self):
        assert (
            openai_model_family_getter("gpt-4o-realtime-preview-2024-10-01")
            == "gpt-4o-realtime"
        )

    def test_strips_preview_before_iso_date_audio(self):
        assert (
            openai_model_family_getter("gpt-4o-audio-preview-2024-10-01")
            == "gpt-4o-audio"
        )

    def test_strips_preview_before_mmdd_suffix(self):
        assert openai_model_family_getter("gpt-4o-mini-preview-0125") == "gpt-4o-mini"

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

    def test_degenerate_preview_returns_original(self):
        assert openai_model_family_getter("-preview") == "-preview"

    def test_degenerate_date_returns_original(self):
        assert openai_model_family_getter("-1234") == "-1234"


class TestCreateOpenAIRedisRateLimiter:
    """Tests for create_openai_redis_rate_limiter factory."""

    def test_returns_rate_limiter_instance(self):
        mock_redis = _async_redis_mock()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            key_prefix="test",
            rpm=100,
            tpm=10000,
        )
        assert isinstance(limiter, RateLimiter)

    def test_config_getter_produces_correct_quotas(self):
        mock_redis = _async_redis_mock()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            key_prefix="test",
            rpm=60,
            tpm=5000,
        )
        # Access the internal config getter to verify quota setup
        config = limiter._config_getter("gpt-4")
        quota_names = set(config.quotas.names)
        assert quota_names == {"requests", "tokens"}

    def test_config_getter_uses_model_family(self):
        mock_redis = _async_redis_mock()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            key_prefix="test",
            rpm=100,
            tpm=10000,
        )
        config = limiter._config_getter("gpt-4-0314")
        # openai_model_family_getter("gpt-4-0314") -> "gpt-4"
        assert config.model_family == "gpt-4"

    def test_config_getter_has_usage_counter(self):
        mock_redis = _async_redis_mock()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            key_prefix="test",
            rpm=100,
            tpm=10000,
        )
        config = limiter._config_getter("gpt-4")
        assert config.usage_counter is not None

    def test_custom_callbacks_are_passed(self):
        mock_redis = _async_redis_mock()

        async def on_wait_start(
            *,
            model_family,
            usage,
            preconsumption_capacities,
        ):
            pass

        custom_callbacks = RateLimiterCallbacks(on_wait_start=on_wait_start)
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            key_prefix="test",
            rpm=100,
            tpm=10000,
            callbacks=custom_callbacks,
        )
        assert limiter._callbacks is not custom_callbacks
        assert limiter._callbacks.on_wait_start is on_wait_start
        assert limiter._callbacks.on_missing_consumption_data is not None

    def test_default_callbacks_when_none_provided(self):
        mock_redis = _async_redis_mock()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            key_prefix="test",
            rpm=100,
            tpm=10000,
        )
        # Default callbacks are created via create_logging_callbacks
        assert limiter._callbacks is not None

    def test_quota_limits_match_parameters(self):
        mock_redis = _async_redis_mock()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            key_prefix="test",
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

    def test_factory_eager_validation_rpm_zero(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            create_openai_redis_rate_limiter(
                _async_redis_mock(), key_prefix="test", rpm=0, tpm=10_000
            )

    def test_factory_eager_validation_rpm_negative(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            create_openai_redis_rate_limiter(
                _async_redis_mock(), key_prefix="test", rpm=-1, tpm=10_000
            )

    def test_factory_eager_validation_rpm_inf_nan(self):
        with pytest.raises(TypeError, match="rpm must be an int"):
            create_openai_redis_rate_limiter(
                _async_redis_mock(), key_prefix="test", rpm=float("inf"), tpm=10_000
            )
        with pytest.raises(TypeError, match="rpm must be an int"):
            create_openai_redis_rate_limiter(
                _async_redis_mock(), key_prefix="test", rpm=float("nan"), tpm=10_000
            )

    def test_factory_eager_validation_rpm_bool(self):
        with pytest.raises(TypeError, match="rpm must be an int"):
            create_openai_redis_rate_limiter(
                _async_redis_mock(), key_prefix="test", rpm=True, tpm=10_000
            )
        with pytest.raises(TypeError, match="tpm must be an int"):
            create_openai_redis_rate_limiter(
                _async_redis_mock(), key_prefix="test", rpm=100, tpm=False
            )

    def test_factory_eager_validation_rpm_non_int(self):
        with pytest.raises(TypeError, match="rpm must be an int"):
            create_openai_redis_rate_limiter(
                _async_redis_mock(), key_prefix="test", rpm=1.5, tpm=10_000
            )
        with pytest.raises(TypeError, match="rpm must be an int"):
            create_openai_redis_rate_limiter(
                _async_redis_mock(), key_prefix="test", rpm="100", tpm=10_000
            )
        with pytest.raises(TypeError, match="tpm must be an int"):
            create_openai_redis_rate_limiter(
                _async_redis_mock(), key_prefix="test", rpm=100, tpm=2.5
            )

    def test_factory_redis_client_type_check(self):
        with pytest.raises(TypeError, match=r"expected redis\.asyncio\.Redis"):
            create_openai_redis_rate_limiter(
                None, key_prefix="test", rpm=100, tpm=10_000
            )
        with pytest.raises(TypeError, match=r"expected redis\.asyncio\.Redis"):
            create_openai_redis_rate_limiter(
                "redis://localhost:6379", key_prefix="test", rpm=100, tpm=10_000
            )
        sync_client = _sync_redis.Redis()
        with pytest.raises(TypeError, match=r"expected redis\.asyncio\.Redis"):
            create_openai_redis_rate_limiter(
                sync_client, key_prefix="test", rpm=100, tpm=10_000
            )

    def test_factory_callbacks_type_check(self):
        with pytest.raises(TypeError, match="callbacks must be a RateLimiterCallbacks"):
            create_openai_redis_rate_limiter(
                _async_redis_mock(),
                key_prefix="test",
                rpm=100,
                tpm=10_000,
                callbacks=False,
            )
        with pytest.raises(TypeError, match="callbacks must be a RateLimiterCallbacks"):
            create_openai_redis_rate_limiter(
                _async_redis_mock(),
                key_prefix="test",
                rpm=100,
                tpm=10_000,
                callbacks={},
            )


class TestL01F04CallbacksDefaultFallbackContract:
    """Regression tests for L01 F04 callbacks default merging.

    User-provided callback bundles merge with the factory defaults slot by slot:
    non-None user callbacks win, and None slots inherit default callbacks.

    See: capsules/260510-r4-full-bugs-and-footguns/findings/01-footgun-catalog.md
    """

    def test_callbacks_none_applies_default_missing_consumption_data_logger(self):
        mock_redis = _async_redis_mock()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            key_prefix="test",
            rpm=100,
            tpm=10000,
        )
        assert limiter._callbacks is not None
        assert limiter._callbacks.on_missing_consumption_data is not None

    def test_empty_callbacks_model_preserves_default_missing_logger(self):
        mock_redis = _async_redis_mock()
        empty = RateLimiterCallbacks()
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            key_prefix="test",
            rpm=100,
            tpm=10000,
            callbacks=empty,
        )
        assert limiter._callbacks is not empty
        assert limiter._callbacks.on_missing_consumption_data is not None

    def test_user_callback_preserved_with_default_logger_merged_in(self):
        async def on_capacity_consumed(
            *,
            model_family: str,
            preconsumption_capacities,
            postconsumption_capacities,
            usage,
            current_time: float,
        ) -> None:
            pass

        mock_redis = _async_redis_mock()
        user_cbs = RateLimiterCallbacks(on_capacity_consumed=on_capacity_consumed)
        limiter = create_openai_redis_rate_limiter(
            mock_redis,
            key_prefix="test",
            rpm=100,
            tpm=10000,
            callbacks=user_cbs,
        )
        assert limiter._callbacks is not user_cbs
        assert limiter._callbacks.on_capacity_consumed is on_capacity_consumed
        assert limiter._callbacks.on_missing_consumption_data is not None
        assert limiter._callbacks.on_wait_start is None
        assert limiter._callbacks.after_wait_end_consumption is None
        assert limiter._callbacks.on_capacity_refunded is None
