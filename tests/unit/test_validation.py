"""Tests for validation logic in token_throttle._validation."""

import math

import pytest
from frozendict import frozendict

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._validation import (
    extract_total_tokens,
    merge_extra_usage,
    resolve_config,
    validate_acquire_usage,
    validate_max_capacity_value,
    validate_metric,
    validate_per_seconds,
    validate_refund_usage,
    validate_timeout,
)


class TestValidateAcquireUsageNonFinite:
    """Cover lines 27-28: NaN/Inf rejection in validate_acquire_usage."""

    def test_nan_usage_raises(self):
        quotas = UsageQuotas([Quota(metric="tokens", limit=1000)])
        with pytest.raises(ValueError, match="must be finite"):
            validate_acquire_usage(frozendict({"tokens": float("nan")}), quotas)

    def test_positive_inf_usage_raises(self):
        quotas = UsageQuotas([Quota(metric="tokens", limit=1000)])
        with pytest.raises(ValueError, match="must be finite"):
            validate_acquire_usage(frozendict({"tokens": float("inf")}), quotas)

    def test_negative_inf_usage_raises(self):
        quotas = UsageQuotas([Quota(metric="tokens", limit=1000)])
        with pytest.raises(ValueError, match="must be finite"):
            validate_acquire_usage(frozendict({"tokens": float("-inf")}), quotas)

    def test_non_numeric_usage_raises(self):
        quotas = UsageQuotas([Quota(metric="tokens", limit=1000)])
        with pytest.raises(ValueError, match="must be finite"):
            validate_acquire_usage(frozendict({"tokens": object()}), quotas)


class TestValidateRefundUsageNonFinite:
    """Cover lines 49-50: NaN/Inf rejection in validate_refund_usage."""

    def test_nan_usage_raises(self):
        with pytest.raises(ValueError, match="must be finite"):
            validate_refund_usage({"tokens": float("nan")}, {"tokens"})

    def test_positive_inf_usage_raises(self):
        with pytest.raises(ValueError, match="must be finite"):
            validate_refund_usage({"tokens": float("inf")}, {"tokens"})

    def test_negative_inf_usage_raises(self):
        with pytest.raises(ValueError, match="must be finite"):
            validate_refund_usage({"tokens": float("-inf")}, {"tokens"})

    def test_non_numeric_usage_raises(self):
        with pytest.raises(ValueError, match="must be finite"):
            validate_refund_usage({"tokens": object()}, {"tokens"})


class TestValidateRefundUsageNegative:
    """Cover line 54: negative value rejection in validate_refund_usage."""

    def test_negative_usage_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_refund_usage({"tokens": -1.0}, {"tokens"})

    def test_negative_usage_raises_with_multiple_metrics(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_refund_usage(
                {"tokens": 100.0, "requests": -0.5},
                {"tokens", "requests"},
            )


class TestValidateTimeout:
    @pytest.mark.parametrize("raw_timeout", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_timeout(self, raw_timeout):
        with pytest.raises(ValueError, match="timeout must be finite"):
            validate_timeout(raw_timeout)

    def test_rejects_boolean_timeout(self):
        raw_timeout = True
        with pytest.raises(ValueError, match="timeout must not be a boolean"):
            validate_timeout(raw_timeout)

    def test_allows_none(self):
        assert validate_timeout(None) is None

    def test_rejects_negative_timeout(self):
        with pytest.raises(ValueError, match="timeout must be non-negative"):
            validate_timeout(-1)
        with pytest.raises(ValueError, match="timeout must be non-negative"):
            validate_timeout(-0.001)

    def test_preserves_valid_numeric_timeout(self):
        assert validate_timeout(0) == 0.0
        assert validate_timeout(1.5) == 1.5
        assert validate_timeout(0.0) == 0.0


class TestValidateMetric:
    def test_boolean_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be a non-empty string"):
            validate_metric(True)  # noqa: FBT003

    def test_non_string_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be a non-empty string"):
            validate_metric(42)

    def test_empty_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be a non-empty string"):
            validate_metric("")

    def test_none_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be a non-empty string"):
            validate_metric(None)

    def test_rejects_metric_containing_colon(self):
        with pytest.raises(ValueError, match="must not contain ':'"):
            validate_metric("requests:per_min")

    def test_valid_metric_returns_string(self):
        assert validate_metric("tokens") == "tokens"
        assert validate_metric("requests") == "requests"


class TestValidatePerSeconds:
    def test_boolean_per_seconds_raises(self):
        with pytest.raises(ValueError, match="per_seconds must not be a boolean"):
            validate_per_seconds(True)  # noqa: FBT003

    def test_float_per_seconds_raises(self):
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            validate_per_seconds(60.5)

    def test_string_per_seconds_raises(self):
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            validate_per_seconds("60")

    def test_zero_per_seconds_raises(self):
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            validate_per_seconds(0)

    def test_negative_per_seconds_raises(self):
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            validate_per_seconds(-1)

    def test_none_per_seconds_raises(self):
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            validate_per_seconds(None)

    def test_nan_per_seconds_raises(self):
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            validate_per_seconds(float("nan"))

    def test_inf_per_seconds_raises(self):
        with pytest.raises(ValueError, match="per_seconds must be a positive integer"):
            validate_per_seconds(float("inf"))

    def test_valid_per_seconds_returns_int(self):
        assert validate_per_seconds(60) == 60
        assert validate_per_seconds(1) == 1

    def test_whole_float_per_seconds_is_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            validate_per_seconds(60.0)


class TestResolveConfig:
    @pytest.mark.filterwarnings(
        "ignore:coroutine '.*' was never awaited:RuntimeWarning"
    )
    def test_rejects_async_config_getter(self):
        async def async_config_getter(model_name: str):
            return PerModelConfig(
                quotas=UsageQuotas([Quota(metric=model_name, limit=1)])
            )

        with pytest.raises(
            ValueError,
            match="cfg must be a synchronous PerModelConfig getter",
        ):
            resolve_config(async_config_getter, "tokens")

    def test_rejects_static_cfg_with_wrong_type(self):
        with pytest.raises(ValueError, match="must resolve to PerModelConfig"):
            resolve_config({"quotas": []}, "tokens")

    def test_rejects_getter_returning_wrong_type(self):
        with pytest.raises(ValueError, match="must resolve to PerModelConfig"):
            resolve_config(lambda _model_name: {"quotas": []}, "tokens")

    def test_rejects_model_name_with_colon_when_defaulting_model_family(self):
        cfg = PerModelConfig(quotas=UsageQuotas([Quota(metric="tokens", limit=1)]))
        with pytest.raises(ValueError, match="must not contain ':'"):
            resolve_config(cfg, "openai:gpt-4")

    def test_rejects_explicit_model_family_with_colon(self):
        with pytest.raises(Exception, match="must not contain ':'"):
            PerModelConfig(
                quotas=UsageQuotas([Quota(metric="tokens", limit=1)]),
                model_family="openai:gpt-4",
            )

    @pytest.mark.parametrize("whitespace_name", [" ", "  ", "\t", "\n", " \t\n "])
    def test_rejects_whitespace_only_model_name(self, whitespace_name):
        # Whitespace-only names would silently become a whitespace-only
        # model_family and let inconsistent callers end up on different
        # backends without noticing. Fail fast instead.
        cfg = PerModelConfig(quotas=UsageQuotas([Quota(metric="tokens", limit=1)]))
        with pytest.raises(ValueError, match="whitespace-only"):
            resolve_config(cfg, whitespace_name)


class _FakeDtype:
    def __str__(self) -> str:
        return "bool"


class _FakeNumpyBool:
    """Simulates numpy.bool_ — has dtype='bool' but is NOT a Python bool subclass."""

    def __init__(self, *, value: object) -> None:
        self._value = value
        self.dtype = _FakeDtype()

    def __float__(self) -> float:
        return float(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)


FAKE_NP_TRUE = _FakeNumpyBool(value=1)


class TestNumpyBoolCoercionInValidation:
    def test_extract_total_tokens_rejects_duck_typed_numpy_bool(self):
        usage = type("Usage", (), {"total_tokens": FAKE_NP_TRUE})()
        with pytest.raises(ValueError, match="int or float"):
            extract_total_tokens(usage)

    def test_validate_timeout_rejects_duck_typed_numpy_bool(self):
        with pytest.raises(ValueError, match="int, float, or None"):
            validate_timeout(FAKE_NP_TRUE)

    def test_validate_max_capacity_value_rejects_duck_typed_numpy_bool(self):
        with pytest.raises(ValueError, match="max_capacity must be finite"):
            validate_max_capacity_value(FAKE_NP_TRUE)

    def test_validate_per_seconds_rejects_non_numeric_duck_typed_numpy_bool(self):
        with pytest.raises(ValueError, match="positive integer"):
            validate_per_seconds(FAKE_NP_TRUE)

    def test_merge_extra_usage_rejects_duck_typed_numpy_bool(self):
        usage = frozendict({"tokens": 100.0})
        with pytest.raises(ValueError, match="int or float"):
            merge_extra_usage(usage, {"tokens": FAKE_NP_TRUE})
