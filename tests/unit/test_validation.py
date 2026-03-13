"""Tests for validation logic in token_throttle._validation."""

import math

import pytest
from frozendict import frozendict

from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._validation import (
    validate_acquire_usage,
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

    def test_preserves_non_boolean_numeric_timeout(self):
        assert validate_timeout(-1) == -1.0
        assert validate_timeout(0) == 0.0
        assert validate_timeout(1.5) == 1.5
