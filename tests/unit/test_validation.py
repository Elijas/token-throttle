"""Tests for validation logic in token_throttle._validation."""

import math

import pytest
from frozendict import frozendict

from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._validation import validate_acquire_usage, validate_refund_usage


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
