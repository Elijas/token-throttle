"""Regression tests for metric/model-family Redis key segment validation."""

import pytest
from pydantic import ValidationError

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
from token_throttle._validation import validate_metric


@pytest.mark.parametrize(
    "metric",
    [
        "token\x00s",
        "token\u200bs",
        "token\ns",
        "token\rs",
        "token\ts",
        "token s",
        " tokens",
        "tokens ",
        "requests:per_minute",
        "tokens{evil}",
        "{evil}tokens",
    ],
)
def test_quota_metric_rejects_unsafe_key_segments(metric):
    with pytest.raises(ValidationError):
        Quota(metric=metric, limit=100.0, per_seconds=60)


@pytest.mark.parametrize(
    "metric",
    [
        "token\x00s",
        "token\u200bs",
        "token\ns",
        "token s",
        " tokens",
        "requests:per_minute",
        "tokens{evil}",
    ],
)
def test_validate_metric_rejects_unsafe_key_segments(metric):
    with pytest.raises(ValueError, match="metric must"):
        validate_metric(metric)


@pytest.mark.parametrize(
    "model_family",
    [
        "gpt\x00-4o",
        "gpt\u200b-4o",
        "gpt\n4o",
        "gpt 4o",
        " gpt-4o",
        "gpt-4o ",
        "org:gpt-4o",
        "gpt{evil}-4o",
        "{evil}gpt-4o",
    ],
)
def test_model_family_rejects_unsafe_key_segments(model_family):
    with pytest.raises(ValidationError):
        PerModelConfig(
            quotas=UsageQuotas([Quota(metric="tokens", limit=100.0)]),
            model_family=model_family,
        )

    with pytest.raises(ValidationError):
        CapacityReservation(usage={"tokens": 1.0}, model_family=model_family)


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [("cafe\u0301", "café"), ("fami\u0301lia", "família")],
)
def test_metric_and_family_normalize_to_nfc(raw, normalized):

    quota = Quota(metric=raw, limit=100.0)
    assert quota.metric == normalized

    cfg = PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=100.0)]),
        model_family=raw,
    )
    assert cfg.model_family == normalized


def test_usage_quotas_empty_requires_explicit_unlimited():
    with pytest.raises(ValueError, match=r"UsageQuotas\.unlimited"):
        UsageQuotas([])

    assert UsageQuotas.unlimited().is_unlimited is True
