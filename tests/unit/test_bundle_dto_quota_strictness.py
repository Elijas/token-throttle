"""Regression tests for FIX-45 DTO-QUOTA-STRICTNESS."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas


def _quota(
    metric: str = "tokens",
    *,
    limit: float = 100.0,
    per_seconds: int = 60,
) -> Quota:
    return Quota(metric=metric, limit=limit, per_seconds=per_seconds)


def test_per_model_config_snapshots_usage_quotas_after_construction() -> None:
    quotas = UsageQuotas([_quota()])
    cfg = PerModelConfig(quotas=quotas, model_family="family")

    quotas.add_metric(_quota("requests", limit=10.0))

    assert cfg.quotas is not quotas
    assert cfg.quotas.names == ["tokens"]
    assert quotas.names == ["tokens", "requests"]


def test_per_model_config_embedded_usage_quotas_snapshot_is_frozen() -> None:
    cfg = PerModelConfig(
        quotas=UsageQuotas([_quota()]),
        model_family="family",
    )

    with pytest.raises(TypeError, match="frozen"):
        cfg.quotas.add_metric(_quota("requests", limit=10.0))


def test_external_usage_quotas_remains_mutable_until_embedded() -> None:
    quotas = UsageQuotas([_quota()])

    quotas.add_metric(_quota("requests", limit=10.0))

    assert quotas.names == ["tokens", "requests"]


def test_unlimited_usage_quotas_snapshot_does_not_track_original_mutation() -> None:
    quotas = UsageQuotas.unlimited()
    cfg = PerModelConfig(quotas=quotas, model_family="family")

    quotas.add_metric(_quota())

    assert cfg.is_unlimited is True
    assert cfg.quotas.names == []
    assert quotas.is_unlimited is False


def test_quota_model_validate_strict_false_still_rejects_lax_payload() -> None:
    payload = {"metric": "tokens", "limit": "100.0", "per_seconds": 60}

    with pytest.raises(ValidationError):
        Quota.model_validate(payload, strict=False)


def test_quota_model_validate_json_strict_false_still_rejects_lax_payload() -> None:
    payload = b'{"metric":"tokens","limit":"100.0","per_seconds":60}'

    with pytest.raises(ValidationError):
        Quota.model_validate_json(payload, strict=False)
