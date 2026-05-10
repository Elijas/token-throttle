"""Regression tests for Round 4 audit lane L02 (model_copy bypass closure).

These tests demonstrate that the recommended replacement pattern
``Model.model_validate({**model.model_dump(), **updates}, strict=True)``
re-runs Pydantic field validators that ``model_copy(update=...)`` skips.
The library's internal call sites at ``_rate_limiter.py``,
``_sync_rate_limiter.py``, and ``_validation.py`` were swapped to this
pattern in the L02 mechanical fix; these tests pin the contract so a
regression at any future internal call site that uses the same pattern
will fail loudly.
"""

import pytest
from pydantic import ValidationError

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas


class TestModelValidateClosesQuotaLimitBypass:
    """V01: ``Quota.limit`` validators re-run on ``model_validate`` swap."""

    def test_negative_limit_rejected_via_model_validate_swap(self):
        # Constructor rejects:
        with pytest.raises(ValidationError):
            Quota(metric="tokens", limit=-1.0, per_seconds=60)
        # The internal swap pattern (now used at _rate_limiter.py) also rejects,
        # whereas the legacy ``model_copy(update={"limit": -1.0})`` would
        # silently accept the value.
        q = Quota(metric="tokens", limit=100.0, per_seconds=60)
        with pytest.raises(ValidationError):
            Quota.model_validate({**q.model_dump(), "limit": -1.0}, strict=True)

    def test_infinite_limit_rejected_via_model_validate_swap(self):
        q = Quota(metric="tokens", limit=100.0, per_seconds=60)
        with pytest.raises(ValidationError):
            Quota.model_validate({**q.model_dump(), "limit": float("inf")}, strict=True)


class TestModelValidateClosesQuotaMetricBypass:
    """V02: ``Quota.metric`` validators re-run on ``model_validate`` swap.

    Direct sibling of R1 HIGH-03 (which only patched the constructor path).
    """

    def test_colon_in_metric_rejected_via_model_validate_swap(self):
        with pytest.raises(ValidationError, match="must not contain ':'"):
            Quota(metric=":bad:", limit=100.0, per_seconds=60)
        q = Quota(metric="tokens", limit=100.0, per_seconds=60)
        with pytest.raises(ValidationError, match="must not contain ':'"):
            Quota.model_validate({**q.model_dump(), "metric": ":bad:"}, strict=True)

    def test_empty_metric_rejected_via_model_validate_swap(self):
        q = Quota(metric="tokens", limit=100.0, per_seconds=60)
        with pytest.raises(ValidationError, match="must not be empty"):
            Quota.model_validate({**q.model_dump(), "metric": ""}, strict=True)

    def test_whitespace_only_metric_rejected_via_model_validate_swap(self):
        q = Quota(metric="tokens", limit=100.0, per_seconds=60)
        with pytest.raises(ValidationError, match="must not be whitespace-only"):
            Quota.model_validate({**q.model_dump(), "metric": "   "}, strict=True)


class TestModelValidateClosesModelFamilyBypass:
    """V04: ``PerModelConfig.model_family`` validators re-run on swap.

    R1 HIGH-03 added a post-copy ``:`` guard at ``resolve_config`` but did
    not re-check empty/whitespace. The mechanical swap closes this for the
    library-internal site at ``_validation.py``.
    """

    def test_whitespace_only_model_family_rejected_via_model_validate_swap(self):
        cfg = PerModelConfig(
            quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=60)]),
            model_family="real",
        )
        with pytest.raises(ValidationError, match="whitespace-only"):
            PerModelConfig.model_validate(
                {**cfg.model_dump(), "model_family": "   "}, strict=True
            )

    def test_colon_in_model_family_rejected_via_model_validate_swap(self):
        cfg = PerModelConfig(
            quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=60)]),
            model_family="real",
        )
        with pytest.raises(ValidationError, match="must not contain ':'"):
            PerModelConfig.model_validate(
                {**cfg.model_dump(), "model_family": ":bad:"}, strict=True
            )
