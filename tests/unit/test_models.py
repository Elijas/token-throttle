"""Tests for data models in token_throttle._interfaces._models."""

import pytest
from frozendict import frozendict
from pydantic import ValidationError

from token_throttle._interfaces._models import (
    CapacityReservation,
    Quota,
    SecondsIn,
    UsageQuotas,
    frozen_usage,
)


class TestSecondsIn:
    def test_minute_value(self):
        assert SecondsIn.MINUTE == 60

    def test_hour_value(self):
        assert SecondsIn.HOUR == 3600

    def test_day_value(self):
        assert SecondsIn.DAY == 86400

    def test_is_int_subclass(self):
        assert isinstance(SecondsIn.MINUTE, int)


class TestQuota:
    def test_construction(self):
        q = Quota(metric="requests", limit=100.0, per_seconds=60)
        assert q.metric == "requests"
        assert q.limit == 100.0
        assert q.per_seconds == 60

    def test_default_per_seconds_is_minute(self):
        q = Quota(metric="tokens", limit=500.0)
        assert q.per_seconds == 60
        assert q.per_seconds == SecondsIn.MINUTE

    def test_rejects_per_seconds_zero(self):
        with pytest.raises(ValidationError, match="per_seconds"):
            Quota(metric="requests", limit=100.0, per_seconds=0)

    def test_rejects_per_seconds_negative(self):
        with pytest.raises(ValidationError, match="per_seconds"):
            Quota(metric="requests", limit=100.0, per_seconds=-10)

    def test_rejects_fractional_per_seconds(self):
        with pytest.raises(ValidationError):
            Quota(metric="requests", limit=100.0, per_seconds=0.5)

    def test_coerces_whole_float_to_int(self):
        q = Quota(metric="requests", limit=100.0, per_seconds=60.0)
        assert q.per_seconds == 60
        assert isinstance(q.per_seconds, int)

    def test_frozen_immutability(self):
        q = Quota(metric="requests", limit=100.0, per_seconds=60)
        with pytest.raises(ValidationError):
            q.metric = "other"


class TestUsageQuotas:
    def test_construction_with_valid_quotas(self):
        quotas = UsageQuotas(
            [
                Quota(metric="requests", limit=100.0, per_seconds=60),
                Quota(metric="tokens", limit=5000.0, per_seconds=60),
            ]
        )
        assert "requests" in quotas.names
        assert "tokens" in quotas.names

    def test_rejects_duplicate_metric_per_seconds(self):
        with pytest.raises(ValueError, match="already exists"):
            UsageQuotas(
                [
                    Quota(metric="requests", limit=100.0, per_seconds=60),
                    Quota(metric="requests", limit=200.0, per_seconds=60),
                ]
            )

    def test_allows_same_metric_different_time_windows(self):
        quotas = UsageQuotas(
            [
                Quota(metric="requests", limit=100.0, per_seconds=60),
                Quota(metric="requests", limit=5000.0, per_seconds=3600),
            ]
        )
        assert quotas.names == ["requests"]
        assert len(quotas.get_quotas("requests")) == 2

    def test_unlimited_creates_empty(self):
        quotas = UsageQuotas.unlimited()
        assert quotas.is_unlimited is True
        assert quotas.names == []

    def test_non_empty_is_not_unlimited(self):
        quotas = UsageQuotas([Quota(metric="requests", limit=100.0)])
        assert quotas.is_unlimited is False

    def test_empty_quotas_warns(self):
        """Cover line 39: empty quotas list triggers UserWarning."""
        with pytest.warns(UserWarning, match="Empty quota list"):
            UsageQuotas([])

    def test_names_property(self):
        quotas = UsageQuotas(
            [
                Quota(metric="requests", limit=100.0),
                Quota(metric="tokens", limit=5000.0),
            ]
        )
        assert sorted(quotas.names) == ["requests", "tokens"]

    def test_get_quotas(self):
        q1 = Quota(metric="requests", limit=100.0, per_seconds=60)
        q2 = Quota(metric="requests", limit=5000.0, per_seconds=3600)
        quotas = UsageQuotas([q1, q2])
        result = quotas.get_quotas("requests")
        assert len(result) == 2
        assert q1 in result
        assert q2 in result

    def test_iteration(self):
        q1 = Quota(metric="requests", limit=100.0, per_seconds=60)
        q2 = Quota(metric="tokens", limit=5000.0, per_seconds=60)
        quotas = UsageQuotas([q1, q2])
        iterated = list(quotas)
        assert q1 in iterated
        assert q2 in iterated
        assert len(iterated) == 2


class TestFrozenUsage:
    def test_converts_dict_to_frozendict(self):
        result = frozen_usage({"requests": 1.0, "tokens": 500.0})
        assert isinstance(result, frozendict)
        assert result["requests"] == 1.0
        assert result["tokens"] == 500.0

    def test_coerces_int_to_float(self):
        result = frozen_usage({"requests": 1, "tokens": 500})
        assert isinstance(result["requests"], float)
        assert isinstance(result["tokens"], float)
        assert result["requests"] == 1.0
        assert result["tokens"] == 500.0


class TestCapacityReservation:
    def test_construction(self):
        reservation = CapacityReservation(
            usage={"requests": 1.0, "tokens": 100.0},
            model_family="gpt-4o",
        )
        assert reservation.model_family == "gpt-4o"
        assert reservation.usage == {"requests": 1.0, "tokens": 100.0}

    def test_get_usage_returns_frozendict(self):
        reservation = CapacityReservation(
            usage={"requests": 1.0, "tokens": 100.0},
            model_family="gpt-4o",
        )
        result = reservation.get_usage()
        assert isinstance(result, frozendict)
        assert result["requests"] == 1.0
        assert result["tokens"] == 100.0

    def test_frozen_immutability(self):
        reservation = CapacityReservation(
            usage={"requests": 1.0},
            model_family="gpt-4o",
        )
        with pytest.raises(ValidationError):
            reservation.model_family = "other"
