"""Tests for data models in token_throttle._interfaces._models."""

import pytest
from frozendict import frozendict
from pydantic import ValidationError

from token_throttle._interfaces._models import (
    CapacityReservation,
    Quota,
    SecondsIn,
    UsageQuotas,
    _coerce_usage_value,
    _is_bool_like,
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

    def test_rejects_empty_metric(self):
        with pytest.raises(ValidationError, match="metric must not be empty"):
            Quota(metric="", limit=100.0, per_seconds=60)

    def test_rejects_boolean_limit(self):
        with pytest.raises(ValidationError, match="must not be a boolean"):
            Quota(metric="requests", limit=True)

    def test_rejects_boolean_per_seconds(self):
        with pytest.raises(ValidationError, match="must not be a boolean"):
            Quota(metric="requests", limit=100.0, per_seconds=True)

    def test_rejects_zero_limit(self):
        with pytest.raises(ValidationError, match="limit"):
            Quota(metric="requests", limit=0)

    def test_rejects_negative_limit(self):
        with pytest.raises(ValidationError, match="limit"):
            Quota(metric="requests", limit=-1)

    def test_rejects_nan_limit(self):
        with pytest.raises(ValidationError, match="limit"):
            Quota(metric="requests", limit=float("nan"))

    def test_rejects_infinite_limit(self):
        with pytest.raises(ValidationError, match="limit"):
            Quota(metric="requests", limit=float("inf"))

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

    def test_rejects_whole_float_per_seconds(self):
        with pytest.raises(ValidationError):
            Quota(metric="requests", limit=100.0, per_seconds=60.0)

    @pytest.mark.parametrize(
        "metric",
        ["requests:per_min", "a:b", ":", "a:b:c"],
        ids=["colon-mid", "single-colon", "bare-colon", "multi-colon"],
    )
    def test_rejects_metric_containing_colon(self, metric):
        with pytest.raises(ValidationError, match="must not contain ':'"):
            Quota(metric=metric, limit=100.0, per_seconds=60)

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

    def test_empty_quotas_raises(self):
        """Bare empty quotas are a typo trap; use UsageQuotas.unlimited()."""
        with pytest.raises(ValueError, match=r"UsageQuotas\.unlimited"):
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

    def test_get_quotas_for_missing_metric_is_non_mutating(self):
        quotas = UsageQuotas([Quota(metric="requests", limit=100.0, per_seconds=60)])

        result = quotas.get_quotas("tokens")

        assert result == []
        assert quotas.names == ["requests"]

    def test_iteration(self):
        q1 = Quota(metric="requests", limit=100.0, per_seconds=60)
        q2 = Quota(metric="tokens", limit=5000.0, per_seconds=60)
        quotas = UsageQuotas([q1, q2])
        iterated = list(quotas)
        assert q1 in iterated
        assert q2 in iterated
        assert len(iterated) == 2

    @pytest.mark.parametrize("raw_quota", [{"metric": "requests"}, object(), "quota"])
    def test_rejects_non_quota_entries(self, raw_quota):
        with pytest.raises(ValueError, match="Each quota must be a Quota instance"):
            UsageQuotas([raw_quota])


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

    def test_rejects_non_numeric_value(self):
        with pytest.raises(ValueError, match="must be finite"):
            frozen_usage({"requests": 1.0, "tokens": object()})

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="must be finite"):
            frozen_usage({"tokens": float("nan")})

    def test_rejects_positive_infinity(self):
        with pytest.raises(ValueError, match="must be finite"):
            frozen_usage({"tokens": float("inf")})

    def test_rejects_negative_infinity(self):
        with pytest.raises(ValueError, match="must be finite"):
            frozen_usage({"tokens": float("-inf")})

    def test_rejects_nan_string(self):
        with pytest.raises(ValueError, match="must be finite"):
            frozen_usage({"tokens": "nan"})

    def test_rejects_inf_string(self):
        with pytest.raises(ValueError, match="must be finite"):
            frozen_usage({"tokens": "inf"})


class TestCapacityReservation:
    def test_construction(self):
        reservation = CapacityReservation(
            usage={"requests": 1.0, "tokens": 100.0},
            model_family="gpt-4o",
            limiter_instance_id="limiter",
        )
        assert reservation.model_family == "gpt-4o"
        assert reservation.usage == {"requests": 1.0, "tokens": 100.0}

    def test_rejects_boolean_usage_value(self):
        with pytest.raises(ValidationError, match="must not be a boolean"):
            CapacityReservation(
                usage={"requests": 1.0, "tokens": True},
                model_family="gpt-4o",
                limiter_instance_id="limiter",
            )

    def test_rejects_non_numeric_usage_value(self):
        with pytest.raises(ValidationError, match="must be finite"):
            CapacityReservation(
                usage={"requests": 1.0, "tokens": object()},
                model_family="gpt-4o",
                limiter_instance_id="limiter",
            )

    @pytest.mark.parametrize(
        ("raw_value", "message"),
        [
            pytest.param(-1.0, "must be non-negative", id="negative"),
            pytest.param(float("nan"), "must be finite", id="nan"),
            pytest.param(float("inf"), "must be finite", id="infinity"),
        ],
    )
    def test_rejects_invalid_usage_values(self, raw_value, message):
        with pytest.raises(ValidationError, match=message):
            CapacityReservation(
                usage={"requests": 1.0, "tokens": raw_value},
                model_family="gpt-4o",
                limiter_instance_id="limiter",
            )

    def test_get_usage_returns_frozendict(self):
        reservation = CapacityReservation(
            usage={"requests": 1.0, "tokens": 100.0},
            model_family="gpt-4o",
            limiter_instance_id="limiter",
        )
        result = reservation.get_usage()
        assert isinstance(result, frozendict)
        assert result["requests"] == 1.0
        assert result["tokens"] == 100.0

    def test_usage_mapping_is_immutable(self):
        reservation = CapacityReservation(
            usage={"requests": 1.0, "tokens": 100.0},
            model_family="gpt-4o",
            limiter_instance_id="limiter",
        )

        assert isinstance(reservation.usage, frozendict)
        with pytest.raises(TypeError):
            reservation.usage["requests"] = 2.0

    def test_bucket_ids_are_normalized(self):
        reservation = CapacityReservation(
            usage={"requests": 1.0},
            model_family="gpt-4o",
            bucket_ids=[("requests", 60)],
            limiter_instance_id="limiter",
        )

        assert reservation.bucket_ids == frozenset({("requests", 60)})

    def test_invalid_bucket_ids_are_rejected(self):
        with pytest.raises(ValidationError, match="bucket_id per_seconds"):
            CapacityReservation(
                usage={"requests": 1.0},
                model_family="gpt-4o",
                bucket_ids=[("requests", 0)],
                limiter_instance_id="limiter",
            )

    def test_frozen_immutability(self):
        reservation = CapacityReservation(
            usage={"requests": 1.0},
            model_family="gpt-4o",
            limiter_instance_id="limiter",
        )
        with pytest.raises(ValidationError):
            reservation.model_family = "other"


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
FAKE_NP_FALSE = _FakeNumpyBool(value=0)


class TestIsBoolLike:
    def test_python_true(self):
        assert _is_bool_like(True) is True  # noqa: FBT003

    def test_python_false(self):
        assert _is_bool_like(False) is True  # noqa: FBT003

    def test_int_not_bool_like(self):
        assert _is_bool_like(1) is False

    def test_float_not_bool_like(self):
        assert _is_bool_like(1.0) is False

    def test_string_not_bool_like(self):
        assert _is_bool_like("true") is False

    def test_none_not_bool_like(self):
        assert _is_bool_like(None) is False

    def test_fake_numpy_bool_true(self):
        assert _is_bool_like(FAKE_NP_TRUE) is False

    def test_fake_numpy_bool_false(self):
        assert _is_bool_like(FAKE_NP_FALSE) is False

    def test_real_numpy_bool(self):
        np = pytest.importorskip("numpy")
        assert _is_bool_like(np.bool_(1)) is False
        assert _is_bool_like(np.bool_(0)) is False

    def test_numpy_int_not_bool_like(self):
        np = pytest.importorskip("numpy")
        assert _is_bool_like(np.int64(1)) is False


class TestNumpyBoolCoercion:
    """Numeric-looking impostors are rejected unless they are int/float."""

    def test_quota_limit_rejects_duck_typed_numpy_bool(self):
        with pytest.raises(ValidationError, match="limit must be int or float"):
            Quota(metric="tokens", limit=FAKE_NP_TRUE)

    def test_quota_per_seconds_rejects_non_numeric_duck_typed_numpy_bool(self):
        with pytest.raises(ValidationError, match="per_seconds must be int or float"):
            Quota(metric="tokens", limit=100.0, per_seconds=FAKE_NP_TRUE)

    def test_coerce_usage_value_rejects_duck_typed_numpy_bool(self):
        with pytest.raises(ValueError, match="int or float"):
            _coerce_usage_value("tokens", FAKE_NP_TRUE)

    def test_frozen_usage_rejects_duck_typed_numpy_bool_value(self):
        with pytest.raises(ValueError, match="int or float"):
            frozen_usage({"tokens": FAKE_NP_TRUE})

    def test_bucket_ids_per_seconds_rejects_duck_typed_numpy_bool(self):
        with pytest.raises(ValidationError, match="bucket_id per_seconds"):
            CapacityReservation(
                usage={"requests": 1.0},
                model_family="gpt-4o",
                bucket_ids=[("requests", FAKE_NP_TRUE)],
                limiter_instance_id="limiter",
            )
