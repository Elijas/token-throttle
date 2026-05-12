"""Regression tests for IEEE 754 math validation hardening."""

from decimal import Decimal
from fractions import Fraction

import pytest
from pydantic import ValidationError

from token_throttle._capacity import calculate_capacity
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._limiter_backends._memory._bucket import MemoryBucket
from token_throttle._limiter_backends._memory._sync_backend import SyncMemoryBackend
from token_throttle._validation import validate_max_capacity_value


def make_bucket(limit: float = 100.0) -> MemoryBucket:
    return MemoryBucket(
        metric="tokens",
        per_seconds=60,
        limit=limit,
        model_family="test-model",
    )


def test_m01_calculate_capacity_rejects_nan_current_time():
    with pytest.raises(ValueError, match="current_time must be finite"):
        calculate_capacity(
            last_checked=1000.0,
            outdated_capacity=50.0,
            current_time=float("nan"),
            max_capacity=100.0,
            rate_per_sec=1.0,
            bucket_id="memory:test-model:tokens:60",
        )


@pytest.mark.parametrize("limit", [float("nan"), float("inf")])
def test_m02_memory_bucket_constructor_rejects_non_finite_limit(limit):
    with pytest.raises(ValueError, match="max_capacity must be finite"):
        make_bucket(limit=limit)


def test_m03_subnormal_limit_rejected_before_zero_rate():
    bucket = make_bucket()
    with pytest.raises(ValueError, match="max_capacity must be finite"):
        bucket.set_max_capacity(5e-324, current_time=1000.0)


@pytest.mark.parametrize(
    "limit",
    [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, 5e-324],
)
def test_m04_memory_bucket_constructor_validates_limit(limit):
    with pytest.raises(ValueError, match="max_capacity"):
        make_bucket(limit=limit)


@pytest.mark.parametrize("rate_per_sec", [float("nan"), float("inf"), -1.0, 0.0])
def test_m08_m11_calculate_capacity_rejects_invalid_rate(rate_per_sec):
    with pytest.raises(ValueError, match="Bucket rate is non-positive/non-finite"):
        calculate_capacity(
            last_checked=1000.0,
            outdated_capacity=50.0,
            current_time=1001.0,
            max_capacity=100.0,
            rate_per_sec=rate_per_sec,
            bucket_id="memory:test-model:tokens:60",
        )


@pytest.mark.parametrize("max_capacity", [0.0, -1.0])
def test_m10_calculate_capacity_rejects_non_positive_max_capacity(max_capacity):
    with pytest.raises(ValueError, match="max_capacity must be finite"):
        calculate_capacity(
            last_checked=1000.0,
            outdated_capacity=50.0,
            current_time=1001.0,
            max_capacity=max_capacity,
            rate_per_sec=1.0,
            bucket_id="memory:test-model:tokens:60",
        )


def test_m13_direct_max_capacity_mutation_is_validated():
    bucket = make_bucket()
    with pytest.raises(ValueError, match="max_capacity must be finite"):
        bucket.max_capacity = float("nan")
    assert bucket.max_capacity == 100.0


def test_m13_direct_rate_mutation_is_validated():
    bucket = make_bucket()
    with pytest.raises(ValueError, match="Bucket rate is non-positive/non-finite"):
        bucket._rate_per_sec = float("nan")
    assert bucket._rate_per_sec == pytest.approx(100.0 / 60.0)


def test_m12_compute_sleep_rejects_poisoned_rate_defensively():
    bucket = make_bucket()
    bucket._rate_per_sec_value = 0.0
    backend = SyncMemoryBackend(
        buckets=[bucket],
        limit_config=PerModelConfig(
            model_family="test-model",
            quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=60)]),
        ),
    )

    with pytest.raises(ValueError, match="Bucket rate is non-positive/non-finite"):
        backend._compute_sleep(
            frozen_usage({"tokens": 10.0}),
            {("tokens", 60): 0.0},
        )


@pytest.mark.parametrize(
    "value",
    [Decimal("1.5"), Fraction(1, 3), b"5"],
)
def test_m18_validate_max_capacity_rejects_non_builtin_numeric(value):
    with pytest.raises(ValueError, match="max_capacity must be finite"):
        validate_max_capacity_value(value)


def test_m20_quota_rejects_subnormal_limit():
    with pytest.raises(ValidationError):
        Quota(metric="tokens", limit=5e-324, per_seconds=60)


def test_m14_set_capacity_normalizes_negative_zero():
    bucket = make_bucket()
    bucket.set_capacity(-0.0, current_time=1000.0, allow_negative=True)
    assert repr(bucket.capacity) == "0.0"
