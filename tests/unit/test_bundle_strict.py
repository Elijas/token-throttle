"""Regression tests for FIX-15 BUNDLE-STRICT.

L15 found that Pydantic lax coercion and runtime duck-typing composed poorly:
bytes could become strings after validators ran, Mocks could masquerade as
numbers or reservations, and some validators preserved hostile subclasses.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
from token_throttle._rate_limiter import RateLimiter
from token_throttle._validation import (
    extract_total_tokens,
    is_unlimited_reservation,
    validate_extra_usage,
    validate_max_capacity_value,
    validate_metric,
    validate_per_seconds,
    validate_timeout,
)


def _limited_config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric="tokens", limit=100.0, per_seconds=60),
                Quota(metric="requests", limit=10.0, per_seconds=60),
            ]
        ),
        model_family="strict-family",
    )


def _limiter() -> RateLimiter:
    return RateLimiter(_limited_config(), backend=MagicMock())


class LyingMapping(Mapping[str, object]):
    def __iter__(self) -> Iterator[str]:
        yield "tokens"

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)


class TestStrictPydanticModels:
    def test_a01_quota_metric_bytes_rejected(self):
        with pytest.raises(ValidationError):
            Quota(metric=b":bad:", limit=10.0, per_seconds=60)

    def test_quota_numeric_string_limit_rejected(self):
        with pytest.raises(ValidationError):
            Quota(metric="tokens", limit="100", per_seconds=60)

    def test_quota_magicmock_limit_rejected(self):
        with pytest.raises(ValidationError, match="limit must be int or float"):
            Quota(metric="tokens", limit=MagicMock(), per_seconds=60)

    def test_a02_config_model_family_bytes_rejected(self):
        with pytest.raises(ValidationError):
            PerModelConfig(
                quotas=UsageQuotas([Quota(metric="tokens", limit=10.0)]),
                model_family=b":bad:",
            )

    def test_a03_reservation_model_family_bytes_rejected(self):
        with pytest.raises(ValidationError):
            CapacityReservation(usage={"tokens": 1.0}, model_family=b":bad:")

    def test_a03_reservation_model_family_colon_rejected(self):
        with pytest.raises(ValidationError, match="must not contain ':'"):
            CapacityReservation(usage={"tokens": 1.0}, model_family=":bad:")


class TestReservationAndResponseGates:
    async def test_a04_magicmock_reservation_rejected(self):
        with pytest.raises(ValueError, match="CapacityReservation"):
            await _limiter().refund_capacity({"tokens": 0.0}, MagicMock())

    async def test_a05_magicmock_response_rejected(self):
        reservation = CapacityReservation(
            usage={"tokens": 50.0, "requests": 1.0},
            model_family="strict-family",
        )
        with pytest.raises(ValueError, match="total_tokens"):
            await _limiter().refund_capacity_from_response(
                reservation,
                response=MagicMock(),
            )

    def test_a06_total_tokens_read_once(self):
        class Usage:
            calls = 0

            @property
            def total_tokens(self) -> int:
                self.calls += 1
                return 12 if self.calls == 1 else 10**9

        usage = Usage()

        assert extract_total_tokens(usage) == 12.0
        assert usage.calls == 1

    def test_a07_truthy_property_override_not_unlimited(self):
        class WeirdReservation(CapacityReservation):
            @property
            def is_unlimited(self):
                return True

        reservation = WeirdReservation(
            usage={"tokens": 50.0},
            model_family="strict-family",
            bucket_ids={("tokens", 60)},
        )

        assert is_unlimited_reservation(reservation) is False


class TestStrictRuntimeValidators:
    def test_a08_magicmock_timeout_rejected(self):
        with pytest.raises(ValueError, match="timeout"):
            validate_timeout(MagicMock())

    def test_a09_magicmock_max_capacity_rejected(self):
        with pytest.raises(ValueError, match="max_capacity"):
            validate_max_capacity_value(MagicMock())

    async def test_a10_magicmock_usage_value_rejected(self):
        with pytest.raises(ValueError, match="int or float"):
            await _limiter().acquire_capacity(
                {"tokens": MagicMock(), "requests": 1.0},
                model="gpt-4",
            )

    def test_a11_lying_mapping_rejected_at_validate_extra_usage(self):
        with pytest.raises(ValueError, match="consistent key/value pairs"):
            validate_extra_usage(LyingMapping())


class TestNumpySubclassRejections:
    def test_a14_string_subclass_metric_rejected(self):
        class StringSubclass(str):
            __slots__ = ()

        with pytest.raises(ValueError, match="metric"):
            validate_metric(StringSubclass("tokens"))

    def test_a14_numpy_str_metric_rejected(self):
        np = pytest.importorskip("numpy")

        with pytest.raises(ValueError, match="metric"):
            validate_metric(np.str_("tokens"))

    def test_a15_float_per_seconds_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            validate_per_seconds(60.0)

    def test_a15_numpy_per_seconds_rejected_consistently(self):
        np = pytest.importorskip("numpy")

        with pytest.raises(ValueError, match="positive integer"):
            validate_per_seconds(np.int64(60))
        with pytest.raises(ValueError, match="positive integer"):
            validate_per_seconds(np.float64(60.0))
