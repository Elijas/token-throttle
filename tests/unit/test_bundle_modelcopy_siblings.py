"""Regression tests for FIX-30 MODELCOPY-SIBLINGS."""

from __future__ import annotations

import copy
import pickle

import pytest
from frozendict import frozendict
from pydantic import ValidationError

from token_throttle._capacity import CalculatedCapacity
from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import (
    CapacityReservation,
    Quota,
    UsageQuotas,
)
from token_throttle._validation import is_unlimited_reservation, resolve_config


def _quota() -> Quota:
    return Quota(metric="tokens", limit=100.0, per_seconds=60)


def _limited_quotas() -> UsageQuotas:
    return UsageQuotas([_quota()])


def _config() -> PerModelConfig:
    return PerModelConfig(quotas=_limited_quotas(), model_family="family")


def _reservation() -> CapacityReservation:
    return CapacityReservation(
        usage=frozendict({"tokens": 5.0}),
        model_family="family",
        bucket_ids=frozenset({("tokens", 60)}),
        model="gpt-4",
    )


async def _async_callback(**_kwargs) -> None:
    return None


def _sync_callback(**_kwargs) -> None:
    return None


class TestQuotaCopyAndState:
    @pytest.mark.parametrize(
        ("update", "message"),
        [
            pytest.param({"metric": "x:y"}, "must not contain ':'", id="colon"),
            pytest.param({"metric": "x{y}"}, "must not contain", id="brace"),
            pytest.param({"limit": -1.0}, "greater than", id="negative-limit"),
        ],
    )
    def test_model_copy_update_revalidates_quota(self, update, message):
        with pytest.raises(ValidationError, match=message):
            _quota().model_copy(update=update)

    def test_model_copy_valid_update_still_succeeds(self):
        copied = _quota().model_copy(update={"limit": 200.0})

        assert copied.limit == 200.0
        assert copied.metric == "tokens"

    def test_model_construct_is_disabled(self):
        with pytest.raises(TypeError, match="model_construct is disabled"):
            Quota.model_construct(metric="x:y", limit=1.0, per_seconds=60)

    def test_pickle_rejects_forged_extra_state(self):
        quota = _quota()
        object.__setattr__(quota, "evil_field", "x:y")

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            pickle.loads(pickle.dumps(quota))  # noqa: S301

    @pytest.mark.parametrize("copy_fn", [copy.copy, copy.deepcopy])
    def test_copy_paths_revalidate_forged_quota(self, copy_fn):
        quota = _quota()
        object.__setattr__(quota, "metric", "x:y")

        with pytest.raises(ValidationError, match="must not contain ':'"):
            copy_fn(quota)


class TestPerModelConfigCopyAndExactTypes:
    def test_model_copy_rejects_fake_unlimited_quotas(self):
        class FakeUnlimited:
            is_unlimited = True

        with pytest.raises(ValidationError):
            _config().model_copy(update={"quotas": FakeUnlimited()})

    def test_model_copy_rejects_usage_quotas_subclass(self):
        class EvilQuotas(UsageQuotas):
            @property
            def is_unlimited(self) -> bool:
                return True

        with pytest.raises(ValidationError, match="quotas must be a UsageQuotas"):
            _config().model_copy(update={"quotas": EvilQuotas.unlimited()})

    def test_model_copy_valid_quotas_update_still_succeeds(self):
        copied = _config().model_copy(update={"quotas": UsageQuotas.unlimited()})

        assert copied.is_unlimited is True

    def test_assignment_is_blocked(self):
        cfg = _config()

        with pytest.raises(ValidationError, match="frozen"):
            cfg.quotas = UsageQuotas.unlimited()

    def test_model_construct_is_disabled(self):
        with pytest.raises(TypeError, match="model_construct is disabled"):
            PerModelConfig.model_construct(quotas=UsageQuotas.unlimited())

    def test_resolve_config_rejects_per_model_config_subclass(self):
        class EvilConfig(PerModelConfig):
            def __getattribute__(self, name: str):
                if name == "is_unlimited":
                    return True
                return super().__getattribute__(name)

        evil = EvilConfig.model_validate(_config().model_dump())

        with pytest.raises(ValueError, match="cfg must resolve to PerModelConfig"):
            resolve_config(evil, "gpt-4")


class TestCallbacksCopyAndAssignment:
    def test_async_model_copy_rejects_sync_callback(self):
        callbacks = RateLimiterCallbacks(on_wait_start=_async_callback)

        with pytest.raises(ValidationError, match="must be an async callable"):
            callbacks.model_copy(update={"on_wait_start": _sync_callback})

    def test_sync_model_copy_rejects_async_callback(self):
        callbacks = SyncRateLimiterCallbacks(on_wait_start=_sync_callback)

        with pytest.raises(ValidationError, match="must be a synchronous callable"):
            callbacks.model_copy(update={"on_wait_start": _async_callback})

    def test_async_valid_model_copy_still_succeeds(self):
        callbacks = RateLimiterCallbacks()

        copied = callbacks.model_copy(update={"on_wait_start": _async_callback})

        assert copied.on_wait_start is _async_callback

    def test_callback_assignment_is_blocked(self):
        callbacks = RateLimiterCallbacks()

        with pytest.raises(ValidationError, match="frozen"):
            callbacks.on_wait_start = _async_callback

    def test_model_construct_is_disabled(self):
        with pytest.raises(TypeError, match="model_construct is disabled"):
            RateLimiterCallbacks.model_construct(on_wait_start=_sync_callback)


class TestCapacityReservationExactTypes:
    @pytest.mark.parametrize(
        "bucket_ids",
        [
            pytest.param(frozenset({("x:y", 60)}), id="colon"),
            pytest.param(frozenset({("x{y}", 60)}), id="brace"),
        ],
    )
    def test_bucket_id_metric_segment_is_validated(self, bucket_ids):
        with pytest.raises(ValidationError, match="bucket_id metric"):
            CapacityReservation(
                usage=frozendict({"tokens": 1.0}),
                model_family="family",
                bucket_ids=bucket_ids,
            )

    def test_model_copy_revalidates_bucket_id_metric(self):
        with pytest.raises(ValidationError, match="bucket_id metric"):
            _reservation().model_copy(update={"bucket_ids": {("x:y", 60)}})

    def test_model_construct_is_disabled(self):
        with pytest.raises(TypeError, match="model_construct is disabled"):
            CapacityReservation.model_construct(
                usage=frozendict({"tokens": 1.0}),
                model_family="family",
                is_unlimited=True,
            )

    def test_unlimited_check_rejects_subclass_override(self):
        class EvilReservation(CapacityReservation):
            def __getattribute__(self, name: str):
                if name == "is_unlimited":
                    return True
                return super().__getattribute__(name)

        evil = EvilReservation.model_validate(_reservation().model_dump())

        with pytest.raises(
            ValueError, match="reservation must be a CapacityReservation"
        ):
            is_unlimited_reservation(evil)


class TestUsageQuotasExactQuota:
    def test_add_metric_rejects_quota_subclass_before_reading_metric(self):
        class EvilQuota(Quota):
            @property
            def metric(self) -> str:
                return "x:y"

        evil = EvilQuota.model_validate(_quota().model_dump())

        with pytest.raises(ValueError, match="Each quota must be a Quota instance"):
            UsageQuotas([evil])


class TestCalculatedCapacity:
    @pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_amount(self, amount):
        with pytest.raises(ValidationError, match="finite"):
            CalculatedCapacity(amount=amount, is_fresh_start=False)

    def test_rejects_string_bool_coercion(self):
        with pytest.raises(ValidationError):
            CalculatedCapacity(amount=1.0, is_fresh_start="yes")

    def test_model_copy_revalidates_amount(self):
        capacity = CalculatedCapacity(amount=1.0, is_fresh_start=False)

        with pytest.raises(ValidationError, match="finite"):
            capacity.model_copy(update={"amount": float("nan")})

    def test_valid_negative_debt_amount_still_succeeds(self):
        capacity = CalculatedCapacity(amount=-1.0, is_fresh_start=False)

        assert capacity.amount == -1.0

    def test_model_construct_is_disabled(self):
        with pytest.raises(TypeError, match="model_construct is disabled"):
            CalculatedCapacity.model_construct(
                amount=float("nan"),
                is_fresh_start="yes",
            )
