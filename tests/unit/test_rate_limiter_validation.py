"""Tests for all ValueError paths in RateLimiter.acquire_capacity, acquire_capacity_for_request, and refund_capacity."""

import gc
import warnings
from collections import UserDict
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
from token_throttle._rate_limiter import RateLimiter

_UNLIMITED_FLAG = "__rate_limiting_disabled__"


def make_mock_backend_builder():
    """Create a mock backend builder that returns a mock backend."""
    mock_backend = AsyncMock()
    mock_backend.await_for_capacity.return_value = None
    mock_backend.refund_capacity.return_value = None

    mock_builder = MagicMock()
    mock_builder.build.return_value = mock_backend
    return mock_builder, mock_backend


def make_limited_config(
    *,
    model_family: str | None = None,
    usage_counter=None,
) -> PerModelConfig:
    """Create a PerModelConfig with tokens and requests quotas."""
    quotas = UsageQuotas(
        [
            Quota(metric="tokens", limit=1000),
            Quota(metric="requests", limit=10),
        ]
    )
    return PerModelConfig(
        quotas=quotas,
        model_family=model_family,
        usage_counter=usage_counter,
    )


def make_unlimited_config() -> PerModelConfig:
    return PerModelConfig(quotas=UsageQuotas.unlimited())


class TestAcquireCapacityValidation:
    """Tests for ValueError paths in acquire_capacity."""

    async def test_usage_must_be_mapping(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="usage must be a mapping"):
            await limiter.acquire_capacity([], model="gpt-4")

    async def test_empty_model_name_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="model_name cannot be empty"):
            await limiter.acquire_capacity({"tokens": 1, "requests": 1}, model="")

    async def test_same_model_cannot_change_model_family(self):
        builder, _ = make_mock_backend_builder()
        model_family = "family-a"

        def config_getter(_model_name: str) -> PerModelConfig:
            return make_limited_config(model_family=model_family)

        limiter = RateLimiter(config_getter, backend=builder)

        await limiter.acquire_capacity({"tokens": 1, "requests": 1}, model="gpt-4")

        model_family = "family-b"
        with pytest.raises(ValueError, match="changed model_family"):
            await limiter.acquire_capacity({"tokens": 1, "requests": 1}, model="gpt-4")

    async def test_same_model_can_toggle_unlimited_without_changing_model_family(
        self,
    ):
        builder, mock_backend = make_mock_backend_builder()
        use_unlimited = True

        def config_getter(_model_name: str) -> PerModelConfig:
            if use_unlimited:
                return PerModelConfig(
                    quotas=UsageQuotas.unlimited(),
                    model_family="family-a",
                )
            return make_limited_config(model_family="family-a")

        limiter = RateLimiter(config_getter, backend=builder)

        first = await limiter.acquire_capacity({}, model="gpt-4")
        assert first.is_unlimited is True

        use_unlimited = False
        second = await limiter.acquire_capacity(
            {"tokens": 1, "requests": 1},
            model="gpt-4",
        )

        use_unlimited = True
        third = await limiter.acquire_capacity({}, model="gpt-4")

        assert second.model_family == "family-a"
        assert third.is_unlimited is True
        builder.build.assert_called_once()
        mock_backend.await_for_capacity.assert_awaited_once()

    async def test_unlimited_config_with_nonempty_usage_returns_unlimited_reservation(
        self,
    ):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        reservation = await limiter.acquire_capacity({"tokens": 5}, model="gpt-4")

        assert reservation.model_family == _UNLIMITED_FLAG
        assert dict(reservation.usage) == {"tokens": 5.0}
        assert reservation.is_unlimited is True

    async def test_unlimited_config_with_empty_usage_returns_unlimited_reservation(
        self,
    ):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        reservation = await limiter.acquire_capacity({}, model="gpt-4")

        assert reservation.model_family == _UNLIMITED_FLAG
        assert dict(reservation.usage) == {}
        assert reservation.is_unlimited is True

    async def test_mismatched_usage_keys_vs_quota_keys_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="do not match quota keys"):
            await limiter.acquire_capacity({"tokens": 1}, model="gpt-4")

    async def test_negative_usage_value_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be non-negative"):
            await limiter.acquire_capacity({"tokens": -1, "requests": 1}, model="gpt-4")

    async def test_boolean_usage_value_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="must not be a boolean"):
            await limiter.acquire_capacity(
                {"tokens": True, "requests": 1},
                model="gpt-4",
            )

    async def test_non_numeric_usage_value_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            await limiter.acquire_capacity(
                {"tokens": object(), "requests": 1},
                model="gpt-4",
            )

    async def test_config_getter_returning_wrong_type_raises_value_error(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(lambda _model_name: {"quotas": []}, backend=builder)

        with pytest.raises(ValueError, match="must resolve to PerModelConfig"):
            await limiter.acquire_capacity(
                {"tokens": 1, "requests": 1},
                model="gpt-4",
            )


class TestAcquireCapacityForRequestValidation:
    """Tests for ValueError paths in acquire_capacity_for_request."""

    async def test_extra_usage_must_be_mapping_or_none(self):
        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = RateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="extra_usage must be a mapping or None"):
            await limiter.acquire_capacity_for_request(
                model="gpt-4",
                extra_usage=[],
            )

    async def test_missing_model_param_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="'model' parameter is required"):
            await limiter.acquire_capacity_for_request(extra_usage=None)

    async def test_false_model_in_request_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="model_name must be a string"):
            await limiter.acquire_capacity_for_request(model=False)

    async def test_empty_string_model_in_request_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="model_name cannot be empty"):
            await limiter.acquire_capacity_for_request(model="")

    async def test_integer_model_in_request_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="model_name must be a string"):
            await limiter.acquire_capacity_for_request(model=0)

    async def test_none_usage_counter_raises(self):
        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=None)
        limiter = RateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="usage_counter cannot be None"):
            await limiter.acquire_capacity_for_request(
                extra_usage=None,
                model="gpt-4",
            )

    async def test_extra_usage_with_unknown_key_raises(self):
        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = RateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="Usage key 'unknown_metric' not found"):
            await limiter.acquire_capacity_for_request(
                extra_usage={"unknown_metric": 5},
                model="gpt-4",
            )

    async def test_extra_usage_with_boolean_value_raises(self):
        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = RateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="must not be a boolean"):
            await limiter.acquire_capacity_for_request(
                extra_usage={"tokens": True},
                model="gpt-4",
            )

    async def test_extra_usage_with_non_numeric_value_raises(self):
        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = RateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="must be a finite number"):
            await limiter.acquire_capacity_for_request(
                extra_usage={"tokens": object()},
                model="gpt-4",
            )

    async def test_extra_usage_with_negative_value_raises(self):
        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = RateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="must be non-negative"):
            await limiter.acquire_capacity_for_request(
                extra_usage={"tokens": -1},
                model="gpt-4",
            )

    async def test_usage_counter_with_non_numeric_value_raises(self):
        def fake_counter(**_kwargs):
            return {"tokens": object(), "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = RateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            await limiter.acquire_capacity_for_request(model="gpt-4")

    async def test_usage_counter_boolean_value_is_not_masked_by_extra_usage(self):
        def fake_counter(**_kwargs):
            return {"tokens": True, "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = RateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="must not be a boolean"):
            await limiter.acquire_capacity_for_request(
                extra_usage={"tokens": 1},
                model="gpt-4",
            )

    async def test_usage_counter_returning_awaitable_raises_clear_error_without_leak(
        self,
    ):
        async def async_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        def fake_counter(**_kwargs):
            return async_counter(**_kwargs)

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = RateLimiter(config, backend=builder)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(
                ValueError,
                match=(
                    "usage_counter must be a synchronous callable returning "
                    "a usage mapping"
                ),
            ):
                await limiter.acquire_capacity_for_request(model="gpt-4")
            gc.collect()

        assert not any("was never awaited" in str(item.message) for item in caught)

    async def test_unlimited_config_returns_unlimited_reservation(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        reservation = await limiter.acquire_capacity_for_request(model="gpt-4")

        assert reservation.model_family == _UNLIMITED_FLAG
        assert dict(reservation.usage) == {}
        assert reservation.is_unlimited is True

    async def test_unlimited_config_with_extra_usage_returns_unlimited_reservation(
        self,
    ):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        reservation = await limiter.acquire_capacity_for_request(
            model="gpt-4", extra_usage={"tokens": 10}
        )

        assert reservation.model_family == _UNLIMITED_FLAG
        assert dict(reservation.usage) == {"tokens": 10.0}
        assert reservation.is_unlimited is True

    async def test_unlimited_config_with_empty_extra_usage_succeeds(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        reservation = await limiter.acquire_capacity_for_request(
            model="gpt-4", extra_usage={}
        )
        assert reservation.model_family == _UNLIMITED_FLAG
        assert reservation.is_unlimited is True

    async def test_unlimited_config_with_counter_and_extra_usage_accepts_new_keys(self):
        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(
            make_limited_config(usage_counter=fake_counter).model_copy(
                update={"quotas": UsageQuotas.unlimited()}
            ),
            backend=builder,
        )

        reservation = await limiter.acquire_capacity_for_request(
            model="gpt-4",
            extra_usage={"images": 2, "tokens": 5},
        )

        assert dict(reservation.usage) == {
            "tokens": 105.0,
            "requests": 1.0,
            "images": 2.0,
        }
        assert reservation.is_unlimited is True


class TestRefundCapacityValidation:
    """Tests for ValueError paths in refund_capacity."""

    async def test_unlimited_reservation_with_nonempty_usage_is_noop(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        reservation = CapacityReservation(
            usage={"tokens": 5},
            model_family=_UNLIMITED_FLAG,
            is_unlimited=True,
        )

        result = await limiter.refund_capacity({"tokens": 5}, reservation)

        assert result is None

    async def test_unlimited_reservation_with_empty_usage_is_noop(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        reservation = CapacityReservation(
            usage={},
            model_family=_UNLIMITED_FLAG,
            is_unlimited=True,
        )

        result = await limiter.refund_capacity({}, reservation)

        assert result is None

    async def test_model_family_matching_unlimited_flag_still_refunds(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(
            make_limited_config(model_family=_UNLIMITED_FLAG),
            backend=builder,
        )

        reservation = await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        await limiter.refund_capacity({"tokens": 80, "requests": 1}, reservation)

        mock_backend.refund_capacity_for_buckets.assert_awaited_once_with(
            reservation.get_usage(),
            {"tokens": 80, "requests": 1},
            bucket_ids=ANY,
        )

    async def test_mismatched_usage_keys_raises(self):
        builder, _mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        # First acquire to populate the backend cache
        await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        with pytest.raises(ValueError, match="do not match reservation usage keys"):
            await limiter.refund_capacity({"tokens": 50}, reservation)

    async def test_unrecognized_model_family_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="nonexistent-family",
        )

        with pytest.raises(ValueError, match="Backend not found for model family"):
            await limiter.refund_capacity(
                {"tokens": 50, "requests": 1},
                reservation,
            )

    async def test_boolean_actual_usage_value_raises(self):
        builder, _mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        with pytest.raises(ValueError, match="must not be a boolean"):
            await limiter.refund_capacity(
                {"tokens": 50, "requests": False},
                reservation,
            )

    async def test_non_numeric_actual_usage_value_raises(self):
        builder, _mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        with pytest.raises(ValueError, match="must be finite"):
            await limiter.refund_capacity(
                {"tokens": object(), "requests": 1},
                reservation,
            )


class TestRefundCapacityFromResponseValidation:
    """Tests for refund_capacity_from_response value paths."""

    async def test_response_with_none_usage_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        class FakeResponse:
            usage = None

        with pytest.raises(ValueError, match=r"response\.usage is None"):
            await limiter.refund_capacity_from_response(
                reservation, response=FakeResponse()
            )

    async def test_unlimited_reservation_is_noop(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={},
            model_family=_UNLIMITED_FLAG,
            is_unlimited=True,
        )
        result = await limiter.refund_capacity_from_response(reservation)
        assert result is None

    async def test_model_family_matching_unlimited_flag_still_refunds_response(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(
            make_limited_config(model_family=_UNLIMITED_FLAG),
            backend=builder,
        )
        reservation = await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        class FakeUsage:
            total_tokens = 80

        class FakeResponse:
            usage = FakeUsage()

        await limiter.refund_capacity_from_response(
            reservation,
            response=FakeResponse(),
        )

        mock_backend.refund_capacity_for_buckets.assert_awaited_once_with(
            reservation.get_usage(),
            {"tokens": 80, "requests": 1},
            bucket_ids=ANY,
        )

    async def test_pydantic_response_object(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        await limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4")
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        class FakeUsage:
            total_tokens = 80

        class FakeResponse:
            usage = FakeUsage()

        await limiter.refund_capacity_from_response(
            reservation, response=FakeResponse()
        )
        mock_backend.refund_capacity_for_buckets.assert_awaited_once()

    async def test_dict_response_object(self):
        """Response.usage is a dict (not object with attributes)."""
        builder, mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        await limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4")
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        class FakeResponse:
            usage = {"total_tokens": 80}  # noqa: RUF012

        await limiter.refund_capacity_from_response(
            reservation, response=FakeResponse()
        )
        mock_backend.refund_capacity_for_buckets.assert_awaited_once()

    async def test_response_dict_with_usage_key(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        await limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4")
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        await limiter.refund_capacity_from_response(
            reservation,
            response={"usage": {"total_tokens": 80}},
        )

        mock_backend.refund_capacity_for_buckets.assert_awaited_once_with(
            reservation.get_usage(),
            {"tokens": 80, "requests": 1},
            bucket_ids=ANY,
        )

    async def test_response_dict_missing_usage_key_raises_value_error(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        with pytest.raises(ValueError, match=r"response.*usage"):
            await limiter.refund_capacity_from_response(
                reservation,
                response={},
            )

    async def test_kwargs_usage_path(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        await limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4")
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        await limiter.refund_capacity_from_response(
            reservation, usage={"total_tokens": 80}
        )
        mock_backend.refund_capacity_for_buckets.assert_awaited_once()

    async def test_kwargs_usage_mapping_path(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        await limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4")
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        await limiter.refund_capacity_from_response(
            reservation,
            usage=UserDict({"total_tokens": 80}),
        )
        mock_backend.refund_capacity_for_buckets.assert_awaited_once_with(
            reservation.get_usage(),
            {"tokens": 80, "requests": 1},
            bucket_ids=ANY,
        )

    async def test_no_response_no_usage_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(
            ValueError, match="Either 'response' or 'usage' keyword argument"
        ):
            await limiter.refund_capacity_from_response(reservation)

    async def test_response_with_none_total_tokens_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        class FakeUsage:
            total_tokens = None

        class FakeResponse:
            usage = FakeUsage()

        with pytest.raises(ValueError, match="total_tokens is None"):
            await limiter.refund_capacity_from_response(
                reservation, response=FakeResponse()
            )

    async def test_kwargs_with_none_total_tokens_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(ValueError, match="total_tokens is None"):
            await limiter.refund_capacity_from_response(
                reservation, usage={"total_tokens": None}
            )

    async def test_kwargs_usage_missing_total_tokens_raises_value_error(self):
        """Missing total_tokens in usage kwarg should raise ValueError, not KeyError."""
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(ValueError, match="total_tokens"):
            await limiter.refund_capacity_from_response(
                reservation, usage={"prompt_tokens": 50}
            )

    async def test_response_dict_usage_missing_total_tokens_raises_value_error(self):
        """response.usage dict missing total_tokens should raise ValueError, not KeyError."""
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        class FakeResponse:
            usage = {"prompt_tokens": 50, "completion_tokens": 30}  # noqa: RUF012

        with pytest.raises(ValueError, match="total_tokens"):
            await limiter.refund_capacity_from_response(
                reservation, response=FakeResponse()
            )


class TestSetMaxCapacityValidation:
    async def test_set_max_capacity_without_prior_backend_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        with pytest.raises(ValueError, match="No backend for model family"):
            await limiter.set_max_capacity("gpt-4", "tokens", 60, 500.0)

    async def test_set_max_capacity_unlimited_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)
        with pytest.raises(ValueError, match="unlimited quotas"):
            await limiter.set_max_capacity("gpt-4", "tokens", 60, 500.0)

    async def test_set_max_capacity_delegates_to_backend(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        await limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4")
        await limiter.set_max_capacity("gpt-4", "tokens", 60, 500.0)
        mock_backend.set_max_capacity.assert_awaited_once_with("tokens", 60, 500.0)


class TestGetBackendValidation:
    async def test_empty_model_family_rejected_at_construction(self):
        """PerModelConfig rejects empty model_family at construction time."""
        with pytest.raises(
            ValidationError, match="model_family must not be an empty string"
        ):
            PerModelConfig(
                quotas=UsageQuotas(
                    [
                        Quota(metric="tokens", limit=1000),
                        Quota(metric="requests", limit=10),
                    ]
                ),
                model_family="",
            )


class TestExtractTotalTokensValidation:
    """_extract_total_tokens should fail-fast with clear errors, not defer to downstream."""

    async def test_boolean_total_tokens_from_response_raises(self):
        """Bool is an int subclass — must be rejected as total_tokens."""
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        await limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4")
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        class FakeUsage:
            total_tokens = True

        class FakeResponse:
            usage = FakeUsage()

        with pytest.raises(ValueError, match="total_tokens"):
            await limiter.refund_capacity_from_response(
                reservation, response=FakeResponse()
            )

    async def test_boolean_total_tokens_from_kwargs_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(ValueError, match="total_tokens"):
            await limiter.refund_capacity_from_response(
                reservation, usage={"total_tokens": True}
            )

    async def test_nan_total_tokens_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(ValueError, match="total_tokens"):
            await limiter.refund_capacity_from_response(
                reservation, usage={"total_tokens": float("nan")}
            )

    async def test_inf_total_tokens_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(ValueError, match="total_tokens"):
            await limiter.refund_capacity_from_response(
                reservation, usage={"total_tokens": float("inf")}
            )

    async def test_negative_total_tokens_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(ValueError, match="total_tokens"):
            await limiter.refund_capacity_from_response(
                reservation, usage={"total_tokens": -5}
            )

    async def test_non_numeric_string_total_tokens_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(ValueError, match="total_tokens"):
            await limiter.refund_capacity_from_response(
                reservation, usage={"total_tokens": "abc"}
            )

    async def test_numeric_string_total_tokens_is_coerced(self):
        """Numeric strings (e.g. from JSON) should be coerced to float."""
        builder, mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        await limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4")
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        await limiter.refund_capacity_from_response(
            reservation, usage={"total_tokens": "80"}
        )
        mock_backend.refund_capacity_for_buckets.assert_awaited_once()

    async def test_zero_total_tokens_is_valid(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        await limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4")
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        await limiter.refund_capacity_from_response(
            reservation, usage={"total_tokens": 0}
        )
        mock_backend.refund_capacity_for_buckets.assert_awaited_once()


class TestModelNameTypeValidation:
    """model parameter must be a string — non-strings should raise ValueError."""

    async def test_boolean_model_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="model_name must be a string"):
            await limiter.acquire_capacity({"tokens": 1, "requests": 1}, model=True)

    async def test_integer_model_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="model_name must be a string"):
            await limiter.acquire_capacity({"tokens": 1, "requests": 1}, model=42)

    async def test_none_model_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="model_name must be a string"):
            await limiter.acquire_capacity({"tokens": 1, "requests": 1}, model=None)


class TestTimeoutValidation:
    """timeout must be validated even for unlimited models (early-return path)."""

    async def test_acquire_capacity_boolean_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must not be a boolean"):
            await limiter.acquire_capacity({}, model="gpt-4", timeout=True)

    async def test_acquire_capacity_nan_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            await limiter.acquire_capacity({}, model="gpt-4", timeout=float("nan"))

    async def test_acquire_capacity_inf_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            await limiter.acquire_capacity({}, model="gpt-4", timeout=float("inf"))

    async def test_acquire_capacity_negative_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be non-negative"):
            await limiter.acquire_capacity({}, model="gpt-4", timeout=-1.0)

    async def test_acquire_capacity_string_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            await limiter.acquire_capacity({}, model="gpt-4", timeout="fast")

    async def test_acquire_capacity_for_request_boolean_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must not be a boolean"):
            await limiter.acquire_capacity_for_request(model="gpt-4", timeout=True)

    async def test_acquire_capacity_for_request_nan_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            await limiter.acquire_capacity_for_request(
                model="gpt-4", timeout=float("nan")
            )

    async def test_acquire_capacity_for_request_inf_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            await limiter.acquire_capacity_for_request(
                model="gpt-4", timeout=float("inf")
            )

    async def test_acquire_capacity_for_request_negative_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be non-negative"):
            await limiter.acquire_capacity_for_request(model="gpt-4", timeout=-1.0)

    async def test_acquire_capacity_for_request_string_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            await limiter.acquire_capacity_for_request(model="gpt-4", timeout="fast")


class TestRecordUsage:
    async def test_record_usage_calls_consume_capacity(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = RateLimiter(make_limited_config(), backend=builder)
        reservation = await limiter.record_usage(
            {"tokens": 100, "requests": 1}, model="gpt-4"
        )
        mock_backend.consume_capacity.assert_awaited_once()
        assert reservation.model_family == "gpt-4"

    async def test_record_usage_unlimited_is_noop(self):
        builder, _ = make_mock_backend_builder()
        limiter = RateLimiter(make_unlimited_config(), backend=builder)
        reservation = await limiter.record_usage({"tokens": 5}, model="gpt-4")
        assert reservation.model_family == _UNLIMITED_FLAG
        assert dict(reservation.usage) == {"tokens": 5.0}
        assert reservation.is_unlimited is True
