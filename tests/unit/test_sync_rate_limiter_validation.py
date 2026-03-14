"""Tests for all ValueError paths in SyncRateLimiter.acquire_capacity and refund_capacity."""

from unittest.mock import MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
from token_throttle._sync_rate_limiter import SyncRateLimiter

_UNLIMITED_FLAG = "__rate_limiting_disabled__"


def make_mock_backend_builder():
    """Create a mock backend builder that returns a mock backend."""
    mock_backend = MagicMock()
    mock_backend.wait_for_capacity.return_value = None
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

    def test_empty_model_name_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="model_name cannot be empty"):
            limiter.acquire_capacity({"tokens": 1, "requests": 1}, model="")

    def test_unlimited_config_with_nonempty_usage_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(
            ValueError, match="Usage must be empty for unlimited capacity"
        ):
            limiter.acquire_capacity({"tokens": 5}, model="gpt-4")

    def test_unlimited_config_with_empty_usage_returns_unlimited_reservation(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        reservation = limiter.acquire_capacity({}, model="gpt-4")

        assert reservation.model_family == _UNLIMITED_FLAG
        assert dict(reservation.usage) == {}

    def test_mismatched_usage_keys_vs_quota_keys_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="do not match quota keys"):
            limiter.acquire_capacity({"tokens": 1}, model="gpt-4")

    def test_negative_usage_value_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be non-negative"):
            limiter.acquire_capacity({"tokens": -1, "requests": 1}, model="gpt-4")

    def test_boolean_usage_value_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="must not be a boolean"):
            limiter.acquire_capacity(
                {"tokens": True, "requests": 1},
                model="gpt-4",
            )

    def test_non_numeric_usage_value_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            limiter.acquire_capacity(
                {"tokens": object(), "requests": 1},
                model="gpt-4",
            )


class TestRefundCapacityValidation:
    """Tests for ValueError paths in refund_capacity."""

    def test_unlimited_reservation_with_nonempty_usage_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        reservation = CapacityReservation(usage={}, model_family=_UNLIMITED_FLAG)

        with pytest.raises(
            ValueError,
            match="Usage must be empty for unlimited capacity reservations",
        ):
            limiter.refund_capacity({"tokens": 5}, reservation)

    def test_unlimited_reservation_with_empty_usage_is_noop(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        reservation = CapacityReservation(usage={}, model_family=_UNLIMITED_FLAG)

        result = limiter.refund_capacity({}, reservation)

        assert result is None

    def test_model_family_matching_unlimited_flag_still_refunds(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = SyncRateLimiter(
            make_limited_config(model_family=_UNLIMITED_FLAG),
            backend=builder,
        )

        reservation = limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        limiter.refund_capacity({"tokens": 80, "requests": 1}, reservation)

        mock_backend.refund_capacity.assert_called_once_with(
            reservation.get_usage(),
            {"tokens": 80, "requests": 1},
        )

    def test_mismatched_usage_keys_raises(self):
        builder, _mock_backend = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        # First acquire to populate the backend cache
        limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        with pytest.raises(ValueError, match="do not match reservation usage keys"):
            limiter.refund_capacity({"tokens": 50}, reservation)

    def test_unrecognized_model_family_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="nonexistent-family",
        )

        with pytest.raises(ValueError, match="Backend not found for model family"):
            limiter.refund_capacity(
                {"tokens": 50, "requests": 1},
                reservation,
            )

    def test_boolean_actual_usage_value_raises(self):
        builder, _mock_backend = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        with pytest.raises(ValueError, match="must not be a boolean"):
            limiter.refund_capacity(
                {"tokens": 50, "requests": False},
                reservation,
            )

    def test_non_numeric_actual_usage_value_raises(self):
        builder, _mock_backend = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        with pytest.raises(ValueError, match="must be finite"):
            limiter.refund_capacity(
                {"tokens": object(), "requests": 1},
                reservation,
            )


class TestAcquireCapacityForRequestValidation:
    """Tests for ValueError paths in acquire_capacity_for_request."""

    def test_missing_model_param_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="'model' parameter is required"):
            limiter.acquire_capacity_for_request(extra_usage=None)

    def test_none_usage_counter_raises(self):
        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=None)
        limiter = SyncRateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="usage_counter cannot be None"):
            limiter.acquire_capacity_for_request(
                extra_usage=None,
                model="gpt-4",
            )

    def test_extra_usage_with_unknown_key_raises(self):
        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = SyncRateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="Usage key 'unknown_metric' not found"):
            limiter.acquire_capacity_for_request(
                extra_usage={"unknown_metric": 5},
                model="gpt-4",
            )

    def test_extra_usage_with_boolean_value_raises(self):
        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = SyncRateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="must not be a boolean"):
            limiter.acquire_capacity_for_request(
                extra_usage={"tokens": True},
                model="gpt-4",
            )

    def test_extra_usage_with_non_numeric_value_raises(self):
        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = SyncRateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="must be a finite number"):
            limiter.acquire_capacity_for_request(
                extra_usage={"tokens": object()},
                model="gpt-4",
            )

    def test_extra_usage_with_negative_value_raises(self):
        def fake_counter(**_kwargs):
            return {"tokens": 100.0, "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = SyncRateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="must be non-negative"):
            limiter.acquire_capacity_for_request(
                extra_usage={"tokens": -1},
                model="gpt-4",
            )

    def test_usage_counter_with_non_numeric_value_raises(self):
        def fake_counter(**_kwargs):
            return {"tokens": object(), "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = SyncRateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            limiter.acquire_capacity_for_request(model="gpt-4")

    def test_usage_counter_boolean_value_is_not_masked_by_extra_usage(self):
        def fake_counter(**_kwargs):
            return {"tokens": True, "requests": 1.0}

        builder, _ = make_mock_backend_builder()
        config = make_limited_config(usage_counter=fake_counter)
        limiter = SyncRateLimiter(config, backend=builder)

        with pytest.raises(ValueError, match="must not be a boolean"):
            limiter.acquire_capacity_for_request(
                extra_usage={"tokens": 1},
                model="gpt-4",
            )

    def test_unlimited_config_returns_unlimited_reservation(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        reservation = limiter.acquire_capacity_for_request(model="gpt-4")

        assert reservation.model_family == _UNLIMITED_FLAG
        assert dict(reservation.usage) == {}

    def test_unlimited_config_with_extra_usage_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(
            ValueError, match="extra_usage must be empty for unlimited capacity"
        ):
            limiter.acquire_capacity_for_request(
                model="gpt-4", extra_usage={"tokens": 10}
            )

    def test_unlimited_config_with_empty_extra_usage_succeeds(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        reservation = limiter.acquire_capacity_for_request(
            model="gpt-4", extra_usage={}
        )
        assert reservation.model_family == _UNLIMITED_FLAG


class TestRefundCapacityFromResponseValidation:
    """Tests for refund_capacity_from_response value paths."""

    def test_unlimited_reservation_is_noop(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        reservation = CapacityReservation(usage={}, model_family=_UNLIMITED_FLAG)

        result = limiter.refund_capacity_from_response(reservation)

        assert result is None

    def test_model_family_matching_unlimited_flag_still_refunds_response(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = SyncRateLimiter(
            make_limited_config(model_family=_UNLIMITED_FLAG),
            backend=builder,
        )

        reservation = limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        class FakeUsage:
            total_tokens = 80

        class FakeResponse:
            usage = FakeUsage()

        limiter.refund_capacity_from_response(
            reservation,
            response=FakeResponse(),
        )

        mock_backend.refund_capacity.assert_called_once_with(
            reservation.get_usage(),
            {"tokens": 80, "requests": 1},
        )

    def test_pydantic_response_object(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        # Acquire first to populate the backend cache
        limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        # Simulate a pydantic response object with .usage.total_tokens
        class FakeUsage:
            total_tokens = 80

        class FakeResponse:
            usage = FakeUsage()

        limiter.refund_capacity_from_response(reservation, response=FakeResponse())

        mock_backend.refund_capacity.assert_called_once()

    def test_response_with_none_usage_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        class FakeResponse:
            usage = None

        with pytest.raises(ValueError, match=r"response\.usage is None"):
            limiter.refund_capacity_from_response(reservation, response=FakeResponse())

    def test_dict_kwargs_usage(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        # Acquire first to populate the backend cache
        limiter.acquire_capacity(
            {"tokens": 100, "requests": 1},
            model="gpt-4",
        )

        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        limiter.refund_capacity_from_response(
            reservation,
            usage={"total_tokens": 80},
        )

        mock_backend.refund_capacity.assert_called_once()

    def test_dict_response_object(self):
        """Response.usage is a dict (not object with attributes)."""
        builder, mock_backend = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4")
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        class FakeResponse:
            usage = {"total_tokens": 80}  # noqa: RUF012

        limiter.refund_capacity_from_response(reservation, response=FakeResponse())
        mock_backend.refund_capacity.assert_called_once()

    def test_no_response_no_usage_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(
            ValueError, match="Either 'response' or 'usage' keyword argument"
        ):
            limiter.refund_capacity_from_response(reservation)

    def test_response_with_none_total_tokens_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        class FakeUsage:
            total_tokens = None

        class FakeResponse:
            usage = FakeUsage()

        with pytest.raises(ValueError, match="total_tokens is None"):
            limiter.refund_capacity_from_response(reservation, response=FakeResponse())

    def test_kwargs_with_none_total_tokens_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(ValueError, match="total_tokens is None"):
            limiter.refund_capacity_from_response(
                reservation, usage={"total_tokens": None}
            )

    def test_kwargs_usage_missing_total_tokens_raises_value_error(self):
        """Missing total_tokens in usage kwarg should raise ValueError, not KeyError."""
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(ValueError, match="total_tokens"):
            limiter.refund_capacity_from_response(
                reservation, usage={"prompt_tokens": 50}
            )

    def test_response_dict_usage_missing_total_tokens_raises_value_error(self):
        """response.usage dict missing total_tokens should raise ValueError, not KeyError."""
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        class FakeResponse:
            usage = {"prompt_tokens": 50, "completion_tokens": 30}  # noqa: RUF012

        with pytest.raises(ValueError, match="total_tokens"):
            limiter.refund_capacity_from_response(
                reservation, response=FakeResponse()
            )


class TestExtractTotalTokensValidation:
    """_extract_total_tokens should fail-fast with clear errors, not defer to downstream."""

    def test_boolean_total_tokens_from_response_raises(self):
        """Bool is an int subclass — must be rejected as total_tokens."""
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4")
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )

        class FakeUsage:
            total_tokens = True

        class FakeResponse:
            usage = FakeUsage()

        with pytest.raises(ValueError, match="total_tokens"):
            limiter.refund_capacity_from_response(
                reservation, response=FakeResponse()
            )

    def test_boolean_total_tokens_from_kwargs_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(ValueError, match="total_tokens"):
            limiter.refund_capacity_from_response(
                reservation, usage={"total_tokens": True}
            )

    def test_nan_total_tokens_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(ValueError, match="total_tokens"):
            limiter.refund_capacity_from_response(
                reservation, usage={"total_tokens": float("nan")}
            )

    def test_inf_total_tokens_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(ValueError, match="total_tokens"):
            limiter.refund_capacity_from_response(
                reservation, usage={"total_tokens": float("inf")}
            )

    def test_negative_total_tokens_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(ValueError, match="total_tokens"):
            limiter.refund_capacity_from_response(
                reservation, usage={"total_tokens": -5}
            )

    def test_non_numeric_string_total_tokens_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        with pytest.raises(ValueError, match="total_tokens"):
            limiter.refund_capacity_from_response(
                reservation, usage={"total_tokens": "abc"}
            )

    def test_numeric_string_total_tokens_is_coerced(self):
        """Numeric strings (e.g. from JSON) should be coerced to float."""
        builder, mock_backend = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4")
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        limiter.refund_capacity_from_response(
            reservation, usage={"total_tokens": "80"}
        )
        mock_backend.refund_capacity.assert_called_once()

    def test_zero_total_tokens_is_valid(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4")
        reservation = CapacityReservation(
            usage={"tokens": 100.0, "requests": 1.0},
            model_family="gpt-4",
        )
        limiter.refund_capacity_from_response(
            reservation, usage={"total_tokens": 0}
        )
        mock_backend.refund_capacity.assert_called_once()


class TestSetMaxCapacityValidation:
    def test_set_max_capacity_without_prior_backend_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        with pytest.raises(ValueError, match="No backend for model family"):
            limiter.set_max_capacity("gpt-4", "tokens", 60, 500.0)

    def test_set_max_capacity_unlimited_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)
        with pytest.raises(ValueError, match="unlimited quotas"):
            limiter.set_max_capacity("gpt-4", "tokens", 60, 500.0)

    def test_set_max_capacity_delegates_to_backend(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        limiter.acquire_capacity({"tokens": 100, "requests": 1}, model="gpt-4")
        limiter.set_max_capacity("gpt-4", "tokens", 60, 500.0)
        mock_backend.set_max_capacity.assert_called_once_with("tokens", 60, 500.0)


class TestGetBackendValidation:
    def test_empty_model_family_raises(self):
        """_get_backend rejects empty model_family (defensive guard)."""
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        cfg = PerModelConfig(
            quotas=UsageQuotas(
                [
                    Quota(metric="tokens", limit=1000),
                    Quota(metric="requests", limit=10),
                ]
            ),
            model_family="",
        )
        with pytest.raises(ValueError, match=r"cfg\.model_family cannot be empty"):
            limiter._get_backend(cfg)


class TestModelNameTypeValidation:
    """model parameter must be a string — non-strings should raise ValueError."""

    def test_boolean_model_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="model_name must be a string"):
            limiter.acquire_capacity(
                {"tokens": 1, "requests": 1}, model=True
            )

    def test_integer_model_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="model_name must be a string"):
            limiter.acquire_capacity(
                {"tokens": 1, "requests": 1}, model=42
            )

    def test_none_model_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)

        with pytest.raises(ValueError, match="model_name must be a string"):
            limiter.acquire_capacity(
                {"tokens": 1, "requests": 1}, model=None
            )


class TestTimeoutValidation:
    """timeout must be validated even for unlimited models (early-return path)."""

    def test_acquire_capacity_boolean_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must not be a boolean"):
            limiter.acquire_capacity({}, model="gpt-4", timeout=True)

    def test_acquire_capacity_nan_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            limiter.acquire_capacity({}, model="gpt-4", timeout=float("nan"))

    def test_acquire_capacity_inf_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            limiter.acquire_capacity({}, model="gpt-4", timeout=float("inf"))

    def test_acquire_capacity_negative_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be non-negative"):
            limiter.acquire_capacity({}, model="gpt-4", timeout=-1.0)

    def test_acquire_capacity_string_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            limiter.acquire_capacity({}, model="gpt-4", timeout="fast")

    def test_acquire_capacity_for_request_boolean_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must not be a boolean"):
            limiter.acquire_capacity_for_request(model="gpt-4", timeout=True)

    def test_acquire_capacity_for_request_nan_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            limiter.acquire_capacity_for_request(
                model="gpt-4", timeout=float("nan")
            )

    def test_acquire_capacity_for_request_inf_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            limiter.acquire_capacity_for_request(
                model="gpt-4", timeout=float("inf")
            )

    def test_acquire_capacity_for_request_negative_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be non-negative"):
            limiter.acquire_capacity_for_request(model="gpt-4", timeout=-1.0)

    def test_acquire_capacity_for_request_string_timeout_raises(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)

        with pytest.raises(ValueError, match="must be finite"):
            limiter.acquire_capacity_for_request(model="gpt-4", timeout="fast")


class TestRecordUsage:
    def test_record_usage_calls_consume_capacity(self):
        builder, mock_backend = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_limited_config(), backend=builder)
        reservation = limiter.record_usage(
            {"tokens": 100, "requests": 1}, model="gpt-4"
        )
        mock_backend.consume_capacity.assert_called_once()
        assert reservation.model_family == "gpt-4"

    def test_record_usage_unlimited_is_noop(self):
        builder, _ = make_mock_backend_builder()
        limiter = SyncRateLimiter(make_unlimited_config(), backend=builder)
        reservation = limiter.record_usage({}, model="gpt-4")
        assert reservation.model_family == _UNLIMITED_FLAG
