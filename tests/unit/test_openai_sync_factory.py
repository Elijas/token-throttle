"""
Tests for the synchronous OpenAI factory.

Source: token_throttle/_factories/_openai/_openai_sync_rate_limiter.py

Closes L04 P01 (sync OpenAI factory missing) and L09 Y01-Y04 (eager
validation of rpm/tpm/redis_client/callbacks at construction time).
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("redis", reason="redis package not installed")

import redis as _sync_redis
import redis.asyncio as _async_redis
from pydantic import ValidationError

from token_throttle._factories._openai._openai_sync_rate_limiter import (
    create_openai_redis_sync_rate_limiter,
)
from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)
from token_throttle._interfaces._models import CapacityReservation
from token_throttle._sync_rate_limiter import SyncRateLimiter
from token_throttle._validation import _UNLIMITED_FLAG


def _sync_redis_mock() -> MagicMock:
    """A MagicMock that passes the factory's isinstance(_, redis.Redis) check."""
    return MagicMock(spec=_sync_redis.Redis)


# 1. Happy-path smoke: factory returns a SyncRateLimiter with the right wiring.
def test_factory_smoke():
    limiter = create_openai_redis_sync_rate_limiter(
        _sync_redis_mock(),
        key_prefix="test",
        rpm=100,
        tpm=10_000,
    )
    assert isinstance(limiter, SyncRateLimiter)
    config = limiter._config_getter("gpt-4-0613")
    assert set(config.quotas.names) == {"requests", "tokens"}
    request_quotas = config.quotas.get_quotas("requests")
    token_quotas = config.quotas.get_quotas("tokens")
    assert request_quotas[0].limit == 100
    assert request_quotas[0].per_seconds == 60
    assert token_quotas[0].limit == 10_000
    assert token_quotas[0].per_seconds == 60
    # model_family resolution still works (date suffix stripped)
    assert config.model_family == "gpt-4"


# 2-6. Y01 closure: rpm/tpm validators fire at construction, not at first acquire.
def test_factory_eager_validation_rpm_zero():
    with pytest.raises(ValidationError, match="greater than 0"):
        create_openai_redis_sync_rate_limiter(
            _sync_redis_mock(), key_prefix="test", rpm=0, tpm=10_000
        )


def test_factory_eager_validation_rpm_negative():
    with pytest.raises(ValidationError, match="greater than 0"):
        create_openai_redis_sync_rate_limiter(
            _sync_redis_mock(), key_prefix="test", rpm=-1, tpm=10_000
        )


def test_factory_eager_validation_rpm_inf_nan():
    # inf/nan are caught by the explicit isinstance(int) gate before any Quota
    # construction (they are floats, not ints).
    with pytest.raises(TypeError, match="rpm must be an int"):
        create_openai_redis_sync_rate_limiter(
            _sync_redis_mock(), key_prefix="test", rpm=float("inf"), tpm=10_000
        )
    with pytest.raises(TypeError, match="rpm must be an int"):
        create_openai_redis_sync_rate_limiter(
            _sync_redis_mock(), key_prefix="test", rpm=float("nan"), tpm=10_000
        )


def test_factory_eager_validation_rpm_bool():
    # `isinstance(True, int)` is True in Python, so the bool gate runs first.
    with pytest.raises(TypeError, match="rpm must be an int"):
        create_openai_redis_sync_rate_limiter(
            _sync_redis_mock(), key_prefix="test", rpm=True, tpm=10_000
        )
    with pytest.raises(TypeError, match="tpm must be an int"):
        create_openai_redis_sync_rate_limiter(
            _sync_redis_mock(), key_prefix="test", rpm=100, tpm=False
        )


def test_factory_eager_validation_rpm_non_int():
    # Y03 closure: floats / strings rejected, not silently coerced.
    with pytest.raises(TypeError, match="rpm must be an int"):
        create_openai_redis_sync_rate_limiter(
            _sync_redis_mock(), key_prefix="test", rpm=1.5, tpm=10_000
        )
    with pytest.raises(TypeError, match="rpm must be an int"):
        create_openai_redis_sync_rate_limiter(
            _sync_redis_mock(), key_prefix="test", rpm="100", tpm=10_000
        )
    with pytest.raises(TypeError, match="tpm must be an int"):
        create_openai_redis_sync_rate_limiter(
            _sync_redis_mock(), key_prefix="test", rpm=100, tpm=2.5
        )


# 7. Y02 closure: non-redis.Redis instances rejected at construction.
def test_factory_redis_client_type_check():
    with pytest.raises(TypeError, match=r"redis_client must be a redis\.Redis"):
        create_openai_redis_sync_rate_limiter(
            None, key_prefix="test", rpm=100, tpm=10_000
        )
    with pytest.raises(TypeError, match=r"redis_client must be a redis\.Redis"):
        create_openai_redis_sync_rate_limiter(
            "redis://localhost:6379", key_prefix="test", rpm=100, tpm=10_000
        )
    # Async client must NOT be accepted (cross-mode misuse footgun).
    async_client = MagicMock(spec=_async_redis.Redis)
    with pytest.raises(TypeError, match=r"redis_client must be a redis\.Redis"):
        create_openai_redis_sync_rate_limiter(
            async_client, key_prefix="test", rpm=100, tpm=10_000
        )


# 8. Y04 closure: non-SyncRateLimiterCallbacks values rejected at construction.
def test_factory_callbacks_type_check():
    with pytest.raises(TypeError, match="callbacks must be a SyncRateLimiterCallbacks"):
        create_openai_redis_sync_rate_limiter(
            _sync_redis_mock(), key_prefix="test", rpm=100, tpm=10_000, callbacks=False
        )
    with pytest.raises(TypeError, match="callbacks must be a SyncRateLimiterCallbacks"):
        create_openai_redis_sync_rate_limiter(
            _sync_redis_mock(), key_prefix="test", rpm=100, tpm=10_000, callbacks={}
        )
    # Async RateLimiterCallbacks must not pass — sync field validators differ.
    async_cbs = RateLimiterCallbacks()
    with pytest.raises(TypeError, match="callbacks must be a SyncRateLimiterCallbacks"):
        create_openai_redis_sync_rate_limiter(
            _sync_redis_mock(),
            key_prefix="test",
            rpm=100,
            tpm=10_000,
            callbacks=async_cbs,
        )


# 9-11. R5 F04 merge contract mirrored from the async factory.
def test_factory_callbacks_none_applies_default_logger():
    limiter = create_openai_redis_sync_rate_limiter(
        _sync_redis_mock(), key_prefix="test", rpm=100, tpm=10_000
    )
    assert limiter._callbacks is not None
    assert limiter._callbacks.on_missing_consumption_data is not None


def test_factory_callbacks_empty_preserves_default_logger():
    empty = SyncRateLimiterCallbacks()
    limiter = create_openai_redis_sync_rate_limiter(
        _sync_redis_mock(), key_prefix="test", rpm=100, tpm=10_000, callbacks=empty
    )
    assert limiter._callbacks is not empty
    assert limiter._callbacks.on_missing_consumption_data is not None


def test_factory_callbacks_one_cb_user_preserved_default_merged():
    def on_capacity_consumed(
        *,
        model_family,
        preconsumption_capacities,
        postconsumption_capacities,
        usage,
        current_time,
    ):
        pass

    user_cbs = SyncRateLimiterCallbacks(on_capacity_consumed=on_capacity_consumed)
    limiter = create_openai_redis_sync_rate_limiter(
        _sync_redis_mock(), key_prefix="test", rpm=100, tpm=10_000, callbacks=user_cbs
    )
    assert limiter._callbacks is not user_cbs
    assert limiter._callbacks.on_capacity_consumed is on_capacity_consumed
    assert limiter._callbacks.on_missing_consumption_data is not None


# 12. Integration smoke: refund_capacity_from_response is wired and short-circuits
#     correctly for an unlimited reservation (no Redis required).  Mirrors step
#     4c from the audit (refund path round-trip).
def test_acquire_refund_round_trip_dict_response():
    limiter = create_openai_redis_sync_rate_limiter(
        _sync_redis_mock(), key_prefix="test", rpm=100, tpm=10_000
    )
    # Unlimited reservation short-circuits the refund path before any backend
    # call — verifying the method exists and accepts a dict-shaped OpenAI
    # response without raising.
    unlimited = CapacityReservation(
        usage={},
        model_family=_UNLIMITED_FLAG,
        bucket_ids=None,
        is_unlimited=True,
        limiter_instance_id=limiter._limiter_instance_id,
    )
    response_dict = {
        "id": "chatcmpl-abc",
        "model": "gpt-4",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    limiter.refund_capacity_from_response(unlimited, response_dict)


# 13. Y16 informational confirmation: typo'd kwargs raise TypeError naturally.
def test_factory_extra_kwargs_rejected():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        create_openai_redis_sync_rate_limiter(
            _sync_redis_mock(),
            key_prefix="test",
            rpm=100,
            tpm=10_000,
            rate_per_minute=60,  # typo'd kwarg
        )
