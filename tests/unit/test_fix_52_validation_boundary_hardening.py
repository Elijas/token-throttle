from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from token_throttle._exceptions import CardinalityLimitExceededError
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import (
    MAX_PER_SECONDS,
    MAX_QUOTAS_PER_USAGE_QUOTAS,
    Quota,
    SecondsIn,
    UsageQuotas,
    frozen_usage,
)
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._redis._ttl import (
    MAX_REDIS_TTL_SECONDS,
    validate_redis_ttl_seconds,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter
from token_throttle._validation import resolve_config, validate_per_seconds


class IntSubclass(int):
    pass


def _config(*, usage_counter=None) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=60)]),
        model_family="family",
        usage_counter=usage_counter,
    )


def _mock_async_backend_builder() -> MagicMock:
    backend = AsyncMock()
    backend.await_for_capacity.return_value = None
    builder = MagicMock()
    builder.build.return_value = backend
    return builder


def test_int_subclass_passes_usage_but_is_rejected_for_capacity_identity() -> None:
    usage = frozen_usage({"tokens": IntSubclass(3)})

    assert usage["tokens"] == 3.0
    with pytest.raises(ValueError, match="exact int"):
        validate_per_seconds(IntSubclass(60))
    with pytest.raises(ValidationError, match="exact int number of seconds"):
        Quota(metric="tokens", limit=100, per_seconds=IntSubclass(60))


def test_seconds_in_enum_is_normalized_to_plain_int() -> None:
    quota = Quota(metric="tokens", limit=100, per_seconds=SecondsIn.MINUTE)

    assert quota.per_seconds == 60
    assert type(quota.per_seconds) is int


def test_usage_quotas_accepts_bounded_list_dict_and_generator() -> None:
    quota = Quota(metric="tokens", limit=100, per_seconds=60)

    assert UsageQuotas([quota]).names == ["tokens"]
    assert UsageQuotas({"tokens": quota}).names == ["tokens"]
    assert UsageQuotas(q for q in [quota]).names == ["tokens"]


def test_usage_quotas_rejects_iterable_beyond_cap() -> None:
    def quotas():
        for index in range(MAX_QUOTAS_PER_USAGE_QUOTAS + 1):
            yield Quota(metric=f"m{index}", limit=100, per_seconds=60)

    with pytest.raises(CardinalityLimitExceededError, match="at most 1000 entries"):
        UsageQuotas(quotas())


async def test_nul_in_model_alias_rejected_before_backend_build() -> None:
    builder = _mock_async_backend_builder()
    limiter = RateLimiter(_config(), backend=builder)

    with pytest.raises(ValueError, match=r"model_name.*control"):
        resolve_config(_config(), "bad\x00alias")

    with pytest.raises(ValueError, match=r"model_name.*control"):
        await limiter.acquire_capacity({"tokens": 1}, model="bad\x00alias")
    builder.build.assert_not_called()


def test_constructor_revalidates_static_config_before_backend_build() -> None:
    cfg = _config()
    cfg.__dict__["model_family"] = "bad\x00family"
    builder = _mock_async_backend_builder()

    with pytest.raises(ValidationError, match="model_family"):
        RateLimiter(cfg, backend=builder)

    builder.build.assert_not_called()


def test_sync_constructor_revalidates_static_config_before_backend_build() -> None:
    cfg = _config()
    cfg.__dict__["model_family"] = "bad\x00family"
    builder = MagicMock()

    with pytest.raises(ValidationError, match="model_family"):
        SyncRateLimiter(cfg, backend=builder)

    builder.build.assert_not_called()


def test_huge_ttl_and_per_seconds_are_rejected_eagerly() -> None:
    with pytest.raises(ValueError, match=r"bucket_ttl_seconds.*choose a smaller"):
        validate_redis_ttl_seconds(10**20, name="bucket_ttl_seconds")
    with pytest.raises(ValueError, match=r"per_seconds.*choose a smaller"):
        validate_per_seconds(10**20)
    with pytest.raises(ValidationError, match=f"{MAX_PER_SECONDS}"):
        Quota(metric="tokens", limit=100, per_seconds=MAX_PER_SECONDS + 1)


def test_ttl_boundary_allows_maximum_plain_int() -> None:
    assert (
        validate_redis_ttl_seconds(MAX_REDIS_TTL_SECONDS, name="bucket_ttl_seconds")
        == MAX_REDIS_TTL_SECONDS
    )


async def test_common_error_messages_include_parameter_value_and_next_action() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())

    with pytest.raises(ValueError, match=r"model.*required") as missing_model:
        await limiter.acquire_capacity_for_request(model_name="gpt-4o")
    message = str(missing_model.value)
    assert "'model' parameter is required" in message
    assert "model_name='gpt-4o'" in message
    assert "Use 'model'" in message

    with pytest.raises(ValueError, match="model_name must be a string") as bad_model:
        await limiter.acquire_capacity({"tokens": 1}, model=False)
    message = str(bad_model.value)
    assert "model_name must be a string" in message
    assert "got bool" in message
    assert "set the 'model' parameter" in message

    with pytest.raises(ValueError, match="model_name cannot be empty") as empty_model:
        await limiter.acquire_capacity({"tokens": 1}, model="")
    message = str(empty_model.value)
    assert "model_name cannot be empty" in message
    assert "non-empty model name string" in message

    with pytest.raises(
        ValueError,
        match="usage_counter cannot be None",
    ) as missing_counter:
        await limiter.acquire_capacity_for_request(model="gpt-4o")
    message = str(missing_counter.value)
    assert "limit_config.usage_counter cannot be None" in message
    assert "set usage_counter" in message
    assert "acquire_capacity" in message


def test_bad_static_cfg_type_rejected_at_constructor() -> None:
    with pytest.raises(ValueError, match="cfg must be a PerModelConfig") as exc_info:
        RateLimiter({"quotas": []}, backend=MemoryBackendBuilder())

    message = str(exc_info.value)
    assert "cfg must be a PerModelConfig or synchronous getter" in message
    assert "got dict" in message
    assert "pass PerModelConfig" in message
