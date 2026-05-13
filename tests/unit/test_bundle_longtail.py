"""Regression tests for FIX-22 long-tail standalone closures."""

import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter
from token_throttle._validation import validate_acquire_usage


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="test-family",
        quotas=UsageQuotas([Quota(metric="requests", limit=100, per_seconds=60)]),
        usage_counter=lambda **_kwargs: {"requests": 1},
    )


def _async_builder():
    backend = AsyncMock()
    backend.await_for_capacity.return_value = None
    builder = MagicMock()
    builder.build.return_value = backend
    return builder


def _sync_builder():
    backend = MagicMock()
    backend.wait_for_capacity.return_value = None
    builder = MagicMock()
    builder.build.return_value = backend
    return builder


async def test_async_request_model_name_typo_suggests_model_keyword():
    limiter = RateLimiter(_config(), backend=_async_builder())

    with pytest.raises(ValueError, match=r"did you mean 'model'.*model_name"):
        await limiter.acquire_capacity_for_request(model_name="gpt-4o")


def test_sync_request_model_name_typo_suggests_model_keyword():
    limiter = SyncRateLimiter(_config(), backend=_sync_builder())

    with pytest.raises(ValueError, match=r"did you mean 'model'.*model_name"):
        limiter.acquire_capacity_for_request(model_name="gpt-4o")


async def test_async_config_getter_is_rejected_at_construction():
    async def async_config_getter(_model_name: str) -> PerModelConfig:
        return _config()

    with pytest.raises(ValueError, match="synchronous PerModelConfig getter"):
        RateLimiter(async_config_getter, backend=_async_builder())


def test_sync_async_config_getter_is_rejected_at_construction():
    async def async_config_getter(_model_name: str) -> PerModelConfig:
        return _config()

    with pytest.raises(ValueError, match="synchronous PerModelConfig getter"):
        SyncRateLimiter(async_config_getter, backend=_sync_builder())


def test_usage_key_mismatch_reports_missing_and_extra_keys():
    quotas = UsageQuotas(
        [
            Quota(metric="input_tokens", limit=100, per_seconds=60),
            Quota(metric="requests", limit=10, per_seconds=60),
        ]
    )

    with pytest.raises(ValueError, match=r"missing=.*extra=") as exc_info:
        validate_acquire_usage({"inputt_tokens": 1, "requests": 1}, quotas)

    message = str(exc_info.value)
    assert "missing=['input_tokens']" in message
    assert "extra=['inputt_tokens']" in message


def test_duplicate_quota_error_includes_conflicting_limits():
    with pytest.raises(ValueError, match=r"existing limit=100.*new limit=200"):
        UsageQuotas(
            [
                Quota(metric="requests", limit=100, per_seconds=60),
                Quota(metric="requests", limit=200, per_seconds=60),
            ]
        )


@pytest.mark.parametrize(
    "builder_cls",
    [MemoryBackendBuilder, SyncMemoryBackendBuilder],
)
def test_memory_backend_builder_rejects_zero_sleep_interval(builder_cls):
    with pytest.raises(ValueError, match="sleep_interval"):
        builder_cls(sleep_interval=0)


def test_redis_backend_builder_rejects_sync_client_shape_for_async_builder():
    redis = pytest.importorskip("redis")
    redis_backend = importlib.import_module(
        "token_throttle._limiter_backends._redis._backend"
    )

    with pytest.raises(TypeError, match=r"redis\.asyncio\.Redis"):
        redis_backend.RedisBackendBuilder(redis.Redis(), key_prefix="test")


def test_sync_redis_backend_builder_rejects_async_client_shape_for_sync_builder():
    redis_async = pytest.importorskip("redis.asyncio")
    redis_sync_backend = importlib.import_module(
        "token_throttle._limiter_backends._redis._sync_backend"
    )

    with pytest.raises(TypeError, match=r"redis\.Redis"):
        redis_sync_backend.SyncRedisBackendBuilder(
            redis_async.Redis(),
            key_prefix="test",
        )
