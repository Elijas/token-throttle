"""
R5 FIX-27 regression tests for OpenAI factory callback default merging.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("redis", reason="redis package not installed")

import redis as _sync_redis
import redis.asyncio as _async_redis

import token_throttle._interfaces._callbacks as _callbacks_module
from token_throttle._factories._openai._openai_rate_limiter import (
    create_openai_redis_rate_limiter,
)
from token_throttle._factories._openai._openai_sync_rate_limiter import (
    create_openai_redis_sync_rate_limiter,
)
from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)


def _async_redis_mock() -> MagicMock:
    return MagicMock(spec=_async_redis.Redis)


def _sync_redis_mock() -> MagicMock:
    return MagicMock(spec=_sync_redis.Redis)


async def test_async_factory_merges_single_user_callback_with_default_logger():
    async def on_wait_start(
        *,
        model_family,
        usage,
        preconsumption_capacities,
    ) -> None:
        pass

    user_callbacks = RateLimiterCallbacks(on_wait_start=on_wait_start)
    limiter = create_openai_redis_rate_limiter(
        _async_redis_mock(),
        key_prefix="test",
        rpm=100,
        tpm=10_000,
        callbacks=user_callbacks,
    )

    assert limiter._callbacks is not user_callbacks
    assert limiter._callbacks.on_wait_start is on_wait_start
    assert limiter._callbacks.on_missing_consumption_data is not None


def test_async_factory_empty_callbacks_preserves_factory_defaults():
    limiter = create_openai_redis_rate_limiter(
        _async_redis_mock(),
        key_prefix="test",
        rpm=100,
        tpm=10_000,
        callbacks=RateLimiterCallbacks(),
    )

    assert limiter._callbacks.on_wait_start is None
    assert limiter._callbacks.after_wait_end_consumption is None
    assert limiter._callbacks.on_capacity_consumed is None
    assert limiter._callbacks.on_capacity_refunded is None
    assert limiter._callbacks.on_missing_consumption_data is not None


async def test_async_factory_none_slot_falls_through_to_default_logger():
    async def on_wait_start(
        *,
        model_family,
        usage,
        preconsumption_capacities,
    ) -> None:
        pass

    user_callbacks = RateLimiterCallbacks(
        on_wait_start=on_wait_start,
        on_missing_consumption_data=None,
    )
    limiter = create_openai_redis_rate_limiter(
        _async_redis_mock(),
        key_prefix="test",
        rpm=100,
        tpm=10_000,
        callbacks=user_callbacks,
    )

    assert limiter._callbacks.on_wait_start is on_wait_start
    assert limiter._callbacks.on_missing_consumption_data is not None


def test_sync_factory_merges_single_user_callback_with_default_logger():
    def on_wait_start(
        *,
        model_family,
        usage,
        preconsumption_capacities,
    ) -> None:
        pass

    user_callbacks = SyncRateLimiterCallbacks(on_wait_start=on_wait_start)
    limiter = create_openai_redis_sync_rate_limiter(
        _sync_redis_mock(),
        key_prefix="test",
        rpm=100,
        tpm=10_000,
        callbacks=user_callbacks,
    )

    assert limiter._callbacks is not user_callbacks
    assert limiter._callbacks.on_wait_start is on_wait_start
    assert limiter._callbacks.on_missing_consumption_data is not None


async def test_async_factory_merged_default_logger_fires(caplog):
    async def on_wait_start(
        *,
        model_family,
        usage,
        preconsumption_capacities,
    ) -> None:
        pass

    limiter = create_openai_redis_rate_limiter(
        _async_redis_mock(),
        key_prefix="test",
        rpm=100,
        tpm=10_000,
        callbacks=RateLimiterCallbacks(on_wait_start=on_wait_start),
    )

    _callbacks_module._loguru_cache["factory"] = _callbacks_module._LOGURU_UNAVAILABLE
    try:
        with caplog.at_level("INFO", logger="token_throttle"):
            await limiter._callbacks.on_missing_consumption_data(
                model_family="gpt-4",
                usage_metric="tokens",
                per_seconds=60,
            )
    finally:
        _callbacks_module._loguru_cache.clear()

    assert "Rate limiter missing consumption data" in caplog.text
    assert "model_family='gpt-4'" in caplog.text
    assert "usage_metric='tokens'" in caplog.text
