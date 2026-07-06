import asyncio
import time
import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

from token_throttle._interfaces import _callbacks
from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter


def _config(*, usage_counter=None) -> PerModelConfig:
    return PerModelConfig(
        model_family="test",
        usage_counter=usage_counter,
        quotas=UsageQuotas([Quota(metric="requests", limit=100, per_seconds=60)]),
    )


async def test_async_hanging_callback_is_bounded_by_callback_timeout(caplog):
    async def on_capacity_consumed(**_kwargs) -> None:
        await asyncio.sleep(60)

    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        callbacks=RateLimiterCallbacks(on_capacity_consumed=on_capacity_consumed),
        callback_timeout=0.02,
    )

    start = time.monotonic()
    await limiter.acquire_capacity({"requests": 1}, model="gpt-test")

    assert time.monotonic() - start < 0.5
    assert "callback exceeded" in caplog.text


def test_sync_hanging_callback_is_bounded_by_callback_timeout(caplog):
    def on_capacity_consumed(**_kwargs) -> None:
        time.sleep(1)

    limiter = SyncRateLimiter(
        _config(),
        backend=SyncMemoryBackendBuilder(),
        callbacks=SyncRateLimiterCallbacks(on_capacity_consumed=on_capacity_consumed),
        callback_timeout=0.02,
    )

    start = time.monotonic()
    limiter.acquire_capacity({"requests": 1}, model="gpt-test")

    assert time.monotonic() - start < 0.5
    assert "callback exceeded" in caplog.text


async def test_async_stubborn_callback_swallowing_cancellation_is_bounded(caplog):
    release = asyncio.Event()
    started = asyncio.Event()
    finished = asyncio.Event()

    async def on_capacity_consumed(**_kwargs) -> None:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # Stubborn: swallow the deadline cancellation and keep working.
            await release.wait()
        finally:
            finished.set()

    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        callbacks=RateLimiterCallbacks(on_capacity_consumed=on_capacity_consumed),
        callback_timeout=0.02,
    )

    start = time.monotonic()
    await limiter.acquire_capacity({"requests": 1}, model="gpt-test")
    elapsed = time.monotonic() - start

    # Bounded by callback_timeout, not by the callback's own runtime.
    assert elapsed < 0.5
    assert "callback exceeded" in caplog.text
    # The callback was abandoned, not awaited: it is still running in the
    # background rather than having blocked the acquire.
    assert started.is_set()
    assert not finished.is_set()

    # It completes once unblocked, with no "Task exception was never retrieved".
    release.set()
    await asyncio.wait_for(finished.wait(), timeout=1.0)
    assert "never retrieved" not in caplog.text


async def test_async_detached_callback_late_exception_is_logged(caplog):
    async def on_capacity_consumed(**_kwargs) -> None:
        await asyncio.sleep(0.05)
        raise RuntimeError("late callback boom")

    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        callbacks=RateLimiterCallbacks(on_capacity_consumed=on_capacity_consumed),
        callback_timeout=0.01,
    )

    before = set(_callbacks._DETACHED_CALLBACK_TASKS)
    await limiter.acquire_capacity({"requests": 1}, model="gpt-test")
    assert "callback exceeded" in caplog.text

    # The callback was abandoned mid-flight and keeps running in the background.
    # Wait for that task to finish; its logging done-callback was registered by
    # the limiter first, so it runs before the one we add here.
    (task,) = _callbacks._DETACHED_CALLBACK_TASKS - before
    done = asyncio.Event()
    task.add_done_callback(lambda _t: done.set())
    await asyncio.wait_for(done.wait(), timeout=1.0)

    # Its late error is surfaced, not dropped as an unretrieved task exception.
    assert "late callback boom" in caplog.text
    assert "never retrieved" not in caplog.text


def test_default_callback_timeout_is_thirty_seconds():
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    sync_limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())

    assert limiter._callback_timeout == 30.0
    assert sync_limiter._callback_timeout == 30.0


async def test_async_callback_warning_filter_error_does_not_escape(caplog):
    async def on_capacity_consumed(**_kwargs) -> None:
        raise RuntimeError("callback boom")

    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        callbacks=RateLimiterCallbacks(on_capacity_consumed=on_capacity_consumed),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        await limiter.acquire_capacity({"requests": 1}, model="gpt-test")

    assert "callback boom" in caplog.text


def test_sync_callback_warning_filter_error_does_not_escape(caplog):
    def on_capacity_consumed(**_kwargs) -> None:
        raise RuntimeError("callback boom")

    limiter = SyncRateLimiter(
        _config(),
        backend=SyncMemoryBackendBuilder(),
        callbacks=SyncRateLimiterCallbacks(on_capacity_consumed=on_capacity_consumed),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        limiter.acquire_capacity({"requests": 1}, model="gpt-test")

    assert "callback boom" in caplog.text


async def test_acquire_timeout_bounds_capacity_wait_not_backend_latency():
    backend = AsyncMock()

    async def slow_backend(_usage, *, timeout=None, **_kwargs) -> None:
        assert timeout == 0.01
        await asyncio.sleep(0.05)

    backend.await_for_capacity.side_effect = slow_backend
    builder = MagicMock()
    builder.build.return_value = backend
    limiter = RateLimiter(_config(), backend=builder)

    start = time.monotonic()
    await limiter.acquire_capacity({"requests": 1}, model="gpt-test", timeout=0.01)

    assert time.monotonic() - start >= 0.05


async def test_redis_exceptions_are_wrapped_at_limiter_boundary():
    redis_connection_error = type(
        "ConnectionError",
        (Exception,),
        {"__module__": "redis.exceptions"},
    )
    backend = AsyncMock()
    backend.await_for_capacity.side_effect = redis_connection_error("down")
    builder = MagicMock()
    builder.build.return_value = backend
    limiter = RateLimiter(_config(), backend=builder)

    with pytest.raises(RuntimeError, match="Redis error") as exc_info:
        await limiter.acquire_capacity({"requests": 1}, model="gpt-test")

    assert type(exc_info.value.__cause__).__module__ == "redis.exceptions"


def test_sync_redis_exceptions_are_wrapped_at_limiter_boundary():
    redis_connection_error = type(
        "ConnectionError",
        (Exception,),
        {"__module__": "redis.exceptions"},
    )
    backend = MagicMock()
    backend.wait_for_capacity.side_effect = redis_connection_error("down")
    builder = MagicMock()
    builder.build.return_value = backend
    limiter = SyncRateLimiter(_config(), backend=builder)

    with pytest.raises(RuntimeError, match="Redis error") as exc_info:
        limiter.acquire_capacity({"requests": 1}, model="gpt-test")

    assert type(exc_info.value.__cause__).__module__ == "redis.exceptions"


async def test_usage_counter_keyerror_is_wrapped_with_model_context():
    def usage_counter(**_kwargs):
        raise KeyError("unknown tokenizer")

    limiter = RateLimiter(
        _config(usage_counter=usage_counter),
        backend=MemoryBackendBuilder(),
    )

    with pytest.raises(ValueError, match="gpt-custom") as exc_info:
        await limiter.acquire_capacity_for_request(model="gpt-custom")

    assert isinstance(exc_info.value.__cause__, KeyError)


def test_sync_usage_counter_keyerror_is_wrapped_with_model_context():
    def usage_counter(**_kwargs):
        raise KeyError("unknown tokenizer")

    limiter = SyncRateLimiter(
        _config(usage_counter=usage_counter),
        backend=SyncMemoryBackendBuilder(),
    )

    with pytest.raises(ValueError, match="gpt-custom") as exc_info:
        limiter.acquire_capacity_for_request(model="gpt-custom")

    assert isinstance(exc_info.value.__cause__, KeyError)
