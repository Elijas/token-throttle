"""Regression tests for FIX-51 callback dispatch hardening."""

from __future__ import annotations

import asyncio
import contextvars
import importlib
import threading
import time
import warnings
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from token_throttle._exceptions import AcquireRefundFailedError
from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
    _invoke_sync_callback_with_timeout,
    with_sync_callback_timeout,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="test/model",
        quotas=UsageQuotas([Quota(metric="requests", limit=20, per_seconds=60)]),
    )


def _reservation() -> CapacityReservation:
    return CapacityReservation(
        usage={"requests": 1},
        model_family="test/model",
        limiter_instance_id="limiter",
    )


def _acquire_refund_error() -> AcquireRefundFailedError:
    return AcquireRefundFailedError(
        reservation=_reservation(),
        refund_error=RuntimeError("refund failed"),
        interrupted_by=asyncio.CancelledError(),
    )


def _async_memory_backend():
    return MemoryBackendBuilder().build(_config())


def _sync_memory_backend():
    return SyncMemoryBackendBuilder().build(_config())


def _async_redis_backend():
    pytest.importorskip("redis", reason="redis package not installed")
    backend_module = importlib.import_module(
        "token_throttle._limiter_backends._redis._backend"
    )
    bucket_module = importlib.import_module(
        "token_throttle._limiter_backends._redis._bucket"
    )
    cfg = _config()
    quota = next(iter(cfg.quotas))
    bucket = bucket_module.RedisBucket(quota, cfg, AsyncMock(), key_prefix="test")
    return backend_module.RedisBackend([bucket], AsyncMock(), cfg, key_prefix="test")


def _sync_redis_backend():
    pytest.importorskip("redis", reason="redis package not installed")
    backend_module = importlib.import_module(
        "token_throttle._limiter_backends._redis._sync_backend"
    )
    bucket_module = importlib.import_module(
        "token_throttle._limiter_backends._redis._sync_bucket"
    )
    cfg = _config()
    quota = next(iter(cfg.quotas))
    bucket = bucket_module.SyncRedisBucket(quota, cfg, MagicMock(), key_prefix="test")
    return backend_module.SyncRedisBackend(
        [bucket], MagicMock(), cfg, key_prefix="test"
    )


ASYNC_BACKENDS: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("async-memory", _async_memory_backend),
    ("async-redis", _async_redis_backend),
)

SYNC_BACKENDS: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("sync-memory", _sync_memory_backend),
    ("sync-redis", _sync_redis_backend),
)

WAIT_START_KWARGS = {
    "model_family": "test/model",
    "usage": {"requests": 1.0},
    "preconsumption_capacities": {("requests", 60): 19.0},
}


def _sync_callback_raising(exc: BaseException):
    def callback(**_kwargs):
        raise exc

    return callback


def _async_callback_raising(exc: BaseException):
    async def callback(**_kwargs):
        raise exc

    return callback


@pytest.mark.parametrize(("backend_name", "backend_factory"), SYNC_BACKENDS)
def test_sync_plain_cancelled_error_propagates(
    backend_name: str, backend_factory: Callable[[], Any]
) -> None:
    backend = backend_factory()
    exc = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError) as raised:
        backend._invoke_callback_safe(_sync_callback_raising(exc))

    assert raised.value is exc


@pytest.mark.parametrize(("backend_name", "backend_factory"), SYNC_BACKENDS)
def test_sync_acquire_refund_failed_error_propagates(
    backend_name: str, backend_factory: Callable[[], Any]
) -> None:
    backend = backend_factory()
    exc = _acquire_refund_error()

    with pytest.raises(AcquireRefundFailedError) as raised:
        backend._invoke_callback_safe(_sync_callback_raising(exc))

    assert raised.value is exc


@pytest.mark.parametrize(("backend_name", "backend_factory"), ASYNC_BACKENDS)
async def test_async_acquire_refund_failed_error_propagates(
    backend_name: str, backend_factory: Callable[[], Any]
) -> None:
    backend = backend_factory()
    exc = _acquire_refund_error()

    with pytest.raises(AcquireRefundFailedError) as raised:
        await backend._invoke_callback_safe(_async_callback_raising(exc))

    assert raised.value is exc


def test_sync_generator_callback_rejected_at_registration() -> None:
    def on_wait_start(**_kwargs):
        yield None

    with pytest.raises(ValidationError, match="must not be a generator"):
        SyncRateLimiterCallbacks(on_wait_start=on_wait_start)


def test_sync_class_generator_callback_rejected_at_registration() -> None:
    class GeneratorCallback:
        def __call__(self, **_kwargs):
            yield None

    with pytest.raises(ValidationError, match="must not be a generator"):
        SyncRateLimiterCallbacks(on_wait_start=GeneratorCallback())


def test_async_generator_callback_rejected_at_registration() -> None:
    async def on_wait_start(**_kwargs):
        yield None

    with pytest.raises(ValidationError, match="must not be a generator"):
        RateLimiterCallbacks(on_wait_start=on_wait_start)


def test_async_class_generator_callback_rejected_at_registration() -> None:
    class AsyncGeneratorCallback:
        async def __call__(self, **_kwargs):
            yield None

    with pytest.raises(ValidationError, match="must not be a generator"):
        RateLimiterCallbacks(on_wait_start=AsyncGeneratorCallback())


def test_sync_callback_returning_coroutine_is_closed_and_rejected() -> None:
    returned = []

    async def accidental_coroutine():
        return None

    def on_wait_start(**_kwargs):
        coroutine = accidental_coroutine()
        returned.append(coroutine)
        return coroutine

    with pytest.raises(TypeError, match="returned an awaitable"):
        _invoke_sync_callback_with_timeout(on_wait_start, None, **WAIT_START_KWARGS)

    assert returned
    assert returned[0].cr_frame is None


def test_timeout_wrapped_sync_callback_copies_contextvars() -> None:
    trace_id = contextvars.ContextVar("trace_id", default="missing")
    seen: list[tuple[str, int]] = []
    caller_thread_id = threading.get_ident()

    def on_wait_start(**_kwargs) -> None:
        seen.append((trace_id.get(), threading.get_ident()))

    callbacks = SyncRateLimiterCallbacks(on_wait_start=on_wait_start)
    wrapped = with_sync_callback_timeout(callbacks, 1.0)

    token = trace_id.set("request-123")
    try:
        wrapped.on_wait_start(**WAIT_START_KWARGS)
    finally:
        trace_id.reset(token)

    assert len(seen) == 1
    assert seen[0][0] == "request-123"
    assert seen[0][1] != caller_thread_id


def test_late_exception_after_sync_callback_timeout_is_reported(caplog) -> None:
    caplog.set_level("WARNING", logger="token_throttle")

    def on_wait_start(**_kwargs) -> None:
        time.sleep(0.05)
        raise ValueError("late boom")

    callbacks = SyncRateLimiterCallbacks(on_wait_start=on_wait_start)
    wrapped = with_sync_callback_timeout(callbacks, 0.01)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapped.on_wait_start(**WAIT_START_KWARGS)

        deadline = time.monotonic() + 1.0
        while (
            "raised after callback_timeout elapsed ValueError: late boom"
            not in caplog.text
        ):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)

    assert "Rate limiter callback exceeded 0.010s timeout; skipping" in caplog.text
    assert "raised after callback_timeout elapsed ValueError: late boom" in caplog.text
    assert any(
        "raised after callback_timeout elapsed ValueError: late boom"
        in str(warning.message)
        for warning in caught
    )
