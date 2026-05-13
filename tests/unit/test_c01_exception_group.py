"""Regression tests for L12 C01/C02 callback exception-group handling."""

import asyncio
import importlib
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="test/model",
        quotas=UsageQuotas([Quota(metric="requests", limit=20, per_seconds=60)]),
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
    return backend_module.RedisBackend([bucket], AsyncMock(), cfg)


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
    return backend_module.SyncRedisBackend([bucket], MagicMock(), cfg)


ASYNC_BACKENDS: tuple[tuple[str, Callable], ...] = (
    ("async-memory", _async_memory_backend),
    ("async-redis", _async_redis_backend),
)

SYNC_BACKENDS: tuple[tuple[str, Callable], ...] = (
    ("sync-memory", _sync_memory_backend),
    ("sync-redis", _sync_redis_backend),
)

PROCESS_SIGNAL_TYPES = (KeyboardInterrupt, SystemExit, GeneratorExit)


def _sync_callback_raising(exc: BaseException):
    def callback(**_kwargs):
        raise exc

    return callback


def _async_callback_raising(exc: BaseException):
    async def callback(**_kwargs):
        raise exc

    return callback


@pytest.mark.parametrize(("backend_name", "backend_factory"), ASYNC_BACKENDS)
async def test_base_exception_group_cancelled_error_propagates_async(
    backend_name: str, backend_factory: Callable
) -> None:
    backend = backend_factory()
    exc = BaseExceptionGroup("g", [asyncio.CancelledError()])

    with pytest.raises(BaseExceptionGroup) as raised:
        await backend._invoke_callback_safe(_async_callback_raising(exc))

    assert raised.value is exc


@pytest.mark.parametrize(("backend_name", "backend_factory"), ASYNC_BACKENDS)
@pytest.mark.parametrize("exc_type", PROCESS_SIGNAL_TYPES)
async def test_base_exception_group_process_signals_propagate_async(
    backend_name: str, backend_factory: Callable, exc_type: type[BaseException]
) -> None:
    backend = backend_factory()
    exc = BaseExceptionGroup("g", [exc_type()])

    with pytest.raises(BaseExceptionGroup) as raised:
        await backend._invoke_callback_safe(_async_callback_raising(exc))

    assert raised.value is exc


@pytest.mark.parametrize(("backend_name", "backend_factory"), SYNC_BACKENDS)
@pytest.mark.parametrize("exc_type", PROCESS_SIGNAL_TYPES)
def test_base_exception_group_process_signals_propagate_sync(
    backend_name: str, backend_factory: Callable, exc_type: type[BaseException]
) -> None:
    backend = backend_factory()
    exc = BaseExceptionGroup("g", [exc_type()])

    with pytest.raises(BaseExceptionGroup) as raised:
        backend._invoke_callback_safe(_sync_callback_raising(exc))

    assert raised.value is exc


@pytest.mark.parametrize(("backend_name", "backend_factory"), ASYNC_BACKENDS)
async def test_base_exception_group_value_error_warns_async(
    backend_name: str, backend_factory: Callable
) -> None:
    backend = backend_factory()

    with pytest.warns(
        RuntimeWarning, match=r"Rate limiter callback raised (?:Base)?ExceptionGroup"
    ):
        await backend._invoke_callback_safe(
            _async_callback_raising(ExceptionGroup("g", [ValueError("x")]))
        )


@pytest.mark.parametrize(("backend_name", "backend_factory"), SYNC_BACKENDS)
def test_base_exception_group_value_error_warns_sync(
    backend_name: str, backend_factory: Callable
) -> None:
    backend = backend_factory()

    with pytest.warns(
        RuntimeWarning, match=r"Rate limiter callback raised (?:Base)?ExceptionGroup"
    ):
        backend._invoke_callback_safe(
            _sync_callback_raising(ExceptionGroup("g", [ValueError("x")]))
        )


@pytest.mark.parametrize(("backend_name", "backend_factory"), ASYNC_BACKENDS)
async def test_plain_cancelled_error_still_propagates_async(
    backend_name: str, backend_factory: Callable
) -> None:
    backend = backend_factory()

    with pytest.raises(asyncio.CancelledError):
        await backend._invoke_callback_safe(
            _async_callback_raising(asyncio.CancelledError())
        )


@pytest.mark.parametrize(("backend_name", "backend_factory"), ASYNC_BACKENDS)
@pytest.mark.parametrize("exc_type", PROCESS_SIGNAL_TYPES)
async def test_plain_process_signals_still_propagate_async(
    backend_name: str, backend_factory: Callable, exc_type: type[BaseException]
) -> None:
    backend = backend_factory()
    exc = exc_type()

    with pytest.raises(exc_type) as raised:
        await backend._invoke_callback_safe(_async_callback_raising(exc))

    assert raised.value is exc


@pytest.mark.parametrize(("backend_name", "backend_factory"), SYNC_BACKENDS)
@pytest.mark.parametrize("exc_type", PROCESS_SIGNAL_TYPES)
def test_plain_process_signals_still_propagate_sync(
    backend_name: str, backend_factory: Callable, exc_type: type[BaseException]
) -> None:
    backend = backend_factory()
    exc = exc_type()

    with pytest.raises(exc_type) as raised:
        backend._invoke_callback_safe(_sync_callback_raising(exc))

    assert raised.value is exc
