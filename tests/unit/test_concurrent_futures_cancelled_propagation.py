"""
Backend behavioral test: ``concurrent.futures.CancelledError`` must propagate
out of every backend's ``_invoke_callback_safe`` dispatcher.

Pre-refactor (archetype-convergence): the sync backends' file-local
``_CRITICAL_CALLBACK_EXCEPTION_TYPES`` tuple omitted
``concurrent.futures.CancelledError`` while the sync rate-limiter's lifecycle
tuple included it (R11-FIX-01). The post-refactor unified
``BACKEND_CALLBACK_CRITICAL_EXCEPTIONS`` includes it for all backends,
closing the drift documented in the refactor plan (Archetype A).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
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


def _sync_callback_raising(exc: BaseException):
    def callback(**_kwargs):
        raise exc

    return callback


def _async_callback_raising(exc: BaseException):
    async def callback(**_kwargs):
        raise exc

    return callback


@pytest.mark.parametrize(("backend_name", "backend_factory"), SYNC_BACKENDS)
def test_sync_backend_propagates_concurrent_futures_cancelled_error(
    backend_name: str, backend_factory: Callable[[], Any]
) -> None:
    """``concurrent.futures.CancelledError`` from a sync user callback must
    propagate. Pre-refactor this was suppressed by sync backends because the
    sync ``_CRITICAL_CALLBACK_EXCEPTION_TYPES`` tuple omitted it.
    Post-refactor the unified ``BACKEND_CALLBACK_CRITICAL_EXCEPTIONS`` makes
    it critical for every backend, matching the sync rate-limiter's
    lifecycle contract added by R11-FIX-01.
    """
    backend = backend_factory()
    exc = concurrent.futures.CancelledError()

    with pytest.raises(concurrent.futures.CancelledError) as raised:
        backend._invoke_callback_safe(_sync_callback_raising(exc))

    assert raised.value is exc


@pytest.mark.parametrize(("backend_name", "backend_factory"), ASYNC_BACKENDS)
async def test_async_backend_propagates_concurrent_futures_cancelled_error(
    backend_name: str, backend_factory: Callable[[], Any]
) -> None:
    """Same propagation invariant on async backends. ``asyncio.CancelledError``
    and ``concurrent.futures.CancelledError`` are distinct types in 3.12+
    (neither subclasses the other); both must propagate from user callbacks.
    """
    backend = backend_factory()
    exc = concurrent.futures.CancelledError()

    with pytest.raises(concurrent.futures.CancelledError) as raised:
        await backend._invoke_callback_safe(_async_callback_raising(exc))

    assert raised.value is exc


def test_concurrent_futures_and_asyncio_cancelled_error_are_distinct() -> None:
    """Sanity check the premise of the refactor: in Python 3.12+ these are
    different exception types. If this test breaks, the rationale for adding
    ``concurrent.futures.CancelledError`` to the critical set needs revisiting.
    """
    assert concurrent.futures.CancelledError is not asyncio.CancelledError, (
        "Premise of the refactor is invalidated; revisit KNOWN UNKNOWN #1."
    )
    assert not issubclass(concurrent.futures.CancelledError, asyncio.CancelledError)
    assert not issubclass(asyncio.CancelledError, concurrent.futures.CancelledError)
