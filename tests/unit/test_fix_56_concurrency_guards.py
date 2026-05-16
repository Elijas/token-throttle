"""Regression tests for FIX-56 concurrency-mode guards and ergonomics."""

import asyncio
import pickle
import threading
import warnings

import pytest

import token_throttle._sync_rate_limiter as sync_rate_limiter_module
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

MODEL = "test-model"
MODEL_FAMILY = "test-family"


def _config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=60)]),
        model_family=MODEL_FAMILY,
    )


class CountingAsyncMemoryBackendBuilder(MemoryBackendBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class CountingSyncMemoryBackendBuilder(SyncMemoryBackendBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


async def test_async_context_manager_closes_with_aclose() -> None:
    builder = CountingAsyncMemoryBackendBuilder()

    async with RateLimiter(_config(), backend=builder) as limiter:
        assert limiter._closed is False

    assert limiter._closed is True
    assert builder.close_calls == 1


def test_sync_context_manager_closes_with_close() -> None:
    builder = CountingSyncMemoryBackendBuilder()

    with SyncRateLimiter(_config(), backend=builder) as limiter:
        assert limiter._closed is False

    assert limiter._closed is True
    assert builder.close_calls == 1


async def test_sync_limiter_close_inside_event_loop_succeeds() -> None:
    builder = CountingSyncMemoryBackendBuilder()
    limiter = SyncRateLimiter(_config(), backend=builder)

    limiter.close()

    assert limiter._closed is True
    assert builder.close_calls == 1


def test_async_limiter_rejects_use_from_different_running_loop() -> None:
    async def make_limiter() -> RateLimiter:
        return RateLimiter(_config(), backend=MemoryBackendBuilder())

    async def use_limiter(limiter: RateLimiter) -> None:
        limiter.clear_unused_model_families(0)

    limiter = asyncio.run(make_limiter())

    with pytest.raises(RuntimeError, match="event-loop-affine"):
        asyncio.run(use_limiter(limiter))


def test_async_limiter_allows_aclose_from_different_running_loop() -> None:
    builder = CountingAsyncMemoryBackendBuilder()

    async def make_limiter() -> RateLimiter:
        return RateLimiter(_config(), backend=builder)

    async def close_limiter(limiter: RateLimiter) -> None:
        await limiter.aclose()

    limiter = asyncio.run(make_limiter())

    asyncio.run(close_limiter(limiter))

    assert limiter._closed is True
    assert builder.close_calls == 1


def test_async_limiter_pid_guard_rejects_changed_process() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    limiter._pid = -1

    with pytest.raises(RuntimeError, match="process-affine"):
        limiter.clear_unused_model_families(0)


async def test_async_limiter_aclose_keeps_pid_guard() -> None:
    builder = CountingAsyncMemoryBackendBuilder()
    limiter = RateLimiter(_config(), backend=builder)
    limiter._pid = -1

    with pytest.raises(RuntimeError, match="process-affine"):
        await limiter.aclose()

    assert limiter._closed is False
    assert builder.close_calls == 0


def test_sync_limiter_pid_guard_rejects_changed_process() -> None:
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    limiter._pid = -1

    with pytest.raises(RuntimeError, match="process-affine"):
        limiter.clear_unused_model_families(0)


def test_sync_limiter_close_keeps_pid_guard() -> None:
    builder = CountingSyncMemoryBackendBuilder()
    limiter = SyncRateLimiter(_config(), backend=builder)
    limiter._pid = -1

    with pytest.raises(RuntimeError, match="process-affine"):
        limiter.close()

    assert limiter._closed is False
    assert builder.close_calls == 0


def test_pid_guard_can_be_disabled() -> None:
    limiter = SyncRateLimiter(
        _config(),
        backend=SyncMemoryBackendBuilder(),
        pid_check=False,
    )
    limiter._pid = -1

    assert limiter.clear_unused_model_families(0) == 0


def test_limiters_are_not_pickleable() -> None:
    async_limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    sync_limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())

    with pytest.raises(TypeError, match="not pickleable"):
        pickle.dumps(async_limiter)
    with pytest.raises(TypeError, match="not pickleable"):
        pickle.dumps(sync_limiter)


async def test_sync_acquire_inside_event_loop_warns_once_per_process() -> None:
    sync_rate_limiter_module._sync_in_async_warning_pids.clear()
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        limiter.acquire_capacity({"tokens": 1}, MODEL)
        limiter.acquire_capacity({"tokens": 1}, MODEL)

    matching = [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
        and "SyncRateLimiter from inside an event loop" in str(warning.message)
    ]
    assert len(matching) == 1


def test_sync_pid_guard_triggers_in_thread_with_mutated_pid() -> None:
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    limiter._pid = -1
    errors: list[BaseException] = []

    def use_limiter() -> None:
        try:
            limiter.acquire_capacity({"tokens": 1}, MODEL)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=use_limiter)
    thread.start()
    thread.join(timeout=1.0)

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "process-affine" in str(errors[0])
