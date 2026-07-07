"""Regression tests for FIX-48 close/set_max lifecycle gating."""

import asyncio
import threading
import time

import pytest

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


async def test_async_close_timeout_is_terminal_closed_state() -> None:
    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        close_drain_timeout_seconds=0.01,
    )
    limiter._pending_acquire_reservations.add("stuck")
    limiter._pending_drained.clear()

    with pytest.raises(TimeoutError, match="pending acquire reservations"):
        await limiter.aclose()

    assert limiter._closed is True
    assert limiter._closing is False
    with pytest.raises(RuntimeError, match="closed"):
        await limiter.acquire_capacity({"tokens": 1}, MODEL)


async def test_async_backend_close_failure_is_terminal_closed_state() -> None:
    builder = MemoryBackendBuilder()
    limiter = RateLimiter(_config(), backend=builder)

    async def fail_close() -> None:
        raise RuntimeError("simulated close failure")

    builder.aclose = fail_close

    with pytest.raises(RuntimeError, match="simulated close failure"):
        await limiter.aclose()

    assert limiter._closed is True
    assert limiter._closing is False
    with pytest.raises(RuntimeError, match="closed"):
        await limiter.acquire_capacity({"tokens": 1}, MODEL)


class _AsyncBuilderWithoutCloseHooks:
    """Minimal builder exposing only build(); aclose()/close() are documented-optional."""

    def __init__(self, inner: MemoryBackendBuilder) -> None:
        self._inner = inner

    def build(self, cfg, *, callbacks=None):
        return self._inner.build(cfg, callbacks=callbacks)


async def test_async_close_tolerates_builder_without_aclose_hook() -> None:
    builder = _AsyncBuilderWithoutCloseHooks(MemoryBackendBuilder())
    assert not hasattr(builder, "aclose")
    limiter = RateLimiter(_config(), backend=builder)
    reservation = await limiter.acquire_capacity({"tokens": 1}, MODEL)
    await limiter.refund_capacity({"tokens": 1}, reservation)

    await limiter.aclose()

    assert limiter._closed is True


async def test_async_close_waits_for_in_flight_set_max_before_backend_close() -> None:
    builder = MemoryBackendBuilder()
    limiter = RateLimiter(_config(), backend=builder)
    reservation = await limiter.acquire_capacity({"tokens": 1}, MODEL)
    await limiter.refund_capacity({"tokens": 1}, reservation)

    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    original_set_max_capacity = backend.set_max_capacity
    set_entered = asyncio.Event()
    release_set = asyncio.Event()
    set_finished = asyncio.Event()
    close_called = asyncio.Event()
    close_called_after_set_finished = False
    calls = 0

    async def controlled_set_max_capacity(metric, per_seconds, value) -> None:
        nonlocal calls
        calls += 1
        set_entered.set()
        await release_set.wait()
        await original_set_max_capacity(metric, per_seconds, value)
        set_finished.set()

    async def controlled_close() -> None:
        nonlocal close_called_after_set_finished
        close_called_after_set_finished = set_finished.is_set()
        close_called.set()

    backend.set_max_capacity = controlled_set_max_capacity
    builder.aclose = controlled_close

    set_task = asyncio.create_task(limiter.set_max_capacity(MODEL, "tokens", 60, 50.0))
    await asyncio.wait_for(set_entered.wait(), timeout=1.0)

    close_task = asyncio.create_task(limiter.aclose())
    await asyncio.sleep(0)
    assert not close_called.is_set()

    release_set.set()
    await asyncio.wait_for(set_task, timeout=1.0)
    await asyncio.wait_for(close_task, timeout=1.0)

    assert close_called_after_set_finished is True
    assert calls == 1
    assert limiter._closed is True
    assert limiter._closing is False
    with pytest.raises(RuntimeError, match="closed"):
        await limiter.set_max_capacity(MODEL, "tokens", 60, 75.0)
    assert calls == 1


def test_sync_close_timeout_is_terminal_closed_state() -> None:
    limiter = SyncRateLimiter(
        _config(),
        backend=SyncMemoryBackendBuilder(),
        close_drain_timeout_seconds=0.01,
    )
    limiter._pending_acquire_reservations.add("stuck")
    limiter._pending_drained.clear()

    with pytest.raises(TimeoutError, match="pending acquire reservations"):
        limiter.close()

    assert limiter._closed is True
    assert limiter._closing is False
    with pytest.raises(RuntimeError, match="closed"):
        limiter.acquire_capacity({"tokens": 1}, MODEL)


def test_sync_backend_close_failure_is_terminal_closed_state() -> None:
    builder = SyncMemoryBackendBuilder()
    limiter = SyncRateLimiter(_config(), backend=builder)

    def fail_close() -> None:
        raise RuntimeError("simulated close failure")

    builder.close = fail_close

    with pytest.raises(RuntimeError, match="simulated close failure"):
        limiter.close()

    assert limiter._closed is True
    assert limiter._closing is False
    with pytest.raises(RuntimeError, match="closed"):
        limiter.acquire_capacity({"tokens": 1}, MODEL)


class _SyncBuilderWithoutCloseHook:
    """Minimal builder exposing only build(); close() is documented-optional."""

    def __init__(self, inner: SyncMemoryBackendBuilder) -> None:
        self._inner = inner

    def build(self, cfg, *, callbacks=None):
        return self._inner.build(cfg, callbacks=callbacks)


def test_sync_close_tolerates_builder_without_close_hook() -> None:
    builder = _SyncBuilderWithoutCloseHook(SyncMemoryBackendBuilder())
    assert not hasattr(builder, "close")
    limiter = SyncRateLimiter(_config(), backend=builder)
    reservation = limiter.acquire_capacity({"tokens": 1}, MODEL)
    limiter.refund_capacity({"tokens": 1}, reservation)

    limiter.close()

    assert limiter._closed is True


def test_sync_close_waits_for_in_flight_set_max_before_backend_close() -> None:
    builder = SyncMemoryBackendBuilder()
    limiter = SyncRateLimiter(_config(), backend=builder)
    reservation = limiter.acquire_capacity({"tokens": 1}, MODEL)
    limiter.refund_capacity({"tokens": 1}, reservation)

    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    original_set_max_capacity = backend.set_max_capacity
    set_entered = threading.Event()
    release_set = threading.Event()
    set_finished = threading.Event()
    close_called = threading.Event()
    close_called_after_set_finished = False
    calls = 0
    errors: list[BaseException] = []

    def controlled_set_max_capacity(metric, per_seconds, value) -> None:
        nonlocal calls
        calls += 1
        set_entered.set()
        release_set.wait(timeout=1.0)
        original_set_max_capacity(metric, per_seconds, value)
        set_finished.set()

    def controlled_close() -> None:
        nonlocal close_called_after_set_finished
        close_called_after_set_finished = set_finished.is_set()
        close_called.set()

    def run_set_max() -> None:
        try:
            limiter.set_max_capacity(MODEL, "tokens", 60, 50.0)
        except BaseException as exc:
            errors.append(exc)

    backend.set_max_capacity = controlled_set_max_capacity
    builder.close = controlled_close

    set_thread = threading.Thread(target=run_set_max)
    set_thread.start()
    assert set_entered.wait(timeout=1.0)

    close_thread = threading.Thread(target=limiter.close)
    close_thread.start()
    time.sleep(0.05)
    assert not close_called.is_set()

    release_set.set()
    set_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert not set_thread.is_alive()
    assert not close_thread.is_alive()
    assert errors == []
    assert close_called_after_set_finished is True
    assert calls == 1
    assert limiter._closed is True
    with pytest.raises(RuntimeError, match="closed"):
        limiter.set_max_capacity(MODEL, "tokens", 60, 75.0)
    assert calls == 1
