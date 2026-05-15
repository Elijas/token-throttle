"""Regression tests for FIX-49 close cancellation lifecycle races."""

import asyncio
import threading
import time

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
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
        quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=3600)]),
        model_family=MODEL_FAMILY,
    )


class CountingAsyncMemoryBackendBuilder(MemoryBackendBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0
        self.close_started: asyncio.Event | None = None
        self.release_close: asyncio.Event | None = None

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_started is not None:
            self.close_started.set()
        if self.release_close is not None:
            await self.release_close.wait()


class CountingSyncMemoryBackendBuilder(SyncMemoryBackendBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0
        self.close_started = threading.Event()
        self.release_close = threading.Event()

    def close(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        self.release_close.wait(timeout=1.0)


async def test_async_cancel_aclose_during_pending_drain_still_terminalizes() -> None:
    builder = CountingAsyncMemoryBackendBuilder()
    limiter = RateLimiter(_config(), backend=builder)
    finalize_entered = asyncio.Event()
    release_finalize = asyncio.Event()
    original_finalize = limiter._finalize_pending_acquire

    async def controlled_finalize(
        reservation: CapacityReservation,
        model: str,
    ) -> None:
        finalize_entered.set()
        await release_finalize.wait()
        await original_finalize(reservation, model)

    limiter._finalize_pending_acquire = controlled_finalize

    acquire_task = asyncio.create_task(limiter.acquire_capacity({"tokens": 10}, MODEL))
    await asyncio.wait_for(finalize_entered.wait(), timeout=1.0)

    close_task = asyncio.create_task(limiter.aclose())
    await asyncio.sleep(0)
    close_task.cancel()
    await asyncio.sleep(0)
    assert not close_task.done()

    release_finalize.set()
    await asyncio.wait_for(acquire_task, timeout=1.0)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(close_task, timeout=1.0)

    assert builder.close_calls == 1
    assert limiter._closed is True
    assert limiter._closing is False


async def test_async_acquire_cancel_cleanup_blocks_close_until_refunded() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    backend = await limiter._get_backend(_config())
    original_refund = backend.refund_capacity_for_buckets
    refund_entered = asyncio.Event()
    release_refund = asyncio.Event()
    refund_calls = 0

    async def controlled_refund(*args, **kwargs) -> None:
        nonlocal refund_calls
        refund_calls += 1
        refund_entered.set()
        await release_refund.wait()
        await original_refund(*args, **kwargs)

    backend.refund_capacity_for_buckets = controlled_refund

    finalize_entered = asyncio.Event()
    release_finalize = asyncio.Event()
    original_finalize = limiter._finalize_pending_acquire

    async def controlled_finalize(
        reservation: CapacityReservation,
        model: str,
    ) -> None:
        finalize_entered.set()
        await release_finalize.wait()
        await original_finalize(reservation, model)

    limiter._finalize_pending_acquire = controlled_finalize

    acquire_task = asyncio.create_task(limiter.acquire_capacity({"tokens": 60}, MODEL))
    await asyncio.wait_for(finalize_entered.wait(), timeout=1.0)
    acquire_task.cancel()
    release_finalize.set()
    await asyncio.wait_for(refund_entered.wait(), timeout=1.0)

    close_task = asyncio.create_task(limiter.aclose())
    await asyncio.sleep(0.05)
    assert not close_task.done()

    release_refund.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(acquire_task, timeout=1.0)
    await asyncio.wait_for(close_task, timeout=1.0)

    assert refund_calls == 1
    assert limiter._closed is True
    assert limiter._closing is False


async def test_async_concurrent_close_calls_builder_once() -> None:
    builder = CountingAsyncMemoryBackendBuilder()
    builder.close_started = asyncio.Event()
    builder.release_close = asyncio.Event()
    limiter = RateLimiter(_config(), backend=builder)

    first = asyncio.create_task(limiter.aclose())
    await asyncio.wait_for(builder.close_started.wait(), timeout=1.0)
    second = asyncio.create_task(limiter.aclose())
    await asyncio.sleep(0)

    builder.release_close.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1.0)

    assert builder.close_calls == 1
    assert limiter._closed is True
    assert limiter._closing is False


def test_sync_keyboard_interrupt_during_drain_is_terminal_closed_state() -> None:
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    limiter._pending_acquire_reservations.add("stuck")

    class InterruptingDrain:
        def clear(self) -> None:
            pass

        def set(self) -> None:
            pass

        def wait(self, timeout=None) -> bool:
            raise KeyboardInterrupt("simulated drain interrupt")

    limiter._pending_drained = InterruptingDrain()

    with pytest.raises(KeyboardInterrupt, match="simulated drain interrupt"):
        limiter.close()

    assert limiter._closed is True
    assert limiter._closing is False
    with pytest.raises(RuntimeError, match="closed"):
        limiter.acquire_capacity({"tokens": 1}, MODEL)


def test_sync_acquire_interrupt_cleanup_blocks_close_until_refunded() -> None:
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    backend = limiter._get_backend(_config())
    original_refund = backend.refund_capacity_for_buckets
    refund_entered = threading.Event()
    release_refund = threading.Event()
    refund_calls = 0

    def controlled_refund(*args, **kwargs) -> None:
        nonlocal refund_calls
        refund_calls += 1
        refund_entered.set()
        release_refund.wait(timeout=1.0)
        original_refund(*args, **kwargs)

    backend.refund_capacity_for_buckets = controlled_refund

    original_finalize = limiter._finalize_pending_acquire
    finalize_calls = 0

    def interrupt_first_finalize(
        reservation: CapacityReservation,
        model: str,
    ) -> None:
        nonlocal finalize_calls
        finalize_calls += 1
        if finalize_calls == 1:
            raise KeyboardInterrupt("simulated post-consume interrupt")
        original_finalize(reservation, model)

    limiter._finalize_pending_acquire = interrupt_first_finalize
    errors: list[BaseException] = []

    def acquire() -> None:
        try:
            limiter.acquire_capacity({"tokens": 60}, MODEL)
        except BaseException as exc:
            errors.append(exc)

    acquire_thread = threading.Thread(target=acquire)
    acquire_thread.start()
    assert refund_entered.wait(timeout=1.0)

    close_thread = threading.Thread(target=limiter.close)
    close_thread.start()
    time.sleep(0.05)
    assert close_thread.is_alive()

    release_refund.set()
    acquire_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert not acquire_thread.is_alive()
    assert not close_thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], KeyboardInterrupt)
    assert refund_calls == 1
    assert limiter._closed is True
    assert limiter._closing is False


def test_sync_concurrent_close_calls_builder_once() -> None:
    builder = CountingSyncMemoryBackendBuilder()
    limiter = SyncRateLimiter(_config(), backend=builder)
    errors: list[BaseException] = []

    def close_limiter() -> None:
        try:
            limiter.close()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=close_limiter)
    second = threading.Thread(target=close_limiter)
    first.start()
    assert builder.close_started.wait(timeout=1.0)
    second.start()
    time.sleep(0.05)

    builder.release_close.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert builder.close_calls == 1
    assert limiter._closed is True
    assert limiter._closing is False
