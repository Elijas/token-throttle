"""TC2-002: async cancel helpers propagate Exception-subclass criticals."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from token_throttle import (
    PerModelConfig,
    Quota,
    RateLimiter,
    RateLimiterCallbacks,
    UsageQuotas,
)
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder

MODEL = "test-model"
MODEL_FAMILY = "test-family"

EXCEPTION_SUBCLASS_CRITICALS = (MemoryError, RecursionError)


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family=MODEL_FAMILY,
        quotas=UsageQuotas([Quota(metric="tokens", limit=10.0, per_seconds=60)]),
    )


async def _raise_after_checkpoint(exc: BaseException) -> None:
    await asyncio.sleep(0)
    raise exc


@pytest.mark.parametrize("exc_type", EXCEPTION_SUBCLASS_CRITICALS)
async def test_backend_task_succeeded_after_cancel_reraises_critical_exception_subclass(
    exc_type: type[BaseException],
) -> None:
    exc = exc_type("forced backend task failure")
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    task = asyncio.create_task(_raise_after_checkpoint(exc))
    await asyncio.sleep(0)

    try:
        with pytest.raises(exc_type) as raised:
            await limiter._backend_task_succeeded_after_cancel(task)
    finally:
        await limiter.aclose()

    assert raised.value is exc


@pytest.mark.parametrize("exc_type", EXCEPTION_SUBCLASS_CRITICALS)
async def test_wait_for_set_max_capacity_cancel_helper_reraises_critical_exception_subclass(
    exc_type: type[BaseException],
) -> None:
    exc = exc_type("forced set_max_capacity failure")
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    task = asyncio.create_task(_raise_after_checkpoint(exc))
    await asyncio.sleep(0)

    try:
        with pytest.raises(exc_type) as raised:
            await limiter._wait_for_set_max_capacity_task_while_cancelled(task)
    finally:
        await limiter.aclose()

    assert raised.value is exc


@pytest.mark.parametrize("exc_type", EXCEPTION_SUBCLASS_CRITICALS)
async def test_cancelled_acquire_surfaces_backend_task_critical_exception_subclass(
    exc_type: type[BaseException],
) -> None:
    exc = exc_type("forced callback failure after cancellation")
    entered_callback = asyncio.Event()

    async def on_capacity_consumed(**_kwargs: object) -> None:
        entered_callback.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise exc from None

    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        callbacks=RateLimiterCallbacks(on_capacity_consumed=on_capacity_consumed),
    )
    task = asyncio.create_task(limiter.acquire_capacity({"tokens": 1.0}, MODEL))

    try:
        await asyncio.wait_for(entered_callback.wait(), timeout=1)
        task.cancel()

        with pytest.raises(exc_type) as raised:
            await asyncio.wait_for(task, timeout=1)

        assert raised.value is exc
        assert limiter.snapshot_state()["in_flight_reservations"] == 0
        await asyncio.wait_for(limiter.aclose(), timeout=1)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        if not limiter._closed:
            with contextlib.suppress(BaseException):
                await limiter.aclose()
