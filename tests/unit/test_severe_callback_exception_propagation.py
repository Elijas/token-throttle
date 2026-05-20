"""Severe callback exceptions propagate through callback dispatch."""

from __future__ import annotations

import warnings

import pytest

from token_throttle import (
    PerModelConfig,
    Quota,
    RateLimiter,
    RateLimiterCallbacks,
    SyncRateLimiter,
    SyncRateLimiterCallbacks,
    UsageQuotas,
)
from token_throttle._interfaces._callbacks import (
    LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
    safe_invoke_async_callback,
    safe_invoke_sync_callback,
)
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)

MODEL = "test-model"
MODEL_FAMILY = "test-family"
SEVERE_CALLBACK_EXCEPTIONS = (MemoryError, RecursionError)


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family=MODEL_FAMILY,
        quotas=UsageQuotas([Quota(metric="tokens", limit=10, per_seconds=3600)]),
    )


@pytest.mark.parametrize("exc_type", SEVERE_CALLBACK_EXCEPTIONS)
async def test_async_lifecycle_callback_dispatch_propagates_severe_exception(
    exc_type: type[BaseException],
) -> None:
    exc = exc_type("forced")

    async def callback() -> None:
        raise exc

    with pytest.raises(exc_type) as raised:
        await safe_invoke_async_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )

    assert raised.value is exc


@pytest.mark.parametrize("exc_type", SEVERE_CALLBACK_EXCEPTIONS)
def test_sync_lifecycle_callback_dispatch_propagates_severe_exception(
    exc_type: type[BaseException],
) -> None:
    exc = exc_type("forced")

    def callback() -> None:
        raise exc

    with pytest.raises(exc_type) as raised:
        safe_invoke_sync_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )

    assert raised.value is exc


async def test_async_lifecycle_callback_dispatch_still_suppresses_exception() -> None:
    async def callback() -> None:
        raise RuntimeError("ordinary")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await safe_invoke_async_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )

    assert any(
        issubclass(w.category, RuntimeWarning)
        and "Rate limiter lifecycle callback raised RuntimeError: ordinary"
        in str(w.message)
        for w in caught
    )


def test_sync_lifecycle_callback_dispatch_still_suppresses_exception() -> None:
    def callback() -> None:
        raise RuntimeError("ordinary")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        safe_invoke_sync_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )

    assert any(
        issubclass(w.category, RuntimeWarning)
        and "Rate limiter lifecycle callback raised RuntimeError: ordinary"
        in str(w.message)
        for w in caught
    )


@pytest.mark.parametrize("exc_type", SEVERE_CALLBACK_EXCEPTIONS)
async def test_async_backend_callback_through_limiter_propagates_severe_exception(
    exc_type: type[BaseException],
) -> None:
    exc = exc_type("forced")
    raise_next = True

    async def on_capacity_consumed(**_kwargs: object) -> None:
        nonlocal raise_next
        if raise_next:
            raise_next = False
            raise exc

    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        callbacks=RateLimiterCallbacks(on_capacity_consumed=on_capacity_consumed),
    )

    with pytest.raises(exc_type) as raised:
        await limiter.acquire_capacity({"tokens": 10}, MODEL)

    assert raised.value is exc
    assert limiter.snapshot_state()["in_flight_reservations"] == 0

    reservation = await limiter.acquire_capacity({"tokens": 10}, MODEL, timeout=0)
    await limiter.refund_capacity({"tokens": 0}, reservation)


@pytest.mark.parametrize("exc_type", SEVERE_CALLBACK_EXCEPTIONS)
def test_sync_backend_callback_through_limiter_propagates_severe_exception(
    exc_type: type[BaseException],
) -> None:
    exc = exc_type("forced")
    raise_next = True

    def on_capacity_consumed(**_kwargs: object) -> None:
        nonlocal raise_next
        if raise_next:
            raise_next = False
            raise exc

    limiter = SyncRateLimiter(
        _config(),
        backend=SyncMemoryBackendBuilder(),
        callbacks=SyncRateLimiterCallbacks(on_capacity_consumed=on_capacity_consumed),
    )

    with pytest.raises(exc_type) as raised:
        limiter.acquire_capacity({"tokens": 10}, MODEL)

    assert raised.value is exc
    assert limiter.snapshot_state()["in_flight_reservations"] == 0

    reservation = limiter.acquire_capacity({"tokens": 10}, MODEL, timeout=0)
    limiter.refund_capacity({"tokens": 0}, reservation)


async def test_async_backend_callback_through_limiter_still_suppresses_exception() -> (
    None
):
    async def on_capacity_consumed(**_kwargs: object) -> None:
        raise RuntimeError("ordinary")

    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        callbacks=RateLimiterCallbacks(on_capacity_consumed=on_capacity_consumed),
    )

    with pytest.warns(RuntimeWarning, match="RuntimeError.*ordinary"):
        reservation = await limiter.acquire_capacity({"tokens": 10}, MODEL)

    assert reservation.model_family == MODEL_FAMILY
    await limiter.refund_capacity({"tokens": 0}, reservation)


def test_sync_backend_callback_through_limiter_still_suppresses_exception() -> None:
    def on_capacity_consumed(**_kwargs: object) -> None:
        raise RuntimeError("ordinary")

    limiter = SyncRateLimiter(
        _config(),
        backend=SyncMemoryBackendBuilder(),
        callbacks=SyncRateLimiterCallbacks(on_capacity_consumed=on_capacity_consumed),
    )

    with pytest.warns(RuntimeWarning, match="RuntimeError.*ordinary"):
        reservation = limiter.acquire_capacity({"tokens": 10}, MODEL)

    assert reservation.model_family == MODEL_FAMILY
    limiter.refund_capacity({"tokens": 0}, reservation)
