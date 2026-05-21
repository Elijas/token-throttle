"""TC2-003: critical cleanup refund failures escape FIX-50 envelopes."""

from __future__ import annotations

import asyncio
import concurrent.futures
import time

import pytest
from frozendict import frozendict

from token_throttle import (
    PerModelConfig,
    Quota,
    RateLimiter,
    RateLimiterCallbacks,
    SyncRateLimiter,
    SyncRateLimiterCallbacks,
    UsageQuotas,
)
from token_throttle._exceptions import AcquireRefundFailedError
from token_throttle._interfaces._models import CapacityReservation, FrozenUsage

MODEL = "test-model"
MODEL_FAMILY = "test-family"

CRITICAL_REFUND_EXCEPTIONS = (
    MemoryError,
    RecursionError,
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
    asyncio.CancelledError,
    concurrent.futures.CancelledError,
)

ASYNC_BACKGROUND_REFUND_CRITICAL_EXCEPTIONS = (
    MemoryError,
    RecursionError,
    asyncio.CancelledError,
    concurrent.futures.CancelledError,
)


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family=MODEL_FAMILY,
        quotas=UsageQuotas([Quota(metric="tokens", limit=10.0, per_seconds=60)]),
    )


def _reservation() -> CapacityReservation:
    return CapacityReservation(
        reservation_id="test-reservation",
        usage=frozendict({"tokens": 1.0}),
        model_family=MODEL_FAMILY,
        bucket_ids=frozenset({("tokens", 60)}),
        model=MODEL,
        limiter_instance_id="test-limiter",
        created_at_seconds=time.time(),
    )


class _AsyncFailingRefundBackend:
    def __init__(self, refund_error: BaseException) -> None:
        self.refund_error = refund_error

    async def await_for_capacity(
        self,
        usage: FrozenUsage,
        *,
        timeout: float | None = None,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float:
        _ = usage, timeout, reservation_id, reservation_lifetime_seconds
        return time.time()

    async def consume_capacity(
        self,
        usage: FrozenUsage,
        *,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float:
        _ = usage, reservation_id, reservation_lifetime_seconds
        return time.time()

    async def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        _ = reserved_usage, actual_usage
        raise self.refund_error

    async def refund_capacity_for_buckets(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        **_kwargs: object,
    ) -> bool:
        await self.refund_capacity(reserved_usage, actual_usage)
        return True

    async def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        _ = metric, per_seconds, value

    def supports_acquire_marker_authority(self) -> bool:
        return False


class _AsyncFailingRefundBuilder:
    def __init__(self, refund_error: BaseException) -> None:
        self.refund_error = refund_error

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> _AsyncFailingRefundBackend:
        _ = cfg, callbacks
        return _AsyncFailingRefundBackend(self.refund_error)

    async def aclose(self) -> None:
        pass


class _SyncFailingRefundBackend:
    def __init__(self, refund_error: BaseException) -> None:
        self.refund_error = refund_error

    def wait_for_capacity(
        self,
        usage: FrozenUsage,
        *,
        timeout: float | None = None,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float:
        _ = usage, timeout, reservation_id, reservation_lifetime_seconds
        return time.time()

    def consume_capacity(
        self,
        usage: FrozenUsage,
        *,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float:
        _ = usage, reservation_id, reservation_lifetime_seconds
        return time.time()

    def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        _ = reserved_usage, actual_usage
        raise self.refund_error

    def refund_capacity_for_buckets(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        **_kwargs: object,
    ) -> bool:
        self.refund_capacity(reserved_usage, actual_usage)
        return True

    def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        _ = metric, per_seconds, value

    def supports_acquire_marker_authority(self) -> bool:
        return False


class _SyncFailingRefundBuilder:
    def __init__(self, refund_error: BaseException) -> None:
        self.refund_error = refund_error

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> _SyncFailingRefundBackend:
        _ = cfg, callbacks
        return _SyncFailingRefundBackend(self.refund_error)

    def close(self) -> None:
        pass


async def _async_interrupted_delivery_callback(**_kwargs: object) -> None:
    raise asyncio.CancelledError("delivery interrupted")


def _sync_interrupted_delivery_callback(**_kwargs: object) -> None:
    raise concurrent.futures.CancelledError("delivery interrupted")


@pytest.mark.parametrize("exc_type", ASYNC_BACKGROUND_REFUND_CRITICAL_EXCEPTIONS)
async def test_async_interrupted_cleanup_refund_critical_propagates_raw(
    exc_type: type[BaseException],
) -> None:
    refund_error = exc_type("critical refund failure")
    limiter = RateLimiter(
        _config(),
        backend=_AsyncFailingRefundBuilder(refund_error),
        callbacks=RateLimiterCallbacks(
            on_lifecycle_event=_async_interrupted_delivery_callback
        ),
    )

    try:
        with pytest.raises(exc_type) as raised:
            await limiter.acquire_capacity(frozendict({"tokens": 1.0}), MODEL)
    finally:
        await limiter.aclose()

    assert raised.value is refund_error


@pytest.mark.parametrize("exc_type", CRITICAL_REFUND_EXCEPTIONS)
async def test_async_refund_wrapper_propagates_all_critical_exceptions_raw(
    exc_type: type[BaseException],
) -> None:
    refund_error = exc_type("critical refund failure")
    limiter = RateLimiter(
        _config(),
        backend=_AsyncFailingRefundBuilder(ValueError("unused")),
    )

    async def fail_refund(_reservation: CapacityReservation) -> None:
        raise refund_error

    limiter._refund_undelivered_acquire = fail_refund

    try:
        with pytest.raises(exc_type) as raised:
            await limiter._refund_undelivered_acquire_or_deliver(
                _reservation(),
                interrupted_by=asyncio.CancelledError("delivery interrupted"),
            )
    finally:
        await limiter.aclose()

    assert raised.value is refund_error


def test_async_interrupted_cleanup_refund_noncritical_still_uses_envelope() -> None:
    async def run() -> None:
        refund_error = ValueError("ordinary refund failure")
        limiter = RateLimiter(
            _config(),
            backend=_AsyncFailingRefundBuilder(refund_error),
            callbacks=RateLimiterCallbacks(
                on_lifecycle_event=_async_interrupted_delivery_callback
            ),
        )

        try:
            with pytest.raises(AcquireRefundFailedError) as raised:
                await limiter.acquire_capacity(frozendict({"tokens": 1.0}), MODEL)
        finally:
            await limiter.aclose()

        assert raised.value.refund_error is refund_error
        assert isinstance(raised.value.interrupted_by, asyncio.CancelledError)

    asyncio.run(run())


@pytest.mark.parametrize("exc_type", CRITICAL_REFUND_EXCEPTIONS)
def test_sync_interrupted_cleanup_refund_critical_propagates_raw(
    exc_type: type[BaseException],
) -> None:
    refund_error = exc_type("critical refund failure")
    limiter = SyncRateLimiter(
        _config(),
        backend=_SyncFailingRefundBuilder(refund_error),
        callbacks=SyncRateLimiterCallbacks(
            on_lifecycle_event=_sync_interrupted_delivery_callback
        ),
    )

    try:
        with pytest.raises(exc_type) as raised:
            limiter.acquire_capacity(frozendict({"tokens": 1.0}), MODEL)
    finally:
        limiter.close()

    assert raised.value is refund_error


def test_sync_interrupted_cleanup_refund_noncritical_still_uses_envelope() -> None:
    refund_error = ValueError("ordinary refund failure")
    limiter = SyncRateLimiter(
        _config(),
        backend=_SyncFailingRefundBuilder(refund_error),
        callbacks=SyncRateLimiterCallbacks(
            on_lifecycle_event=_sync_interrupted_delivery_callback
        ),
    )

    try:
        with pytest.raises(AcquireRefundFailedError) as raised:
            limiter.acquire_capacity(frozendict({"tokens": 1.0}), MODEL)
    finally:
        limiter.close()

    assert raised.value.refund_error is refund_error
    assert isinstance(raised.value.interrupted_by, concurrent.futures.CancelledError)
