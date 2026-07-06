"""
Behavioral tests for ``RateLimiter.reserve`` / ``SyncRateLimiter.reserve``.

``reserve`` is a context manager over the acquire -> call -> refund cycle. The
async and sync flavors share identical semantics (parity is separately pinned
by ``tests/conformance/test_sync_async_surface_parity.py``); this suite pins the
runtime behavior on the memory backend for both:

* happy path: ``set_actual_usage`` returns the unused remainder to the pool,
* forgot-to-set path: a ``RuntimeWarning`` fires and no capacity is fabricated,
* exception path (default): conservative close, original exception re-raised,
* exception path (``usage_on_error``): the override is honored,
* a refund failure while handling an in-block exception does not mask it,
* ``timeout`` passthrough: ``timeout=0`` raises ``TimeoutError`` cleanly.
"""

import pytest

from token_throttle import (
    MemoryBackendBuilder,
    PerModelConfig,
    Quota,
    RateLimiter,
    SyncMemoryBackendBuilder,
    SyncRateLimiter,
    UsageQuotas,
)

_MODEL = "demo"


def _cfg() -> PerModelConfig:
    # Large per_seconds so refill is negligible over a test's lifetime, letting
    # capacity math be asserted exactly.
    return PerModelConfig(
        model_family="demo-family",
        quotas=UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=3600)]),
    )


def _async_limiter() -> RateLimiter:
    return RateLimiter(_cfg(), backend=MemoryBackendBuilder())


def _sync_limiter() -> SyncRateLimiter:
    return SyncRateLimiter(_cfg(), backend=SyncMemoryBackendBuilder())


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


async def test_async_happy_path_returns_unused_capacity() -> None:
    limiter = _async_limiter()
    try:
        async with limiter.reserve({"tokens": 100}, _MODEL) as handle:
            assert handle.reservation.usage["tokens"] == 100.0
            handle.set_actual_usage({"tokens": 40})

        assert limiter.snapshot_state()["in_flight_reservations"] == 0
        # 100 reserved - 40 actual => 60 returned to the pool.
        await limiter.acquire_capacity({"tokens": 60}, _MODEL, timeout=0)
        with pytest.raises(TimeoutError):
            await limiter.acquire_capacity({"tokens": 1}, _MODEL, timeout=0)
    finally:
        await limiter.aclose()


async def test_async_forgot_set_actual_usage_warns_and_fabricates_nothing() -> None:
    limiter = _async_limiter()
    try:
        with pytest.warns(RuntimeWarning, match="set_actual_usage"):
            async with limiter.reserve({"tokens": 100}, _MODEL):
                pass

        assert limiter.snapshot_state()["in_flight_reservations"] == 0
        # Full reserved usage was refunded: nothing returns to the pool.
        with pytest.raises(TimeoutError):
            await limiter.acquire_capacity({"tokens": 1}, _MODEL, timeout=0)
    finally:
        await limiter.aclose()


async def test_async_exception_path_default_is_conservative() -> None:
    limiter = _async_limiter()
    try:
        with pytest.raises(ValueError, match="boom"):
            async with limiter.reserve({"tokens": 100}, _MODEL):
                raise ValueError("boom")

        assert limiter.snapshot_state()["in_flight_reservations"] == 0
        # Conservative close refunds actual == reserved: nothing returns.
        with pytest.raises(TimeoutError):
            await limiter.acquire_capacity({"tokens": 1}, _MODEL, timeout=0)
    finally:
        await limiter.aclose()


async def test_async_exception_path_honors_usage_on_error() -> None:
    limiter = _async_limiter()
    try:
        with pytest.raises(ValueError, match="boom"):
            async with limiter.reserve(
                {"tokens": 100}, _MODEL, usage_on_error={"tokens": 30}
            ):
                raise ValueError("boom")

        assert limiter.snapshot_state()["in_flight_reservations"] == 0
        # usage_on_error=30 => 70 returned to the pool.
        await limiter.acquire_capacity({"tokens": 70}, _MODEL, timeout=0)
        with pytest.raises(TimeoutError):
            await limiter.acquire_capacity({"tokens": 1}, _MODEL, timeout=0)
    finally:
        await limiter.aclose()


async def test_async_refund_failure_does_not_mask_original_exception() -> None:
    limiter = _async_limiter()
    captured: dict[str, str] = {}

    async def _refund_raises(actual_usage, reservation) -> None:
        raise RuntimeError("refund exploded")

    try:
        with pytest.raises(ValueError, match="original"):  # noqa: PT012
            async with limiter.reserve({"tokens": 100}, _MODEL) as handle:
                captured["rid"] = handle.reservation.reservation_id
                limiter.refund_capacity = _refund_raises  # type: ignore[method-assign]
                raise ValueError("original")

        # The non-critical refund failure was swallowed; the reservation was
        # therefore never refunded and remains in flight.
        assert captured["rid"] in limiter._in_flight_reservation_ids
    finally:
        await limiter.aclose()


async def test_async_timeout_zero_raises_cleanly_without_leaking() -> None:
    limiter = _async_limiter()
    try:
        # Consume all capacity but leave nothing in flight: acquire then refund
        # actual == reserved so the bucket stays at 0 and in_flight returns to 0.
        reservation = await limiter.acquire_capacity({"tokens": 100}, _MODEL)
        await limiter.refund_capacity({"tokens": 100}, reservation)
        assert limiter.snapshot_state()["in_flight_reservations"] == 0

        with pytest.raises(TimeoutError):
            async with limiter.reserve({"tokens": 1}, _MODEL, timeout=0):
                pytest.fail("body must not run when acquire times out")

        assert limiter.snapshot_state()["in_flight_reservations"] == 0
    finally:
        await limiter.aclose()


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


def test_sync_happy_path_returns_unused_capacity() -> None:
    limiter = _sync_limiter()
    try:
        with limiter.reserve({"tokens": 100}, _MODEL) as handle:
            assert handle.reservation.usage["tokens"] == 100.0
            handle.set_actual_usage({"tokens": 40})

        assert limiter.snapshot_state()["in_flight_reservations"] == 0
        limiter.acquire_capacity({"tokens": 60}, _MODEL, timeout=0)
        with pytest.raises(TimeoutError):
            limiter.acquire_capacity({"tokens": 1}, _MODEL, timeout=0)
    finally:
        limiter.close()


def test_sync_forgot_set_actual_usage_warns_and_fabricates_nothing() -> None:
    limiter = _sync_limiter()
    try:
        with (
            pytest.warns(RuntimeWarning, match="set_actual_usage"),
            limiter.reserve({"tokens": 100}, _MODEL),
        ):
            pass

        assert limiter.snapshot_state()["in_flight_reservations"] == 0
        with pytest.raises(TimeoutError):
            limiter.acquire_capacity({"tokens": 1}, _MODEL, timeout=0)
    finally:
        limiter.close()


def test_sync_exception_path_default_is_conservative() -> None:
    limiter = _sync_limiter()
    try:
        with (
            pytest.raises(ValueError, match="boom"),
            limiter.reserve({"tokens": 100}, _MODEL),
        ):
            raise ValueError("boom")

        assert limiter.snapshot_state()["in_flight_reservations"] == 0
        with pytest.raises(TimeoutError):
            limiter.acquire_capacity({"tokens": 1}, _MODEL, timeout=0)
    finally:
        limiter.close()


def test_sync_exception_path_honors_usage_on_error() -> None:
    limiter = _sync_limiter()
    try:
        with (
            pytest.raises(ValueError, match="boom"),
            limiter.reserve({"tokens": 100}, _MODEL, usage_on_error={"tokens": 30}),
        ):
            raise ValueError("boom")

        assert limiter.snapshot_state()["in_flight_reservations"] == 0
        limiter.acquire_capacity({"tokens": 70}, _MODEL, timeout=0)
        with pytest.raises(TimeoutError):
            limiter.acquire_capacity({"tokens": 1}, _MODEL, timeout=0)
    finally:
        limiter.close()


def test_sync_refund_failure_does_not_mask_original_exception() -> None:
    limiter = _sync_limiter()
    captured: dict[str, str] = {}

    def _refund_raises(actual_usage, reservation) -> None:
        raise RuntimeError("refund exploded")

    try:
        with (  # noqa: PT012
            pytest.raises(ValueError, match="original"),
            limiter.reserve({"tokens": 100}, _MODEL) as handle,
        ):
            captured["rid"] = handle.reservation.reservation_id
            limiter.refund_capacity = _refund_raises  # type: ignore[method-assign]
            raise ValueError("original")

        assert captured["rid"] in limiter._in_flight_reservation_ids
    finally:
        limiter.close()


def test_sync_timeout_zero_raises_cleanly_without_leaking() -> None:
    limiter = _sync_limiter()
    try:
        reservation = limiter.acquire_capacity({"tokens": 100}, _MODEL)
        limiter.refund_capacity({"tokens": 100}, reservation)
        assert limiter.snapshot_state()["in_flight_reservations"] == 0

        with (
            pytest.raises(TimeoutError),
            limiter.reserve({"tokens": 1}, _MODEL, timeout=0),
        ):
            pytest.fail("body must not run when acquire times out")

        assert limiter.snapshot_state()["in_flight_reservations"] == 0
    finally:
        limiter.close()
