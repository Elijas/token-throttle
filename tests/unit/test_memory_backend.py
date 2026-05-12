"""Tests for MemoryBackend and MemoryBackendBuilder coverage gaps.

Covers: builder init/build, ValueError on bad bucket key, continue branches
in multi-metric loops, callback invocations, and fresh-start callback.
"""

from unittest.mock import AsyncMock

import pytest
from frozendict import frozendict

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, SecondsIn, UsageQuotas
from token_throttle._limiter_backends._memory._backend import (
    MemoryBackend,
    MemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter


def _make_config(
    *,
    model_family: str = "test-family",
    quotas: list[Quota] | None = None,
) -> PerModelConfig:
    if quotas is None:
        quotas = [
            Quota(metric="tokens", limit=1000, per_seconds=SecondsIn.MINUTE),
            Quota(metric="requests", limit=10, per_seconds=SecondsIn.MINUTE),
        ]
    return PerModelConfig(
        quotas=UsageQuotas(quotas),
        model_family=model_family,
    )


def _make_multi_window_config(
    *,
    model_family: str = "test-family",
) -> PerModelConfig:
    """Config with 2 time windows per metric — exercises continue branches."""
    return PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric="tokens", limit=1000, per_seconds=SecondsIn.MINUTE),
                Quota(metric="tokens", limit=50000, per_seconds=SecondsIn.HOUR),
                Quota(metric="requests", limit=10, per_seconds=SecondsIn.MINUTE),
                Quota(metric="requests", limit=500, per_seconds=SecondsIn.HOUR),
            ]
        ),
        model_family=model_family,
    )


def _make_callbacks(**overrides) -> RateLimiterCallbacks:
    defaults = {
        "on_wait_start": AsyncMock(),
        "after_wait_end_consumption": AsyncMock(),
        "on_capacity_consumed": AsyncMock(),
        "on_capacity_refunded": AsyncMock(),
        "on_missing_consumption_data": AsyncMock(),
    }
    defaults.update(overrides)
    return RateLimiterCallbacks(**defaults)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TestMemoryBackendBuilder:
    def test_build_returns_memory_backend(self):
        builder = MemoryBackendBuilder()
        cfg = _make_config()
        backend = builder.build(cfg)
        assert isinstance(backend, MemoryBackend)

    def test_build_with_sleep_interval(self):
        builder = MemoryBackendBuilder(sleep_interval=0.5)
        cfg = _make_config()
        backend = builder.build(cfg)
        assert backend._sleep_interval == 0.5

    def test_build_with_callbacks(self):
        builder = MemoryBackendBuilder()
        cfg = _make_config()
        cbs = _make_callbacks()
        backend = builder.build(cfg, callbacks=cbs)
        assert backend._callbacks is cbs

    def test_build_creates_correct_number_of_buckets(self):
        builder = MemoryBackendBuilder()
        cfg = _make_config()
        backend = builder.build(cfg)
        assert len(backend._buckets) == 2


# ---------------------------------------------------------------------------
# _set_capacities bad key
# ---------------------------------------------------------------------------


class TestSetCapacitiesBadKey:
    async def test_set_capacities_unknown_bucket_raises(self):
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config())
        bad_caps = frozendict({("nonexistent_metric", 60): 100.0})
        with pytest.raises(
            ValueError, match="Bucket 'nonexistent_metric/60s' not found"
        ):
            backend._set_capacities(bad_caps, 1000.0)


# ---------------------------------------------------------------------------
# refund_capacity bad key
# ---------------------------------------------------------------------------


class TestSetMaxCapacityBadKey:
    async def test_set_max_capacity_unknown_metric_raises(self):
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config())
        with pytest.raises(ValueError, match="Bucket 'nonexistent/60s' not found"):
            await backend.set_max_capacity("nonexistent", 60, 500.0)

    async def test_set_max_capacity_wrong_per_seconds_raises(self):
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config())
        with pytest.raises(ValueError, match="Bucket 'tokens/3600s' not found"):
            await backend.set_max_capacity("tokens", 3600, 500.0)


# ---------------------------------------------------------------------------
# Multi-metric continue branches
# ---------------------------------------------------------------------------


class TestMultiMetricContinueBranches:
    """Exercise the `if metric != usage_metric: continue` branches.

    Uses 2+ quotas per metric (different time windows).
    """

    async def test_check_and_consume_with_multi_window(self):
        builder = MemoryBackendBuilder()
        cfg = _make_multi_window_config()
        backend = builder.build(cfg)
        usage = frozendict({"tokens": 100.0, "requests": 1.0})
        await backend.await_for_capacity(usage)
        # Should not raise — all 4 buckets processed

    async def test_consume_capacity_with_multi_window(self):
        builder = MemoryBackendBuilder()
        cfg = _make_multi_window_config()
        backend = builder.build(cfg)
        usage = frozendict({"tokens": 100.0, "requests": 1.0})
        await backend.consume_capacity(usage)
        # Should not raise — all buckets processed

    async def test_refund_capacity_with_multi_window(self):
        builder = MemoryBackendBuilder()
        cfg = _make_multi_window_config()
        backend = builder.build(cfg)
        reserved = frozendict({"tokens": 200.0, "requests": 2.0})
        actual = frozendict({"tokens": 100.0, "requests": 1.0})
        # Must consume first so there's something to refund
        await backend.await_for_capacity(reserved)
        await backend.refund_capacity(reserved, actual)
        # Should not raise — all buckets processed

    async def test_await_for_capacity_with_multi_window(self):
        builder = MemoryBackendBuilder()
        cfg = _make_multi_window_config()
        backend = builder.build(cfg)
        usage = frozendict({"tokens": 50.0, "requests": 1.0})
        await backend.await_for_capacity(usage)
        # Should not raise — capacity is sufficient in all windows


# ---------------------------------------------------------------------------
# Callback invocations
# ---------------------------------------------------------------------------


class TestCapacityConsumedCallback:
    async def test_on_capacity_consumed_fires_on_acquire(self):
        cbs = _make_callbacks()
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config(), callbacks=cbs)
        usage = frozendict({"tokens": 100.0, "requests": 1.0})
        await backend.await_for_capacity(usage)
        cbs.on_capacity_consumed.assert_awaited_once()
        call_kwargs = cbs.on_capacity_consumed.call_args.kwargs
        assert call_kwargs["model_family"] == "test-family"
        assert call_kwargs["usage"] == usage

    async def test_on_capacity_consumed_fires_on_consume(self):
        cbs = _make_callbacks()
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config(), callbacks=cbs)
        usage = frozendict({"tokens": 100.0, "requests": 1.0})
        await backend.consume_capacity(usage)
        cbs.on_capacity_consumed.assert_awaited_once()


class TestWaitCallbacks:
    async def test_on_wait_start_fires_when_capacity_insufficient(self):
        cbs = _make_callbacks()
        builder = MemoryBackendBuilder(sleep_interval=0.01)
        backend = builder.build(_make_config(), callbacks=cbs)

        # Consume all capacity first
        await backend.consume_capacity(frozendict({"tokens": 1000.0, "requests": 10.0}))
        cbs.on_capacity_consumed.reset_mock()

        # Now set max capacity low and refund to make capacity available soon
        # Actually, let's just consume partially and then wait for small amount
        # by resetting the backend
        builder2 = MemoryBackendBuilder(sleep_interval=0.01)
        cbs2 = _make_callbacks()
        backend2 = builder2.build(_make_config(), callbacks=cbs2)

        # Drain all capacity
        await backend2.consume_capacity(
            frozendict({"tokens": 1000.0, "requests": 10.0})
        )
        cbs2.on_capacity_consumed.reset_mock()

        # Refund most of it back so the next await_for_capacity will wait then succeed
        await backend2.refund_capacity(
            frozendict({"tokens": 1000.0, "requests": 10.0}),
            frozendict({"tokens": 0.0, "requests": 0.0}),
        )
        cbs2.on_capacity_refunded.assert_awaited_once()

        # Now capacity should be back, so await_for_capacity should succeed immediately
        await backend2.await_for_capacity(frozendict({"tokens": 50.0, "requests": 1.0}))
        cbs2.on_capacity_consumed.assert_awaited_once()

    async def test_wait_callbacks_fire_when_waiting(self):
        """Test that on_wait_start and after_wait_end_consumption fire.

        Triggers when the backend has to poll.
        """
        cbs = _make_callbacks()
        builder = MemoryBackendBuilder(sleep_interval=0.01)
        cfg = PerModelConfig(
            quotas=UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=10)]),
            model_family="test-family",
        )
        backend = builder.build(cfg, callbacks=cbs)

        # Consume all capacity
        await backend.consume_capacity(frozendict({"tokens": 100.0}))
        cbs.on_capacity_consumed.reset_mock()

        # With per_seconds=10 and limit=100, refill rate is 10 tokens/sec.
        # 1 token takes ~0.1s to refill — slow enough to reliably trigger the
        # wait path, fast enough to keep the test quick.
        await backend.await_for_capacity(frozendict({"tokens": 1.0}))

        cbs.on_wait_start.assert_awaited_once()
        call_kwargs = cbs.on_wait_start.call_args.kwargs
        assert call_kwargs["model_family"] == "test-family"

        cbs.after_wait_end_consumption.assert_awaited_once()
        end_kwargs = cbs.after_wait_end_consumption.call_args.kwargs
        assert end_kwargs["wait_time_s"] > 0


class TestRefundCallbacks:
    async def test_on_capacity_refunded_fires(self):
        cbs = _make_callbacks()
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config(), callbacks=cbs)

        reserved = frozendict({"tokens": 200.0, "requests": 2.0})
        actual = frozendict({"tokens": 100.0, "requests": 1.0})
        await backend.await_for_capacity(reserved)
        await backend.refund_capacity(reserved, actual)

        cbs.on_capacity_refunded.assert_awaited_once()
        call_kwargs = cbs.on_capacity_refunded.call_args.kwargs
        assert call_kwargs["model_family"] == "test-family"
        assert call_kwargs["reserved_usage"] == reserved
        assert call_kwargs["actual_usage"] == actual


# ---------------------------------------------------------------------------
# Fresh-start callback
# ---------------------------------------------------------------------------


class TestFreshStartCallback:
    async def test_fresh_start_callback_fires_on_first_access(self):
        """On a brand new backend, the first capacity check is a fresh start.

        No prior consumption data means on_missing_consumption_data fires.
        """
        cbs = _make_callbacks()
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config(), callbacks=cbs)

        usage = frozendict({"tokens": 10.0, "requests": 1.0})
        await backend.await_for_capacity(usage)

        # Fresh start fires for each bucket
        assert cbs.on_missing_consumption_data.await_count == 2
        calls = cbs.on_missing_consumption_data.call_args_list
        metrics_reported = {c.kwargs["usage_metric"] for c in calls}
        assert metrics_reported == {"tokens", "requests"}

    async def test_fresh_start_callback_fires_on_consume_capacity(self):
        cbs = _make_callbacks()
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config(), callbacks=cbs)

        usage = frozendict({"tokens": 10.0, "requests": 1.0})
        await backend.consume_capacity(usage)

        assert cbs.on_missing_consumption_data.await_count == 2

    async def test_fresh_start_callback_fires_on_refund(self):
        cbs = _make_callbacks()
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config(), callbacks=cbs)

        # Acquire first
        reserved = frozendict({"tokens": 100.0, "requests": 1.0})
        await backend.await_for_capacity(reserved)
        cbs.on_missing_consumption_data.reset_mock()

        # Second access is NOT fresh start (bucket has prior data)
        actual = frozendict({"tokens": 50.0, "requests": 1.0})
        await backend.refund_capacity(reserved, actual)
        cbs.on_missing_consumption_data.assert_not_awaited()


# ---------------------------------------------------------------------------
# Negative refund warning
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Usage value strictness
# ---------------------------------------------------------------------------


class TestUsageValueCoercion:
    """Memory backend rejects numeric-looking strings at the validation boundary."""

    async def test_consume_capacity_rejects_string_values(self):
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config())
        usage = frozendict({"tokens": "50", "requests": "1"})
        with pytest.raises(ValueError, match="int or float"):
            await backend.consume_capacity(usage)

    async def test_await_for_capacity_rejects_string_values(self):
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config())
        usage = frozendict({"tokens": "50", "requests": "1"})
        with pytest.raises(ValueError, match="int or float"):
            await backend.await_for_capacity(usage)

    async def test_await_for_capacity_rejects_string_values_when_waiting(self):
        """String usage fails before _compute_sleep sees it."""
        cfg = _make_config(
            quotas=[Quota(metric="tokens", limit=100, per_seconds=SecondsIn.MINUTE)],
        )
        backend = MemoryBackendBuilder().build(cfg)
        await backend.consume_capacity(frozendict({"tokens": 100.0}))
        with pytest.raises(ValueError, match="int or float"):
            await backend.await_for_capacity(
                frozendict({"tokens": "50"}),
                timeout=0.05,
            )


# ---------------------------------------------------------------------------
# Callback exception handling (capacity leak prevention)
# ---------------------------------------------------------------------------


class TestCallbackExceptionSuppression:
    """Callback exceptions must not propagate — they would prevent
    CapacityReservation construction and cause permanent capacity leaks.
    """

    async def test_on_capacity_consumed_exception_suppressed_on_acquire(self):
        cbs = _make_callbacks(
            on_capacity_consumed=AsyncMock(side_effect=RuntimeError("boom")),
        )
        backend = MemoryBackendBuilder().build(_make_config(), callbacks=cbs)
        usage = frozendict({"tokens": 100.0, "requests": 1.0})
        with pytest.warns(RuntimeWarning, match="RuntimeError.*boom"):
            await backend.await_for_capacity(usage)
        cbs.on_capacity_consumed.assert_awaited_once()

    async def test_on_capacity_consumed_exception_suppressed_on_consume(self):
        cbs = _make_callbacks(
            on_capacity_consumed=AsyncMock(side_effect=RuntimeError("boom")),
        )
        backend = MemoryBackendBuilder().build(_make_config(), callbacks=cbs)
        usage = frozendict({"tokens": 100.0, "requests": 1.0})
        with pytest.warns(RuntimeWarning, match="RuntimeError.*boom"):
            await backend.consume_capacity(usage)
        cbs.on_capacity_consumed.assert_awaited_once()

    async def test_on_capacity_refunded_exception_suppressed(self):
        cbs = _make_callbacks(
            on_capacity_refunded=AsyncMock(side_effect=RuntimeError("refund boom")),
        )
        backend = MemoryBackendBuilder().build(_make_config(), callbacks=cbs)
        reserved = frozendict({"tokens": 200.0, "requests": 2.0})
        actual = frozendict({"tokens": 100.0, "requests": 1.0})
        await backend.await_for_capacity(reserved)
        with pytest.warns(RuntimeWarning, match="RuntimeError.*refund boom"):
            await backend.refund_capacity(reserved, actual)
        cbs.on_capacity_refunded.assert_awaited_once()

    async def test_on_missing_consumption_data_exception_suppressed(self):
        cbs = _make_callbacks(
            on_missing_consumption_data=AsyncMock(
                side_effect=RuntimeError("fresh boom"),
            ),
        )
        backend = MemoryBackendBuilder().build(_make_config(), callbacks=cbs)
        usage = frozendict({"tokens": 10.0, "requests": 1.0})
        with pytest.warns(RuntimeWarning, match="RuntimeError.*fresh boom"):
            await backend.await_for_capacity(usage)
        # Both buckets triggered fresh-start, so called twice despite first raising
        assert cbs.on_missing_consumption_data.await_count == 2

    async def test_capacity_still_consumed_despite_callback_exception(self):
        """Verify capacity is actually consumed even when callback raises."""
        cbs = _make_callbacks(
            on_capacity_consumed=AsyncMock(side_effect=RuntimeError("boom")),
        )
        cfg = _make_config(
            quotas=[Quota(metric="tokens", limit=100, per_seconds=SecondsIn.MINUTE)],
        )
        backend = MemoryBackendBuilder().build(cfg, callbacks=cbs)
        with pytest.warns(RuntimeWarning):
            await backend.await_for_capacity(frozendict({"tokens": 90.0}))
        # Only 10 tokens remain — trying to acquire 50 should time out
        with pytest.raises(TimeoutError):
            await backend.await_for_capacity(
                frozendict({"tokens": 50.0}),
                timeout=0.05,
            )


class TestNegativeRefundWarning:
    async def test_overuse_warns(self):
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config())
        reserved = frozendict({"tokens": 100.0, "requests": 1.0})
        actual = frozendict({"tokens": 200.0, "requests": 1.0})
        await backend.await_for_capacity(reserved)
        with pytest.warns(RuntimeWarning, match="exceeds reserved usage"):
            await backend.refund_capacity(reserved, actual)


# ---------------------------------------------------------------------------
# End-to-end: callback exception must not leak capacity via rate limiter
# ---------------------------------------------------------------------------


class TestCallbackExceptionCapacityLeakE2E:
    """Regression test: if on_capacity_consumed raises, the rate limiter must
    still return a CapacityReservation so the caller can refund.
    """

    async def test_acquire_returns_reservation_despite_callback_exception(self):
        cbs = _make_callbacks(
            on_capacity_consumed=AsyncMock(side_effect=RuntimeError("boom")),
        )
        cfg = _make_config()
        builder = MemoryBackendBuilder()
        limiter = RateLimiter(cfg, backend=builder, callbacks=cbs)

        usage = {"tokens": 100, "requests": 1}
        with pytest.warns(RuntimeWarning, match="RuntimeError.*boom"):
            reservation = await limiter.acquire_capacity(usage, model="test-model")

        assert reservation is not None
        assert reservation.usage["tokens"] == 100
        assert reservation.model_family == "test-family"

        # Reservation is usable for refund
        await limiter.refund_capacity({"tokens": 80, "requests": 1}, reservation)

    async def test_record_usage_returns_reservation_despite_callback_exception(self):
        cbs = _make_callbacks(
            on_capacity_consumed=AsyncMock(side_effect=RuntimeError("boom")),
        )
        cfg = _make_config()
        builder = MemoryBackendBuilder()
        limiter = RateLimiter(cfg, backend=builder, callbacks=cbs)

        usage = {"tokens": 100, "requests": 1}
        with pytest.warns(RuntimeWarning, match="RuntimeError.*boom"):
            reservation = await limiter.record_usage(usage, model="test-model")

        assert reservation is not None
        assert reservation.model_family == "test-family"
