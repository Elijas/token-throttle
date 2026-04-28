"""
Sync cancellation tests — verify that KeyboardInterrupt during callbacks
triggers capacity refund and propagates correctly.

Covers the sync BaseException handlers added in Round 1 CRIT-02/03 that
had zero test coverage (R2 F18.01 / F18.02).
"""

import time

import pytest

from token_throttle._interfaces._callbacks import SyncRateLimiterCallbacks
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)


def _make_config(
    *,
    metric: str = "requests",
    limit: float = 100,
    per_seconds: float = 60,
) -> PerModelConfig:
    return PerModelConfig(
        model_family="test",
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
    )


def _read_capacity(backend, metric: str = "requests", per_seconds: int = 60) -> float:
    with backend._condition:
        caps, _ = backend._get_capacities(time.time())
    return caps[(metric, per_seconds)]


class TestKeyboardInterruptDuringOnCapacityConsumed:
    """KeyboardInterrupt in on_capacity_consumed must refund capacity."""

    def test_capacity_refunded_after_interrupt(self):
        call_count = 0

        def exploding_callback(**kwargs):
            nonlocal call_count
            call_count += 1
            raise KeyboardInterrupt

        callbacks = SyncRateLimiterCallbacks(on_capacity_consumed=exploding_callback)
        config = _make_config(limit=100, per_seconds=60)
        backend = SyncMemoryBackendBuilder().build(config, callbacks=callbacks)

        cap_before = _read_capacity(backend)
        assert cap_before == pytest.approx(100, abs=1)

        with pytest.raises(KeyboardInterrupt):
            backend.wait_for_capacity(frozen_usage({"requests": 10}))

        assert call_count == 1

        cap_after = _read_capacity(backend)
        assert cap_after == pytest.approx(100, abs=1)


class TestKeyboardInterruptDuringAfterWaitEnd:
    """KeyboardInterrupt in after_wait_end_consumption must refund capacity."""

    def test_capacity_refunded_after_interrupt_post_wait(self):
        call_count = 0

        def exploding_after_wait(**kwargs):
            nonlocal call_count
            call_count += 1
            raise KeyboardInterrupt

        callbacks = SyncRateLimiterCallbacks(
            after_wait_end_consumption=exploding_after_wait,
        )
        config = _make_config(limit=10, per_seconds=1)
        backend = SyncMemoryBackendBuilder().build(config, callbacks=callbacks)

        backend.consume_capacity(frozen_usage({"requests": 10}))

        with pytest.raises(KeyboardInterrupt):
            backend.wait_for_capacity(frozen_usage({"requests": 1}))

        assert call_count == 1

        cap_after = _read_capacity(backend, per_seconds=1)
        assert cap_after > 0


class TestRefundCancelledConsumption:
    """_refund_cancelled_consumption acquires lock, reads capacities, and restores."""

    def test_refund_restores_capacity(self):
        config = _make_config(limit=100, per_seconds=60)
        backend = SyncMemoryBackendBuilder().build(config)

        backend.wait_for_capacity(frozen_usage({"requests": 40}))
        cap_mid = _read_capacity(backend)
        assert cap_mid == pytest.approx(60, abs=1)

        backend._refund_cancelled_consumption(frozen_usage({"requests": 40}))

        cap_after = _read_capacity(backend)
        assert cap_after == pytest.approx(100, abs=1)

    def test_refund_does_not_exceed_max_capacity(self):
        config = _make_config(limit=100, per_seconds=60)
        backend = SyncMemoryBackendBuilder().build(config)

        backend._refund_cancelled_consumption(frozen_usage({"requests": 50}))

        cap_after = _read_capacity(backend)
        assert cap_after == pytest.approx(100, abs=1)
