"""Tests for SyncMemoryBackend and SyncMemoryBackendBuilder coverage gaps.

Mirror of test_memory_backend.py for the synchronous backend.
"""

from unittest.mock import MagicMock

import pytest
from frozendict import frozendict

from token_throttle._interfaces._callbacks import SyncRateLimiterCallbacks
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, SecondsIn, UsageQuotas
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackend,
    SyncMemoryBackendBuilder,
)


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


def _make_callbacks(**overrides) -> SyncRateLimiterCallbacks:
    defaults = {
        "on_wait_start": MagicMock(),
        "after_wait_end_consumption": MagicMock(),
        "on_capacity_consumed": MagicMock(),
        "on_capacity_refunded": MagicMock(),
        "on_missing_consumption_data": MagicMock(),
    }
    defaults.update(overrides)
    return SyncRateLimiterCallbacks(**defaults)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TestSyncMemoryBackendBuilder:
    def test_build_returns_sync_memory_backend(self):
        builder = SyncMemoryBackendBuilder()
        cfg = _make_config()
        backend = builder.build(cfg)
        assert isinstance(backend, SyncMemoryBackend)

    def test_build_with_sleep_interval(self):
        builder = SyncMemoryBackendBuilder(sleep_interval=0.5)
        cfg = _make_config()
        backend = builder.build(cfg)
        assert backend._sleep_interval == 0.5

    def test_build_with_callbacks(self):
        builder = SyncMemoryBackendBuilder()
        cfg = _make_config()
        cbs = _make_callbacks()
        backend = builder.build(cfg, callbacks=cbs)
        assert backend._callbacks is cbs

    def test_build_creates_correct_number_of_buckets(self):
        builder = SyncMemoryBackendBuilder()
        cfg = _make_config()
        backend = builder.build(cfg)
        assert len(backend._buckets) == 2


# ---------------------------------------------------------------------------
# _set_capacities bad key
# ---------------------------------------------------------------------------


class TestSetCapacitiesBadKey:
    def test_set_capacities_unknown_bucket_raises(self):
        builder = SyncMemoryBackendBuilder()
        backend = builder.build(_make_config())
        bad_caps = frozendict({("nonexistent_metric", 60): 100.0})
        with pytest.raises(
            ValueError, match="Bucket 'nonexistent_metric/60s' not found"
        ):
            backend._set_capacities(bad_caps, 1000.0)


# ---------------------------------------------------------------------------
# set_max_capacity bad key
# ---------------------------------------------------------------------------


class TestSetMaxCapacityBadKey:
    def test_set_max_capacity_unknown_metric_raises(self):
        builder = SyncMemoryBackendBuilder()
        backend = builder.build(_make_config())
        with pytest.raises(ValueError, match="Bucket 'nonexistent/60s' not found"):
            backend.set_max_capacity("nonexistent", 60, 500.0)

    def test_set_max_capacity_wrong_per_seconds_raises(self):
        builder = SyncMemoryBackendBuilder()
        backend = builder.build(_make_config())
        with pytest.raises(ValueError, match="Bucket 'tokens/3600s' not found"):
            backend.set_max_capacity("tokens", 3600, 500.0)


# ---------------------------------------------------------------------------
# Multi-metric continue branches
# ---------------------------------------------------------------------------


class TestMultiMetricContinueBranches:
    def test_check_and_consume_with_multi_window(self):
        builder = SyncMemoryBackendBuilder()
        cfg = _make_multi_window_config()
        backend = builder.build(cfg)
        usage = frozendict({"tokens": 100.0, "requests": 1.0})
        ok, pre, post = backend._check_and_consume_capacity(usage)
        assert ok is True
        assert len(pre) == 4
        assert len(post) == 4

    def test_consume_capacity_with_multi_window(self):
        builder = SyncMemoryBackendBuilder()
        cfg = _make_multi_window_config()
        backend = builder.build(cfg)
        usage = frozendict({"tokens": 100.0, "requests": 1.0})
        backend.consume_capacity(usage)

    def test_refund_capacity_with_multi_window(self):
        builder = SyncMemoryBackendBuilder()
        cfg = _make_multi_window_config()
        backend = builder.build(cfg)
        reserved = frozendict({"tokens": 200.0, "requests": 2.0})
        actual = frozendict({"tokens": 100.0, "requests": 1.0})
        backend.wait_for_capacity(reserved)
        backend.refund_capacity(reserved, actual)

    def test_wait_for_capacity_with_multi_window(self):
        builder = SyncMemoryBackendBuilder()
        cfg = _make_multi_window_config()
        backend = builder.build(cfg)
        usage = frozendict({"tokens": 50.0, "requests": 1.0})
        backend.wait_for_capacity(usage)


# ---------------------------------------------------------------------------
# Callback invocations
# ---------------------------------------------------------------------------


class TestCapacityConsumedCallback:
    def test_on_capacity_consumed_fires_on_acquire(self):
        cbs = _make_callbacks()
        builder = SyncMemoryBackendBuilder()
        backend = builder.build(_make_config(), callbacks=cbs)
        usage = frozendict({"tokens": 100.0, "requests": 1.0})
        backend.wait_for_capacity(usage)
        cbs.on_capacity_consumed.assert_called_once()
        call_kwargs = cbs.on_capacity_consumed.call_args.kwargs
        assert call_kwargs["model_family"] == "test-family"
        assert call_kwargs["usage"] == usage

    def test_on_capacity_consumed_fires_on_consume(self):
        cbs = _make_callbacks()
        builder = SyncMemoryBackendBuilder()
        backend = builder.build(_make_config(), callbacks=cbs)
        usage = frozendict({"tokens": 100.0, "requests": 1.0})
        backend.consume_capacity(usage)
        cbs.on_capacity_consumed.assert_called_once()


class TestWaitCallbacks:
    def test_wait_callbacks_fire_when_waiting(self):
        cbs = _make_callbacks()
        builder = SyncMemoryBackendBuilder(sleep_interval=0.01)
        cfg = PerModelConfig(
            quotas=UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=1)]),
            model_family="test-family",
        )
        backend = builder.build(cfg, callbacks=cbs)

        backend.consume_capacity(frozendict({"tokens": 100.0}))
        cbs.on_capacity_consumed.reset_mock()

        # rate is 100/sec, so 0.01s sleep should refill ~1 token
        backend.wait_for_capacity(frozendict({"tokens": 1.0}))

        cbs.on_wait_start.assert_called_once()
        call_kwargs = cbs.on_wait_start.call_args.kwargs
        assert call_kwargs["model_family"] == "test-family"

        cbs.after_wait_end_consumption.assert_called_once()
        end_kwargs = cbs.after_wait_end_consumption.call_args.kwargs
        assert end_kwargs["wait_time_s"] > 0


class TestRefundCallbacks:
    def test_on_capacity_refunded_fires(self):
        cbs = _make_callbacks()
        builder = SyncMemoryBackendBuilder()
        backend = builder.build(_make_config(), callbacks=cbs)

        reserved = frozendict({"tokens": 200.0, "requests": 2.0})
        actual = frozendict({"tokens": 100.0, "requests": 1.0})
        backend.wait_for_capacity(reserved)
        backend.refund_capacity(reserved, actual)

        cbs.on_capacity_refunded.assert_called_once()
        call_kwargs = cbs.on_capacity_refunded.call_args.kwargs
        assert call_kwargs["model_family"] == "test-family"
        assert call_kwargs["reserved_usage"] == reserved
        assert call_kwargs["actual_usage"] == actual


# ---------------------------------------------------------------------------
# Fresh-start callback
# ---------------------------------------------------------------------------


class TestFreshStartCallback:
    def test_fresh_start_callback_fires_on_first_access(self):
        cbs = _make_callbacks()
        builder = SyncMemoryBackendBuilder()
        backend = builder.build(_make_config(), callbacks=cbs)

        usage = frozendict({"tokens": 10.0, "requests": 1.0})
        backend.wait_for_capacity(usage)

        assert cbs.on_missing_consumption_data.call_count == 2
        calls = cbs.on_missing_consumption_data.call_args_list
        metrics_reported = {c.kwargs["usage_metric"] for c in calls}
        assert metrics_reported == {"tokens", "requests"}

    def test_fresh_start_callback_fires_on_consume_capacity(self):
        cbs = _make_callbacks()
        builder = SyncMemoryBackendBuilder()
        backend = builder.build(_make_config(), callbacks=cbs)

        usage = frozendict({"tokens": 10.0, "requests": 1.0})
        backend.consume_capacity(usage)

        assert cbs.on_missing_consumption_data.call_count == 2

    def test_fresh_start_callback_not_on_second_access(self):
        cbs = _make_callbacks()
        builder = SyncMemoryBackendBuilder()
        backend = builder.build(_make_config(), callbacks=cbs)

        reserved = frozendict({"tokens": 100.0, "requests": 1.0})
        backend.wait_for_capacity(reserved)
        cbs.on_missing_consumption_data.reset_mock()

        actual = frozendict({"tokens": 50.0, "requests": 1.0})
        backend.refund_capacity(reserved, actual)
        cbs.on_missing_consumption_data.assert_not_called()


# ---------------------------------------------------------------------------
# Negative refund warning
# ---------------------------------------------------------------------------


class TestNegativeRefundWarning:
    def test_overuse_warns(self):
        builder = SyncMemoryBackendBuilder()
        backend = builder.build(_make_config())
        reserved = frozendict({"tokens": 100.0, "requests": 1.0})
        actual = frozendict({"tokens": 200.0, "requests": 1.0})
        backend.wait_for_capacity(reserved)
        with pytest.warns(RuntimeWarning, match="exceeds reserved usage"):
            backend.refund_capacity(reserved, actual)
