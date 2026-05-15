"""Regression tests for Round 3 audit findings."""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import (
    Quota,
    SecondsIn,
    UsageQuotas,
)
from token_throttle._limiter_backends._memory._backend import (
    MemoryBackendBuilder,
)
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter


def _make_config(
    *,
    model_family: str = "test-family",
    quotas: list[Quota] | None = None,
) -> PerModelConfig:
    if quotas is None:
        quotas = [Quota(metric="tokens", limit=100, per_seconds=SecondsIn.MINUTE)]
    return PerModelConfig(
        quotas=UsageQuotas(quotas),
        model_family=model_family,
    )


def _make_mock_backend_builder():
    mock_backend = AsyncMock()
    mock_backend.await_for_capacity.return_value = None
    mock_backend.refund_capacity.return_value = None
    mock_builder = MagicMock()
    mock_builder.build.return_value = mock_backend
    return mock_builder, mock_backend


def _make_sync_mock_backend_builder():
    mock_backend = MagicMock()
    mock_backend.await_for_capacity.return_value = None
    mock_backend.refund_capacity.return_value = None
    mock_builder = MagicMock()
    mock_builder.build.return_value = mock_backend
    return mock_builder, mock_backend


# ── F02.R3.01: Concurrent duplicate refund TOCTOU ──


class TestConcurrentDuplicateRefundAsync:
    """F02.R3.01: Two concurrent refund_capacity calls for the same
    reservation must not both credit the backend.
    """

    async def test_concurrent_refund_deduplicates_second_caller(self):
        builder, mock_backend = _make_mock_backend_builder()

        async def slow_refund(*args, **kwargs):
            await asyncio.sleep(0.01)

        mock_backend.refund_capacity_for_buckets.side_effect = slow_refund

        limiter = RateLimiter(_make_config(), backend=builder)
        reservation = await limiter.acquire_capacity({"tokens": 10}, model="test-model")

        async def refund_once():
            await limiter.refund_capacity({"tokens": 5}, reservation)

        with pytest.warns(UserWarning, match="has already been refunded"):
            await asyncio.gather(refund_once(), refund_once())
        assert mock_backend.refund_capacity_for_buckets.await_count == 1

    async def test_failed_refund_retry_is_not_deduplicated_until_success(self):
        builder, mock_backend = _make_mock_backend_builder()

        call_count = 0

        async def fail_after_commit(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("backend failure")

        mock_backend.refund_capacity_for_buckets.side_effect = fail_after_commit

        limiter = RateLimiter(_make_config(), backend=builder)
        reservation = await limiter.acquire_capacity({"tokens": 10}, model="test-model")

        with pytest.raises(RuntimeError, match="backend failure"):
            await limiter.refund_capacity({"tokens": 5}, reservation)

        with pytest.raises(RuntimeError, match="backend failure"):
            await limiter.refund_capacity({"tokens": 5}, reservation)
        assert call_count == 2


class TestConcurrentDuplicateRefundSync:
    """F02.R3.01 (sync): Same TOCTOU fix for SyncRateLimiter."""

    def test_concurrent_refund_deduplicates_second_caller(self):
        builder, mock_backend = _make_sync_mock_backend_builder()
        hit_refund = threading.Event()

        def slow_refund(*args, **kwargs):
            hit_refund.set()
            time.sleep(0.01)

        mock_backend.refund_capacity_for_buckets.side_effect = slow_refund

        limiter = SyncRateLimiter(_make_config(), backend=builder)
        reservation = limiter.acquire_capacity({"tokens": 10}, model="test-model")

        errors: list[Exception] = []

        def refund_b():
            hit_refund.wait()
            try:
                with pytest.warns(UserWarning, match="has already been refunded"):
                    limiter.refund_capacity({"tokens": 5}, reservation)
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=refund_b)
        t.start()
        limiter.refund_capacity({"tokens": 5}, reservation)
        t.join(timeout=5)

        assert errors == []
        assert mock_backend.refund_capacity_for_buckets.call_count == 1


# ── F04.R3.02: _bucket_registry bounded growth ──


class TestBucketRegistryEviction:
    """F04.R3.02: Reconfiguration must prune stale entries from _bucket_registry."""

    async def test_async_registry_pruned_on_reconfigure(self):
        cfg_a = _make_config(
            quotas=[
                Quota(metric="tokens", limit=100, per_seconds=SecondsIn.MINUTE),
                Quota(metric="requests", limit=10, per_seconds=SecondsIn.MINUTE),
            ],
        )
        builder = MemoryBackendBuilder()
        backend_a = builder.build(cfg_a)
        assert len(backend_a._bucket_registry) == 2

        cfg_b = _make_config(
            quotas=[Quota(metric="tokens", limit=200, per_seconds=SecondsIn.MINUTE)],
        )
        new_backend = builder.build(cfg_b)
        await backend_a.prepare_reconfigured_backend(new_backend, cfg_b)

        assert ("tokens", 60) in backend_a._bucket_registry
        assert ("requests", 60) not in backend_a._bucket_registry
        assert len(backend_a._bucket_registry) == 1

    def test_sync_registry_pruned_on_reconfigure(self):
        cfg_a = _make_config(
            quotas=[
                Quota(metric="tokens", limit=100, per_seconds=SecondsIn.MINUTE),
                Quota(metric="requests", limit=10, per_seconds=SecondsIn.MINUTE),
            ],
        )
        builder = SyncMemoryBackendBuilder()
        backend_a = builder.build(cfg_a)
        assert len(backend_a._bucket_registry) == 2

        cfg_b = _make_config(
            quotas=[Quota(metric="tokens", limit=200, per_seconds=SecondsIn.MINUTE)],
        )
        new_backend = builder.build(cfg_b)
        backend_a.prepare_reconfigured_backend(new_backend, cfg_b)

        assert ("tokens", 60) in backend_a._bucket_registry
        assert ("requests", 60) not in backend_a._bucket_registry
        assert len(backend_a._bucket_registry) == 1
