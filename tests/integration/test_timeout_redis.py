"""Integration tests for the timeout parameter on await_for_capacity / wait_for_capacity.

Parameterized across all backends (memory + redis) via the backend_builder
and sync_backend_builder fixtures.

Tests cover timeout=0 (try-acquire), timeout=N (bounded wait), and
timeout=None (default).  These tests will fail until timeout is implemented
(TDD red phase).
"""

import time

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage


def _make_config(
    *, limit: float = 100, per_seconds: int = 3600, metric: str = "requests"
) -> PerModelConfig:
    return PerModelConfig(
        model_family="test",
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
    )


# ---------------------------------------------------------------------------
# Async -- timeout=0 (try-acquire)
# ---------------------------------------------------------------------------


async def test_timeout_zero_fails_when_no_capacity(backend_builder):
    """timeout=0 should raise TimeoutError immediately when capacity is exhausted."""
    config = _make_config(limit=10, per_seconds=3600)
    backend = backend_builder.build(config)

    # Exhaust all capacity
    await backend.await_for_capacity(frozen_usage({"requests": 10}))

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        await backend.await_for_capacity(frozen_usage({"requests": 1}), timeout=0)
    elapsed = time.monotonic() - start

    # Should fail nearly instantly
    assert elapsed < 0.5, f"timeout=0 should fail immediately, took {elapsed:.2f}s"


async def test_timeout_zero_succeeds_with_capacity(backend_builder):
    """timeout=0 should succeed immediately when capacity is available."""
    config = _make_config(limit=100, per_seconds=3600)
    backend = backend_builder.build(config)

    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1}), timeout=0)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5


# ---------------------------------------------------------------------------
# Async -- timeout=N (bounded wait)
# ---------------------------------------------------------------------------


async def test_timeout_n_raises_after_bounded_wait(backend_builder):
    """timeout=N should raise TimeoutError after ~N seconds."""
    # Very slow refill: 100 tokens over 3600s = 0.028/s
    config = _make_config(limit=100, per_seconds=3600)
    backend = backend_builder.build(config)

    # Exhaust all capacity
    await backend.await_for_capacity(frozen_usage({"requests": 100}))

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        await backend.await_for_capacity(frozen_usage({"requests": 50}), timeout=0.5)
    elapsed = time.monotonic() - start

    # Should have waited approximately 0.5s (bounded, not indefinite)
    assert elapsed >= 0.4, f"Expected wait of ~0.5s, got {elapsed:.2f}s"
    assert elapsed < 2.0, (
        f"Should not wait much longer than timeout, got {elapsed:.2f}s"
    )


# ---------------------------------------------------------------------------
# Async -- timeout=None (default behavior unchanged)
# ---------------------------------------------------------------------------


async def test_timeout_none_blocks_until_available(backend_builder):
    """timeout=None should block until capacity is available (same as default)."""
    # Fast refill: 100/s
    config = _make_config(limit=100, per_seconds=1)
    backend = backend_builder.build(config)

    # Exhaust all capacity
    await backend.consume_capacity(frozen_usage({"requests": 100}))

    # timeout=None should block until refill, then succeed
    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1}), timeout=None)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"timeout=None should eventually succeed, took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Sync -- timeout=0 (try-acquire)
# ---------------------------------------------------------------------------


def test_sync_timeout_zero_fails_when_no_capacity(sync_backend_builder):
    """timeout=0 should raise TimeoutError immediately (sync)."""
    config = _make_config(limit=10, per_seconds=3600)
    backend = sync_backend_builder.build(config)

    # Exhaust all capacity
    backend.wait_for_capacity(frozen_usage({"requests": 10}))

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        backend.wait_for_capacity(frozen_usage({"requests": 1}), timeout=0)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, f"timeout=0 should fail immediately, took {elapsed:.2f}s"


def test_sync_timeout_zero_succeeds_with_capacity(sync_backend_builder):
    """timeout=0 should succeed immediately when capacity is available (sync)."""
    config = _make_config(limit=100, per_seconds=3600)
    backend = sync_backend_builder.build(config)

    start = time.monotonic()
    backend.wait_for_capacity(frozen_usage({"requests": 1}), timeout=0)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5


# ---------------------------------------------------------------------------
# Sync -- timeout=N (bounded wait)
# ---------------------------------------------------------------------------


def test_sync_timeout_n_raises_after_bounded_wait(sync_backend_builder):
    """timeout=N should raise TimeoutError after ~N seconds (sync)."""
    config = _make_config(limit=100, per_seconds=3600)
    backend = sync_backend_builder.build(config)

    # Exhaust all capacity
    backend.wait_for_capacity(frozen_usage({"requests": 100}))

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        backend.wait_for_capacity(frozen_usage({"requests": 50}), timeout=0.5)
    elapsed = time.monotonic() - start

    assert elapsed >= 0.4, f"Expected wait of ~0.5s, got {elapsed:.2f}s"
    assert elapsed < 2.0, (
        f"Should not wait much longer than timeout, got {elapsed:.2f}s"
    )


# ---------------------------------------------------------------------------
# Sync -- timeout=None (default behavior unchanged)
# ---------------------------------------------------------------------------


def test_sync_timeout_none_blocks_until_available(sync_backend_builder):
    """timeout=None should block until capacity is available (sync)."""
    # Fast refill: 100/s
    config = _make_config(limit=100, per_seconds=1)
    backend = sync_backend_builder.build(config)

    # Exhaust all capacity
    backend.consume_capacity(frozen_usage({"requests": 100}))

    # timeout=None should block until refill, then succeed
    start = time.monotonic()
    backend.wait_for_capacity(frozen_usage({"requests": 1}), timeout=None)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"timeout=None should eventually succeed, took {elapsed:.2f}s"
