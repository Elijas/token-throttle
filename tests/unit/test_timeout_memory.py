"""Tests for the timeout parameter on await_for_capacity / wait_for_capacity.

Tests cover timeout=0 (try-acquire), timeout=N (bounded wait), and
timeout=None (default, blocks indefinitely).  These tests will fail until
the timeout parameter is implemented (TDD red phase).

Covers: async MemoryBackend and sync SyncMemoryBackend.
"""

import asyncio
import math
import time

import pytest
from frozendict import frozendict

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)


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
# Async MemoryBackend -- timeout=0 (try-acquire)
# ---------------------------------------------------------------------------


class TestAsyncTimeoutZero:
    async def test_timeout_zero_fails_when_no_capacity(self):
        """timeout=0 should raise TimeoutError immediately when capacity is exhausted."""
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config(limit=10, per_seconds=3600))

        # Exhaust all capacity
        await backend.await_for_capacity(frozendict({"requests": 10.0}))

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await backend.await_for_capacity(frozendict({"requests": 1.0}), timeout=0)
        elapsed = time.monotonic() - start

        # Should fail nearly instantly (generous CI tolerance)
        assert elapsed < 0.5, f"timeout=0 should fail immediately, took {elapsed:.2f}s"

    async def test_timeout_zero_succeeds_with_capacity(self):
        """timeout=0 should succeed immediately when capacity is available."""
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config(limit=100, per_seconds=3600))

        # Capacity is available -- should succeed immediately
        start = time.monotonic()
        await backend.await_for_capacity(frozendict({"requests": 1.0}), timeout=0)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5


# ---------------------------------------------------------------------------
# Async MemoryBackend -- timeout=N (bounded wait)
# ---------------------------------------------------------------------------


class TestAsyncTimeoutBounded:
    async def test_timeout_n_raises_after_bounded_wait(self):
        """timeout=N should raise TimeoutError after ~N seconds."""
        builder = MemoryBackendBuilder()
        # Very slow refill: 100 tokens over 3600s = 0.028/s
        backend = builder.build(_make_config(limit=100, per_seconds=3600))

        # Exhaust all capacity
        await backend.await_for_capacity(frozendict({"requests": 100.0}))

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await backend.await_for_capacity(
                frozendict({"requests": 50.0}), timeout=0.5
            )
        elapsed = time.monotonic() - start

        # Should have waited approximately 0.5s (bounded, not indefinite)
        assert elapsed >= 0.4, f"Expected wait of ~0.5s, got {elapsed:.2f}s"
        assert elapsed < 2.0, (
            f"Should not wait much longer than timeout, got {elapsed:.2f}s"
        )


class TestAsyncTimeoutInvalid:
    async def test_timeout_nan_raises_instead_of_hanging(self):
        builder = MemoryBackendBuilder()
        backend = builder.build(_make_config(limit=10, per_seconds=3600))

        await backend.await_for_capacity(frozendict({"requests": 10.0}))

        with pytest.raises(ValueError, match="timeout must be finite"):
            await asyncio.wait_for(
                backend.await_for_capacity(
                    frozendict({"requests": 1.0}),
                    timeout=math.nan,
                ),
                timeout=0.2,
            )


# ---------------------------------------------------------------------------
# Async MemoryBackend -- timeout=None (default behavior)
# ---------------------------------------------------------------------------


class TestAsyncTimeoutDefault:
    async def test_timeout_none_blocks_until_available(self):
        """timeout=None (explicit) should block indefinitely, same as current behavior."""
        builder = MemoryBackendBuilder(sleep_interval=0.01)
        # Fast refill: 100/s
        backend = builder.build(_make_config(limit=100, per_seconds=1))

        # Exhaust all capacity
        await backend.consume_capacity(frozendict({"requests": 100.0}))

        # timeout=None should block until refill, then succeed
        start = time.monotonic()
        await backend.await_for_capacity(frozendict({"requests": 1.0}), timeout=None)
        elapsed = time.monotonic() - start

        # Should succeed after brief refill wait
        assert elapsed < 2.0, (
            f"timeout=None should eventually succeed, took {elapsed:.2f}s"
        )


# ---------------------------------------------------------------------------
# Sync SyncMemoryBackend -- timeout=0 (try-acquire)
# ---------------------------------------------------------------------------


class TestSyncTimeoutZero:
    def test_timeout_zero_fails_when_no_capacity(self):
        """timeout=0 should raise TimeoutError immediately when capacity is exhausted."""
        builder = SyncMemoryBackendBuilder()
        backend = builder.build(_make_config(limit=10, per_seconds=3600))

        # Exhaust all capacity
        backend.wait_for_capacity(frozendict({"requests": 10.0}))

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            backend.wait_for_capacity(frozendict({"requests": 1.0}), timeout=0)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, f"timeout=0 should fail immediately, took {elapsed:.2f}s"

    def test_timeout_zero_succeeds_with_capacity(self):
        """timeout=0 should succeed immediately when capacity is available."""
        builder = SyncMemoryBackendBuilder()
        backend = builder.build(_make_config(limit=100, per_seconds=3600))

        start = time.monotonic()
        backend.wait_for_capacity(frozendict({"requests": 1.0}), timeout=0)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5


# ---------------------------------------------------------------------------
# Sync SyncMemoryBackend -- timeout=N (bounded wait)
# ---------------------------------------------------------------------------


class TestSyncTimeoutBounded:
    def test_timeout_n_raises_after_bounded_wait(self):
        """timeout=N should raise TimeoutError after ~N seconds."""
        builder = SyncMemoryBackendBuilder()
        backend = builder.build(_make_config(limit=100, per_seconds=3600))

        # Exhaust all capacity
        backend.wait_for_capacity(frozendict({"requests": 100.0}))

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            backend.wait_for_capacity(frozendict({"requests": 50.0}), timeout=0.5)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.4, f"Expected wait of ~0.5s, got {elapsed:.2f}s"
        assert elapsed < 2.0, (
            f"Should not wait much longer than timeout, got {elapsed:.2f}s"
        )


# ---------------------------------------------------------------------------
# Sync SyncMemoryBackend -- timeout=None (default behavior)
# ---------------------------------------------------------------------------


class TestSyncTimeoutDefault:
    def test_timeout_none_blocks_until_available(self):
        """timeout=None (explicit) should block indefinitely, same as current behavior."""
        builder = SyncMemoryBackendBuilder(sleep_interval=0.01)
        # Fast refill: 100/s
        backend = builder.build(_make_config(limit=100, per_seconds=1))

        # Exhaust all capacity
        backend.consume_capacity(frozendict({"requests": 100.0}))

        # timeout=None should block until refill, then succeed
        start = time.monotonic()
        backend.wait_for_capacity(frozendict({"requests": 1.0}), timeout=None)
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, (
            f"timeout=None should eventually succeed, took {elapsed:.2f}s"
        )
