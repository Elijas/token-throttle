"""
Regression tests: rate changes (``set_max_capacity``,
``prepare_reconfigured_backend``) must NOT apply the new refill rate retroactively to
time that elapsed under the old rate.

Context: ``calculate_capacity`` integrates a single ``rate_per_sec`` across
``[last_checked, current_time]``. Mutating the rate without snapshotting
``capacity``/``last_checked`` would retroactively grant (or revoke) tokens that
were never actually accrued under the new rate.
"""

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._limiter_backends._memory import _backend as memory_backend_module
from token_throttle._limiter_backends._memory import (
    _sync_backend as sync_memory_backend_module,
)
from token_throttle._limiter_backends._memory._backend import (
    MemoryBackendBuilder,
)
from token_throttle._limiter_backends._memory._bucket import MemoryBucket
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

# ---------------------------------------------------------------------------
# Direct MemoryBucket: piecewise behavior
# ---------------------------------------------------------------------------


class TestMemoryBucketSnapshotOnRateChange:
    def test_raise_max_capacity_does_not_backfill_at_new_rate(self):
        """T=0: drain to 0. T=60: raise 10->1000. Capacity at T=60 must be ~10
        (accrued under OLD rate), not 1000 (would be backfill at NEW rate).
        """
        bucket = MemoryBucket(
            metric="tokens", per_seconds=60, limit=10.0, model_family="test"
        )
        bucket.set_capacity(0.0, current_time=0.0)
        # Old rate = 10/60. Over [0, 60] that gives 10 tokens, capped at old max 10.
        bucket.set_max_capacity(1000.0, current_time=60.0)
        result = bucket.get_capacity(current_time=60.0)
        assert result.amount == pytest.approx(10.0)
        # After the swap, new rate 1000/60 applies going forward.
        # Snapshot=10 + 60 * 1000/60 = 1010, capped at new max 1000.
        result = bucket.get_capacity(current_time=120.0)
        assert result.amount == pytest.approx(1000.0)

    def test_lower_max_capacity_does_not_retroactively_reduce_accrual(self):
        """T=0: drain to 0 (limit=1000). T=30 (half-refilled): lower 1000->10.
        Accrued under old rate by T=30 = 500. After snapshot, it should be
        capped at the NEW max 10, not reflect a new-rate re-integration.
        """
        bucket = MemoryBucket(
            metric="tokens", per_seconds=60, limit=1000.0, model_family="test"
        )
        bucket.set_capacity(0.0, current_time=0.0)
        bucket.set_max_capacity(10.0, current_time=30.0)
        # Snapshot=min(1000, 500) = 500 clamped by the new max at the next read.
        result = bucket.get_capacity(current_time=30.0)
        assert result.amount == pytest.approx(10.0)

    def test_fresh_bucket_skips_snapshot(self):
        """No prior state => no retroactive issue; change rate and max, move on."""
        bucket = MemoryBucket(
            metric="tokens", per_seconds=60, limit=10.0, model_family="test"
        )
        bucket.set_max_capacity(1000.0, current_time=100.0)
        result = bucket.get_capacity(current_time=100.0)
        assert result.amount == 1000.0
        assert result.is_fresh_start is True


# ---------------------------------------------------------------------------
# MemoryBackend (async + sync) via time.time() monkeypatch
# ---------------------------------------------------------------------------


class _ClockStub:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock_async(monkeypatch):
    clock = _ClockStub(1000.0)
    monkeypatch.setattr(memory_backend_module.time, "time", clock)
    return clock


@pytest.fixture
def clock_sync(monkeypatch):
    clock = _ClockStub(1000.0)
    monkeypatch.setattr(sync_memory_backend_module.time, "time", clock)
    return clock


def _make_quota(limit: float, per_seconds: int = 60, metric: str = "tokens") -> Quota:
    return Quota(metric=metric, limit=limit, per_seconds=per_seconds)


def _make_config(limit: float, per_seconds: int = 60) -> PerModelConfig:
    return PerModelConfig(
        model_family="test",
        quotas=UsageQuotas([_make_quota(limit, per_seconds)]),
    )


class TestMemoryBackendSnapshotOnRateChange:
    async def test_async_set_max_capacity_snapshots_under_old_rate(self, clock_async):
        backend = MemoryBackendBuilder().build(_make_config(limit=10, per_seconds=60))

        # Drain all 10 tokens at T=1000.
        clock_async.t = 1000.0
        await backend.await_for_capacity(frozen_usage({"tokens": 10}))

        # T=1060: raise limit from 10 to 1000.
        clock_async.t = 1060.0
        await backend.set_max_capacity("tokens", 60, 1000.0)

        # Immediately (still T=1060): must NOT have 1000 tokens free.
        # Accrued under old rate across [1000, 1060] = 10, capped at old max 10.
        with pytest.raises(TimeoutError):
            await backend.await_for_capacity(frozen_usage({"tokens": 500}), timeout=0.0)
        # But 10 tokens should be available (that's the snapshotted amount).
        await backend.await_for_capacity(frozen_usage({"tokens": 10}), timeout=0.0)

    def test_sync_set_max_capacity_snapshots_under_old_rate(self, clock_sync):
        backend = SyncMemoryBackendBuilder().build(
            _make_config(limit=10, per_seconds=60)
        )

        clock_sync.t = 1000.0
        backend.wait_for_capacity(frozen_usage({"tokens": 10}))

        clock_sync.t = 1060.0
        backend.set_max_capacity("tokens", 60, 1000.0)

        with pytest.raises(TimeoutError):
            backend.wait_for_capacity(frozen_usage({"tokens": 500}), timeout=0.0)
        backend.wait_for_capacity(frozen_usage({"tokens": 10}), timeout=0.0)


# ---------------------------------------------------------------------------
# prepare_reconfigured_backend: reused bucket must also snapshot
# ---------------------------------------------------------------------------


class TestPrepareReconfiguredBackendSnapshot:
    async def test_async_reconfigure_snapshots_reused_bucket(self, clock_async):
        backend = MemoryBackendBuilder().build(_make_config(limit=10, per_seconds=60))
        clock_async.t = 1000.0
        await backend.await_for_capacity(frozen_usage({"tokens": 10}))

        # Build the new backend (itself using the same clock).
        clock_async.t = 1060.0
        new_backend = MemoryBackendBuilder().build(
            _make_config(limit=1000, per_seconds=60)
        )
        await backend.prepare_reconfigured_backend(
            new_backend, _make_config(limit=1000, per_seconds=60)
        )

        # At T=1060 the reused bucket was drained at T=1000 at rate 10/60.
        # Over [1000, 1060] => accrued 10 tokens (capped at old max 10). Despite
        # the new max of 1000, we must NOT get free tokens retroactively.
        with pytest.raises(TimeoutError):
            await backend.await_for_capacity(frozen_usage({"tokens": 500}), timeout=0.0)
        await backend.await_for_capacity(frozen_usage({"tokens": 10}), timeout=0.0)

    def test_sync_reconfigure_snapshots_reused_bucket(self, clock_sync):
        backend = SyncMemoryBackendBuilder().build(
            _make_config(limit=10, per_seconds=60)
        )
        clock_sync.t = 1000.0
        backend.wait_for_capacity(frozen_usage({"tokens": 10}))

        clock_sync.t = 1060.0
        new_backend = SyncMemoryBackendBuilder().build(
            _make_config(limit=1000, per_seconds=60)
        )
        backend.prepare_reconfigured_backend(
            new_backend, _make_config(limit=1000, per_seconds=60)
        )

        with pytest.raises(TimeoutError):
            backend.wait_for_capacity(frozen_usage({"tokens": 500}), timeout=0.0)
        backend.wait_for_capacity(frozen_usage({"tokens": 10}), timeout=0.0)


# ---------------------------------------------------------------------------
# Public API regression: the exact observable-consequence scenario from the bug
# ---------------------------------------------------------------------------


def _make_public_api_config(limit: float, per_seconds: int = 60) -> PerModelConfig:
    return PerModelConfig(
        model_family="m1",
        quotas=UsageQuotas(
            [Quota(metric="tokens", limit=limit, per_seconds=per_seconds)]
        ),
    )


class TestPublicApiRateChangeSnapshot:
    """
    Scenario from the bug report:

      T=0:  acquire 10 tokens (limit=10/min, rate≈0.167/s) -> capacity=0
      T=60: set_max_capacity("m1", "tokens", 60, 1000.0)  (limit=1000/min)
      T=60: acquire(500) with timeout=0 must raise TimeoutError.
    """

    def test_sync_rate_limiter_reports_timeout_after_limit_raise(self, clock_sync):
        cfg = _make_public_api_config(limit=10, per_seconds=60)
        limiter = SyncRateLimiter(cfg, backend=SyncMemoryBackendBuilder())

        clock_sync.t = 0.0
        limiter.acquire_capacity({"tokens": 10}, model="m1")

        clock_sync.t = 60.0
        limiter.set_max_capacity(
            model="m1", metric="tokens", per_seconds=60, value=1000.0
        )

        with pytest.raises(TimeoutError):
            limiter.acquire_capacity({"tokens": 500}, model="m1", timeout=0.0)

    async def test_async_rate_limiter_reports_timeout_after_limit_raise(
        self, clock_async
    ):
        cfg = _make_public_api_config(limit=10, per_seconds=60)
        limiter = RateLimiter(cfg, backend=MemoryBackendBuilder())

        clock_async.t = 0.0
        await limiter.acquire_capacity({"tokens": 10}, model="m1")

        clock_async.t = 60.0
        await limiter.set_max_capacity(
            model="m1", metric="tokens", per_seconds=60, value=1000.0
        )

        with pytest.raises(TimeoutError):
            await limiter.acquire_capacity({"tokens": 500}, model="m1", timeout=0.0)
