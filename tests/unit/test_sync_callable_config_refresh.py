"""Tests for SyncRateLimiter callable config refresh (stale-callable-config fix)."""

import contextlib
import threading
import time
import warnings

import pytest

from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    SyncRateLimiterBackend,
    SyncRateLimiterBackendBuilderInterface,
)
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._bucket import MemoryBucket
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackend,
    SyncMemoryBackendBuilder,
)
from token_throttle._sync_rate_limiter import SyncRateLimiter


class TestSyncCallableConfigQuotaRefresh:
    """When a callable config getter returns changed quotas, the backend must update."""

    def test_limit_decrease_is_enforced(self):
        """After the callable lowers a limit, requests exceeding the new limit must fail."""
        current_limit = 100

        def config_getter(model_name: str) -> PerModelConfig:
            return PerModelConfig(
                quotas=UsageQuotas(
                    [Quota(metric="tokens", limit=current_limit, per_seconds=60)]
                ),
                model_family="test-family",
            )

        limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())

        # First call with limit=100 — acquire 50 tokens, then refund
        reservation = limiter.acquire_capacity({"tokens": 50}, "test-model")
        limiter.refund_capacity({"tokens": 0}, reservation)

        # Lower the limit to 10
        current_limit = 10

        # Now requesting 50 tokens should fail because max_capacity is 10
        with pytest.raises(ValueError, match=r"exceeds.*max.capacity"):
            limiter.acquire_capacity({"tokens": 50}, "test-model")

    def test_limit_increase_is_applied(self):
        """After the callable raises a limit, larger requests must be allowed."""
        current_limit = 10

        def config_getter(model_name: str) -> PerModelConfig:
            return PerModelConfig(
                quotas=UsageQuotas(
                    [Quota(metric="tokens", limit=current_limit, per_seconds=60)]
                ),
                model_family="test-family",
            )

        limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())

        # First call with limit=10 — acquire 5
        reservation = limiter.acquire_capacity({"tokens": 5}, "test-model")
        limiter.refund_capacity({"tokens": 0}, reservation)

        # Raise the limit to 100
        current_limit = 100

        # Now requesting 50 tokens should succeed
        reservation2 = limiter.acquire_capacity({"tokens": 50}, "test-model")
        assert reservation2.usage["tokens"] == 50

    def test_unchanged_config_returns_same_backend(self):
        """When the callable returns the same quotas, the backend object is reused."""

        def config_getter(model_name: str) -> PerModelConfig:
            return PerModelConfig(
                quotas=UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=60)]),
                model_family="test-family",
            )

        limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())

        limiter.acquire_capacity({"tokens": 10}, "test-model")
        backend_after_first = limiter._model_family_to_backend["test-family"]

        limiter.acquire_capacity({"tokens": 10}, "test-model")
        backend_after_second = limiter._model_family_to_backend["test-family"]

        assert backend_after_first is backend_after_second

    def test_static_config_works_as_before(self):
        """A non-callable static config still works correctly."""
        config = PerModelConfig(
            quotas=UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=60)]),
            model_family="test-family",
        )
        limiter = SyncRateLimiter(config, backend=SyncMemoryBackendBuilder())

        reservation = limiter.acquire_capacity({"tokens": 50}, "test-model")
        assert reservation.usage["tokens"] == 50

    def test_multi_metric_limit_change(self):
        """Limit changes across multiple metrics are all applied."""
        limits = {"tokens": 100, "requests": 10}

        def config_getter(model_name: str) -> PerModelConfig:
            return PerModelConfig(
                quotas=UsageQuotas(
                    [
                        Quota(metric="tokens", limit=limits["tokens"], per_seconds=60),
                        Quota(
                            metric="requests", limit=limits["requests"], per_seconds=60
                        ),
                    ]
                ),
                model_family="test-family",
            )

        limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())

        # First call establishes the backend
        reservation = limiter.acquire_capacity({"tokens": 50, "requests": 5}, "m")
        limiter.refund_capacity({"tokens": 0, "requests": 0}, reservation)

        # Lower both limits
        limits["tokens"] = 20
        limits["requests"] = 3

        # Requesting more than the new limits should fail
        with pytest.raises(ValueError, match=r"exceeds.*max.capacity"):
            limiter.acquire_capacity({"tokens": 50, "requests": 1}, "m")


class TestSyncCallableConfigMetricSetChange:
    """When the callable changes metric names, the backend must be rebuilt."""

    def test_metric_set_change_triggers_rebuild(self):
        """Changing metric names causes a new backend to be built."""
        use_new_metrics = False

        def config_getter(model_name: str) -> PerModelConfig:
            if use_new_metrics:
                quotas = UsageQuotas(
                    [Quota(metric="requests", limit=50, per_seconds=60)]
                )
            else:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=60)]
                )
            return PerModelConfig(quotas=quotas, model_family="test-family")

        limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())

        # Establish the backend with "tokens" metric
        reservation = limiter.acquire_capacity({"tokens": 10}, "test-model")
        limiter.refund_capacity({"tokens": 0}, reservation)
        old_backend = limiter._model_family_to_backend["test-family"]

        # Switch to "requests" metric
        use_new_metrics = True

        # Next call should rebuild the backend and warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reservation2 = limiter.acquire_capacity({"requests": 5}, "test-model")

        assert len(w) == 1
        assert "changed metric set" in str(w[0].message)

        new_backend = limiter._model_family_to_backend["test-family"]
        assert new_backend is not old_backend
        assert reservation2.usage["requests"] == 5

    def test_metric_set_change_new_limits_enforced(self):
        """After metric set rebuild, the new limits are enforced."""
        use_new_metrics = False

        def config_getter(model_name: str) -> PerModelConfig:
            if use_new_metrics:
                quotas = UsageQuotas(
                    [Quota(metric="requests", limit=5, per_seconds=60)]
                )
            else:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=60)]
                )
            return PerModelConfig(quotas=quotas, model_family="test-family")

        limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())

        reservation = limiter.acquire_capacity({"tokens": 10}, "test-model")
        limiter.refund_capacity({"tokens": 0}, reservation)

        use_new_metrics = True

        # New limit is 5, so requesting 10 should fail
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match=r"exceeds.*max.capacity"):
                limiter.acquire_capacity({"requests": 10}, "test-model")


class TestSyncCallableConfigMetricSetStateTransfer:
    """When metric set changes, consumption state for surviving metrics must be preserved."""

    def test_metric_expansion_preserves_surviving_state(self):
        """Expanding {tokens} → {tokens, requests}: tokens consumption state preserved."""
        use_expanded = False

        def config_getter(model_name: str) -> PerModelConfig:
            if use_expanded:
                quotas = UsageQuotas(
                    [
                        Quota(metric="tokens", limit=100, per_seconds=60),
                        Quota(metric="requests", limit=10, per_seconds=60),
                    ]
                )
            else:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=60)]
                )
            return PerModelConfig(quotas=quotas, model_family="test-family")

        limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())

        # Consume 90 tokens → ~10 remaining
        reservation = limiter.acquire_capacity({"tokens": 90}, "test-model")
        limiter.refund_capacity({"tokens": 90}, reservation)

        # Expand metric set to include requests
        use_expanded = True

        # 20 tokens should fail — only ~10 remain from old backend
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(TimeoutError):
                limiter.acquire_capacity(
                    {"tokens": 20, "requests": 1},
                    "test-model",
                    timeout=0,
                )

    def test_metric_expansion_new_metric_starts_fresh(self):
        """Expanding {tokens} → {tokens, requests}: new 'requests' metric has full capacity."""
        use_expanded = False

        def config_getter(model_name: str) -> PerModelConfig:
            if use_expanded:
                quotas = UsageQuotas(
                    [
                        Quota(metric="tokens", limit=100, per_seconds=60),
                        Quota(metric="requests", limit=10, per_seconds=60),
                    ]
                )
            else:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=60)]
                )
            return PerModelConfig(quotas=quotas, model_family="test-family")

        limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())

        # Consume 5 tokens → ~95 remaining
        reservation = limiter.acquire_capacity({"tokens": 5}, "test-model")
        limiter.refund_capacity({"tokens": 5}, reservation)

        # Expand metric set
        use_expanded = True

        # 5 tokens + 10 requests should succeed (tokens ~95 remaining, requests fresh 10)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reservation2 = limiter.acquire_capacity(
                {"tokens": 5, "requests": 10},
                "test-model",
            )
        assert reservation2.usage["tokens"] == 5
        assert reservation2.usage["requests"] == 10

    def test_metric_contraction_preserves_surviving_state(self):
        """Contracting {tokens, requests} → {tokens}: tokens consumption state preserved."""
        use_contracted = False

        def config_getter(model_name: str) -> PerModelConfig:
            if use_contracted:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=60)]
                )
            else:
                quotas = UsageQuotas(
                    [
                        Quota(metric="tokens", limit=100, per_seconds=60),
                        Quota(metric="requests", limit=10, per_seconds=60),
                    ]
                )
            return PerModelConfig(quotas=quotas, model_family="test-family")

        limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())

        # Consume 90 tokens + 5 requests
        reservation = limiter.acquire_capacity(
            {"tokens": 90, "requests": 5},
            "test-model",
        )
        limiter.refund_capacity({"tokens": 90, "requests": 5}, reservation)

        # Contract metric set — drop requests
        use_contracted = True

        # 20 tokens should fail — only ~10 remain
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(TimeoutError):
                limiter.acquire_capacity({"tokens": 20}, "test-model", timeout=0)

    def test_metric_replacement_no_surviving_state(self):
        """Replacing {tokens} → {requests}: no common metrics, requests starts fresh."""
        use_new = False

        def config_getter(model_name: str) -> PerModelConfig:
            if use_new:
                quotas = UsageQuotas(
                    [Quota(metric="requests", limit=10, per_seconds=60)]
                )
            else:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=60)]
                )
            return PerModelConfig(quotas=quotas, model_family="test-family")

        limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())

        # Consume 90 tokens
        reservation = limiter.acquire_capacity({"tokens": 90}, "test-model")
        limiter.refund_capacity({"tokens": 90}, reservation)

        # Replace with completely different metric
        use_new = True

        # Requests starts fresh — full 10 available
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reservation2 = limiter.acquire_capacity({"requests": 10}, "test-model")
        assert reservation2.usage["requests"] == 10


class RacingSyncMemoryBackendBuilder(SyncMemoryBackendBuilder):
    """Block rebuilds long enough for duplicate refreshes to overlap."""

    def __init__(self) -> None:
        super().__init__()
        self.build_calls = 0
        self._build_calls_lock = threading.Lock()
        self._rebuild_barrier = threading.Barrier(2)

    def build(self, cfg, *, callbacks=None):
        with self._build_calls_lock:
            self.build_calls += 1
            build_call = self.build_calls
        if build_call >= 2:
            with contextlib.suppress(threading.BrokenBarrierError):
                self._rebuild_barrier.wait(timeout=0.2)
        return super().build(cfg, callbacks=callbacks)


class BlockingPrepareSyncMemoryBackend(SyncMemoryBackend):
    """Pause metric-set reconfiguration so concurrent callers can run."""

    def __init__(
        self,
        *,
        prepare_started: threading.Event,
        release_prepare: threading.Event,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._prepare_started = prepare_started
        self._release_prepare = release_prepare

    def prepare_reconfigured_backend(
        self,
        new_backend: SyncRateLimiterBackend,
        cfg: PerModelConfig,
    ) -> SyncRateLimiterBackend:
        self._prepare_started.set()
        assert self._release_prepare.wait(timeout=2.0)
        return super().prepare_reconfigured_backend(new_backend, cfg)


class BlockingPrepareSyncMemoryBackendBuilder(SyncMemoryBackendBuilder):
    """Build backends that can pause right before a metric-set swap."""

    def __init__(self) -> None:
        super().__init__()
        self.prepare_started = threading.Event()
        self.release_prepare = threading.Event()

    def build(self, cfg, *, callbacks=None):
        buckets = [
            MemoryBucket(
                metric=quota.metric,
                per_seconds=quota.per_seconds,
                limit=float(quota.limit),
                model_family=cfg.get_model_family(),
            )
            for quota in cfg.quotas
        ]
        return BlockingPrepareSyncMemoryBackend(
            buckets=buckets,
            limit_config=cfg,
            sleep_interval=self._sleep_interval,
            callbacks=callbacks,
            prepare_started=self.prepare_started,
            release_prepare=self.release_prepare,
        )


class TestSyncCallableConfigMetricSetConcurrency:
    """Concurrent refreshes must not rebuild and consume from split state."""

    def test_metric_expansion_refresh_is_serialized(self):
        use_expanded = False

        def config_getter(model_name: str) -> PerModelConfig:
            if use_expanded:
                quotas = UsageQuotas(
                    [
                        Quota(metric="tokens", limit=100, per_seconds=60),
                        Quota(metric="requests", limit=10, per_seconds=60),
                    ]
                )
            else:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=60)]
                )
            return PerModelConfig(quotas=quotas, model_family="test-family")

        builder = RacingSyncMemoryBackendBuilder()
        limiter = SyncRateLimiter(config_getter, backend=builder)

        reservation = limiter.acquire_capacity({"tokens": 90}, "test-model")
        limiter.refund_capacity({"tokens": 90}, reservation)
        old_backend = limiter._model_family_to_backend["test-family"]

        start_barrier = threading.Barrier(3)
        results: list[str] = []
        results_lock = threading.Lock()

        def worker() -> None:
            start_barrier.wait()
            try:
                limiter.acquire_capacity(
                    {"tokens": 8, "requests": 1},
                    "test-model",
                    timeout=0,
                )
            except TimeoutError:
                result = "TimeoutError"
            else:
                result = "success"
            with results_lock:
                results.append(result)

        use_expanded = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with old_backend._condition:
                threads = [threading.Thread(target=worker) for _ in range(2)]
                for thread in threads:
                    thread.start()
                start_barrier.wait()
            for thread in threads:
                thread.join()

        assert builder.build_calls == 2
        assert sorted(results) == ["TimeoutError", "success"]


class TestSyncCallableConfigMetricSetRefunds:
    """Refunds created before a rebuild must still update surviving metrics."""

    def test_metric_expansion_refund_after_rebuild_updates_surviving_metric(self):
        use_expanded = False

        def config_getter(model_name: str) -> PerModelConfig:
            if use_expanded:
                quotas = UsageQuotas(
                    [
                        Quota(metric="tokens", limit=100, per_seconds=60),
                        Quota(metric="requests", limit=10, per_seconds=60),
                    ]
                )
            else:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=60)]
                )
            return PerModelConfig(quotas=quotas, model_family="test-family")

        limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())

        reservation = limiter.acquire_capacity({"tokens": 90}, "test-model")
        use_expanded = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            limiter.acquire_capacity({"tokens": 0, "requests": 0}, "test-model")

        limiter.refund_capacity({"tokens": 20}, reservation)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reservation2 = limiter.acquire_capacity(
                {"tokens": 50, "requests": 10},
                "test-model",
                timeout=0,
            )
        assert reservation2.usage["tokens"] == 50

    def test_metric_contraction_refund_after_rebuild_ignores_dropped_metric(self):
        use_contracted = False

        def config_getter(model_name: str) -> PerModelConfig:
            if use_contracted:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=60)]
                )
            else:
                quotas = UsageQuotas(
                    [
                        Quota(metric="tokens", limit=100, per_seconds=60),
                        Quota(metric="requests", limit=10, per_seconds=60),
                    ]
                )
            return PerModelConfig(quotas=quotas, model_family="test-family")

        limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())

        reservation = limiter.acquire_capacity(
            {"tokens": 90, "requests": 5},
            "test-model",
        )
        use_contracted = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            limiter.acquire_capacity({"tokens": 0}, "test-model")

        limiter.refund_capacity({"tokens": 20, "requests": 5}, reservation)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reservation2 = limiter.acquire_capacity(
                {"tokens": 50},
                "test-model",
                timeout=0,
            )
        assert reservation2.usage["tokens"] == 50


class TestSyncCallableConfigMetricSetRebuildIntegrity:
    """Metric-set refresh must not lose live in-memory state."""

    def test_metric_expansion_preserves_old_backend_consumption(self):
        use_expanded = False

        def config_getter(model_name: str) -> PerModelConfig:
            if use_expanded:
                quotas = UsageQuotas(
                    [
                        Quota(metric="tokens", limit=100, per_seconds=60),
                        Quota(metric="requests", limit=10, per_seconds=60),
                    ]
                )
            else:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=60)]
                )
            return PerModelConfig(quotas=quotas, model_family="test-family")

        builder = BlockingPrepareSyncMemoryBackendBuilder()
        limiter = SyncRateLimiter(config_getter, backend=builder)

        reservation = limiter.acquire_capacity({"tokens": 90}, "test-model")
        limiter.refund_capacity({"tokens": 90}, reservation)

        use_expanded = True

        def trigger_rebuild() -> None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                limiter.acquire_capacity(
                    {"tokens": 0, "requests": 0},
                    "test-model",
                )

        rebuild_thread = threading.Thread(target=trigger_rebuild)
        rebuild_thread.start()
        assert builder.prepare_started.wait(timeout=2.0)

        use_expanded = False
        limiter.acquire_capacity({"tokens": 10}, "test-model", timeout=0)

        use_expanded = True
        builder.release_prepare.set()
        rebuild_thread.join(timeout=2.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(TimeoutError):
                limiter.acquire_capacity(
                    {"tokens": 10, "requests": 1},
                    "test-model",
                    timeout=0,
                )

    def test_metric_expansion_preserves_refill_during_prepare_delay(self):
        use_expanded = False

        def config_getter(model_name: str) -> PerModelConfig:
            if use_expanded:
                quotas = UsageQuotas(
                    [
                        Quota(metric="tokens", limit=100, per_seconds=1),
                        Quota(metric="requests", limit=10, per_seconds=60),
                    ]
                )
            else:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=1)]
                )
            return PerModelConfig(quotas=quotas, model_family="test-family")

        builder = BlockingPrepareSyncMemoryBackendBuilder()
        limiter = SyncRateLimiter(config_getter, backend=builder)

        reservation = limiter.acquire_capacity({"tokens": 100}, "test-model")
        limiter.refund_capacity({"tokens": 100}, reservation)

        use_expanded = True

        def trigger_rebuild() -> None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                limiter.acquire_capacity(
                    {"tokens": 0, "requests": 0},
                    "test-model",
                )

        rebuild_thread = threading.Thread(target=trigger_rebuild)
        rebuild_thread.start()
        assert builder.prepare_started.wait(timeout=2.0)
        time.sleep(0.5)
        builder.release_prepare.set()
        rebuild_thread.join(timeout=2.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reservation2 = limiter.acquire_capacity(
                {"tokens": 50, "requests": 1},
                "test-model",
                timeout=0,
            )
        assert reservation2.usage["tokens"] == 50
        assert reservation2.usage["requests"] == 1


class SimpleSyncBackend(SyncRateLimiterBackend):
    def wait_for_capacity(self, usage, *, timeout=None) -> None:
        return None

    def consume_capacity(self, usage) -> None:
        return None

    def refund_capacity(self, reserved_usage, actual_usage) -> None:
        return None

    def set_max_capacity(self, metric, per_seconds, value) -> None:
        return None


class SimpleSyncBackendBuilder(SyncRateLimiterBackendBuilderInterface):
    def build(self, cfg, *, callbacks=None) -> SyncRateLimiterBackend:
        return SimpleSyncBackend()


class TestSyncCallableConfigMetricSetBackendSupport:
    """Metric-set changes should fail fast for unsupported backend types."""

    def test_metric_set_change_without_backend_support_raises(self):
        use_new_metrics = False

        def config_getter(model_name: str) -> PerModelConfig:
            if use_new_metrics:
                quotas = UsageQuotas(
                    [Quota(metric="requests", limit=10, per_seconds=60)]
                )
            else:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=60)]
                )
            return PerModelConfig(quotas=quotas, model_family="test-family")

        limiter = SyncRateLimiter(config_getter, backend=SimpleSyncBackendBuilder())

        limiter.acquire_capacity({"tokens": 1}, "test-model")
        use_new_metrics = True

        with pytest.raises(RuntimeError, match="does not support metric-set changes"):
            limiter.acquire_capacity({"requests": 0}, "test-model")
