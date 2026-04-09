"""Tests for RateLimiter (async) callable config refresh (stale-callable-config fix)."""

import asyncio
import warnings

import pytest

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    RateLimiterBackend,
    RateLimiterBackendBuilderInterface,
)
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import (
    MemoryBackend,
    MemoryBackendBuilder,
)
from token_throttle._limiter_backends._memory._bucket import MemoryBucket
from token_throttle._rate_limiter import RateLimiter


class TestCallableConfigQuotaRefresh:
    """When a callable config getter returns changed quotas, the backend must update."""

    async def test_limit_decrease_is_enforced(self):
        """After the callable lowers a limit, requests exceeding the new limit must fail."""
        current_limit = 100

        def config_getter(model_name: str) -> PerModelConfig:
            return PerModelConfig(
                quotas=UsageQuotas(
                    [Quota(metric="tokens", limit=current_limit, per_seconds=60)]
                ),
                model_family="test-family",
            )

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        # First call with limit=100 — acquire 50 tokens, then refund
        reservation = await limiter.acquire_capacity({"tokens": 50}, "test-model")
        await limiter.refund_capacity({"tokens": 0}, reservation)

        # Lower the limit to 10
        current_limit = 10

        # Now requesting 50 tokens should fail because max_capacity is 10
        with pytest.raises(ValueError, match=r"exceeds.*max.capacity"):
            await limiter.acquire_capacity({"tokens": 50}, "test-model")

    async def test_limit_increase_is_applied(self):
        """After the callable raises a limit, larger requests must be allowed."""
        current_limit = 10

        def config_getter(model_name: str) -> PerModelConfig:
            return PerModelConfig(
                quotas=UsageQuotas(
                    [Quota(metric="tokens", limit=current_limit, per_seconds=60)]
                ),
                model_family="test-family",
            )

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        # First call with limit=10 — acquire 5
        reservation = await limiter.acquire_capacity({"tokens": 5}, "test-model")
        await limiter.refund_capacity({"tokens": 0}, reservation)

        # Raise the limit to 100
        current_limit = 100

        # Now requesting 50 tokens should succeed
        reservation2 = await limiter.acquire_capacity({"tokens": 50}, "test-model")
        assert reservation2.usage["tokens"] == 50

    async def test_unchanged_config_returns_same_backend(self):
        """When the callable returns the same quotas, the backend object is reused."""

        def config_getter(model_name: str) -> PerModelConfig:
            return PerModelConfig(
                quotas=UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=60)]),
                model_family="test-family",
            )

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        await limiter.acquire_capacity({"tokens": 10}, "test-model")
        backend_after_first = limiter._model_family_to_backend["test-family"]

        await limiter.acquire_capacity({"tokens": 10}, "test-model")
        backend_after_second = limiter._model_family_to_backend["test-family"]

        assert backend_after_first is backend_after_second

    async def test_static_config_works_as_before(self):
        """A non-callable static config still works correctly."""
        config = PerModelConfig(
            quotas=UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=60)]),
            model_family="test-family",
        )
        limiter = RateLimiter(config, backend=MemoryBackendBuilder())

        reservation = await limiter.acquire_capacity({"tokens": 50}, "test-model")
        assert reservation.usage["tokens"] == 50

    async def test_multi_metric_limit_change(self):
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

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        # First call establishes the backend
        reservation = await limiter.acquire_capacity({"tokens": 50, "requests": 5}, "m")
        await limiter.refund_capacity({"tokens": 0, "requests": 0}, reservation)

        # Lower both limits
        limits["tokens"] = 20
        limits["requests"] = 3

        # Requesting more than the new limits should fail
        with pytest.raises(ValueError, match=r"exceeds.*max.capacity"):
            await limiter.acquire_capacity({"tokens": 50, "requests": 1}, "m")


class TestCallableConfigMetricSetChange:
    """When the callable changes metric names, the backend must be refreshed."""

    async def test_metric_set_change_reconfigures_cached_backend_in_place(self):
        """Changing metric names updates the cached backend object in place."""
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

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        # Establish the backend with "tokens" metric
        reservation = await limiter.acquire_capacity({"tokens": 10}, "test-model")
        await limiter.refund_capacity({"tokens": 0}, reservation)
        old_backend = limiter._model_family_to_backend["test-family"]

        # Switch to "requests" metric
        use_new_metrics = True

        # Next call should refresh the backend and warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reservation2 = await limiter.acquire_capacity({"requests": 5}, "test-model")

        assert len(w) == 1
        assert "changed metric set" in str(w[0].message)

        new_backend = limiter._model_family_to_backend["test-family"]
        assert new_backend is old_backend
        assert reservation2.usage["requests"] == 5

    async def test_metric_set_change_new_limits_enforced(self):
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

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        reservation = await limiter.acquire_capacity({"tokens": 10}, "test-model")
        await limiter.refund_capacity({"tokens": 0}, reservation)

        use_new_metrics = True

        # New limit is 5, so requesting 10 should fail
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match=r"exceeds.*max.capacity"):
                await limiter.acquire_capacity({"requests": 10}, "test-model")

    async def test_refund_refreshes_backend_before_metric_drop_is_applied(self):
        """Refunds must not credit metrics removed by the latest callable config."""
        state = "both"

        def config_getter(model_name: str) -> PerModelConfig:
            quotas = [Quota(metric="requests", limit=1, per_seconds=3600)]
            if state == "both":
                quotas.append(Quota(metric="tokens", limit=10, per_seconds=3600))
            return PerModelConfig(
                quotas=UsageQuotas(quotas), model_family="test-family"
            )

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        reservation = await limiter.acquire_capacity(
            {"requests": 1, "tokens": 10},
            "test-model",
        )

        state = "requests"
        await limiter.refund_capacity({"requests": 1, "tokens": 0}, reservation)
        await limiter.record_usage({"requests": 0}, "test-model")

        state = "both"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            await limiter.record_usage({"requests": 0, "tokens": 0}, "test-model")

        with pytest.raises(TimeoutError):
            await limiter.acquire_capacity(
                {"requests": 0, "tokens": 10},
                "test-model",
                timeout=0,
            )

    async def test_set_max_capacity_refreshes_backend_after_metric_expansion(self):
        """set_max_capacity must rebuild cached backends before mutating new buckets."""
        use_expanded = False

        def config_getter(model_name: str) -> PerModelConfig:
            quotas = [Quota(metric="requests", limit=10, per_seconds=60)]
            if use_expanded:
                quotas.append(Quota(metric="tokens", limit=100, per_seconds=60))
            return PerModelConfig(
                quotas=UsageQuotas(quotas), model_family="test-family"
            )

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())
        await limiter.acquire_capacity({"requests": 1}, "test-model")

        use_expanded = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await limiter.set_max_capacity("test-model", "tokens", 60, 200)

        assert any("changed metric set" in str(item.message) for item in caught)

        reservation = await limiter.acquire_capacity(
            {"requests": 0, "tokens": 150},
            "test-model",
        )
        assert reservation.usage["tokens"] == 150

    async def test_metric_expansion_preserves_runtime_max_capacity_when_static_unchanged(
        self,
    ):
        """Metric-set rebuilds must keep live set_max_capacity overrides when quota limits stay the same."""
        use_expanded = False

        def config_getter(model_name: str) -> PerModelConfig:
            quotas = [Quota(metric="tokens", limit=100, per_seconds=60)]
            if use_expanded:
                quotas.append(Quota(metric="requests", limit=10, per_seconds=60))
            return PerModelConfig(
                quotas=UsageQuotas(quotas), model_family="test-family"
            )

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())
        await limiter.acquire_capacity({"tokens": 0}, "test-model")

        await limiter.set_max_capacity("test-model", "tokens", 60, 20)
        with pytest.raises(ValueError, match=r"exceeds.*max.capacity"):
            await limiter.acquire_capacity({"tokens": 30}, "test-model", timeout=0)

        use_expanded = True
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match=r"exceeds.*max.capacity"):
                await limiter.acquire_capacity(
                    {"tokens": 30, "requests": 0},
                    "test-model",
                    timeout=0,
                )

    async def test_runtime_max_capacity_stays_visible_while_metric_set_rebuilds(self):
        """Waiters must not observe the stale static quota during a rebuild."""
        use_expanded = False
        wait_started = asyncio.Event()

        def config_getter(model_name: str) -> PerModelConfig:
            quotas = [Quota(metric="tokens", limit=100, per_seconds=1)]
            if use_expanded:
                quotas.append(Quota(metric="requests", limit=10, per_seconds=60))
            return PerModelConfig(
                quotas=UsageQuotas(quotas), model_family="test-family"
            )

        async def on_wait_start(**_kwargs) -> None:
            wait_started.set()

        limiter = RateLimiter(
            config_getter,
            backend=DelayedPrepareMemoryBackendBuilder(sleep_interval=0.01),
            callbacks=RateLimiterCallbacks(on_wait_start=on_wait_start),
        )
        await limiter.acquire_capacity({"tokens": 100}, "test-model")
        await limiter.set_max_capacity("test-model", "tokens", 1, 200)

        waiter = asyncio.create_task(
            limiter.acquire_capacity({"tokens": 150}, "test-model", timeout=1.0)
        )
        await asyncio.wait_for(wait_started.wait(), timeout=0.2)

        use_expanded = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await limiter.acquire_capacity(
                {"tokens": 0, "requests": 0},
                "test-model",
            )

        reservation = await waiter
        assert reservation.usage["tokens"] == 150


class TestCallableConfigMetricSetStateTransfer:
    """When metric set changes, consumption state for surviving metrics must be preserved."""

    async def test_metric_expansion_preserves_surviving_state(self):
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

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        # Consume 90 tokens → ~10 remaining
        reservation = await limiter.acquire_capacity({"tokens": 90}, "test-model")
        await limiter.refund_capacity({"tokens": 90}, reservation)

        # Expand metric set to include requests
        use_expanded = True

        # 20 tokens should fail — only ~10 remain from old backend
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(TimeoutError):
                await limiter.acquire_capacity(
                    {"tokens": 20, "requests": 1},
                    "test-model",
                    timeout=0,
                )

    async def test_metric_expansion_new_metric_starts_fresh(self):
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

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        # Consume 5 tokens → ~95 remaining
        reservation = await limiter.acquire_capacity({"tokens": 5}, "test-model")
        await limiter.refund_capacity({"tokens": 5}, reservation)

        # Expand metric set
        use_expanded = True

        # 5 tokens + 10 requests should succeed (tokens ~95 remaining, requests fresh 10)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reservation2 = await limiter.acquire_capacity(
                {"tokens": 5, "requests": 10},
                "test-model",
            )
        assert reservation2.usage["tokens"] == 5
        assert reservation2.usage["requests"] == 10

    async def test_metric_contraction_preserves_surviving_state(self):
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

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        # Consume 90 tokens + 5 requests
        reservation = await limiter.acquire_capacity(
            {"tokens": 90, "requests": 5},
            "test-model",
        )
        await limiter.refund_capacity({"tokens": 90, "requests": 5}, reservation)

        # Contract metric set — drop requests
        use_contracted = True

        # 20 tokens should fail — only ~10 remain
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(TimeoutError):
                await limiter.acquire_capacity({"tokens": 20}, "test-model", timeout=0)

    async def test_metric_replacement_no_surviving_state(self):
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

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        # Consume 90 tokens
        reservation = await limiter.acquire_capacity({"tokens": 90}, "test-model")
        await limiter.refund_capacity({"tokens": 90}, reservation)

        # Replace with completely different metric
        use_new = True

        # Requests starts fresh — full 10 available
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reservation2 = await limiter.acquire_capacity(
                {"requests": 10}, "test-model"
            )
        assert reservation2.usage["requests"] == 10

    async def test_metric_readdition_preserves_dormant_state(self):
        """Re-adding a removed bucket must restore its previous consumption state."""
        phase = 0

        def config_getter(model_name: str) -> PerModelConfig:
            if phase in {0, 2}:
                quotas = UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=3600)]
                )
            else:
                quotas = UsageQuotas(
                    [Quota(metric="requests", limit=10, per_seconds=3600)]
                )
            return PerModelConfig(quotas=quotas, model_family="test-family")

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        reservation = await limiter.acquire_capacity({"tokens": 90}, "test-model")
        await limiter.refund_capacity({"tokens": 90}, reservation)

        phase = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await limiter.acquire_capacity({"requests": 0}, "test-model")

        phase = 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(TimeoutError):
                await limiter.acquire_capacity({"tokens": 50}, "test-model", timeout=0)


class TestCallableConfigWindowChangeHandling:
    """Window-only changes must update blocked acquires and later refunds correctly."""

    async def test_window_change_applies_to_existing_blocked_waiter(self):
        current_window = 1

        def config_getter(model_name: str) -> PerModelConfig:
            return PerModelConfig(
                quotas=UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=current_window)]
                ),
                model_family="test-family",
            )

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        await limiter.acquire_capacity({"tokens": 100}, "test-model")

        async def waiter():
            started_at = asyncio.get_running_loop().time()
            await limiter.acquire_capacity({"tokens": 50}, "test-model", timeout=0.6)
            return asyncio.get_running_loop().time() - started_at

        waiter = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)

        current_window = 3600
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await limiter.acquire_capacity({"tokens": 0}, "test-model")

        elapsed = await waiter
        assert elapsed < 0.25

    async def test_refund_after_window_replacement_does_not_credit_new_window(self):
        current_window = 60

        def config_getter(model_name: str) -> PerModelConfig:
            return PerModelConfig(
                quotas=UsageQuotas(
                    [Quota(metric="tokens", limit=100, per_seconds=current_window)]
                ),
                model_family="test-family",
            )

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        reservation = await limiter.acquire_capacity({"tokens": 50}, "test-model")

        current_window = 3600
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await limiter.acquire_capacity({"tokens": 0}, "test-model")

        await limiter.acquire_capacity({"tokens": 80}, "test-model")
        await limiter.refund_capacity({"tokens": 0}, reservation)

        with pytest.raises(TimeoutError):
            await limiter.acquire_capacity({"tokens": 70}, "test-model", timeout=0)

    async def test_refund_only_credits_surviving_same_metric_windows(self):
        use_expanded_windows = False

        def config_getter(model_name: str) -> PerModelConfig:
            quotas = [Quota(metric="tokens", limit=100, per_seconds=60)]
            if use_expanded_windows:
                quotas.append(Quota(metric="tokens", limit=20, per_seconds=3600))
            return PerModelConfig(
                quotas=UsageQuotas(quotas),
                model_family="test-family",
            )

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        reservation = await limiter.acquire_capacity({"tokens": 20}, "test-model")

        use_expanded_windows = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await limiter.acquire_capacity({"tokens": 0}, "test-model")

        await limiter.acquire_capacity({"tokens": 10}, "test-model")
        await limiter.refund_capacity({"tokens": 0}, reservation)

        with pytest.raises(TimeoutError):
            await limiter.acquire_capacity({"tokens": 15}, "test-model", timeout=0)


class TestCallableConfigMetricSetWaiters:
    """Blocked acquires must handle metric-set refreshes without hitting invariants."""

    async def test_metric_expansion_preserves_existing_blocked_waiter(self):
        use_expanded = False

        def config_getter(model_name: str) -> PerModelConfig:
            quotas = [Quota(metric="tokens", limit=100, per_seconds=1)]
            if use_expanded:
                quotas.append(Quota(metric="requests", limit=10, per_seconds=60))
            return PerModelConfig(
                quotas=UsageQuotas(quotas),
                model_family="test-family",
            )

        limiter = RateLimiter(
            config_getter,
            backend=MemoryBackendBuilder(sleep_interval=0.01),
        )

        await limiter.acquire_capacity({"tokens": 100}, "test-model")

        waiter = asyncio.create_task(
            limiter.acquire_capacity({"tokens": 1}, "test-model", timeout=1.0)
        )
        await asyncio.sleep(0.05)

        use_expanded = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await limiter.acquire_capacity(
                {"tokens": 0, "requests": 0},
                "test-model",
            )

        reservation = await waiter
        assert reservation.usage["tokens"] == 1


class GateCondition:
    """Gate later condition acquisitions without blocking the initial state transfer."""

    def __init__(self, release_event: asyncio.Event) -> None:
        self._condition = asyncio.Condition()
        self._release_event = release_event
        self._enter_count = 0

    async def __aenter__(self):
        if self._enter_count:
            await self._release_event.wait()
        self._enter_count += 1
        return await self._condition.__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        return await self._condition.__aexit__(exc_type, exc, tb)

    async def wait(self):
        return await self._condition.wait()

    def notify_all(self) -> None:
        self._condition.notify_all()


class RacingMemoryBackendBuilder(MemoryBackendBuilder):
    """Inject a gate into rebuilt backends so concurrent refreshes can overlap."""

    def __init__(self) -> None:
        super().__init__()
        self.build_calls = 0
        self.release_new_backend_condition = asyncio.Event()

    def build(self, cfg, *, callbacks=None):
        self.build_calls += 1
        backend = super().build(cfg, callbacks=callbacks)
        if self.build_calls >= 2:
            backend._condition = GateCondition(self.release_new_backend_condition)
        return backend


class BlockingPrepareMemoryBackend(MemoryBackend):
    """Pause metric-set reconfiguration so concurrent callers can run."""

    def __init__(
        self,
        *,
        prepare_started: asyncio.Event,
        release_prepare: asyncio.Event,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._prepare_started = prepare_started
        self._release_prepare = release_prepare

    async def prepare_reconfigured_backend(
        self,
        new_backend: RateLimiterBackend,
        cfg: PerModelConfig,
    ) -> RateLimiterBackend:
        self._prepare_started.set()
        await self._release_prepare.wait()
        return await super().prepare_reconfigured_backend(new_backend, cfg)


class BlockingPrepareMemoryBackendBuilder(MemoryBackendBuilder):
    """Build backends that can pause right before a metric-set swap."""

    def __init__(self) -> None:
        super().__init__()
        self.prepare_started = asyncio.Event()
        self.release_prepare = asyncio.Event()

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
        return BlockingPrepareMemoryBackend(
            buckets=buckets,
            limit_config=cfg,
            sleep_interval=self._sleep_interval,
            callbacks=callbacks,
            prepare_started=self.prepare_started,
            release_prepare=self.release_prepare,
        )


class DelayedPrepareMemoryBackend(MemoryBackend):
    """Keep the post-prepare state visible long enough to expose rebuild races."""

    async def prepare_reconfigured_backend(
        self,
        new_backend: RateLimiterBackend,
        cfg: PerModelConfig,
    ) -> RateLimiterBackend:
        backend = await super().prepare_reconfigured_backend(new_backend, cfg)
        await asyncio.sleep(0.05)
        return backend


class DelayedPrepareMemoryBackendBuilder(MemoryBackendBuilder):
    """Build backends that pause after installing rebuilt bucket state."""

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
        return DelayedPrepareMemoryBackend(
            buckets=buckets,
            limit_config=cfg,
            sleep_interval=self._sleep_interval,
            callbacks=callbacks,
        )


class TestCallableConfigMetricSetConcurrency:
    """Concurrent refreshes must not rebuild and consume from split state."""

    async def test_metric_expansion_refresh_is_serialized(self):
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

        builder = RacingMemoryBackendBuilder()
        limiter = RateLimiter(config_getter, backend=builder)

        reservation = await limiter.acquire_capacity({"tokens": 90}, "test-model")
        await limiter.refund_capacity({"tokens": 90}, reservation)
        old_backend = limiter._model_family_to_backend["test-family"]

        async def worker() -> str:
            try:
                await limiter.acquire_capacity(
                    {"tokens": 8, "requests": 1},
                    "test-model",
                    timeout=0,
                )
            except TimeoutError:
                return "TimeoutError"
            return "success"

        use_expanded = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            async with old_backend._condition:
                task1 = asyncio.create_task(worker())
                task2 = asyncio.create_task(worker())
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            builder.release_new_backend_condition.set()
            results = await asyncio.gather(task1, task2)

        assert builder.build_calls == 2
        assert sorted(results) == ["TimeoutError", "success"]


class TestCallableConfigMetricSetRefunds:
    """Refunds created before a rebuild must still update surviving metrics."""

    async def test_metric_expansion_refund_after_rebuild_updates_surviving_metric(self):
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

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        reservation = await limiter.acquire_capacity({"tokens": 90}, "test-model")
        use_expanded = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await limiter.acquire_capacity(
                {"tokens": 0, "requests": 0},
                "test-model",
            )

        await limiter.refund_capacity({"tokens": 20}, reservation)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reservation2 = await limiter.acquire_capacity(
                {"tokens": 50, "requests": 10},
                "test-model",
                timeout=0,
            )
        assert reservation2.usage["tokens"] == 50
        assert reservation2.usage["requests"] == 10

    async def test_metric_contraction_refund_after_rebuild_ignores_dropped_metric(
        self,
    ):
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

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        reservation = await limiter.acquire_capacity(
            {"tokens": 90, "requests": 5},
            "test-model",
        )
        use_contracted = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await limiter.acquire_capacity({"tokens": 0}, "test-model")

        await limiter.refund_capacity({"tokens": 20, "requests": 5}, reservation)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reservation2 = await limiter.acquire_capacity(
                {"tokens": 50},
                "test-model",
                timeout=0,
            )
        assert reservation2.usage["tokens"] == 50


class TestCallableConfigMetricSetRebuildIntegrity:
    """Metric-set refresh must not lose live in-memory state."""

    async def test_metric_expansion_preserves_old_backend_consumption(self):
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

        builder = BlockingPrepareMemoryBackendBuilder()
        limiter = RateLimiter(config_getter, backend=builder)

        reservation = await limiter.acquire_capacity({"tokens": 90}, "test-model")
        await limiter.refund_capacity({"tokens": 90}, reservation)

        use_expanded = True

        async def trigger_rebuild() -> None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                await limiter.acquire_capacity(
                    {"tokens": 0, "requests": 0},
                    "test-model",
                )

        rebuild_task = asyncio.create_task(trigger_rebuild())
        await builder.prepare_started.wait()

        use_expanded = False
        await limiter.acquire_capacity({"tokens": 10}, "test-model", timeout=0)

        use_expanded = True
        builder.release_prepare.set()
        await rebuild_task

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(TimeoutError):
                await limiter.acquire_capacity(
                    {"tokens": 10, "requests": 1},
                    "test-model",
                    timeout=0,
                )

    async def test_metric_expansion_preserves_refill_during_prepare_delay(self):
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
                quotas = UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=1)])
            return PerModelConfig(quotas=quotas, model_family="test-family")

        builder = BlockingPrepareMemoryBackendBuilder()
        limiter = RateLimiter(config_getter, backend=builder)

        reservation = await limiter.acquire_capacity({"tokens": 100}, "test-model")
        await limiter.refund_capacity({"tokens": 100}, reservation)

        use_expanded = True

        async def trigger_rebuild() -> None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                await limiter.acquire_capacity(
                    {"tokens": 0, "requests": 0},
                    "test-model",
                )

        rebuild_task = asyncio.create_task(trigger_rebuild())
        await builder.prepare_started.wait()
        await asyncio.sleep(0.5)
        builder.release_prepare.set()
        await rebuild_task

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reservation2 = await limiter.acquire_capacity(
                {"tokens": 50, "requests": 1},
                "test-model",
                timeout=0,
            )
        assert reservation2.usage["tokens"] == 50
        assert reservation2.usage["requests"] == 1


class SimpleAsyncBackend(RateLimiterBackend):
    async def await_for_capacity(self, usage, **kwargs) -> None:
        return None

    async def consume_capacity(self, usage) -> None:
        return None

    async def refund_capacity(self, reserved_usage, actual_usage) -> None:
        return None

    async def set_max_capacity(self, metric, per_seconds, value) -> None:
        return None


class SimpleAsyncBackendBuilder(RateLimiterBackendBuilderInterface):
    def build(self, cfg, *, callbacks=None) -> RateLimiterBackend:
        return SimpleAsyncBackend()


class TestCallableConfigMetricSetBackendSupport:
    """Metric-set changes should fail fast for unsupported backend types."""

    async def test_metric_set_change_without_backend_support_raises(self):
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

        limiter = RateLimiter(config_getter, backend=SimpleAsyncBackendBuilder())

        await limiter.acquire_capacity({"tokens": 1}, "test-model")
        use_new_metrics = True

        with pytest.raises(RuntimeError, match="does not support metric-set changes"):
            await limiter.acquire_capacity({"requests": 0}, "test-model")
