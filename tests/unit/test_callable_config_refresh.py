"""Tests for RateLimiter (async) callable config refresh (stale-callable-config fix)."""

import warnings

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._rate_limiter import RateLimiter


class TestCallableConfigQuotaRefresh:
    """When a callable config getter returns changed quotas, the backend must update."""

    async def test_limit_decrease_is_enforced(self):
        """After the callable lowers a limit, requests exceeding the new limit must fail."""
        current_limit = 100

        def config_getter(model_name: str) -> PerModelConfig:
            return PerModelConfig(
                quotas=UsageQuotas([Quota(metric="tokens", limit=current_limit, per_seconds=60)]),
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
                quotas=UsageQuotas([Quota(metric="tokens", limit=current_limit, per_seconds=60)]),
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
                quotas=UsageQuotas([
                    Quota(metric="tokens", limit=limits["tokens"], per_seconds=60),
                    Quota(metric="requests", limit=limits["requests"], per_seconds=60),
                ]),
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
    """When the callable changes metric names, the backend must be rebuilt."""

    async def test_metric_set_change_triggers_rebuild(self):
        """Changing metric names causes a new backend to be built."""
        use_new_metrics = False

        def config_getter(model_name: str) -> PerModelConfig:
            if use_new_metrics:
                quotas = UsageQuotas([Quota(metric="requests", limit=50, per_seconds=60)])
            else:
                quotas = UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=60)])
            return PerModelConfig(quotas=quotas, model_family="test-family")

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())

        # Establish the backend with "tokens" metric
        reservation = await limiter.acquire_capacity({"tokens": 10}, "test-model")
        await limiter.refund_capacity({"tokens": 0}, reservation)
        old_backend = limiter._model_family_to_backend["test-family"]

        # Switch to "requests" metric
        use_new_metrics = True

        # Next call should rebuild the backend and warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reservation2 = await limiter.acquire_capacity({"requests": 5}, "test-model")

        assert len(w) == 1
        assert "changed metric set" in str(w[0].message)

        new_backend = limiter._model_family_to_backend["test-family"]
        assert new_backend is not old_backend
        assert reservation2.usage["requests"] == 5

    async def test_metric_set_change_new_limits_enforced(self):
        """After metric set rebuild, the new limits are enforced."""
        use_new_metrics = False

        def config_getter(model_name: str) -> PerModelConfig:
            if use_new_metrics:
                quotas = UsageQuotas([Quota(metric="requests", limit=5, per_seconds=60)])
            else:
                quotas = UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=60)])
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
