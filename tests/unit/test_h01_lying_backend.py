"""Regression tests for L18 H01 lying backend detection."""

import pytest

from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    RateLimiterBackend,
    SyncRateLimiterBackend,
    backend_uses_default_prepare_reconfigured_backend,
    sync_backend_uses_default_prepare_reconfigured_backend,
)
from token_throttle._interfaces._models import Quota, SecondsIn, UsageQuotas
from token_throttle._limiter_backends._memory._backend import (
    MemoryBackend,
    MemoryBackendBuilder,
)
from token_throttle._limiter_backends._memory._bucket import MemoryBucket
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackend,
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter


def _config(metrics: tuple[str, ...]) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric=metric, limit=100, per_seconds=SecondsIn.MINUTE)
                for metric in metrics
            ]
        ),
        model_family="h01-family",
    )


def _memory_buckets(cfg: PerModelConfig) -> list[MemoryBucket]:
    return [
        MemoryBucket(
            metric=quota.metric,
            per_seconds=quota.per_seconds,
            limit=float(quota.limit),
            model_family=cfg.get_model_family(),
        )
        for quota in cfg.quotas
    ]


class LyingBackend(MemoryBackend):
    prepare_reconfigured_backend = RateLimiterBackend.prepare_reconfigured_backend


class LyingBackendBuilder(MemoryBackendBuilder):
    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> RateLimiterBackend:
        return LyingBackend(
            buckets=_memory_buckets(cfg),
            sleep_interval=self._sleep_interval,
            callbacks=callbacks,
            limit_config=cfg,
        )


class SyncLyingBackend(SyncMemoryBackend):
    prepare_reconfigured_backend = SyncRateLimiterBackend.prepare_reconfigured_backend


class SyncLyingBackendBuilder(SyncMemoryBackendBuilder):
    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: SyncRateLimiterCallbacks | None = None,
    ) -> SyncRateLimiterBackend:
        return SyncLyingBackend(
            buckets=_memory_buckets(cfg),
            sleep_interval=self._sleep_interval,
            callbacks=callbacks,
            limit_config=cfg,
        )


def test_built_in_memory_backends_override_prepare_reconfigured_backend():
    async_backend = MemoryBackendBuilder().build(_config(("tokens", "requests")))
    sync_backend = SyncMemoryBackendBuilder().build(_config(("tokens", "requests")))

    assert not backend_uses_default_prepare_reconfigured_backend(async_backend)
    assert not sync_backend_uses_default_prepare_reconfigured_backend(sync_backend)


async def test_async_limiter_rejects_lying_backend_on_metric_set_rebuild():
    expanded = False

    def config_getter(_model_name: str) -> PerModelConfig:
        metrics = ("tokens", "requests") if not expanded else ("tokens", "characters")
        return _config(metrics)

    limiter = RateLimiter(config_getter, backend=LyingBackendBuilder())
    await limiter.acquire_capacity({"tokens": 50, "requests": 0}, "test-model")

    expanded = True

    with pytest.raises(
        RuntimeError,
        match=(
            r"claims supports_metric_set_change=True.*"
            r"did not override prepare_reconfigured_backend.*"
            r"silent state drop would occur"
        ),
    ):
        await limiter.acquire_capacity({"tokens": 1, "characters": 0}, "test-model")


def test_sync_limiter_rejects_lying_backend_on_metric_set_rebuild():
    expanded = False

    def config_getter(_model_name: str) -> PerModelConfig:
        metrics = ("tokens", "requests") if not expanded else ("tokens", "characters")
        return _config(metrics)

    limiter = SyncRateLimiter(config_getter, backend=SyncLyingBackendBuilder())
    limiter.acquire_capacity({"tokens": 50, "requests": 0}, "test-model")

    expanded = True

    with pytest.raises(
        RuntimeError,
        match=(
            r"claims supports_metric_set_change=True.*"
            r"did not override prepare_reconfigured_backend.*"
            r"silent state drop would occur"
        ),
    ):
        limiter.acquire_capacity({"tokens": 1, "characters": 0}, "test-model")
