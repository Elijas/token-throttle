import time

import pytest
from pydantic import ValidationError

from token_throttle import (
    CardinalityLimitExceededError,
    PerModelConfig,
    Quota,
    RateLimiter,
    SyncRateLimiter,
    UsageQuotas,
)
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)


def _config(*, model_family: str | None = None, metrics: list[str] | None = None):
    return PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric=metric, limit=10_000_000, per_seconds=60)
                for metric in (metrics or ["tokens"])
            ]
        ),
        model_family=model_family,
    )


async def test_async_max_model_families_is_fail_closed():
    limiter = RateLimiter(
        lambda model: _config(model_family=f"family-{model}"),
        backend=MemoryBackendBuilder(),
        max_model_families=10,
    )

    for index in range(10):
        await limiter.record_usage({"tokens": 1}, model=f"model-{index}")

    with pytest.raises(CardinalityLimitExceededError, match="max_model_families"):
        await limiter.record_usage({"tokens": 1}, model="model-10")


async def test_async_max_metrics_per_family_is_fail_closed():
    limiter = RateLimiter(
        _config(metrics=["m0", "m1"]),
        backend=MemoryBackendBuilder(),
        max_metrics_per_family=1,
    )

    with pytest.raises(CardinalityLimitExceededError, match="max_metrics_per_family"):
        await limiter.record_usage({"m0": 1, "m1": 1}, model="model")


async def test_async_max_aliases_is_fail_closed():
    limiter = RateLimiter(
        lambda _model: _config(model_family="shared"),
        backend=MemoryBackendBuilder(),
        max_aliases=2,
    )

    await limiter.record_usage({"tokens": 1}, model="alias-0")
    await limiter.record_usage({"tokens": 1}, model="alias-1")

    with pytest.raises(CardinalityLimitExceededError, match="max_aliases"):
        await limiter.record_usage({"tokens": 1}, model="alias-2")


async def test_async_max_in_flight_reservations_is_fail_closed_before_consume():
    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        max_in_flight_reservations=2,
    )

    await limiter.acquire_capacity({"tokens": 1}, model="model")
    await limiter.acquire_capacity({"tokens": 1}, model="model")

    with pytest.raises(CardinalityLimitExceededError, match="max_in_flight"):
        await limiter.acquire_capacity({"tokens": 1}, model="model")


async def test_model_family_length_cap_raises():
    with pytest.raises((CardinalityLimitExceededError, ValidationError)):
        _config(model_family="f" * 257)
    with pytest.raises((CardinalityLimitExceededError, ValidationError)):
        Quota(metric="m" * 65, limit=1)

    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        max_alias_length=10,
    )
    with pytest.raises(CardinalityLimitExceededError, match="max_alias_length"):
        await limiter.record_usage({"tokens": 1}, model="m" * 11)

    metric_limiter = RateLimiter(
        _config(metrics=["tokens"]),
        backend=MemoryBackendBuilder(),
        max_metric_length=5,
    )
    with pytest.raises(CardinalityLimitExceededError, match="max_metric_length"):
        await metric_limiter.record_usage({"tokens": 1}, model="model")


async def test_clear_unused_model_families_evicts_only_idle_rows():
    limiter = RateLimiter(
        lambda model: _config(model_family=model),
        backend=MemoryBackendBuilder(),
    )

    idle = await limiter.acquire_capacity({"tokens": 1}, model="idle")
    active = await limiter.acquire_capacity({"tokens": 1}, model="active")
    await limiter.refund_capacity({"tokens": 0}, idle)

    assert limiter.clear_unused_model_families(0) == 1
    assert "idle" not in limiter._model_family_to_backend
    assert "active" in limiter._model_family_to_backend

    await limiter.refund_capacity({"tokens": 0}, active)


async def test_async_shared_family_cold_start_validation_is_linear():
    calls = 0

    def config_getter(_model: str) -> PerModelConfig:
        nonlocal calls
        calls += 1
        return _config(model_family="shared")

    limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())
    started = time.perf_counter()
    for index in range(1000):
        await limiter.record_usage({"tokens": 1}, model=f"alias-{index}")
    elapsed = time.perf_counter() - started

    assert calls == 1000
    assert elapsed < 5.0


def test_sync_caps_and_cleanup():
    limiter = SyncRateLimiter(
        lambda model: _config(model_family=model),
        backend=SyncMemoryBackendBuilder(),
        max_model_families=2,
        max_in_flight_reservations=2,
    )

    first = limiter.acquire_capacity({"tokens": 1}, model="family-0")
    second = limiter.acquire_capacity({"tokens": 1}, model="family-1")

    with pytest.raises(CardinalityLimitExceededError, match="max_model_families"):
        limiter.acquire_capacity({"tokens": 1}, model="family-2")
    with pytest.raises(CardinalityLimitExceededError, match="max_in_flight"):
        limiter.acquire_capacity({"tokens": 1}, model="family-0")

    limiter.refund_capacity({"tokens": 0}, first)
    assert limiter.clear_unused_model_families(0) == 1
    assert "family-0" not in limiter._model_family_to_backend
    assert "family-1" in limiter._model_family_to_backend
    limiter.refund_capacity({"tokens": 0}, second)


def test_sync_shared_family_cold_start_validation_is_linear():
    calls = 0

    def config_getter(_model: str) -> PerModelConfig:
        nonlocal calls
        calls += 1
        return _config(model_family="shared")

    limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())
    started = time.perf_counter()
    for index in range(1000):
        limiter.record_usage({"tokens": 1}, model=f"alias-{index}")
    elapsed = time.perf_counter() - started

    assert calls == 1000
    assert elapsed < 5.0
