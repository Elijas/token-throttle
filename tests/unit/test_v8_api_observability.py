"""v8 public API hardening and observability regression coverage."""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import warnings

import pytest

from token_throttle import CardinalityLimitExceededError
from token_throttle._exceptions import DuplicateRefundError, UnknownReservationError
from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
    create_logging_callbacks,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

MODEL = "gpt-test"
FAMILY = "test-family"


def _config(
    *,
    model_family: str = FAMILY,
    metric: str = "tokens",
    limit: float = 100.0,
    usage_counter=None,
) -> PerModelConfig:
    return PerModelConfig(
        model_family=model_family,
        usage_counter=usage_counter,
        quotas=UsageQuotas([Quota(metric=metric, limit=limit, per_seconds=60)]),
    )


async def test_async_constructor_rejects_backend_without_build_methods() -> None:
    with pytest.raises(TypeError, match=r"backend must be a BackendBuilder"):
        RateLimiter(_config(), backend=object())


def test_sync_constructor_rejects_backend_without_build_methods() -> None:
    with pytest.raises(TypeError, match=r"backend must be a BackendBuilder"):
        SyncRateLimiter(_config(), backend=object())


def test_redis_builders_reject_cross_mode_clients() -> None:
    redis = pytest.importorskip("redis", reason="redis package not installed")
    async_redis = pytest.importorskip(
        "redis.asyncio",
        reason="redis package not installed",
    )
    redis_backend = importlib.import_module(
        "token_throttle._limiter_backends._redis._backend"
    )
    redis_sync_backend = importlib.import_module(
        "token_throttle._limiter_backends._redis._sync_backend"
    )

    with pytest.raises(TypeError, match=r"expected redis\.asyncio\.Redis"):
        redis_backend.RedisBackendBuilder(object(), key_prefix="tenant")
    with pytest.raises(TypeError, match=r"for sync use redis\.Redis"):
        redis_backend.RedisBackendBuilder(redis.Redis(), key_prefix="tenant")
    with pytest.raises(TypeError, match=r"expected redis\.Redis"):
        redis_sync_backend.SyncRedisBackendBuilder(object(), key_prefix="tenant")
    with pytest.raises(TypeError, match=r"for async use redis\.asyncio\.Redis"):
        redis_sync_backend.SyncRedisBackendBuilder(
            async_redis.Redis(), key_prefix="tenant"
        )


async def test_async_close_from_callback_raises_runtime_error() -> None:
    seen: list[str] = []
    limiter: RateLimiter

    async def on_lifecycle_event(**_kwargs) -> None:
        with pytest.raises(
            RuntimeError,
            match=r"close\(\)/aclose\(\) cannot be called",
        ):
            await limiter.aclose()
        seen.append("blocked")

    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        callbacks=RateLimiterCallbacks(on_lifecycle_event=on_lifecycle_event),
        close_drain_timeout_seconds=0.01,
    )

    reservation = await limiter.acquire_capacity({"tokens": 1}, MODEL)
    await limiter.refund_capacity({"tokens": 0}, reservation)
    await limiter.aclose()

    assert seen


def test_sync_close_from_callback_raises_runtime_error() -> None:
    seen: list[str] = []
    limiter: SyncRateLimiter

    def on_lifecycle_event(**_kwargs) -> None:
        with pytest.raises(
            RuntimeError,
            match=r"close\(\)/aclose\(\) cannot be called",
        ):
            limiter.close()
        seen.append("blocked")

    limiter = SyncRateLimiter(
        _config(),
        backend=SyncMemoryBackendBuilder(),
        callbacks=SyncRateLimiterCallbacks(on_lifecycle_event=on_lifecycle_event),
        close_drain_timeout_seconds=0.01,
    )

    reservation = limiter.acquire_capacity({"tokens": 1}, MODEL)
    limiter.refund_capacity({"tokens": 0}, reservation)
    limiter.close()

    assert seen


async def test_mapping_usage_counter_result_is_accepted() -> None:
    def usage_counter(**_kwargs) -> dict[str, int]:
        return {"tokens": 1}

    limiter = RateLimiter(
        _config(usage_counter=usage_counter),
        backend=MemoryBackendBuilder(),
    )

    reservation = await limiter.acquire_capacity_for_request(model=MODEL)
    await limiter.refund_capacity({"tokens": 0}, reservation)

    assert reservation.usage == {"tokens": 1.0}


async def test_create_logging_callbacks_uses_stdlib_even_when_loguru_importable(
    caplog,
    monkeypatch,
) -> None:
    class _LoguruSentinel:
        def __getattr__(self, name: str):
            raise AssertionError(f"loguru should not be used: {name}")

    monkeypatch.setitem(sys.modules, "loguru", _LoguruSentinel())
    callbacks = create_logging_callbacks(wait_start="INFO")

    caplog.set_level(logging.INFO, logger="token_throttle")
    await callbacks.on_wait_start(
        model_family=FAMILY,
        model_alias=MODEL,
        request_id="req-1",
        reservation_id="res-1",
        usage={"tokens": 1.0},
        preconsumption_capacities={("tokens", 60): 99.0},
    )

    record = next(
        record for record in caplog.records if record.name == "token_throttle"
    )
    assert record.model_family == FAMILY
    assert record.model_alias == MODEL
    assert record.request_id == "req-1"
    assert record.reservation_id == "res-1"
    assert "model_family=" in record.getMessage()


@pytest.mark.parametrize("level", ["TRACE", "SUCCESS"])
def test_create_logging_callbacks_rejects_unknown_log_level(level: str) -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        create_logging_callbacks(wait_start=level)


async def test_callback_failure_log_includes_slot_and_reservation_context(
    caplog,
) -> None:
    async def on_lifecycle_event(**_kwargs) -> None:
        raise RuntimeError("callback boom")

    limiter = RateLimiter(
        _config(),
        backend=MemoryBackendBuilder(),
        callbacks=RateLimiterCallbacks(on_lifecycle_event=on_lifecycle_event),
    )

    caplog.set_level(logging.WARNING, logger="token_throttle")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reservation = await limiter.acquire_capacity({"tokens": 1}, MODEL)

    record = next(
        record for record in caplog.records if "callback boom" in record.getMessage()
    )
    assert record.callback_slot == "on_lifecycle_event"
    assert record.reservation_id == reservation.reservation_id
    assert record.model_family == FAMILY
    assert record.bucket_id == ("tokens", 60)


async def test_wait_callbacks_receive_request_context() -> None:
    captured: list[dict[str, object]] = []
    wait_started = asyncio.Event()

    async def on_wait_start(**kwargs) -> None:
        captured.append(kwargs)
        wait_started.set()

    limiter = RateLimiter(
        _config(limit=1, usage_counter=lambda **_kwargs: {"tokens": 1}),
        backend=MemoryBackendBuilder(sleep_interval=0.01),
        callbacks=RateLimiterCallbacks(on_wait_start=on_wait_start),
    )
    first = await limiter.acquire_capacity({"tokens": 1}, MODEL)
    second_task = asyncio.create_task(
        limiter.acquire_capacity_for_request(model=MODEL, request_id="req-1")
    )

    await asyncio.wait_for(wait_started.wait(), timeout=1)
    await limiter.refund_capacity({"tokens": 0}, first)
    second = await asyncio.wait_for(second_task, timeout=1)
    await limiter.refund_capacity({"tokens": 0}, second)

    payload = captured[0]
    assert payload["model_alias"] == MODEL
    assert payload["request_id"] == "req-1"
    assert isinstance(payload["reservation_id"], str)


async def test_refund_dropped_warning_also_logs_bucket_context(caplog) -> None:
    use_new_metric = False

    def config_getter(_model: str) -> PerModelConfig:
        return _config(metric="requests" if use_new_metric else "tokens")

    limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 1}, MODEL)
    use_new_metric = True
    await limiter.record_usage({"requests": 0}, MODEL)

    caplog.set_level(logging.WARNING, logger="token_throttle")
    with pytest.warns(RuntimeWarning, match="Refund dropped"):
        await limiter.refund_capacity({"tokens": 0}, reservation)

    record = next(
        record for record in caplog.records if "Refund dropped" in record.getMessage()
    )
    assert record.reservation_id == reservation.reservation_id
    assert record.model_family == FAMILY
    assert record.old_bucket_ids == [("tokens", 60)]
    assert record.active_bucket_ids == [("requests", 60)]


async def test_refund_refresh_failure_also_logs_reservation_context(caplog) -> None:
    should_fail = False

    def config_getter(_model: str) -> PerModelConfig:
        if should_fail:
            raise RuntimeError("config unavailable")
        return _config()

    limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 1}, MODEL)
    should_fail = True

    caplog.set_level(logging.WARNING, logger="token_throttle")
    with pytest.warns(RuntimeWarning, match="Failed to refresh backend during refund"):
        await limiter.refund_capacity({"tokens": 0}, reservation)

    record = next(
        record
        for record in caplog.records
        if "Failed to refresh backend during refund" in record.getMessage()
    )
    assert record.reservation_id == reservation.reservation_id
    assert record.model_family == FAMILY
    assert record.model_name == MODEL


async def test_metric_set_rebuild_warning_also_logs_context(caplog) -> None:
    use_new_metric = False

    def config_getter(_model: str) -> PerModelConfig:
        return _config(metric="requests" if use_new_metric else "tokens")

    limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())
    await limiter.record_usage({"tokens": 1}, MODEL)
    use_new_metric = True

    caplog.set_level(logging.WARNING, logger="token_throttle")
    with pytest.warns(UserWarning, match="changed metric set"):
        await limiter.record_usage({"requests": 1}, MODEL)

    record = next(
        record
        for record in caplog.records
        if "changed metric set" in record.getMessage()
    )
    assert record.model_family == FAMILY
    assert record.old_bucket_ids == [("tokens", 60)]
    assert record.active_bucket_ids == [("requests", 60)]


async def test_blocking_acquire_timeout_includes_bottleneck_context() -> None:
    limiter = RateLimiter(
        _config(limit=1),
        backend=MemoryBackendBuilder(sleep_interval=0.01),
    )

    await limiter.acquire_capacity({"tokens": 1}, MODEL)

    with pytest.raises(
        TimeoutError,
        match=(
            r"Timed out waiting for capacity "
            r"\(bottleneck=\('tokens', 60\), available=.*"
            r"requested=1\.0, computed_sleep="
        ),
    ):
        await limiter.acquire_capacity({"tokens": 1}, MODEL, timeout=0.01)


async def test_refund_exceptions_include_reservation_attributes() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 1}, MODEL)
    await limiter.refund_capacity({"tokens": 0}, reservation)

    with pytest.raises(DuplicateRefundError) as duplicate:
        await limiter.refund_capacity({"tokens": 0}, reservation)

    assert duplicate.value.reservation_id == reservation.reservation_id
    assert duplicate.value.model_family == FAMILY
    assert reservation.reservation_id in str(duplicate.value)
    assert FAMILY in str(duplicate.value)

    forged = CapacityReservation(
        reservation_id="foreign-reservation",
        usage={"tokens": 1},
        model_family=FAMILY,
        bucket_ids=frozenset({("tokens", 60)}),
        model=MODEL,
        limiter_instance_id="foreign-limiter",
    )
    with pytest.raises(UnknownReservationError) as unknown:
        await limiter.refund_capacity({"tokens": 0}, forged)

    assert unknown.value.reservation_id == "foreign-reservation"
    assert unknown.value.model_family == FAMILY
    assert "foreign-reservation" in str(unknown.value)
    assert FAMILY in str(unknown.value)


async def test_cardinality_cap_errors_include_offending_value_and_count() -> None:
    family_limiter = RateLimiter(
        lambda model: _config(model_family=f"family-{model}"),
        backend=MemoryBackendBuilder(),
        max_model_families=1,
    )
    await family_limiter.record_usage({"tokens": 1}, "a")

    with pytest.raises(CardinalityLimitExceededError) as family_error:
        await family_limiter.record_usage({"tokens": 1}, "b")
    assert "offending model_family='family-b'" in str(family_error.value)
    assert "current_count=1" in str(family_error.value)

    alias_limiter = RateLimiter(
        lambda _model: _config(model_family="shared-family"),
        backend=MemoryBackendBuilder(),
        max_aliases=1,
    )
    await alias_limiter.record_usage({"tokens": 1}, "a")

    with pytest.raises(CardinalityLimitExceededError) as alias_error:
        await alias_limiter.record_usage({"tokens": 1}, "b")
    assert "offending alias='b'" in str(alias_error.value)
    assert "current_count=1" in str(alias_error.value)
