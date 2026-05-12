"""Regression tests for FIX-07 bundle-locktrx."""

import asyncio
import concurrent.futures
import contextlib
import contextvars
import threading

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

MODEL = "test-model"
MODEL_FAMILY = "test-family"
BUCKET_ID = ("tokens", 60)


def _config(*, tokens_limit: float = 100.0, metrics: tuple[str, ...] = ("tokens",)):
    return PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric=metric, limit=tokens_limit, per_seconds=60)
                for metric in metrics
            ]
        ),
        model_family=MODEL_FAMILY,
    )


def _two_metric_config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric="tokens", limit=100.0, per_seconds=60),
                Quota(metric="requests", limit=100.0, per_seconds=60),
            ]
        ),
        model_family=MODEL_FAMILY,
    )


def _bucket_max_capacity(limiter, bucket_id=BUCKET_ID) -> float:
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    return backend._bucket_registry[bucket_id].max_capacity


def _runtime_override(limiter, bucket_id=BUCKET_ID) -> float | None:
    return limiter._model_family_to_runtime_max_capacity.get(MODEL_FAMILY, {}).get(
        bucket_id
    )


class TestRefundLockTransaction:
    async def test_async_post_write_failure_keeps_refund_deduped(self):
        limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
        reservation = await limiter.acquire_capacity({"tokens": 30}, MODEL)
        backend = limiter._model_family_to_backend[MODEL_FAMILY]
        original_refund = backend.refund_capacity_for_buckets
        calls = 0

        async def write_then_fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            await original_refund(*args, **kwargs)
            raise RuntimeError("simulated post-write failure")

        backend.refund_capacity_for_buckets = write_then_fail

        with pytest.raises(RuntimeError, match="simulated post-write failure"):
            await limiter.refund_capacity({"tokens": 10}, reservation)

        with pytest.warns(UserWarning, match="has already been refunded"):
            await limiter.refund_capacity({"tokens": 10}, reservation)

        assert calls == 1

    async def test_async_cancel_after_backend_write_keeps_refund_deduped(self):
        limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
        reservation = await limiter.acquire_capacity({"tokens": 30}, MODEL)
        backend = limiter._model_family_to_backend[MODEL_FAMILY]
        original_refund = backend.refund_capacity_for_buckets
        write_done = asyncio.Event()
        keep_open = asyncio.Event()
        calls = 0

        async def write_then_pause(*args, **kwargs):
            nonlocal calls
            calls += 1
            await original_refund(*args, **kwargs)
            write_done.set()
            await keep_open.wait()

        backend.refund_capacity_for_buckets = write_then_pause

        task = asyncio.create_task(limiter.refund_capacity({"tokens": 10}, reservation))
        await asyncio.wait_for(write_done.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        with pytest.warns(UserWarning, match="has already been refunded"):
            await limiter.refund_capacity({"tokens": 10}, reservation)

        assert calls == 1

    async def test_async_empty_projection_keeps_refund_deduped(self):
        current_config = _two_metric_config()

        def config_getter(_model: str) -> PerModelConfig:
            return current_config

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())
        reservation = await limiter.acquire_capacity(
            {"tokens": 30, "requests": 3}, MODEL
        )

        current_config = _config(metrics=("compute",))
        with pytest.warns(RuntimeWarning, match="Refund dropped"):
            await limiter.refund_capacity({"tokens": 10, "requests": 1}, reservation)
        assert reservation.reservation_id in limiter._refunded_reservation_ids

        current_config = _two_metric_config()
        with pytest.warns(UserWarning, match="has already been refunded"):
            await limiter.refund_capacity({"tokens": 10, "requests": 1}, reservation)

    def test_sync_post_write_failure_keeps_refund_deduped(self):
        limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
        reservation = limiter.acquire_capacity({"tokens": 30}, MODEL)
        backend = limiter._model_family_to_backend[MODEL_FAMILY]
        original_refund = backend.refund_capacity_for_buckets
        calls = 0

        def write_then_fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            original_refund(*args, **kwargs)
            raise RuntimeError("simulated post-write failure")

        backend.refund_capacity_for_buckets = write_then_fail

        with pytest.raises(RuntimeError, match="simulated post-write failure"):
            limiter.refund_capacity({"tokens": 10}, reservation)

        with pytest.warns(UserWarning, match="has already been refunded"):
            limiter.refund_capacity({"tokens": 10}, reservation)

        assert calls == 1


class TestSetMaxCapacityLockTransaction:
    async def test_async_concurrent_set_and_rebuild_keep_override_consistent(self):
        old_config = _config(tokens_limit=100.0)
        new_config = _config(tokens_limit=200.0)
        config_var = contextvars.ContextVar("config", default=old_config)

        def config_getter(_model: str) -> PerModelConfig:
            return config_var.get()

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())
        await limiter.acquire_capacity({"tokens": 1}, MODEL)
        backend = limiter._model_family_to_backend[MODEL_FAMILY]

        apply_entered = asyncio.Event()
        override_remembered = asyncio.Event()
        original_apply = backend.apply_configured_max_capacity
        original_remember = limiter._remember_runtime_max_capacity

        async def controlled_apply(*args, **kwargs):
            await original_apply(*args, **kwargs)
            apply_entered.set()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(override_remembered.wait(), timeout=0.05)

        def remember_and_signal(*args, **kwargs):
            original_remember(*args, **kwargs)
            override_remembered.set()

        backend.apply_configured_max_capacity = controlled_apply
        limiter._remember_runtime_max_capacity = remember_and_signal

        async def actor_rebuild():
            token = config_var.set(new_config)
            try:
                await limiter.record_usage({"tokens": 1}, MODEL)
            finally:
                config_var.reset(token)

        async def actor_set():
            await asyncio.wait_for(apply_entered.wait(), timeout=1)
            token = config_var.set(old_config)
            try:
                await limiter.set_max_capacity(MODEL, "tokens", 60, 50.0)
            finally:
                config_var.reset(token)

        await asyncio.gather(actor_rebuild(), actor_set())

        bucket_value = _bucket_max_capacity(limiter)
        override_value = _runtime_override(limiter)
        assert (bucket_value == 50.0) == (override_value == 50.0)

    def test_sync_concurrent_set_and_rebuild_keep_override_consistent(self):
        old_config = _config(tokens_limit=100.0)
        new_config = _config(tokens_limit=200.0)
        thread_config = threading.local()

        def config_getter(_model: str) -> PerModelConfig:
            return getattr(thread_config, "config", old_config)

        limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())
        thread_config.config = old_config
        limiter.acquire_capacity({"tokens": 1}, MODEL)
        backend = limiter._model_family_to_backend[MODEL_FAMILY]

        apply_entered = threading.Event()
        override_remembered = threading.Event()
        original_apply = backend.apply_configured_max_capacity
        original_remember = limiter._remember_runtime_max_capacity

        def controlled_apply(*args, **kwargs):
            original_apply(*args, **kwargs)
            apply_entered.set()
            override_remembered.wait(timeout=0.05)

        def remember_and_signal(*args, **kwargs):
            original_remember(*args, **kwargs)
            override_remembered.set()

        backend.apply_configured_max_capacity = controlled_apply
        limiter._remember_runtime_max_capacity = remember_and_signal

        def actor_rebuild():
            thread_config.config = new_config
            limiter.record_usage({"tokens": 1}, MODEL)

        def actor_set():
            assert apply_entered.wait(timeout=1)
            thread_config.config = old_config
            limiter.set_max_capacity(MODEL, "tokens", 60, 50.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(actor_rebuild),
                executor.submit(actor_set),
            ]
            for future in futures:
                future.result(timeout=5)

        bucket_value = _bucket_max_capacity(limiter)
        override_value = _runtime_override(limiter)
        assert (bucket_value == 50.0) == (override_value == 50.0)
