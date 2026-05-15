"""Regression tests for FIX-35 PERF-REFUND-GUARD-NARROW."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from token_throttle._exceptions import DuplicateRefundError
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


def _config(*, metrics: tuple[str, ...] = ("tokens",)) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=100.0, per_seconds=60) for metric in metrics]
        ),
        model_family=MODEL_FAMILY,
    )


def _two_metric_config() -> PerModelConfig:
    return _config(metrics=("tokens", "requests"))


class TestRefundCorrectnessStillMatchesFix07:
    async def test_async_post_write_failure_keeps_refund_deduped(self) -> None:
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

        with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
            await limiter.refund_capacity({"tokens": 10}, reservation)

        assert calls == 1
        assert limiter._refund_locks == {}

    async def test_async_cancel_after_backend_write_keeps_refund_deduped(self) -> None:
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

        with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
            await limiter.refund_capacity({"tokens": 10}, reservation)

        assert calls == 1
        assert limiter._refund_locks == {}

    async def test_async_empty_projection_keeps_refund_deduped(self) -> None:
        current_config = _two_metric_config()

        def config_getter(_model: str) -> PerModelConfig:
            return current_config

        limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())
        reservation = await limiter.acquire_capacity(
            {"tokens": 30, "requests": 3},
            MODEL,
        )

        current_config = _config(metrics=("compute",))
        with pytest.warns(RuntimeWarning, match="Refund dropped"):
            await limiter.refund_capacity({"tokens": 10, "requests": 1}, reservation)
        assert reservation.reservation_id in limiter._refunded_reservation_ids

        current_config = _two_metric_config()
        with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
            await limiter.refund_capacity({"tokens": 10, "requests": 1}, reservation)

        assert limiter._refund_locks == {}

    def test_sync_post_write_failure_keeps_refund_deduped(self) -> None:
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

        with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
            limiter.refund_capacity({"tokens": 10}, reservation)

        assert calls == 1
        assert limiter._refund_locks == {}


class TestPerReservationRefundLocks:
    async def test_async_different_reservations_overlap_backend_refund(self) -> None:
        limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
        first = await limiter.acquire_capacity({"tokens": 10}, MODEL)
        second = await limiter.acquire_capacity({"tokens": 10}, MODEL)
        backend = limiter._model_family_to_backend[MODEL_FAMILY]
        original_refund = backend.refund_capacity_for_buckets
        entered: set[str] = set()
        all_entered = asyncio.Event()
        release = asyncio.Event()

        async def controlled_refund(*args, reservation_id=None, **kwargs):
            entered.add(reservation_id)
            if len(entered) == 2:
                all_entered.set()
            await release.wait()
            return await original_refund(
                *args,
                reservation_id=reservation_id,
                **kwargs,
            )

        backend.refund_capacity_for_buckets = controlled_refund

        tasks = [
            asyncio.create_task(limiter.refund_capacity({"tokens": 0}, first)),
            asyncio.create_task(limiter.refund_capacity({"tokens": 0}, second)),
        ]
        try:
            await asyncio.wait_for(all_entered.wait(), timeout=0.5)
        finally:
            release.set()
            await asyncio.gather(*tasks)

        assert entered == {first.reservation_id, second.reservation_id}
        assert limiter._refund_locks == {}

    async def test_async_same_reservation_does_not_double_issue_refund(self) -> None:
        limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
        reservation = await limiter.acquire_capacity({"tokens": 10}, MODEL)
        backend = limiter._model_family_to_backend[MODEL_FAMILY]
        original_refund = backend.refund_capacity_for_buckets
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def controlled_refund(*args, **kwargs):
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return await original_refund(*args, **kwargs)

        backend.refund_capacity_for_buckets = controlled_refund

        first_task = asyncio.create_task(
            limiter.refund_capacity({"tokens": 0}, reservation)
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        second_task = asyncio.create_task(
            limiter.refund_capacity({"tokens": 0}, reservation)
        )
        await asyncio.sleep(0.05)
        assert calls == 1

        async def release_and_collect_refunds() -> None:
            release.set()
            results = await asyncio.gather(
                first_task, second_task, return_exceptions=True
            )
            assert (
                sum(isinstance(result, DuplicateRefundError) for result in results) == 1
            )

        await release_and_collect_refunds()

        assert calls == 1
        assert limiter._refund_locks == {}

    def test_sync_different_reservations_overlap_backend_refund(self) -> None:
        limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
        first = limiter.acquire_capacity({"tokens": 10}, MODEL)
        second = limiter.acquire_capacity({"tokens": 10}, MODEL)
        backend = limiter._model_family_to_backend[MODEL_FAMILY]
        original_refund = backend.refund_capacity_for_buckets
        entered: set[str] = set()
        entered_lock = threading.Lock()
        all_entered = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []

        def controlled_refund(*args, reservation_id=None, **kwargs):
            with entered_lock:
                entered.add(reservation_id)
                if len(entered) == 2:
                    all_entered.set()
            assert release.wait(timeout=5)
            return original_refund(*args, reservation_id=reservation_id, **kwargs)

        def refund_once(reservation) -> None:
            try:
                limiter.refund_capacity({"tokens": 0}, reservation)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        backend.refund_capacity_for_buckets = controlled_refund
        threads = [
            threading.Thread(target=refund_once, args=(first,)),
            threading.Thread(target=refund_once, args=(second,)),
        ]
        try:
            for thread in threads:
                thread.start()
            assert all_entered.wait(timeout=0.5)
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=5)

        assert errors == []
        assert entered == {first.reservation_id, second.reservation_id}
        assert limiter._refund_locks == {}

    def test_sync_same_reservation_does_not_double_issue_refund(self) -> None:
        limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
        reservation = limiter.acquire_capacity({"tokens": 10}, MODEL)
        backend = limiter._model_family_to_backend[MODEL_FAMILY]
        original_refund = backend.refund_capacity_for_buckets
        entered = threading.Event()
        release = threading.Event()
        calls = 0
        calls_lock = threading.Lock()
        errors: list[BaseException] = []

        def controlled_refund(*args, **kwargs):
            nonlocal calls
            with calls_lock:
                calls += 1
            entered.set()
            assert release.wait(timeout=5)
            return original_refund(*args, **kwargs)

        def first_refund() -> None:
            try:
                limiter.refund_capacity({"tokens": 0}, reservation)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        def duplicate_refund() -> None:
            try:
                with pytest.raises(DuplicateRefundError):
                    limiter.refund_capacity({"tokens": 0}, reservation)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        backend.refund_capacity_for_buckets = controlled_refund
        first_thread = threading.Thread(target=first_refund)
        duplicate_thread = threading.Thread(target=duplicate_refund)

        first_thread.start()
        assert entered.wait(timeout=1)
        duplicate_thread.start()
        time.sleep(0.05)
        with calls_lock:
            assert calls == 1

        release.set()
        first_thread.join(timeout=5)
        duplicate_thread.join(timeout=5)

        assert errors == []
        with calls_lock:
            assert calls == 1
        assert limiter._refund_locks == {}
