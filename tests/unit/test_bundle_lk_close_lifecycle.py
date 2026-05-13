"""Regression tests for FIX-32 close/acquire lifecycle hardening."""

import asyncio
import importlib
import threading
from unittest.mock import AsyncMock, Mock, patch

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import CapacityReservation, Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

MODEL = "test-model"
MODEL_FAMILY = "test-family"


def _config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=60)]),
        model_family=MODEL_FAMILY,
    )


class TestAsyncCloseLifecycle:
    async def test_close_waits_for_pending_acquire_before_returning(self):
        limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
        finalize_entered = asyncio.Event()
        release_finalize = asyncio.Event()
        original_finalize = limiter._finalize_pending_acquire

        async def controlled_finalize(
            reservation: CapacityReservation,
            model: str,
        ) -> None:
            finalize_entered.set()
            await release_finalize.wait()
            await original_finalize(reservation, model)

        limiter._finalize_pending_acquire = controlled_finalize

        acquire_task = asyncio.create_task(
            limiter.acquire_capacity({"tokens": 10}, MODEL)
        )
        await asyncio.wait_for(finalize_entered.wait(), timeout=1.0)

        close_task = asyncio.create_task(limiter.aclose())
        await asyncio.sleep(0)
        assert not close_task.done()

        with pytest.raises(RuntimeError, match="closed"):
            await limiter.acquire_capacity({"tokens": 1}, MODEL)

        release_finalize.set()
        reservation = await asyncio.wait_for(acquire_task, timeout=1.0)
        if not close_task.done():
            await limiter.refund_capacity({"tokens": 0}, reservation)
        await asyncio.wait_for(close_task, timeout=1.0)

    async def test_close_after_cancelled_acquire_drains_immediately(self):
        limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
        backend = await limiter._get_backend(_config())
        backend_entered = asyncio.Event()
        release_backend = asyncio.Event()

        async def blocked_backend(*_args, **_kwargs) -> None:
            backend_entered.set()
            await release_backend.wait()

        backend.await_for_capacity = blocked_backend
        acquire_task = asyncio.create_task(
            limiter.acquire_capacity({"tokens": 10}, MODEL)
        )
        await asyncio.wait_for(backend_entered.wait(), timeout=1.0)
        assert len(limiter._pending_acquire_reservations) == 1

        acquire_task.cancel()
        release_backend.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(acquire_task, timeout=1.0)
        assert limiter._pending_acquire_reservations == set()

        await asyncio.wait_for(limiter.aclose(), timeout=1.0)

    async def test_close_pending_drain_timeout_raises(self):
        limiter = RateLimiter(
            _config(),
            backend=MemoryBackendBuilder(),
            close_drain_timeout_seconds=0.01,
        )
        limiter._pending_acquire_reservations.add("stuck")
        limiter._pending_drained.clear()

        with pytest.raises(TimeoutError, match="pending acquire reservations"):
            await limiter.aclose()


class TestSyncCloseLifecycle:
    def test_close_waits_for_pending_acquire_before_returning(self):
        limiter = SyncRateLimiter(
            _config(),
            backend=SyncMemoryBackendBuilder(),
            close_drain_timeout_seconds=1.0,
        )
        backend = limiter._get_backend(_config())
        original_wait = backend.wait_for_capacity
        backend_entered = threading.Event()
        release_backend = threading.Event()
        errors: list[BaseException] = []

        def controlled_wait(*args, **kwargs) -> None:
            original_wait(*args, **kwargs)
            backend_entered.set()
            release_backend.wait(timeout=1.0)

        backend.wait_for_capacity = controlled_wait

        def acquire() -> None:
            try:
                limiter.acquire_capacity({"tokens": 10}, MODEL)
            except BaseException as exc:
                errors.append(exc)

        acquire_thread = threading.Thread(target=acquire)
        acquire_thread.start()
        assert backend_entered.wait(timeout=1.0)

        close_thread = threading.Thread(target=limiter.close)
        close_thread.start()
        close_thread.join(timeout=0.05)
        assert close_thread.is_alive()

        with pytest.raises(RuntimeError, match="closed"):
            limiter.acquire_capacity({"tokens": 1}, MODEL)

        release_backend.set()
        acquire_thread.join(timeout=1.0)
        close_thread.join(timeout=1.0)
        assert not acquire_thread.is_alive()
        assert not close_thread.is_alive()
        assert errors == []


class TestRedisClientOwnership:
    async def test_async_redis_builder_closes_owned_client(self):
        redis_async = pytest.importorskip("redis.asyncio")
        redis_backend = importlib.import_module(
            "token_throttle._limiter_backends._redis._backend"
        )

        client = redis_async.Redis()
        with patch.object(client, "aclose", new=AsyncMock()) as aclose:
            builder = redis_backend.RedisBackendBuilder(
                client,
                key_prefix="test",
                owns_redis_client=True,
            )
            await builder.aclose()

        aclose.assert_awaited_once_with()

    async def test_async_redis_builder_leaves_borrowed_client_open(self):
        redis_async = pytest.importorskip("redis.asyncio")
        redis_backend = importlib.import_module(
            "token_throttle._limiter_backends._redis._backend"
        )

        client = redis_async.Redis()
        with patch.object(client, "aclose", new=AsyncMock()) as aclose:
            builder = redis_backend.RedisBackendBuilder(
                client,
                key_prefix="test",
                owns_redis_client=False,
            )
            await builder.aclose()

        aclose.assert_not_awaited()

    def test_sync_redis_builder_closes_owned_client(self):
        redis = pytest.importorskip("redis")
        redis_backend = importlib.import_module(
            "token_throttle._limiter_backends._redis._sync_backend"
        )

        client = redis.Redis()
        with patch.object(client, "close", new=Mock()) as close:
            builder = redis_backend.SyncRedisBackendBuilder(
                client,
                key_prefix="test",
                owns_redis_client=True,
            )
            builder.close()

        close.assert_called_once_with()

    def test_sync_redis_builder_leaves_borrowed_client_open(self):
        redis = pytest.importorskip("redis")
        redis_backend = importlib.import_module(
            "token_throttle._limiter_backends._redis._sync_backend"
        )

        client = redis.Redis()
        with patch.object(client, "close", new=Mock()) as close:
            builder = redis_backend.SyncRedisBackendBuilder(
                client,
                key_prefix="test",
                owns_redis_client=False,
            )
            builder.close()

        close.assert_not_called()
