"""Bug 1: Redis await_for_capacity/wait_for_capacity missing float coercion.

Commit a2074c6 added usage coercion to memory backends' wait methods but
missed the Redis backends.  Without coercion at the top of the wait method,
callbacks receive raw (int) values instead of floats, and _check_and_consume
re-coerces on every loop iteration instead of once upfront.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from frozendict import frozendict

from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, SecondsIn, UsageQuotas

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._limiter_backends._redis._backend import RedisBackend
from token_throttle._limiter_backends._redis._sync_backend import SyncRedisBackend


def _make_config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [Quota(metric="tokens", limit=100, per_seconds=SecondsIn.MINUTE)]
        ),
        model_family="test-family",
    )


def _make_async_backend(
    callbacks: RateLimiterCallbacks,
) -> RedisBackend:
    cfg = _make_config()
    backend = object.__new__(RedisBackend)
    backend._callbacks = callbacks
    backend._limit_config = cfg
    backend._usage_metric_names = {"tokens"}
    backend._sleep_interval = 0.01
    backend.MAX_CROSS_WORKER_POLL = 1.0
    backend._local_condition = asyncio.Condition()
    return backend


def _make_sync_backend(
    callbacks: SyncRateLimiterCallbacks,
) -> SyncRedisBackend:
    cfg = _make_config()
    backend = object.__new__(SyncRedisBackend)
    backend._callbacks = callbacks
    backend._limit_config = cfg
    backend._usage_metric_names = {"tokens"}
    backend._sleep_interval = 0.01
    backend.MAX_CROSS_WORKER_POLL = 1.0
    backend._local_condition = threading.Condition()
    backend.sorted_buckets = []
    return backend


def _make_async_fake_check():
    """Return an async callable that fails on first call, succeeds on second."""
    call_count = 0

    async def _fn(_usage, **_kw):
        nonlocal call_count
        call_count += 1
        caps = frozendict({("tokens", 60): 50.0})
        if call_count == 1:
            return False, caps, frozendict(), None
        return True, caps, frozendict({("tokens", 60): 0.0}), 0.0

    return _fn


def _make_sync_fake_check():
    """Return a sync callable that fails on first call, succeeds on second."""
    call_count = 0

    def _fn(_usage, **_kw):
        nonlocal call_count
        call_count += 1
        caps = frozendict({("tokens", 60): 50.0})
        if call_count == 1:
            return False, caps, frozendict(), None
        return True, caps, frozendict({("tokens", 60): 0.0}), 0.0

    return _fn


# ---------------------------------------------------------------------------
# Async Redis backend
# ---------------------------------------------------------------------------


class TestAsyncRedisWaitCoercion:
    """await_for_capacity must coerce usage values to float before entering
    the wait loop, so callbacks see float and _check_and_consume_capacity
    doesn't re-coerce on every iteration.
    """

    async def test_on_wait_start_receives_float_usage(self):
        """on_wait_start callback must receive float-typed usage values."""
        cbs = RateLimiterCallbacks(on_wait_start=AsyncMock())
        backend = _make_async_backend(cbs)
        backend._check_and_consume_capacity = _make_async_fake_check()
        backend._compute_sleep = lambda _u, _p: 0.001

        # Pass int values — should be coerced to float before callback
        await backend.await_for_capacity(frozendict({"tokens": 50}))

        cbs.on_wait_start.assert_awaited_once()
        received_usage = cbs.on_wait_start.call_args.kwargs["usage"]
        assert isinstance(received_usage["tokens"], float), (
            f"on_wait_start received {type(received_usage['tokens']).__name__}, expected float"
        )

    async def test_after_wait_end_receives_float_usage(self):
        """after_wait_end_consumption callback must receive float-typed usage values."""
        cbs = RateLimiterCallbacks(after_wait_end_consumption=AsyncMock())
        backend = _make_async_backend(cbs)
        backend._check_and_consume_capacity = _make_async_fake_check()
        backend._compute_sleep = lambda _u, _p: 0.001

        await backend.await_for_capacity(frozendict({"tokens": 50}))

        cbs.after_wait_end_consumption.assert_awaited_once()
        received_usage = cbs.after_wait_end_consumption.call_args.kwargs["usage"]
        assert isinstance(received_usage["tokens"], float), (
            f"after_wait_end_consumption received {type(received_usage['tokens']).__name__}, expected float"
        )


# ---------------------------------------------------------------------------
# Sync Redis backend
# ---------------------------------------------------------------------------


class TestSyncRedisWaitCoercion:
    """wait_for_capacity must coerce usage values to float before entering
    the wait loop.
    """

    def test_on_wait_start_receives_float_usage(self):
        """on_wait_start callback must receive float-typed usage values."""
        cbs = SyncRateLimiterCallbacks(on_wait_start=MagicMock())
        backend = _make_sync_backend(cbs)
        backend._check_and_consume_capacity = _make_sync_fake_check()
        backend._compute_sleep = lambda _u, _p: 0.001

        backend.wait_for_capacity(frozendict({"tokens": 50}))

        cbs.on_wait_start.assert_called_once()
        received_usage = cbs.on_wait_start.call_args.kwargs["usage"]
        assert isinstance(received_usage["tokens"], float), (
            f"on_wait_start received {type(received_usage['tokens']).__name__}, expected float"
        )

    def test_after_wait_end_receives_float_usage(self):
        """after_wait_end_consumption callback must receive float-typed usage values."""
        cbs = SyncRateLimiterCallbacks(after_wait_end_consumption=MagicMock())
        backend = _make_sync_backend(cbs)
        backend._check_and_consume_capacity = _make_sync_fake_check()
        backend._compute_sleep = lambda _u, _p: 0.001

        backend.wait_for_capacity(frozendict({"tokens": 50}))

        cbs.after_wait_end_consumption.assert_called_once()
        received_usage = cbs.after_wait_end_consumption.call_args.kwargs["usage"]
        assert isinstance(received_usage["tokens"], float), (
            f"after_wait_end_consumption received {type(received_usage['tokens']).__name__}, expected float"
        )
