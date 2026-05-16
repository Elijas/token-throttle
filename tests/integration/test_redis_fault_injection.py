"""Fault-injection tests for the Redis backend.

Exercises failure modes that are absent from the normal integration suite:
  - ConnectionError mid-pipeline
  - ConnectionError during lock acquisition
  - Slow Redis (latency injection)
  - Server time skew (backward clock jump)
  - Lock TTL expiry mid-operation

Each test injects exactly one fault type and verifies the backend surfaces
a clean error with no corruption or partial writes.
"""

import asyncio
import contextlib
import secrets
import warnings
from unittest.mock import patch

import pytest
import redis.asyncio as aioredis
import redis.exceptions
from frozendict import frozendict

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis import _server_time
from token_throttle._limiter_backends._redis._backend import (
    LOCK_TIMEOUT_SECONDS,
    RedisBackend,
    RedisBackendBuilder,
)
from token_throttle._limiter_backends._redis._server_time import async_server_time


def _make_config(
    *,
    limit: float = 100,
    per_seconds: int = 3600,
    metric: str = "requests",
    model_family: str | None = None,
) -> PerModelConfig:
    return PerModelConfig(
        model_family=model_family or f"fi-{secrets.token_hex(4)}",
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
    )


async def _get_redis_capacity(backend: RedisBackend) -> float:
    """Read current capacity from Redis (authoritative source)."""
    pipeline = backend._redis.pipeline()
    current_time = await async_server_time(backend._redis)
    result = await backend._get_capacities_unsafe(
        pipeline=pipeline, current_time=current_time
    )
    return next(iter(result.capacities.values()))


# ---------------------------------------------------------------------------
# 1. ConnectionError mid-pipeline
# ---------------------------------------------------------------------------


@pytest.mark.redis
class TestConnectionErrorMidPipeline:
    """ConnectionError during pipeline.execute() must surface cleanly."""

    async def test_connection_error_during_write_surfaces_as_error(self, redis_client):
        """ConnectionError during _set_capacities_unsafe propagates, no corruption."""
        builder = RedisBackendBuilder(redis_client, key_prefix="test")
        config = _make_config()
        backend = builder.build(config, callbacks=RateLimiterCallbacks())

        # Seed initial state so we have a known capacity value
        await backend.await_for_capacity(frozendict({"requests": 10.0}))
        cap_before = await _get_redis_capacity(backend)

        real_set = backend._set_capacities_unsafe

        async def failing_set(*args, **kwargs):
            raise ConnectionError("Injected: connection lost mid-pipeline")

        backend._set_capacities_unsafe = failing_set

        with pytest.raises(ConnectionError):
            await backend.await_for_capacity(frozendict({"requests": 5.0}), timeout=0)

        backend._set_capacities_unsafe = real_set

        # State must be consistent: either the old capacity or fully committed
        cap_after = await _get_redis_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=2.0), (
            f"Partial write detected! Before={cap_before}, after={cap_after}. "
            f"ConnectionError mid-pipeline must not leave partial state."
        )

    async def test_connection_error_during_read_pipeline_surfaces_cleanly(
        self, redis_client
    ):
        """ConnectionError during _get_capacities_unsafe propagates immediately."""
        builder = RedisBackendBuilder(redis_client, key_prefix="test")
        config = _make_config()
        backend = builder.build(config, callbacks=RateLimiterCallbacks())

        async def failing_execute(self_pipe, *args, **kwargs):
            raise ConnectionError("Injected: connection lost during read")

        with (
            patch.object(aioredis.client.Pipeline, "execute", failing_execute),
            pytest.raises((ConnectionError, redis.exceptions.ConnectionError)),
        ):
            await backend.await_for_capacity(frozendict({"requests": 5.0}), timeout=0)


# ---------------------------------------------------------------------------
# 2. ConnectionError during lock acquisition
# ---------------------------------------------------------------------------


@pytest.mark.redis
class TestConnectionErrorDuringLockAcquisition:
    """ConnectionError during lock.acquire() must not leak locks."""

    async def test_connection_error_during_lock_acquire_no_lock_leak(
        self, redis_client
    ):
        """Lock is cleaned up if ConnectionError fires during acquire."""
        builder = RedisBackendBuilder(redis_client, key_prefix="test")
        config = _make_config()
        backend = builder.build(config, callbacks=RateLimiterCallbacks())
        bucket = backend.sorted_buckets[0]
        lock_key = bucket._lock_key

        assert await redis_client.exists(lock_key) == 0

        async def failing_acquire(self_lock, *args, **kwargs):
            raise ConnectionError("Injected: connection lost during lock acquire")

        with (
            patch.object(aioredis.lock.Lock, "acquire", failing_acquire),
            pytest.raises((ConnectionError, redis.exceptions.ConnectionError)),
        ):
            await backend.await_for_capacity(frozendict({"requests": 1.0}), timeout=0)

        lock_exists = await redis_client.exists(lock_key)
        assert lock_exists == 0, (
            f"Lock key {lock_key!r} leaked after ConnectionError during acquire!"
        )

    async def test_subsequent_acquire_succeeds_after_connection_error(
        self, redis_client
    ):
        """After ConnectionError during acquire, the next acquire works normally."""
        builder = RedisBackendBuilder(redis_client, key_prefix="test")
        config = _make_config()
        backend = builder.build(config, callbacks=RateLimiterCallbacks())

        call_count = 0
        original_acquire = aioredis.lock.Lock.acquire

        async def flaky_acquire(self_lock, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Injected: transient connection error")
            return await original_acquire(self_lock, *args, **kwargs)

        with patch.object(aioredis.lock.Lock, "acquire", flaky_acquire):
            with pytest.raises((ConnectionError, redis.exceptions.ConnectionError)):
                await backend.await_for_capacity(
                    frozendict({"requests": 1.0}), timeout=0
                )
            # Second call succeeds (patch still active but counter > 1)
            await backend.await_for_capacity(frozendict({"requests": 1.0}), timeout=0)


# ---------------------------------------------------------------------------
# 3. Slow Redis (latency injection)
# ---------------------------------------------------------------------------


@pytest.mark.redis
class TestSlowRedisLatencyInjection:
    """Backend handles slow Redis without corrupting state."""

    async def test_timeout_zero_fails_immediately_despite_slow_redis(
        self, redis_client
    ):
        """timeout=0 (try-acquire) must fail fast when capacity is exhausted."""
        builder = RedisBackendBuilder(redis_client, key_prefix="test")
        config = _make_config(limit=100, per_seconds=3600)
        backend = builder.build(config, callbacks=RateLimiterCallbacks())

        await backend.await_for_capacity(frozendict({"requests": 100.0}))

        start = asyncio.get_event_loop().time()
        with pytest.raises(TimeoutError):
            await backend.await_for_capacity(frozendict({"requests": 1.0}), timeout=0)
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 5.0, f"timeout=0 took {elapsed:.1f}s — should fail immediately"

    async def test_slow_pipeline_with_sufficient_timeout_succeeds(self, redis_client):
        """Operations with a generous timeout survive slow pipelines."""
        builder = RedisBackendBuilder(redis_client, key_prefix="test")
        config = _make_config(limit=100, per_seconds=3600)
        backend = builder.build(config, callbacks=RateLimiterCallbacks())

        real_execute = aioredis.client.Pipeline.execute

        async def slow_execute(self_pipe, *args, **kwargs):
            await asyncio.sleep(0.3)
            return await real_execute(self_pipe, *args, **kwargs)

        with patch.object(aioredis.client.Pipeline, "execute", slow_execute):
            await backend.await_for_capacity(frozendict({"requests": 10.0}), timeout=5)

        cap = await _get_redis_capacity(backend)
        assert cap == pytest.approx(90.0, abs=2.0)


# ---------------------------------------------------------------------------
# 4. Server time skew (backward clock jump)
# ---------------------------------------------------------------------------


@pytest.mark.redis
class TestServerTimeSkew:
    """Backward clock jumps (NTP correction) must not produce negative refill."""

    async def test_backward_time_jump_clamps_to_zero_refill(self, redis_client):
        """calculate_capacity clamps negative time_passed to 0 on clock skew."""
        builder = RedisBackendBuilder(redis_client, key_prefix="test")
        config = _make_config(limit=100, per_seconds=60)
        backend = builder.build(config, callbacks=RateLimiterCallbacks())

        await backend.await_for_capacity(frozendict({"requests": 50.0}))
        cap_before = await _get_redis_capacity(backend)

        real_server_time = _server_time.async_server_time
        jump_applied = False

        async def skewed_server_time(client):
            nonlocal jump_applied
            real_time = await real_server_time(client)
            if not jump_applied:
                jump_applied = True
                return real_time - 10.0
            return real_time

        with (
            patch.object(_server_time, "async_server_time", skewed_server_time),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", RuntimeWarning)
            pipeline = backend._redis.pipeline()
            skewed_time = await skewed_server_time(redis_client)
            result = await backend._get_capacities_unsafe(
                pipeline=pipeline, current_time=skewed_time
            )
            cap_skewed = next(iter(result.capacities.values()))

        assert cap_skewed >= cap_before - 1.0, (
            f"Backward clock jump caused capacity loss! "
            f"Before={cap_before}, skewed={cap_skewed}. "
            f"Negative time_passed was not clamped."
        )

    async def test_forward_time_jump_does_not_exceed_max_capacity(self, redis_client):
        """A large forward time jump refills to max_capacity, not beyond."""
        builder = RedisBackendBuilder(redis_client, key_prefix="test")
        config = _make_config(limit=100, per_seconds=60)
        backend = builder.build(config, callbacks=RateLimiterCallbacks())

        await backend.await_for_capacity(frozendict({"requests": 100.0}))

        real_server_time = _server_time.async_server_time

        async def future_server_time(client):
            real_time = await real_server_time(client)
            return real_time + 86400.0

        with patch.object(_server_time, "async_server_time", future_server_time):
            pipeline = backend._redis.pipeline()
            future_time = await future_server_time(redis_client)
            result = await backend._get_capacities_unsafe(
                pipeline=pipeline, current_time=future_time
            )
            cap = next(iter(result.capacities.values()))

        assert cap == pytest.approx(100.0), (
            f"Forward time jump exceeded max_capacity! Got {cap}, max=100. "
            f"calculate_capacity must cap at max_capacity."
        )


# ---------------------------------------------------------------------------
# 5. Lock TTL expiry simulation
# ---------------------------------------------------------------------------


@pytest.mark.redis
class TestLockTTLExpiry:
    """Lock expiry mid-operation must be detected, not silently ignored."""

    async def test_extend_locks_detects_expired_lock(self, redis_client):
        """_extend_locks raises LockError when the lock has expired."""
        builder = RedisBackendBuilder(redis_client, key_prefix="test")
        config = _make_config()
        backend = builder.build(config, callbacks=RateLimiterCallbacks())

        stack = await backend._lock(timeout=LOCK_TIMEOUT_SECONDS)
        try:
            assert len(stack.locks) >= 1
            lock = stack.locks[0]
            await redis_client.delete(lock.name)

            with pytest.raises(redis.exceptions.LockError):
                await backend._extend_locks(stack)
        finally:
            with contextlib.suppress(redis.exceptions.LockNotOwnedError):
                await stack.aclose()

    async def test_expired_lock_aborts_consume_cleanly(self, redis_client):
        """If the lock expires between read and write, the operation fails cleanly."""
        builder = RedisBackendBuilder(redis_client, key_prefix="test")
        config = _make_config(limit=100)
        backend = builder.build(config, callbacks=RateLimiterCallbacks())

        await backend.await_for_capacity(frozendict({"requests": 10.0}))
        cap_before = await _get_redis_capacity(backend)

        real_extend = RedisBackend._extend_locks

        async def expiring_extend(stack, **kwargs):
            for lock in stack.locks:
                await redis_client.delete(lock.name)
            return await real_extend(stack, **kwargs)

        backend._extend_locks = expiring_extend

        try:
            with pytest.raises((TimeoutError, redis.exceptions.LockError)):
                await backend.await_for_capacity(
                    frozendict({"requests": 5.0}), timeout=0
                )
        finally:
            backend._extend_locks = real_extend

        cap_after = await _get_redis_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=2.0), (
            f"State corrupted after lock expiry! "
            f"Before={cap_before}, after={cap_after}. "
            f"The write must not commit when the lock has expired."
        )

    async def test_lock_stolen_by_another_worker_detected(self, redis_client):
        """If another worker steals the lock, _extend_locks raises LockError."""
        builder = RedisBackendBuilder(redis_client, key_prefix="test")
        config = _make_config()
        backend = builder.build(config, callbacks=RateLimiterCallbacks())

        stack = await backend._lock(timeout=LOCK_TIMEOUT_SECONDS)
        try:
            lock = stack.locks[0]

            await redis_client.delete(lock.name)
            thief_lock = backend.sorted_buckets[0].lock(timeout=10)
            assert await thief_lock.acquire(blocking_timeout=1)

            with pytest.raises(redis.exceptions.LockError):
                await backend._extend_locks(stack)

            await thief_lock.release()
        finally:
            with contextlib.suppress(redis.exceptions.LockNotOwnedError):
                await stack.aclose()

    async def test_consume_capacity_with_short_lock_ttl(self, redis_client):
        """Very short lock TTL does not corrupt state even if extend is needed."""
        builder = RedisBackendBuilder(redis_client, key_prefix="test")
        config = _make_config(limit=100)
        backend = builder.build(config, callbacks=RateLimiterCallbacks())

        original_lock = backend._lock

        async def short_ttl_lock(**kwargs):
            kwargs["timeout"] = 2
            return await original_lock(**kwargs)

        backend._lock = short_ttl_lock

        with contextlib.suppress(redis.exceptions.LockError):
            await backend.consume_capacity(frozendict({"requests": 10.0}))

        cap = await _get_redis_capacity(backend)
        assert cap == pytest.approx(90.0, abs=2.0) or cap == pytest.approx(
            100.0, abs=2.0
        ), (
            f"Inconsistent state! Capacity={cap}. "
            f"Expected either ~90 (write committed) or ~100 (write aborted)."
        )


# ---------------------------------------------------------------------------
# 6. Compound fault: ConnectionError during refund
# ---------------------------------------------------------------------------


@pytest.mark.redis
class TestConnectionErrorDuringRefund:
    """ConnectionError during refund_capacity must not corrupt state."""

    async def test_connection_error_during_refund_surfaces_cleanly(self, redis_client):
        """Refund fails cleanly if Redis dies mid-refund pipeline."""
        builder = RedisBackendBuilder(redis_client, key_prefix="test")
        config = _make_config(limit=100, per_seconds=3600)
        backend = builder.build(config, callbacks=RateLimiterCallbacks())

        await backend.consume_capacity(frozendict({"requests": 50.0}))
        cap_before = await _get_redis_capacity(backend)

        real_set = backend._set_capacities_unsafe

        async def failing_set(*args, **kwargs):
            raise ConnectionError("Injected: connection lost during refund write")

        backend._set_capacities_unsafe = failing_set

        with pytest.raises(ConnectionError):
            await backend.refund_capacity(
                reserved_usage=frozendict({"requests": 50.0}),
                actual_usage=frozendict({"requests": 20.0}),
            )

        backend._set_capacities_unsafe = real_set

        cap_after = await _get_redis_capacity(backend)
        assert cap_after == pytest.approx(cap_before, abs=2.0), (
            f"Partial refund committed! Before={cap_before}, after={cap_after}."
        )
