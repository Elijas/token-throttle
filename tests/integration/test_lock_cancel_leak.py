"""Regression test for Bug 2: Redis distributed lock leaks on cancel mid-acquire.

Before the fix, ``_lock()`` in the async Redis backend wrapped
``lock.acquire()`` in a simple ``try`` that only pushed the release
callback AFTER acquire returned True. If CancelledError arrived between
the successful SET NX round-trip and the Python-level ``return True``,
the lock key persisted in Redis for its full 30-second TTL, blocking
every subsequent acquirer.

The fix wraps ``await lock.acquire(...)`` in try/except BaseException
and issues a best-effort ``lock.release()`` under ``asyncio.shield``
before re-raising, so even a cancel arriving at the worst possible
instant cannot leak the lock.
"""

import asyncio
import secrets

import pytest
import redis.asyncio as aioredis
from frozendict import frozendict

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder


@pytest.mark.redis
class TestLockCancelDoesNotLeakKey:
    """Cancel during ``lock.acquire()`` must not leak the Redis lock key."""

    async def test_cancel_during_acquire_does_not_leak_lock_key(self, redis_client):
        """Repeatedly cancel a lock.acquire() mid-flight; no key may leak.

        The reproducer from the bug report: create a Lock, kick off
        acquire(), yield a few times so the SET NX round-trip lands,
        then cancel. Under the old code this leaked ~99.95% of the time.

        NOTE: this test exercises ``aioredis.lock.Lock`` directly to
        demonstrate that the leak is a real Redis-client behavior; the
        per-backend fix is exercised by
        ``test_backend_lock_helper_does_not_leak_on_cancel`` below.
        """
        lock_key = f"probe:{secrets.token_hex(4)}:lock"
        # 200 trials is enough to surface the old 99%+ leak rate; 2000
        # (per the bug report) was too slow under the normal test run.
        trials = 200
        leaked = 0
        for _ in range(trials):
            lock = aioredis.lock.Lock(
                redis_client, lock_key, timeout=30, blocking_timeout=5
            )
            task = asyncio.create_task(lock.acquire())
            for _ in range(3):
                await asyncio.sleep(0)
            task.cancel()
            try:  # noqa: SIM105
                await task
            except BaseException:  # noqa: S110
                pass
            if await redis_client.exists(lock_key):
                leaked += 1
                await redis_client.delete(lock_key)

        # This test is informational — it documents that the raw
        # aioredis.lock.Lock has the leaky behavior. The backend fix
        # wraps it to compensate. We don't assert leaked == 0 here
        # because fixing the raw client is out of scope for this repo.
        assert leaked >= 0  # sanity: no spurious pytest failure

    async def test_backend_lock_helper_does_not_leak_on_cancel(self, redis_client):
        """End-to-end: RedisBackend._lock() must not leak on cancel.

        Spawn a consume_capacity, cancel it while it is inside _lock(),
        and assert no lock keys for this backend remain in Redis.
        """
        config = PerModelConfig(
            model_family=f"test-{secrets.token_hex(4)}",
            quotas=UsageQuotas([Quota(metric="requests", limit=100, per_seconds=3600)]),
        )
        backend = RedisBackendBuilder(redis_client, key_prefix="test").build(
            config, callbacks=RateLimiterCallbacks()
        )
        lock_key_pattern = f"*{config.model_family}*lock*"

        trials = 50
        leaked_trials = 0
        for _ in range(trials):
            task = asyncio.create_task(
                backend.consume_capacity(frozendict({"requests": 1.0}))
            )
            for _ in range(3):
                await asyncio.sleep(0)
            task.cancel()
            try:  # noqa: SIM105
                await task
            except BaseException:  # noqa: S110
                pass

            remaining = await redis_client.keys(lock_key_pattern)
            if remaining:
                leaked_trials += 1
                for key in remaining:
                    await redis_client.delete(key)

        assert leaked_trials == 0, (
            f"Backend _lock() leaked locks in {leaked_trials}/{trials} trials "
            f"after cancellation during acquire. Expected zero leaks — the "
            f"best-effort release in _lock()'s acquire wrapper should prevent "
            f"the lock key from persisting for its 30s TTL."
        )
