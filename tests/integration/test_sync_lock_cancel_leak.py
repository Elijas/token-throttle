"""Sync counterpart to test_lock_cancel_leak.py (F18.03).

Verifies that the sync Redis backend's _lock() method does not leak
lock keys when KeyboardInterrupt arrives after the SET NX succeeds
but before the Python-level acquire() returns.

The fix: _lock() pre-generates the token and performs a best-effort
CAS release (lua_release) in its except BaseException handler.
"""

import secrets
from unittest.mock import patch

import pytest
import redis as sync_redis
from frozendict import frozendict

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)


@pytest.mark.redis
class TestSyncLockCancelDoesNotLeakKey:
    """KeyboardInterrupt during sync lock.acquire() must not leak the Redis lock key."""

    def test_raw_lock_leaks_on_interrupt(self, sync_redis_client):
        """Informational: raw redis.lock.Lock leaks when interrupted after SET NX.

        This documents the underlying behavior that the backend's
        _lock() wrapper compensates for. We don't assert leaked == 0
        because fixing the raw client is out of scope.
        """
        lock_key = f"probe:{secrets.token_hex(4)}:lock"
        original_acquire = sync_redis.lock.Lock.acquire
        trials = 50
        leaked = 0

        for _ in range(trials):
            lock = sync_redis.lock.Lock(
                sync_redis_client, lock_key, timeout=30, blocking_timeout=5
            )

            def acquire_then_interrupt(self_lock, *args, **kwargs):
                result = original_acquire(self_lock, *args, **kwargs)
                if result:
                    raise KeyboardInterrupt("Simulated interrupt after SET NX")
                return result

            with patch.object(sync_redis.lock.Lock, "acquire", acquire_then_interrupt):
                try:  # noqa: SIM105
                    lock.acquire()
                except KeyboardInterrupt:
                    pass

            if sync_redis_client.exists(lock_key):
                leaked += 1
                sync_redis_client.delete(lock_key)

        assert leaked >= 0  # informational only

    def test_backend_lock_helper_does_not_leak_on_interrupt(self, sync_redis_client):
        """End-to-end: SyncRedisBackend._lock() must not leak on KeyboardInterrupt.

        Triggers consume_capacity with lock.acquire monkeypatched to raise
        KeyboardInterrupt right after SET NX succeeds. The pre-generated
        token + CAS lua_release in the except handler should clean up the key.
        """
        config = PerModelConfig(
            model_family=f"test-{secrets.token_hex(4)}",
            quotas=UsageQuotas([Quota(metric="requests", limit=100, per_seconds=3600)]),
        )
        backend = SyncRedisBackendBuilder(sync_redis_client).build(config)
        lock_key_pattern = f"*{config.model_family}*lock*"

        original_acquire = sync_redis.lock.Lock.acquire
        trials = 50
        leaked_trials = 0

        for _ in range(trials):

            def acquire_then_interrupt(self_lock, *args, **kwargs):
                result = original_acquire(self_lock, *args, **kwargs)
                if result:
                    raise KeyboardInterrupt("Simulated interrupt after SET NX")
                return result

            with patch.object(sync_redis.lock.Lock, "acquire", acquire_then_interrupt):
                try:  # noqa: SIM105
                    backend.consume_capacity(frozendict({"requests": 1.0}))
                except KeyboardInterrupt:
                    pass

            remaining = sync_redis_client.keys(lock_key_pattern)
            if remaining:
                leaked_trials += 1
                for key in remaining:
                    sync_redis_client.delete(key)

        assert leaked_trials == 0, (
            f"Sync _lock() leaked locks in {leaked_trials}/{trials} trials "
            f"after KeyboardInterrupt during acquire. Expected zero leaks — "
            f"the best-effort CAS release should prevent the lock key from "
            f"persisting for its 30s TTL."
        )
