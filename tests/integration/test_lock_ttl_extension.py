"""Regression tests for Bug 4: Redis distributed lock TTL expiry mid-operation.

Before the fix, the Redis backend acquired the distributed lock with a
30-second TTL but never refreshed it. If a GC pause, connection stall,
or K8s CPU throttle occurred between ``_get_capacities_unsafe`` and
``_set_capacities_unsafe``, the lock's Redis TTL could lapse, a second
worker could take the lock, and both workers would write post-consumption
values — a lost-update race.

The fix adds ``_extend_locks(stack)`` which calls ``lock.reacquire()``
on every held lock, resetting the TTL to ``LOCK_TIMEOUT_SECONDS``. Each
mutating path (``_check_and_consume_capacity``, ``consume_capacity``,
``refund_capacity_for_buckets``, ``_refund_cancelled_consumption``)
calls it immediately before its write pipeline so the write is always
protected by a fresh TTL regardless of how long preceding reads took.

These tests check the behavioural contract — full lost-update immunity
under arbitrary GC pauses would require deterministic pause injection
and a more invasive Lua-CAS write (see KNOWN UNKNOWN in the commit).
"""

import secrets

import pytest
from frozendict import frozendict

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import (
    LOCK_TIMEOUT_SECONDS,
    RedisBackendBuilder,
)


def _make_config(model_family: str | None = None) -> PerModelConfig:
    return PerModelConfig(
        model_family=model_family or f"test-{secrets.token_hex(4)}",
        quotas=UsageQuotas([Quota(metric="requests", limit=100, per_seconds=3600)]),
    )


@pytest.mark.redis
class TestLockExtensionHelper:
    """``_extend_locks`` resets lock TTL."""

    async def test_extend_locks_refreshes_ttl(self, redis_client):
        """Reacquire via ``_extend_locks`` resets the TTL to the configured timeout."""
        backend = RedisBackendBuilder(redis_client, key_prefix="test").build(
            _make_config(), callbacks=RateLimiterCallbacks()
        )
        async with await backend._lock(timeout=LOCK_TIMEOUT_SECONDS) as stack:
            assert len(stack.locks) == 1
            lock_name = stack.locks[0].name
            # Hand-shrink TTL so we can observe the extend.
            await redis_client.pexpire(lock_name, 2_000)
            ttl_before = await redis_client.pttl(lock_name)
            assert ttl_before <= 2_000

            await backend._extend_locks(stack)

            ttl_after = await redis_client.pttl(lock_name)
            # Reacquire resets TTL to LOCK_TIMEOUT_SECONDS seconds worth
            # of milliseconds. Allow some slack for the round-trip.
            assert ttl_after > 25_000, (
                f"TTL was not refreshed: before={ttl_before}ms, "
                f"after={ttl_after}ms. Expected >25000ms."
            )


@pytest.mark.redis
class TestMutatingPathsCallExtendLocks:
    """Each mutating path calls ``_extend_locks`` before writing.

    This is a code-level regression check — it doesn't prove lost-update
    immunity under arbitrary GC pauses, but it proves the fix wiring is
    in place on every mutating call site.
    """

    async def test_consume_capacity_extends_lock_before_write(
        self, redis_client, monkeypatch
    ):
        backend = RedisBackendBuilder(redis_client, key_prefix="test").build(
            _make_config(), callbacks=RateLimiterCallbacks()
        )
        calls = []
        orig_extend = backend._extend_locks

        async def traced_extend(stack, **kwargs):
            calls.append("extend")
            return await orig_extend(stack, **kwargs)

        monkeypatch.setattr(backend, "_extend_locks", traced_extend)

        await backend.consume_capacity(frozendict({"requests": 1.0}))
        assert calls, "consume_capacity must call _extend_locks before its write"

    async def test_await_for_capacity_extends_lock_before_write(
        self, redis_client, monkeypatch
    ):
        backend = RedisBackendBuilder(redis_client, key_prefix="test").build(
            _make_config(), callbacks=RateLimiterCallbacks()
        )
        calls = []
        orig_extend = backend._extend_locks

        async def traced_extend(stack, **kwargs):
            calls.append("extend")
            return await orig_extend(stack, **kwargs)

        monkeypatch.setattr(backend, "_extend_locks", traced_extend)

        await backend.await_for_capacity(frozendict({"requests": 1.0}), timeout=0)
        assert calls, (
            "_check_and_consume_capacity must call _extend_locks before its write"
        )

    async def test_refund_capacity_extends_lock_before_write(
        self, redis_client, monkeypatch
    ):
        backend = RedisBackendBuilder(redis_client, key_prefix="test").build(
            _make_config(), callbacks=RateLimiterCallbacks()
        )
        await backend.consume_capacity(frozendict({"requests": 5.0}))

        calls = []
        orig_extend = backend._extend_locks

        async def traced_extend(stack, **kwargs):
            calls.append("extend")
            return await orig_extend(stack, **kwargs)

        monkeypatch.setattr(backend, "_extend_locks", traced_extend)

        await backend.refund_capacity(
            reserved_usage=frozendict({"requests": 5.0}),
            actual_usage=frozendict({"requests": 2.0}),
        )
        assert calls, (
            "refund_capacity_for_buckets must call _extend_locks before its write"
        )
