"""Public reservation authority remains valid across callable config changes."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

import pytest

from tests._redis_guard import ensure_flush_allowed
from token_throttle import (
    MemoryBackendBuilder,
    PerModelConfig,
    Quota,
    RateLimiter,
    RateLimiterCallbacks,
    SqliteBackendBuilder,
    SyncMemoryBackendBuilder,
    SyncRateLimiter,
    SyncRateLimiterCallbacks,
    SyncSqliteBackendBuilder,
    UsageQuotas,
)


def _config(*windows: int) -> PerModelConfig:
    return PerModelConfig(
        model_family="pending-rebuild",
        quotas=UsageQuotas(
            [Quota(metric="tokens", limit=100, per_seconds=w) for w in windows]
        ),
    )


@dataclass
class Harness:
    limiter: RateLimiter | SyncRateLimiter
    configs: list[PerModelConfig]
    entered: threading.Event
    release: threading.Event

    async def call(self, method: str, *args, **kwargs):
        operation = getattr(self.limiter, method)
        if isinstance(self.limiter, SyncRateLimiter):
            return await asyncio.to_thread(operation, *args, **kwargs)
        return await operation(*args, **kwargs)

    async def capacities(self) -> dict[int, float]:
        backend = self.limiter._model_family_to_backend["pending-rebuild"]
        if isinstance(self.limiter, SyncRateLimiter):
            snapshot = await asyncio.to_thread(backend.introspect)
        else:
            snapshot = await backend.introspect()
        return {b.per_seconds: b.current_capacity for b in snapshot.buckets}


@pytest.fixture(params=["memory", "sqlite", "redis"])
async def harness(request, tmp_path):  # noqa: PLR0915
    sync = request.node.callspec.params["sync"]
    entered, release = threading.Event(), threading.Event()

    def sync_wait(**_kwargs):
        entered.set()
        assert release.wait(timeout=5)

    async def async_wait(**_kwargs):
        entered.set()
        assert await asyncio.to_thread(release.wait, 5)

    client = None
    if request.param == "memory":
        builder = (SyncMemoryBackendBuilder if sync else MemoryBackendBuilder)()
    elif request.param == "sqlite":
        builder = (SyncSqliteBackendBuilder if sync else SqliteBackendBuilder)(
            tmp_path / "pending.sqlite3", key_prefix="pending-rebuild"
        )
    else:
        redis = pytest.importorskip("redis")
        redis_async = pytest.importorskip("redis.asyncio")
        from token_throttle import (  # noqa: PLC0415
            RedisBackendBuilder,
            SyncRedisBackendBuilder,
        )

        url = request.config.getoption("--redis-url")
        client = (redis if sync else redis_async).Redis.from_url(
            url, socket_connect_timeout=0.2, socket_timeout=2
        )
        try:
            if sync:
                await asyncio.to_thread(client.ping)
            else:
                await client.ping()
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            if sync:
                client.close()
            else:
                await client.aclose()
            pytest.skip("Redis unavailable")
        ensure_flush_allowed(url)
        if sync:
            client.flushdb()
        else:
            await client.flushdb()
        builder = (SyncRedisBackendBuilder if sync else RedisBackendBuilder)(
            client, key_prefix="pending-rebuild"
        )
    configs = [_config(10000)]
    callbacks = (
        SyncRateLimiterCallbacks(on_wait_start=sync_wait)
        if sync
        else RateLimiterCallbacks(on_wait_start=async_wait)
    )
    limiter = (SyncRateLimiter if sync else RateLimiter)(
        lambda _: configs[0], backend=builder, callbacks=callbacks
    )
    value = Harness(limiter, configs, entered, release)
    try:
        yield value
    finally:
        release.set()
        await value.call("close" if sync else "aclose")
        if client is not None:
            if sync:
                client.flushdb()
                client.close()
            else:
                await client.flushdb()
                await client.aclose()


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_pending_acquire_full_refund_matches_every_debit(harness, sync):
    first = await harness.call("acquire_capacity", {"tokens": 100}, "model")
    waiter = asyncio.create_task(
        harness.call("acquire_capacity", {"tokens": 100}, "model", timeout=4)
    )
    try:
        assert await asyncio.to_thread(harness.entered.wait, 2)
        harness.configs[0] = _config(10000, 20000)
        await harness.call("refund_capacity", {"tokens": 0}, first)
        harness.release.set()
        reservation = await waiter
        await harness.call("refund_capacity", {"tokens": 0}, reservation)
        assert await harness.capacities() == pytest.approx({10000: 100, 20000: 100})
        assert harness.limiter.snapshot_state()["in_flight_reservations"] == 0
    finally:
        harness.release.set()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_rebuild_rejects_new_acquire_until_pending_acquire_finishes(
    harness, sync
):
    first = await harness.call("acquire_capacity", {"tokens": 100}, "model")
    waiter = asyncio.create_task(
        harness.call("acquire_capacity", {"tokens": 100}, "model", timeout=4)
    )
    try:
        assert await asyncio.to_thread(harness.entered.wait, 2)
        harness.configs[0] = _config(10000, 20000)
        with pytest.raises(ValueError, match="pending acquisitions"):
            await harness.call("acquire_capacity", {"tokens": 0}, "model", timeout=0)
        assert set(await harness.capacities()) == {10000}
        await harness.call("refund_capacity", {"tokens": 0}, first)
        harness.release.set()
        reservation = await waiter
        await harness.call("refund_capacity", {"tokens": 0}, reservation)
        retry = await harness.call(
            "acquire_capacity", {"tokens": 100}, "model", timeout=0
        )
        assert retry.bucket_ids == frozenset({("tokens", 10000), ("tokens", 20000)})
        await harness.call("refund_capacity", {"tokens": 0}, retry)
    finally:
        harness.release.set()
        await asyncio.gather(waiter, return_exceptions=True)
