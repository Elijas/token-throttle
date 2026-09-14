"""Exact accounting traces for all six built-in backend implementations.

Only accounting clocks are replaced. Redis still executes real transactions;
its expiry clock and lock deadlines remain real. TTL-specific contracts belong
in backend-specific tests because their semantics intentionally differ.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import time
import uuid
from types import SimpleNamespace

import pytest

from token_throttle import (
    MemoryBackendBuilder,
    PerModelConfig,
    Quota,
    SqliteBackendBuilder,
    SyncMemoryBackendBuilder,
    SyncSqliteBackendBuilder,
    UsageQuotas,
    frozen_usage,
)


class _Trace:
    def __init__(self, backend, clock, *, asynchronous):
        self.backend = backend
        self.clock = clock
        self.asynchronous = asynchronous

    async def call(self, method, *args, **kwargs):
        result = getattr(self.backend, method)(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    async def acquire(self, requests, tokens):
        return await self.call(
            "await_for_capacity" if self.asynchronous else "wait_for_capacity",
            frozen_usage({"requests": requests, "tokens": tokens}),
            timeout=0,
        )

    async def capacities(self, requests, tokens):
        snapshot = await self.call("introspect")
        actual = {bucket.metric: bucket.current_capacity for bucket in snapshot.buckets}
        assert actual == pytest.approx({"requests": requests, "tokens": tokens})


@pytest.fixture(
    params=[
        "memory-sync",
        "memory-async",
        "sqlite-sync",
        "sqlite-async",
        "redis-sync",
        "redis-async",
    ]
)
async def trace(request, tmp_path, monkeypatch):  # noqa: PLR0915
    kind, mode = request.param.split("-")
    asynchronous = mode == "async"
    clock = [time.time()]
    config = PerModelConfig(
        model_family="trace",
        quotas=UsageQuotas(
            [
                Quota(metric="requests", limit=10, per_seconds=10),
                Quota(metric="tokens", limit=20, per_seconds=20),
            ]
        ),
    )
    client = None
    if kind == "memory":
        builder = MemoryBackendBuilder() if asynchronous else SyncMemoryBackendBuilder()
        module = importlib.import_module(
            f"token_throttle._limiter_backends._memory.{'_backend' if asynchronous else '_sync_backend'}"
        )
        monkeypatch.setattr(
            module,
            "time",
            SimpleNamespace(
                time=lambda: clock[0],
                monotonic=time.monotonic,
                sleep=time.sleep,
            ),
        )
    elif kind == "sqlite":
        cls = SqliteBackendBuilder if asynchronous else SyncSqliteBackendBuilder
        builder = cls(tmp_path / "trace.db", key_prefix="trace")
    else:
        redis = pytest.importorskip("redis")
        redis_async = pytest.importorskip("redis.asyncio")
        url = request.config.getoption("--redis-url")
        client = (redis_async if asynchronous else redis).from_url(url)
        try:
            ping = client.ping()
            if asynchronous:
                await ping
        except redis.exceptions.RedisError:
            if asynchronous:
                await client.aclose()
            else:
                client.close()
            pytest.skip("Redis unavailable at the explicitly configured test URL")
        backend_module = importlib.import_module(
            f"token_throttle._limiter_backends._redis.{'_backend' if asynchronous else '_sync_backend'}"
        )
        bucket_module = importlib.import_module(
            f"token_throttle._limiter_backends._redis.{'_bucket' if asynchronous else '_sync_bucket'}"
        )

        async def async_clock(_client):
            return clock[0]

        for module in (backend_module, bucket_module):
            monkeypatch.setattr(
                module,
                "async_server_time" if asynchronous else "sync_server_time",
                async_clock if asynchronous else lambda _client: clock[0],
            )
        cls = (
            backend_module.RedisBackendBuilder
            if asynchronous
            else backend_module.SyncRedisBackendBuilder
        )
        # Unique namespaces avoid flushing data and interference with other tests.
        prefix = f"trace-{uuid.uuid4().hex}"
        builder = cls(client, key_prefix=prefix)
    try:
        backend = builder.build(config)
        if kind == "sqlite":
            for method in (
                "try_consume",
                "consume",
                "refund",
                "set_max_capacity",
                "inspect_snapshot",
            ):
                monkeypatch.setattr(
                    backend._engine,
                    method,
                    functools.partial(
                        getattr(backend._engine, method), clock=lambda: clock[0]
                    ),
                )
        yield _Trace(backend, clock, asynchronous=asynchronous)
    finally:
        if asynchronous:
            await builder.aclose()
        else:
            builder.close()
        if client is not None:
            if asynchronous:
                keys = [key async for key in client.scan_iter(match=f"{prefix}:*")]
                if keys:
                    await client.delete(*keys)
                await client.aclose()
            else:
                keys = list(client.scan_iter(match=f"{prefix}:*"))
                if keys:
                    client.delete(*keys)
                client.close()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
async def test_refill_refund_debt_and_rate_changes_follow_exact_trace(trace):
    await trace.acquire(6, 4)
    await trace.capacities(4, 16)
    trace.clock[0] += 2
    await trace.capacities(6, 18)
    await trace.call(
        "refund_capacity",
        frozen_usage({"requests": 6, "tokens": 4}),
        frozen_usage({"requests": 8, "tokens": 2}),
    )
    await trace.capacities(4, 20)
    await trace.call("consume_capacity", frozen_usage({"requests": 20, "tokens": 25}))
    await trace.capacities(-10, -5)
    trace.clock[0] += 5
    await trace.capacities(-5, 0)
    with pytest.raises(TimeoutError):
        await trace.acquire(0, 0)
    await trace.capacities(-5, 0)
    trace.clock[0] += 6
    await trace.acquire(1, 6)
    await trace.capacities(0, 0)
    await trace.call("set_max_capacity", "requests", 10, 5)
    await trace.call("set_max_capacity", "tokens", 20, 40)
    trace.clock[0] += 2
    await trace.capacities(1, 4)


async def test_failed_multi_metric_acquire_does_not_charge_other_bucket(trace):
    await trace.acquire(9, 1)
    for _ in range(3):
        with pytest.raises(TimeoutError):
            await trace.acquire(2, 10)
        await trace.capacities(1, 19)
    with pytest.raises(ValueError, match="exceeds bucket max capacity"):
        await trace.acquire(0, 21)
    await trace.capacities(1, 19)
    await trace.acquire(1, 19)
    await trace.capacities(0, 0)


async def test_cap_changes_preserve_overflow_until_a_capacity_write(trace):
    await trace.acquire(2, 5)
    await trace.call("set_max_capacity", "requests", 10, 3)
    await trace.capacities(3, 15)
    await trace.call("set_max_capacity", "requests", 10, 10)
    await trace.capacities(8, 15)
    await trace.call("set_max_capacity", "requests", 10, 3)
    await trace.acquire(1, 0)
    await trace.call("set_max_capacity", "requests", 10, 10)
    await trace.capacities(2, 15)
