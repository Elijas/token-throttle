"""Regression tests for FIX-33 LK-SETMAX-RECONCILE."""

import asyncio
import json
import time

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

try:
    import token_throttle._limiter_backends._redis._backend as redis_backend_module
    import token_throttle._limiter_backends._redis._sync_backend as sync_redis_backend_module
    from token_throttle._limiter_backends._redis._backend import RedisBackend
    from token_throttle._limiter_backends._redis._bucket import RedisBucket
    from token_throttle._limiter_backends._redis._sync_backend import SyncRedisBackend
    from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket
except ImportError:
    redis_backend_module = None
    sync_redis_backend_module = None
    RedisBackend = None
    RedisBucket = None
    SyncRedisBackend = None
    SyncRedisBucket = None

MODEL = "test-model"
MODEL_FAMILY = "test-family"
BUCKET_ID = ("tokens", 60)


def _config(limit: float = 100.0) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=limit, per_seconds=60)]),
        model_family=MODEL_FAMILY,
    )


def _runtime_override(limiter) -> float | None:
    return limiter._model_family_to_runtime_max_capacity.get(MODEL_FAMILY, {}).get(
        BUCKET_ID
    )


def _bucket_max_capacity(limiter) -> float:
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    return backend._bucket_registry[BUCKET_ID].max_capacity


async def test_async_post_write_failure_reconciles_limiter_override() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    await limiter.acquire_capacity({"tokens": 1}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    original_set_max_capacity = backend.set_max_capacity

    async def write_then_fail(metric, per_seconds, value) -> None:
        await original_set_max_capacity(metric, per_seconds, value)
        raise RuntimeError("simulated post-write failure")

    backend.set_max_capacity = write_then_fail

    with pytest.raises(RuntimeError, match="simulated post-write failure"):
        await limiter.set_max_capacity(MODEL, "tokens", 60, 50.0)

    assert _bucket_max_capacity(limiter) == pytest.approx(50.0)
    assert _runtime_override(limiter) == pytest.approx(50.0)


async def test_async_cancel_after_backend_write_reconciles_limiter_override() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    await limiter.acquire_capacity({"tokens": 1}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    original_set_max_capacity = backend.set_max_capacity
    write_done = asyncio.Event()
    release_backend = asyncio.Event()

    async def write_then_pause(metric, per_seconds, value) -> None:
        await original_set_max_capacity(metric, per_seconds, value)
        write_done.set()
        await release_backend.wait()

    backend.set_max_capacity = write_then_pause

    task = asyncio.create_task(limiter.set_max_capacity(MODEL, "tokens", 60, 50.0))
    await asyncio.wait_for(write_done.wait(), timeout=1.0)
    task.cancel()
    release_backend.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert _bucket_max_capacity(limiter) == pytest.approx(50.0)
    assert _runtime_override(limiter) == pytest.approx(50.0)


def test_sync_post_write_failure_reconciles_limiter_override() -> None:
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    limiter.acquire_capacity({"tokens": 1}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    original_set_max_capacity = backend.set_max_capacity

    def write_then_fail(metric, per_seconds, value) -> None:
        original_set_max_capacity(metric, per_seconds, value)
        raise RuntimeError("simulated post-write failure")

    backend.set_max_capacity = write_then_fail

    with pytest.raises(RuntimeError, match="simulated post-write failure"):
        limiter.set_max_capacity(MODEL, "tokens", 60, 50.0)

    assert _bucket_max_capacity(limiter) == pytest.approx(50.0)
    assert _runtime_override(limiter) == pytest.approx(50.0)


def test_sync_interrupt_after_backend_write_reconciles_limiter_override() -> None:
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    limiter.acquire_capacity({"tokens": 1}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    original_set_max_capacity = backend.set_max_capacity

    def write_then_interrupt(metric, per_seconds, value) -> None:
        original_set_max_capacity(metric, per_seconds, value)
        raise KeyboardInterrupt

    backend.set_max_capacity = write_then_interrupt

    with pytest.raises(KeyboardInterrupt):
        limiter.set_max_capacity(MODEL, "tokens", 60, 50.0)

    assert _bucket_max_capacity(limiter) == pytest.approx(50.0)
    assert _runtime_override(limiter) == pytest.approx(50.0)


class _AsyncRedisPipeline:
    def __init__(self, redis_client):
        self._redis = redis_client
        self._ops: list[tuple[str, str, object, dict[str, object]]] = []

    def get(self, key):
        self._ops.append(("get", key, None, {}))

    def set(self, key, value, **kwargs):
        self._ops.append(("set", key, value, kwargs))

    def expire(self, key, seconds):
        self._ops.append(("expire", key, seconds, {}))

    async def execute(self):
        results = []
        for op, key, value, kwargs in self._ops:
            if op == "get":
                results.append(self._redis.store.get(key))
            elif op == "set":
                self._redis.store[key] = value
                self._redis.set_calls.append((key, value, kwargs))
                results.append(True)
            elif op == "expire":
                self._redis.expire_calls.append((key, value))
                results.append(key in self._redis.store)
        return results


class _AsyncRedis:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.get_calls: list[str] = []
        self.expire_calls: list[tuple[str, object]] = []
        self.set_calls: list[tuple[str, object, dict[str, object]]] = []

    async def get(self, key):
        self.get_calls.append(key)
        return self.store.get(key)

    async def expire(self, key, seconds):
        self.expire_calls.append((key, seconds))
        return key in self.store

    def pipeline(self):
        return _AsyncRedisPipeline(self)


class _SyncRedisPipeline:
    def __init__(self, redis_client):
        self._redis = redis_client
        self._ops: list[tuple[str, str, object, dict[str, object]]] = []

    def get(self, key):
        self._ops.append(("get", key, None, {}))

    def set(self, key, value, **kwargs):
        self._ops.append(("set", key, value, kwargs))

    def expire(self, key, seconds):
        self._ops.append(("expire", key, seconds, {}))

    def execute(self):
        results = []
        for op, key, value, kwargs in self._ops:
            if op == "get":
                results.append(self._redis.store.get(key))
            elif op == "set":
                self._redis.store[key] = value
                self._redis.set_calls.append((key, value, kwargs))
                results.append(True)
            elif op == "expire":
                self._redis.expire_calls.append((key, value))
                results.append(key in self._redis.store)
        return results


class _SyncRedis:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.get_calls: list[str] = []
        self.expire_calls: list[tuple[str, object]] = []
        self.set_calls: list[tuple[str, object, dict[str, object]]] = []

    def get(self, key):
        self.get_calls.append(key)
        return self.store.get(key)

    def expire(self, key, seconds):
        self.expire_calls.append((key, seconds))
        return key in self.store

    def pipeline(self):
        return _SyncRedisPipeline(self)


def _redis_config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=60)]),
        model_family="redis-family",
    )


def _redis_override_payload(value: float) -> str:
    return json.dumps(
        {"configured_max_capacity": 100.0, "override_max_capacity": value}
    )


async def test_async_redis_snapshot_force_refreshes_stale_local_override_cache(
    monkeypatch,
) -> None:
    if redis_backend_module is None or RedisBackend is None or RedisBucket is None:
        pytest.skip("redis package not installed")

    async def fake_server_time(_redis) -> float:
        return 100.0

    monkeypatch.setattr(redis_backend_module, "async_server_time", fake_server_time)
    redis_client = _AsyncRedis()
    cfg = _redis_config()
    quota = next(iter(cfg.quotas))
    bucket = RedisBucket(
        quota=quota,
        limit_config=cfg,
        redis_client=redis_client,
        key_prefix="test",
    )
    backend = RedisBackend(
        buckets=[bucket],
        redis=redis_client,
        limit_config=cfg,
        key_prefix="test",
    )

    bucket._set_cached_max_capacity_override(None)
    bucket._max_capacity_cache_time = time.time()
    redis_client.store[bucket._max_capacity_key] = _redis_override_payload(200.0)
    redis_client.store[bucket._last_checked_key] = 70.0
    redis_client.store[bucket._capacity_key] = 0.0

    await backend._snapshot_bucket_state(bucket)

    assert redis_client.get_calls == [bucket._max_capacity_key]
    assert bucket.max_capacity == pytest.approx(200.0)
    assert redis_client.store[bucket._capacity_key] == pytest.approx(100.0)


def test_sync_redis_snapshot_force_refreshes_stale_local_override_cache(
    monkeypatch,
) -> None:
    if (
        sync_redis_backend_module is None
        or SyncRedisBackend is None
        or SyncRedisBucket is None
    ):
        pytest.skip("redis package not installed")

    monkeypatch.setattr(
        sync_redis_backend_module, "sync_server_time", lambda _redis: 100.0
    )
    redis_client = _SyncRedis()
    cfg = _redis_config()
    quota = next(iter(cfg.quotas))
    bucket = SyncRedisBucket(
        quota=quota,
        limit_config=cfg,
        redis_client=redis_client,
        key_prefix="test",
    )
    backend = SyncRedisBackend(
        buckets=[bucket],
        redis=redis_client,
        limit_config=cfg,
        key_prefix="test",
    )

    bucket._set_cached_max_capacity_override(None)
    bucket._max_capacity_cache_time = time.time()
    redis_client.store[bucket._max_capacity_key] = _redis_override_payload(200.0)
    redis_client.store[bucket._last_checked_key] = 70.0
    redis_client.store[bucket._capacity_key] = 0.0

    backend._snapshot_bucket_state(bucket)

    assert redis_client.get_calls == [bucket._max_capacity_key]
    assert bucket.max_capacity == pytest.approx(200.0)
    assert redis_client.store[bucket._capacity_key] == pytest.approx(100.0)
