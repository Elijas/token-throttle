"""Regression coverage for mandatory Redis bucket-state TTLs."""

import math

import pytest

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder
from token_throttle._limiter_backends._redis._bucket import RedisBucket
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)


def _redis_get_value(value: object) -> object:
    if value is None or isinstance(value, (bytes, str)):
        return value
    return str(value)


class _FakeAsyncPipeline:
    def __init__(self, redis: "_FakeAsyncRedis") -> None:
        self._redis = redis
        self._commands: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def get(self, key: str) -> None:
        self._commands.append(("get", (key,), {}))

    def set(self, key: str, value: object, **kwargs: object) -> None:
        self._commands.append(("set", (key, value), kwargs))

    def expire(self, key: str, seconds: int) -> None:
        self._commands.append(("expire", (key, seconds), {}))

    async def execute(self) -> list[object]:
        results = []
        for name, args, kwargs in self._commands:
            method = getattr(self._redis, name)
            results.append(await method(*args, **kwargs))
        self._commands.clear()
        return results


class _FakeAsyncRedis:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.deadlines: dict[str, float | None] = {}
        self.now = 1000.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def _purge_if_expired(self, key: str) -> None:
        deadline = self.deadlines.get(key)
        if deadline is not None and deadline <= self.now:
            self.store.pop(key, None)
            self.deadlines.pop(key, None)

    async def get(self, key: str) -> object:
        self._purge_if_expired(key)
        return _redis_get_value(self.store.get(key))

    async def set(
        self,
        key: str,
        value: object,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        self._purge_if_expired(key)
        if nx and key in self.store:
            return None
        self.store[key] = value
        self.deadlines[key] = None if ex is None else self.now + ex
        return True

    async def expire(self, key: str, seconds: int) -> bool:
        self._purge_if_expired(key)
        if key not in self.store:
            return False
        self.deadlines[key] = self.now + seconds
        return True

    async def delete(self, key: str) -> int:
        self._purge_if_expired(key)
        existed = key in self.store
        self.store.pop(key, None)
        self.deadlines.pop(key, None)
        return int(existed)

    async def ttl(self, key: str) -> int:
        self._purge_if_expired(key)
        if key not in self.store:
            return -2
        deadline = self.deadlines.get(key)
        if deadline is None:
            return -1
        return max(0, math.ceil(deadline - self.now))

    def pipeline(self) -> _FakeAsyncPipeline:
        return _FakeAsyncPipeline(self)


class _FakeSyncRedis:
    pass


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="test/model",
        quotas=UsageQuotas([Quota(metric="tokens", limit=100.0, per_seconds=60)]),
    )


def _bucket(redis_client: _FakeAsyncRedis, *, ttl: int = 10) -> RedisBucket:
    quota = next(iter(_config().quotas))
    return RedisBucket(
        quota=quota,
        limit_config=_config(),
        redis_client=redis_client,
        key_prefix="test",
        bucket_ttl_seconds=ttl,
    )


@pytest.mark.parametrize("ttl", [0, -1])
def test_async_redis_builder_rejects_non_positive_bucket_ttl(ttl: int) -> None:
    with pytest.raises(ValueError, match="bucket_ttl_seconds"):
        RedisBackendBuilder(
            _FakeAsyncRedis(),
            key_prefix="test",
            bucket_ttl_seconds=ttl,
        )


@pytest.mark.parametrize("ttl", [0, -1])
def test_sync_redis_builder_rejects_non_positive_bucket_ttl(ttl: int) -> None:
    with pytest.raises(ValueError, match="bucket_ttl_seconds"):
        SyncRedisBackendBuilder(
            _FakeSyncRedis(),
            key_prefix="test",
            bucket_ttl_seconds=ttl,
        )


async def test_bucket_write_applies_ttl_to_capacity_and_last_checked() -> None:
    redis_client = _FakeAsyncRedis()
    bucket = _bucket(redis_client, ttl=30)

    await bucket.set_capacity(25.0, current_time=redis_client.now)

    assert await redis_client.ttl(bucket._last_checked_key) == 30
    assert await redis_client.ttl(bucket._capacity_key) == 30


async def test_bucket_read_refreshes_existing_state_ttl() -> None:
    redis_client = _FakeAsyncRedis()
    bucket = _bucket(redis_client, ttl=10)
    await bucket.set_capacity(25.0, current_time=redis_client.now)
    redis_client.advance(4.0)

    assert await redis_client.ttl(bucket._capacity_key) == 6

    await bucket.get_capacity(current_time=redis_client.now)

    assert await redis_client.ttl(bucket._last_checked_key) == 10
    assert await redis_client.ttl(bucket._capacity_key) == 10


async def test_schema_version_key_is_exempt_from_ttl() -> None:
    redis_client = _FakeAsyncRedis()
    bucket = _bucket(redis_client, ttl=30)

    await bucket.set_max_capacity(50.0)

    assert await redis_client.ttl(bucket._schema_version_key) == -1
    assert await redis_client.ttl(bucket._max_capacity_key) == 30


async def test_idle_bucket_state_expires_after_ttl() -> None:
    redis_client = _FakeAsyncRedis()
    bucket = _bucket(redis_client, ttl=1)
    await bucket.set_capacity(25.0, current_time=redis_client.now)

    redis_client.advance(1.1)

    assert await redis_client.ttl(bucket._last_checked_key) == -2
    assert await redis_client.ttl(bucket._capacity_key) == -2
