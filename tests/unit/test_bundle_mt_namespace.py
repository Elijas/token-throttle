"""Regression coverage for Redis multi-tenant namespace isolation."""

import json

import pytest
from pydantic import ValidationError

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._factories._openai._openai_rate_limiter import (
    create_openai_redis_rate_limiter,
)
from token_throttle._factories._openai._openai_sync_rate_limiter import (
    create_openai_redis_sync_rate_limiter,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)


class _AsyncRedisStore:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    async def get(self, key: str) -> object:
        return self.store.get(key)

    async def set(self, key: str, value: object, **kwargs: object) -> bool | None:
        if kwargs.get("nx") and key in self.store:
            return None
        self.store[key] = value
        return True

    async def delete(self, key: str) -> int:
        existed = key in self.store
        self.store.pop(key, None)
        return int(existed)


class _SyncRedisStore:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    def get(self, key: str) -> object:
        return self.store.get(key)

    def set(self, key: str, value: object, **kwargs: object) -> bool | None:
        if kwargs.get("nx") and key in self.store:
            return None
        self.store[key] = value
        return True

    def delete(self, key: str) -> int:
        existed = key in self.store
        self.store.pop(key, None)
        return int(existed)


def _config(limit: float = 100.0) -> PerModelConfig:
    quota = Quota(metric="tokens", limit=limit, per_seconds=60)
    return PerModelConfig(
        model_family="gpt-4",
        quotas=UsageQuotas([quota]),
    )


def test_async_redis_builder_requires_key_prefix() -> None:
    with pytest.raises(TypeError, match="key_prefix"):
        RedisBackendBuilder(_AsyncRedisStore())


def test_sync_redis_builder_requires_key_prefix() -> None:
    with pytest.raises(TypeError, match="key_prefix"):
        SyncRedisBackendBuilder(_SyncRedisStore())


def test_openai_redis_factories_require_key_prefix() -> None:
    with pytest.raises(TypeError, match="key_prefix"):
        create_openai_redis_rate_limiter(_AsyncRedisStore(), rpm=1, tpm=1)
    with pytest.raises(TypeError, match="key_prefix"):
        create_openai_redis_sync_rate_limiter(_SyncRedisStore(), rpm=1, tpm=1)


@pytest.mark.parametrize(
    "builder_cls, client",
    [
        (RedisBackendBuilder, _AsyncRedisStore()),
        (SyncRedisBackendBuilder, _SyncRedisStore()),
    ],
)
@pytest.mark.parametrize("key_prefix", ["", " ", "a:b", "{a}", "control\x00char"])
def test_redis_builders_reject_invalid_key_prefixes(
    builder_cls: type[RedisBackendBuilder] | type[SyncRedisBackendBuilder],
    client: _AsyncRedisStore | _SyncRedisStore,
    key_prefix: str,
) -> None:
    with pytest.raises(ValueError, match="key_prefix"):
        builder_cls(client, key_prefix=key_prefix)


@pytest.mark.parametrize("model_family", ["x{y}", "{x}y"])
def test_model_family_rejects_cluster_hash_tag_braces(model_family: str) -> None:
    with pytest.raises(ValidationError, match="hash tag"):
        PerModelConfig(
            model_family=model_family,
            quotas=UsageQuotas([Quota(metric="tokens", limit=100.0)]),
        )


def test_two_prefixes_build_distinct_bucket_keys_for_same_family() -> None:
    redis_client = _AsyncRedisStore()
    cfg = _config()

    backend_a = RedisBackendBuilder(redis_client, key_prefix="tenant-a").build(cfg)
    backend_b = RedisBackendBuilder(redis_client, key_prefix="tenant-b").build(cfg)

    bucket_a = backend_a.sorted_buckets[0]
    bucket_b = backend_b.sorted_buckets[0]

    assert bucket_a.full_redis_key == "tenant-a:rate_limiting:bucket:gpt-4:tokens:60"
    assert bucket_b.full_redis_key == "tenant-b:rate_limiting:bucket:gpt-4:tokens:60"
    assert bucket_a._capacity_key != bucket_b._capacity_key
    assert bucket_a._lock_key != bucket_b._lock_key
    assert bucket_a._max_capacity_key != bucket_b._max_capacity_key


async def test_set_max_capacity_is_scoped_by_key_prefix() -> None:
    redis_client = _AsyncRedisStore()
    cfg = _config()

    backend_a = RedisBackendBuilder(redis_client, key_prefix="tenant-a").build(cfg)
    backend_b = RedisBackendBuilder(redis_client, key_prefix="tenant-b").build(cfg)
    bucket_a = backend_a.sorted_buckets[0]
    bucket_b = backend_b.sorted_buckets[0]

    await bucket_a.set_max_capacity(50.0)

    assert bucket_a._max_capacity_key in redis_client.store
    assert bucket_b._max_capacity_key not in redis_client.store
    assert json.loads(redis_client.store[bucket_a._max_capacity_key]) == {
        "configured_max_capacity": 100.0,
        "override_max_capacity": 50.0,
    }
    assert await bucket_b.get_max_capacity() == pytest.approx(100.0)


def test_sync_builder_uses_the_same_namespaced_bucket_shape() -> None:
    backend = SyncRedisBackendBuilder(
        _SyncRedisStore(),
        key_prefix="tenant-a",
    ).build(_config())

    bucket = backend.sorted_buckets[0]

    assert bucket.full_redis_key == "tenant-a:rate_limiting:bucket:gpt-4:tokens:60"
    assert bucket._schema_version_key == "tenant-a:rate_limiting:schema_version"
