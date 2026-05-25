"""Regression coverage for Redis multi-tenant namespace isolation."""

import json

import pytest
from pydantic import ValidationError

pytest.importorskip("redis", reason="redis package not installed")

import redis as _sync_redis
import redis.asyncio as _async_redis

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


class _AsyncRedisStore(_async_redis.Redis):
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    async def get(self, key: str) -> object:
        return self.store.get(key)

    async def set(self, key: str, value: object, **kwargs: object) -> bool | None:
        if kwargs.get("nx") and key in self.store:
            return None
        self.store[key] = value
        return True

    async def expire(self, key: str, _seconds: int) -> bool:
        return key in self.store

    async def delete(self, key: str) -> int:
        existed = key in self.store
        self.store.pop(key, None)
        return int(existed)


class _SyncRedisStore(_sync_redis.Redis):
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    def get(self, key: str) -> object:
        return self.store.get(key)

    def set(self, key: str, value: object, **kwargs: object) -> bool | None:
        if kwargs.get("nx") and key in self.store:
            return None
        self.store[key] = value
        return True

    def expire(self, key: str, _seconds: int) -> bool:
        return key in self.store

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


def _multi_quota_config(model_family: str, metric_suffix: str) -> PerModelConfig:
    return PerModelConfig(
        model_family=model_family,
        quotas=UsageQuotas(
            [
                Quota(metric=f"tokens-{metric_suffix}", limit=100.0, per_seconds=60),
                Quota(metric=f"requests-{metric_suffix}", limit=10.0, per_seconds=1),
            ]
        ),
    )


def _redis_keys_for_backend_bucket(bucket) -> set[str]:
    bucket_keys = {
        bucket.full_redis_key,
        bucket._capacity_key,
        bucket._last_checked_key,
        bucket._lock_key,
        bucket._max_capacity_key,
    }
    legacy_key = getattr(bucket, "_legacy_max_capacity_key", None)
    if legacy_key is not None:
        bucket_keys.add(legacy_key)
    return bucket_keys


def _redis_keys_for_backend(backend) -> set[str]:
    keys: set[str] = set()
    for bucket in backend.sorted_buckets:
        keys.update(_redis_keys_for_backend_bucket(bucket))
    return keys


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
def test_stock_redis_builders_produce_non_colliding_keys_for_valid_inputs(
    builder_cls: type[RedisBackendBuilder] | type[SyncRedisBackendBuilder],
    client: _AsyncRedisStore | _SyncRedisStore,
) -> None:
    configs = [
        _multi_quota_config("gpt-4o", "primary"),
        _multi_quota_config("gpt-4o-mini", "secondary"),
        _multi_quota_config("anthropic/claude-3.5", "tertiary"),
    ]
    prefixes = ["tenant-a", "tenant-b"]

    key_to_owner: dict[str, str] = {}
    keys_per_bucket: int | None = None
    for prefix in prefixes:
        builder = builder_cls(client, key_prefix=prefix)
        for cfg in configs:
            backend = builder.build(cfg)
            owner = f"{prefix}/{cfg.model_family}"
            for bucket in backend.sorted_buckets:
                bucket_keys = _redis_keys_for_backend_bucket(bucket)
                if keys_per_bucket is None:
                    keys_per_bucket = len(bucket_keys)
                for key in bucket_keys:
                    assert key not in key_to_owner, (
                        f"{key!r} generated by both {key_to_owner[key]} and {owner}"
                    )
                    key_to_owner[key] = owner

    assert keys_per_bucket is not None
    expected_bucket_key_count = len(prefixes) * len(configs) * 2 * keys_per_bucket
    assert len(key_to_owner) == expected_bucket_key_count


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
