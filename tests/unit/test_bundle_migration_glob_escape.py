"""Regression coverage for Redis glob-metacharacter escaping in bucket cleanup.

``cleanup_legacy_buckets``/``async_cleanup_legacy_buckets`` build a Redis
``SCAN ... MATCH`` pattern from the caller's ``key_prefix``. Redis MATCH
patterns are glob-style: unescaped ``*``, ``?``, ``[``, ``]`` are
metacharacters. A ``key_prefix`` that legitimately contains one of those
characters must not turn into a wildcard that shadows a sibling deployment's
keys. These tests run against a real Redis instance because the escaping
contract is about Redis's own glob matching, not a Python approximation of it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("redis", reason="redis package not installed")

import redis as _sync_redis
import redis.asyncio as _async_redis
from redis.exceptions import RedisError

from token_throttle.migration import (
    async_cleanup_legacy_buckets,
    cleanup_legacy_buckets,
)

_REDIS_URL = "redis://localhost:6379"

# The "attacker" prefix legitimately contains a literal Redis glob
# metacharacter; unescaped, its scan pattern shadows the "victim" prefix's
# keys too (both prefixes are fixlane2-namespaced test fixtures, not real
# deployments).
_VICTIM_PREFIX = "fixlane2-wildcard-fixes-victim"
_ATTACKER_PREFIX = "fixlane2-wildcard-fixes-vict*"


def _legacy_last_checked_key(key_prefix: str) -> str:
    return f"{key_prefix}:rate_limiting:bucket:fam:tokens:60:last_checked"


@pytest.fixture
def sync_client():
    client = _sync_redis.from_url(_REDIS_URL)
    try:
        client.ping()
    except RedisError as exc:
        client.close()
        pytest.skip(f"Redis unavailable at {_REDIS_URL}: {exc}")
    created_keys: set[str] = set()
    try:
        yield client, created_keys
    finally:
        remaining = [key for key in created_keys if client.exists(key)]
        if remaining:
            client.delete(*remaining)
        client.close()


@pytest.fixture
async def async_client():
    client = _async_redis.from_url(_REDIS_URL)
    try:
        await client.ping()
    except RedisError as exc:
        await client.aclose()
        pytest.skip(f"Redis unavailable at {_REDIS_URL}: {exc}")
    created_keys: set[str] = set()
    try:
        yield client, created_keys
    finally:
        remaining = [key for key in created_keys if await client.exists(key)]
        if remaining:
            await client.delete(*remaining)
        await client.aclose()


def test_cleanup_legacy_buckets_does_not_shadow_sibling_prefix(sync_client) -> None:
    client, created_keys = sync_client
    victim_key = _legacy_last_checked_key(_VICTIM_PREFIX)
    attacker_key = _legacy_last_checked_key(_ATTACKER_PREFIX)
    created_keys.update({victim_key, attacker_key})
    client.set(victim_key, "0")
    client.set(attacker_key, "0")

    deleted = cleanup_legacy_buckets(client, _ATTACKER_PREFIX)

    assert deleted == 1
    assert client.exists(attacker_key) == 0
    assert client.exists(victim_key) == 1


async def test_async_cleanup_legacy_buckets_does_not_shadow_sibling_prefix(
    async_client,
) -> None:
    client, created_keys = async_client
    victim_key = _legacy_last_checked_key(_VICTIM_PREFIX)
    attacker_key = _legacy_last_checked_key(_ATTACKER_PREFIX)
    created_keys.update({victim_key, attacker_key})
    await client.set(victim_key, "0")
    await client.set(attacker_key, "0")

    deleted = await async_cleanup_legacy_buckets(client, _ATTACKER_PREFIX)

    assert deleted == 1
    assert await client.exists(attacker_key) == 0
    assert await client.exists(victim_key) == 1
