"""
Run the Redis backends under the public conformance suite.

``tests/conformance/test_backend_conformance.py`` runs the memory and SQLite
backends under ``conformance_test_for`` / ``sync_conformance_test_for`` but the
Redis backends were never exercised by the public suite. This module closes
that gap using the repository's ``--redis-url`` option, a disposable key prefix
and prefix-scoped cleanup (no ``flushdb``).
"""

from __future__ import annotations

import uuid

import pytest

from token_throttle import conformance_test_for, sync_conformance_test_for


def _redis_url(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--redis-url"))


def _delete_prefix_sync(client, prefix: str) -> None:
    cursor = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, match=f"{prefix}:*", count=500)
        if keys:
            client.delete(*keys)
        if cursor == 0:
            break


async def test_async_redis_backend_passes_public_conformance_suite(
    request: pytest.FixtureRequest,
) -> None:
    redis_async = pytest.importorskip("redis.asyncio")
    sync_redis = pytest.importorskip("redis")
    redis_exceptions = pytest.importorskip("redis.exceptions")
    from token_throttle._limiter_backends._redis._backend import (  # noqa: PLC0415
        RedisBackendBuilder,
    )

    url = _redis_url(request)
    client = redis_async.from_url(url)
    try:
        await client.ping()
    except redis_exceptions.RedisError as exc:
        await client.aclose()
        pytest.skip(f"Redis unavailable at {url}: {exc}")
    prefix = f"conformance-async-{uuid.uuid4().hex}"
    try:
        await conformance_test_for(
            RedisBackendBuilder(client, key_prefix=prefix, sleep_interval=0.01)
        )
    finally:
        await client.aclose()
        cleanup = sync_redis.from_url(url)
        try:
            _delete_prefix_sync(cleanup, prefix)
        finally:
            cleanup.close()


def test_sync_redis_backend_passes_public_conformance_suite(
    request: pytest.FixtureRequest,
) -> None:
    sync_redis = pytest.importorskip("redis")
    redis_exceptions = pytest.importorskip("redis.exceptions")
    from token_throttle._limiter_backends._redis._sync_backend import (  # noqa: PLC0415
        SyncRedisBackendBuilder,
    )

    url = _redis_url(request)
    client = sync_redis.from_url(url)
    try:
        client.ping()
    except redis_exceptions.RedisError as exc:
        client.close()
        pytest.skip(f"Redis unavailable at {url}: {exc}")
    prefix = f"conformance-sync-{uuid.uuid4().hex}"
    try:
        sync_conformance_test_for(
            SyncRedisBackendBuilder(client, key_prefix=prefix, sleep_interval=0.01)
        )
    finally:
        cleanup = sync_redis.from_url(url)
        try:
            _delete_prefix_sync(cleanup, prefix)
        finally:
            cleanup.close()
        client.close()
