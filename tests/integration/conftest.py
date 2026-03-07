import pytest
import redis as sync_redis
import redis.asyncio as redis

from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)


@pytest.fixture
def redis_url(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--redis-url")


@pytest.fixture
async def redis_client(redis_url: str):
    client = redis.from_url(redis_url)
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


@pytest.fixture(params=["redis", "memory"])
def backend_builder(request: pytest.FixtureRequest):
    """Parameterized backend builder — runs tests against all backends."""
    if request.param == "redis":
        redis_client = request.getfixturevalue("redis_client")
        return RedisBackendBuilder(redis_client)
    if request.param == "memory":
        return MemoryBackendBuilder()
    raise ValueError(f"Unknown backend: {request.param}")


@pytest.fixture
def sync_redis_client(redis_url: str):
    client = sync_redis.from_url(redis_url)
    try:
        yield client
    finally:
        client.flushdb()
        client.close()


@pytest.fixture(params=["memory", "redis"])
def sync_backend_builder(request: pytest.FixtureRequest):
    """Parameterized sync backend builder — runs tests against all sync backends."""
    if request.param == "memory":
        return SyncMemoryBackendBuilder()
    if request.param == "redis":
        client = request.getfixturevalue("sync_redis_client")
        return SyncRedisBackendBuilder(client)
    raise ValueError(f"Unknown backend: {request.param}")
