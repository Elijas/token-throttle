import pytest

from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)


@pytest.fixture
def redis_url(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--redis-url")


@pytest.fixture
async def redis_client(redis_url: str):
    redis = pytest.importorskip("redis.asyncio", reason="redis package not installed")

    client = redis.from_url(redis_url)
    try:
        await client.ping()
    except redis.exceptions.RedisError as exc:
        await client.aclose()
        pytest.skip(f"Redis unavailable at {redis_url}: {exc}")
    try:
        await client.flushdb()
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


@pytest.fixture(params=["redis", "memory"])
def backend_builder(request: pytest.FixtureRequest):
    """Parameterized backend builder — runs tests against all backends."""
    if request.param == "redis":
        pytest.importorskip("redis", reason="redis package not installed")
        from token_throttle._limiter_backends._redis._backend import (  # noqa: PLC0415
            RedisBackendBuilder,
        )

        redis_client = request.getfixturevalue("redis_client")
        return RedisBackendBuilder(redis_client, key_prefix="test")
    if request.param == "memory":
        return MemoryBackendBuilder()
    raise ValueError(f"Unknown backend: {request.param}")


@pytest.fixture
def sync_redis_client(redis_url: str):
    sync_redis = pytest.importorskip("redis", reason="redis package not installed")

    client = sync_redis.from_url(redis_url)
    try:
        client.ping()
    except sync_redis.exceptions.RedisError as exc:
        client.close()
        pytest.skip(f"Redis unavailable at {redis_url}: {exc}")
    try:
        client.flushdb()
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
        pytest.importorskip("redis", reason="redis package not installed")
        from token_throttle._limiter_backends._redis._sync_backend import (  # noqa: PLC0415
            SyncRedisBackendBuilder,
        )

        client = request.getfixturevalue("sync_redis_client")
        return SyncRedisBackendBuilder(client, key_prefix="test")
    raise ValueError(f"Unknown backend: {request.param}")
