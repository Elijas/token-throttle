import pytest
import redis.asyncio as redis

from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder


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


@pytest.fixture(params=["redis"])
def backend_builder(request: pytest.FixtureRequest, redis_client: redis.Redis):
    """Parameterized backend builder — add future backends here."""
    if request.param == "redis":
        return RedisBackendBuilder(redis_client)
    raise ValueError(f"Unknown backend: {request.param}")
