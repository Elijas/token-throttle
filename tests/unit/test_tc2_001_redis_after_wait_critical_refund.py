"""TC2-001: Redis after-wait critical callbacks refund consumed capacity."""

from __future__ import annotations

import asyncio

import pytest
from frozendict import frozendict

from token_throttle import PerModelConfig, Quota, RateLimiterCallbacks, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder

redis_async = pytest.importorskip("redis.asyncio", reason="redis package not installed")

pytestmark = pytest.mark.redis

MODEL_FAMILY = "test-family"

AFTER_WAIT_CRITICAL_EXCEPTIONS = (
    MemoryError,
    RecursionError,
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
)


@pytest.fixture
async def redis_client(request: pytest.FixtureRequest):
    client = redis_async.from_url(request.config.getoption("--redis-url"))
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        pytest.skip(f"Redis server unavailable: {exc!r}")
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family=MODEL_FAMILY,
        quotas=UsageQuotas([Quota(metric="tokens", limit=10.0, per_seconds=10)]),
    )


@pytest.mark.parametrize("exc_type", AFTER_WAIT_CRITICAL_EXCEPTIONS)
async def test_async_redis_after_wait_end_critical_refunds_consumed_capacity(
    redis_client,
    exc_type: type[BaseException],
) -> None:
    exc = exc_type("forced after_wait_end_consumption failure")

    async def after_wait_end_consumption(**_kwargs: object) -> None:
        raise exc

    backend = RedisBackendBuilder(
        redis_client,
        key_prefix="tc2-001",
        sleep_interval=0.01,
    ).build(
        _config(),
        callbacks=RateLimiterCallbacks(
            after_wait_end_consumption=after_wait_end_consumption
        ),
    )

    await backend.await_for_capacity(frozendict({"tokens": 10.0}), timeout=0)

    with pytest.raises(exc_type) as raised:
        await backend.await_for_capacity(frozendict({"tokens": 1.0}), timeout=2)

    assert raised.value is exc

    await asyncio.wait_for(
        backend.await_for_capacity(frozendict({"tokens": 1.0}), timeout=0),
        timeout=1,
    )
