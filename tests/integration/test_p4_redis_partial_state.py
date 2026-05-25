import pytest

pytest.importorskip("redis")

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder
from token_throttle._rate_limiter import RateLimiter

MODEL = "redis-partial-state-model"
MODEL_FAMILY = "redis-partial-state-family"


def _config(*, limit: float = 10.0, per_seconds: int = 3600) -> PerModelConfig:
    return PerModelConfig(
        model_family=MODEL_FAMILY,
        quotas=UsageQuotas(
            [Quota(metric="requests", limit=limit, per_seconds=per_seconds)]
        ),
    )


async def _async_capacity(redis_client, key: str) -> float:
    raw = await redis_client.get(key)
    assert raw is not None
    return float(raw)


@pytest.mark.redis
async def test_p4_redis_01_partial_state_blocks_acquire_and_reports_missing_key(
    redis_client,
):
    events: list[dict[str, object]] = []

    async def on_missing_consumption_data(**kwargs: object) -> None:
        events.append(kwargs)

    callbacks = RateLimiterCallbacks(
        on_missing_consumption_data=on_missing_consumption_data
    )
    limiter = RateLimiter(
        _config(limit=10.0),
        backend=RedisBackendBuilder(redis_client, key_prefix="v8-p4-01"),
        callbacks=callbacks,
    )
    backend = await limiter._get_backend(_config(limit=10.0))
    bucket = backend.sorted_buckets[0]

    await limiter.acquire_capacity({"requests": 10.0}, MODEL, timeout=0)
    await redis_client.delete(bucket._last_checked_key)

    with pytest.raises(TimeoutError):
        await limiter.acquire_capacity({"requests": 1.0}, MODEL, timeout=0)

    partial_events = [
        event
        for event in events
        if event.get("missing_state_reason") == "partial_state_drained"
    ]
    assert partial_events
    assert partial_events[-1]["usage_metric"] == "requests"
    assert partial_events[-1]["per_seconds"] == 3600
    assert partial_events[-1]["missing_state_keys"] == ("last_checked",)
    assert partial_events[-1]["present_state_keys"] == ("capacity",)
    assert await redis_client.exists(bucket._last_checked_key) == 1
    assert await _async_capacity(redis_client, bucket._capacity_key) == pytest.approx(
        0.0
    )
