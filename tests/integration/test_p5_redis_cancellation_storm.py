import asyncio

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder
from token_throttle._rate_limiter import RateLimiter

MODEL = "redis-cancel-storm-model"
MODEL_FAMILY = "redis-cancel-storm-family"


def _config(*, limit: float = 1.0, per_seconds: int = 3600) -> PerModelConfig:
    return PerModelConfig(
        model_family=MODEL_FAMILY,
        quotas=UsageQuotas(
            [Quota(metric="requests", limit=limit, per_seconds=per_seconds)]
        ),
    )


@pytest.mark.redis
async def test_p5_01_cancelled_redis_waiters_drain_pending_bookkeeping(
    redis_client,
):
    limiter = RateLimiter(
        _config(limit=1.0),
        backend=RedisBackendBuilder(redis_client, key_prefix="v8-p5-01"),
    )
    backend = await limiter._get_backend(_config(limit=1.0))
    cleanup_release = asyncio.Event()
    started = 0
    started_event = asyncio.Event()

    class _LuaRelease:
        async def __call__(self, **_kwargs: object) -> None:
            await cleanup_release.wait()

    class _HangingAcquireLock:
        name = "v8-p5-01:hanging-lock"
        lua_release = _LuaRelease()

        async def acquire(self, **_kwargs: object) -> bool:
            nonlocal started
            started += 1
            if started == 100:
                started_event.set()
            await asyncio.Future()
            return True

    def hanging_lock(**_kwargs: object) -> _HangingAcquireLock:
        return _HangingAcquireLock()

    for bucket in backend.sorted_buckets:
        bucket.lock = hanging_lock

    tasks = [
        asyncio.create_task(limiter.acquire_capacity({"requests": 1.0}, MODEL))
        for _ in range(100)
    ]
    try:
        await asyncio.wait_for(started_event.wait(), timeout=1.0)
        for task in tasks:
            task.cancel()
        done, pending = await asyncio.wait(tasks, timeout=1.0)
        assert not pending
        assert len(done) == 100
        assert limiter._pending_acquire_reservations == set()
        assert all(task.cancelled() for task in done)
    finally:
        cleanup_release.set()
        await asyncio.sleep(0)
