import asyncio
import threading
import time

import pytest

pytest.importorskip("redis")

import redis as sync_redis
import redis.asyncio as redis

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

MODEL = "redis-close-refund-model"
MODEL_FAMILY = "redis-close-refund-family"


def _config(*, limit: float, per_seconds: int = 3600) -> PerModelConfig:
    return PerModelConfig(
        model_family=MODEL_FAMILY,
        quotas=UsageQuotas(
            [Quota(metric="requests", limit=limit, per_seconds=per_seconds)]
        ),
    )


async def _async_capacity(observer: redis.Redis, key: str) -> float:
    raw = await observer.get(key)
    assert raw is not None
    return float(raw)


def _sync_capacity(observer: sync_redis.Redis, key: str) -> float:
    raw = observer.get(key)
    assert raw is not None
    return float(raw)


@pytest.mark.redis
async def test_p5_02_aclose_waits_for_in_flight_redis_refunds(redis_url: str):
    owned_client = redis.from_url(redis_url)
    observer = redis.from_url(redis_url)
    await observer.flushdb()
    count = 25
    limiter = RateLimiter(
        _config(limit=float(count)),
        backend=RedisBackendBuilder(
            owned_client,
            key_prefix="v8-p5-02",
            owns_redis_client=True,
        ),
        close_drain_timeout_seconds=2.0,
    )
    backend = await limiter._get_backend(_config(limit=float(count)))
    bucket = backend.sorted_buckets[0]
    reservations = [
        await limiter.acquire_capacity({"requests": 1.0}, MODEL, timeout=0)
        for _ in range(count)
    ]
    original_refund = backend.refund_capacity_for_buckets
    entered = 0
    all_entered = asyncio.Event()
    release_refunds = asyncio.Event()

    async def slow_refund(*args: object, **kwargs: object) -> None:
        nonlocal entered
        entered += 1
        if entered == count:
            all_entered.set()
        await release_refunds.wait()
        await original_refund(*args, **kwargs)

    backend.refund_capacity_for_buckets = slow_refund
    refund_tasks = [
        asyncio.create_task(limiter.refund_capacity({"requests": 0.0}, reservation))
        for reservation in reservations
    ]
    try:
        await asyncio.wait_for(all_entered.wait(), timeout=2.0)
        close_task = asyncio.create_task(limiter.aclose())
        await asyncio.sleep(0.05)
        assert not close_task.done()
        release_refunds.set()
        await asyncio.wait_for(asyncio.gather(*refund_tasks), timeout=2.0)
        await asyncio.wait_for(close_task, timeout=2.0)
        assert await _async_capacity(observer, bucket._capacity_key) == pytest.approx(
            float(count),
            abs=1.0,
        )
    finally:
        release_refunds.set()
        await observer.flushdb()
        await observer.aclose()


@pytest.mark.redis
def test_p5_02_close_waits_for_in_flight_sync_redis_refunds(redis_url: str):
    owned_client = sync_redis.from_url(redis_url)
    observer = sync_redis.from_url(redis_url)
    observer.flushdb()
    count = 25
    limiter = SyncRateLimiter(
        _config(limit=float(count)),
        backend=SyncRedisBackendBuilder(
            owned_client,
            key_prefix="v8-p5-02-sync",
            owns_redis_client=True,
        ),
        close_drain_timeout_seconds=2.0,
    )
    backend = limiter._get_backend(_config(limit=float(count)))
    bucket = backend.sorted_buckets[0]
    reservations = [
        limiter.acquire_capacity({"requests": 1.0}, MODEL, timeout=0)
        for _ in range(count)
    ]
    original_refund = backend.refund_capacity_for_buckets
    entered = 0
    all_entered = threading.Event()
    release_refunds = threading.Event()

    def slow_refund(*args: object, **kwargs: object) -> None:
        nonlocal entered
        entered += 1
        if entered == count:
            all_entered.set()
        release_refunds.wait()
        original_refund(*args, **kwargs)

    backend.refund_capacity_for_buckets = slow_refund
    errors: list[BaseException] = []

    def refund_worker(reservation) -> None:
        try:
            limiter.refund_capacity({"requests": 0.0}, reservation)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=refund_worker, args=(res,)) for res in reservations
    ]
    for thread in threads:
        thread.start()
    try:
        assert all_entered.wait(timeout=2.0)
        close_thread = threading.Thread(target=limiter.close)
        close_thread.start()
        time.sleep(0.05)
        assert close_thread.is_alive()
        release_refunds.set()
        for thread in threads:
            thread.join(timeout=2.0)
            assert not thread.is_alive()
        close_thread.join(timeout=2.0)
        assert not close_thread.is_alive()
        assert errors == []
        assert _sync_capacity(observer, bucket._capacity_key) == pytest.approx(
            float(count),
            abs=1.0,
        )
    finally:
        release_refunds.set()
        observer.flushdb()
        observer.close()
