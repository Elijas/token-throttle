"""Regression tests for FIX-49 SETMAX-RECONCILE-PARITY."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._rate_limiter import RateLimiter

try:
    from token_throttle._limiter_backends._redis._bucket import RedisBucket
    from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket
except ImportError:
    RedisBucket = None
    SyncRedisBucket = None

MODEL = "test-model"
MODEL_FAMILY = "test-family"
BUCKET_ID = ("tokens", 60)


def _config(limit: float = 100.0) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=limit, per_seconds=60)]),
        model_family=MODEL_FAMILY,
    )


def _runtime_override(limiter: RateLimiter) -> float | None:
    return limiter._model_family_to_runtime_max_capacity.get(MODEL_FAMILY, {}).get(
        BUCKET_ID
    )


def _redis_bucket_config() -> tuple[Quota, PerModelConfig]:
    quota = Quota(metric="tokens", limit=100.0, per_seconds=60)
    cfg = PerModelConfig(
        quotas=UsageQuotas([quota]),
        model_family="redis-family",
    )
    return quota, cfg


async def test_async_set_max_commits_when_success_readback_would_fail() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    await limiter.acquire_capacity({"tokens": 1}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]

    async def readback_fails(_metric: str, _per_seconds: int) -> float:
        raise RuntimeError("readback unavailable")

    backend._runtime_max_capacity_for_reconciliation = readback_fails

    await limiter.set_max_capacity(MODEL, "tokens", 60, 50.0)

    assert _runtime_override(limiter) == pytest.approx(50.0)


async def test_async_set_max_cancelled_backend_write_reconciles_override() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    await limiter.acquire_capacity({"tokens": 1}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    original_set_max_capacity = backend.set_max_capacity

    async def write_then_cancel(metric: str, per_seconds: int, value: float) -> None:
        await original_set_max_capacity(metric, per_seconds, value)
        raise asyncio.CancelledError

    backend.set_max_capacity = write_then_cancel

    with pytest.raises(asyncio.CancelledError):
        await limiter.set_max_capacity(MODEL, "tokens", 60, 50.0)

    assert _runtime_override(limiter) == pytest.approx(50.0)


async def test_async_redis_post_write_readback_failure_does_not_abort_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if RedisBucket is None:
        pytest.skip("redis package not installed")
    quota, cfg = _redis_bucket_config()
    redis_client = MagicMock()
    redis_client.set = AsyncMock(return_value=True)
    redis_client.get.side_effect = RuntimeError("readback unavailable")
    bucket = RedisBucket(quota, cfg, redis_client, key_prefix="test")

    def cache_update_fails(_value: float | None) -> None:
        raise RuntimeError("cache repair failed")

    monkeypatch.setattr(bucket, "_set_cached_max_capacity_override", cache_update_fails)

    await bucket.set_max_capacity(50.0)

    assert redis_client.set.call_count == 2


@pytest.mark.parametrize(
    "payload",
    [
        b"50.0",
        json.dumps({"override_max_capacity": 50.0}).encode(),
    ],
)
async def test_async_redis_bad_legacy_override_does_not_refresh_ttl(
    payload: bytes,
) -> None:
    if RedisBucket is None:
        pytest.skip("redis package not installed")
    quota, cfg = _redis_bucket_config()
    redis_client = MagicMock()
    redis_client.get.return_value = payload
    bucket = RedisBucket(quota, cfg, redis_client, key_prefix="test")

    assert await bucket.refresh_max_capacity_from_redis() == pytest.approx(100.0)
    redis_client.expire.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        b"50.0",
        json.dumps({"override_max_capacity": 50.0}).encode(),
    ],
)
def test_sync_redis_bad_legacy_override_does_not_refresh_ttl(payload: bytes) -> None:
    if SyncRedisBucket is None:
        pytest.skip("redis package not installed")
    quota, cfg = _redis_bucket_config()
    redis_client = MagicMock()
    redis_client.get.return_value = payload
    bucket = SyncRedisBucket(quota, cfg, redis_client, key_prefix="test")

    assert bucket.refresh_max_capacity_from_redis() == pytest.approx(100.0)
    redis_client.expire.assert_not_called()
