"""Regression coverage for Redis pipeline defensive validation."""

# ruff: noqa: E402

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

redis = pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis import _backend as redis_backend_module
from token_throttle._limiter_backends._redis._backend import RedisBackend
from token_throttle._limiter_backends._redis._bucket import (
    MaxCapacityOverrideParseError,
    RedisBucket,
    RedisPipelineResultError,
)
from token_throttle._limiter_backends._redis._sync_backend import SyncRedisBackend
from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket


@pytest.fixture
def limit_config() -> PerModelConfig:
    return PerModelConfig(
        model_family="test/model",
        quotas=UsageQuotas([Quota(metric="requests", limit=20, per_seconds=60)]),
    )


@pytest.fixture
def quota(limit_config: PerModelConfig) -> Quota:
    return next(iter(limit_config.quotas))


class AsyncPipeline:
    def __init__(self, result: object = None, exc: BaseException | None = None) -> None:
        self.result = result
        self.exc = exc

    def get(self, _key: str) -> None:
        return None

    def expire(self, _key: str, _seconds: int) -> None:
        return None

    async def execute(self) -> object:
        if self.exc is not None:
            raise self.exc
        return self.result


class SyncPipeline:
    def __init__(self, result: object = None, exc: BaseException | None = None) -> None:
        self.result = result
        self.exc = exc

    def get(self, _key: str) -> None:
        return None

    def expire(self, _key: str, _seconds: int) -> None:
        return None

    def execute(self) -> object:
        if self.exc is not None:
            raise self.exc
        return self.result


async def test_partial_none_bucket_state_is_normalized_to_cold_start(
    quota: Quota, limit_config: PerModelConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis_client = AsyncMock()
    bucket = RedisBucket(quota, limit_config, redis_client, key_prefix="test")
    backend = RedisBackend([bucket], redis_client, limit_config, key_prefix="test")
    pipeline = AsyncPipeline(result=[b"1000.0", None, False, False, None, False])
    seen: list[tuple[object, object]] = []

    def capture_calculate_capacity(
        last_checked: object, capacity: object, current_time: float
    ):
        seen.append((last_checked, capacity))
        return original_calculate_capacity(last_checked, capacity, current_time)

    original_calculate_capacity = bucket.calculate_capacity
    monkeypatch.setattr(bucket, "calculate_capacity", capture_calculate_capacity)

    result = await backend._get_capacities_unsafe(
        pipeline=pipeline,
        current_time=1234.0,
    )

    assert seen == [(None, None)]
    assert result.capacities[("requests", 60)] == pytest.approx(20.0)
    assert result.fresh_start_buckets == [bucket]


def test_update_max_capacity_from_result_rejects_non_canonical_input(
    quota: Quota, limit_config: PerModelConfig
) -> None:
    bucket = RedisBucket(quota, limit_config, AsyncMock(), key_prefix="test")

    bucket.update_max_capacity_from_result(b"not-json")
    assert bucket.max_capacity == pytest.approx(20.0)

    with pytest.raises(MaxCapacityOverrideParseError, match="must be bytes"):
        bucket.update_max_capacity_from_result(42)


@pytest.mark.parametrize("override_value", [True, "10.0"])
def test_json_override_rejects_bool_and_numeric_string(
    quota: Quota, limit_config: PerModelConfig, override_value: object
) -> None:
    bucket = RedisBucket(quota, limit_config, AsyncMock(), key_prefix="test")
    payload = json.dumps(
        {
            "configured_max_capacity": 20.0,
            "override_max_capacity": override_value,
        }
    ).encode()

    bucket.update_max_capacity_from_result(payload)
    assert bucket.max_capacity == pytest.approx(20.0)


async def test_pipeline_response_error_is_translated(
    quota: Quota, limit_config: PerModelConfig
) -> None:
    redis_client = AsyncMock()
    bucket = RedisBucket(quota, limit_config, redis_client, key_prefix="test")
    backend = RedisBackend([bucket], redis_client, limit_config, key_prefix="test")
    pipeline = AsyncPipeline(
        exc=redis.exceptions.ResponseError("WRONGTYPE invalid key type")
    )

    with pytest.raises(RedisPipelineResultError, match="Redis pipeline failed"):
        await backend._get_capacities_unsafe(pipeline=pipeline, current_time=1234.0)


async def test_snapshot_bucket_state_logs_hostile_data(
    quota: Quota,
    limit_config: PerModelConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fake_server_time(_redis) -> float:
        return 1234.0

    redis_client = MagicMock()
    redis_client.pipeline.return_value = AsyncPipeline(
        result=[b"abc", b"1.0", True, True]
    )
    bucket = RedisBucket(quota, limit_config, redis_client, key_prefix="test")
    bucket.get_max_capacity = AsyncMock(return_value=bucket.max_capacity)
    bucket.refresh_max_capacity_from_redis = AsyncMock(return_value=bucket.max_capacity)
    backend = RedisBackend([bucket], redis_client, limit_config, key_prefix="test")
    monkeypatch.setattr(redis_backend_module, "async_server_time", fake_server_time)

    with caplog.at_level("WARNING", logger="token_throttle"):
        await backend._snapshot_bucket_state(bucket)

    assert "snapshot skipped due to unparseable Redis state" in caplog.text


def test_sync_pipeline_response_error_is_translated(
    quota: Quota, limit_config: PerModelConfig
) -> None:
    redis_client = MagicMock()
    bucket = SyncRedisBucket(quota, limit_config, redis_client, key_prefix="test")
    backend = SyncRedisBackend([bucket], redis_client, limit_config, key_prefix="test")
    pipeline = SyncPipeline(
        exc=redis.exceptions.ResponseError("WRONGTYPE invalid key type")
    )

    with pytest.raises(RuntimeError, match="Redis pipeline failed"):
        backend._get_capacities_unsafe(pipeline=pipeline, current_time=1234.0)
