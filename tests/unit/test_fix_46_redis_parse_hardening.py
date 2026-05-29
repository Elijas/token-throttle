"""FIX-46 Redis reply parsing and pipeline slot validation regressions."""

# ruff: noqa: E402

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from frozendict import frozendict

redis = pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis import (
    _backend,
    _server_time,
    _sync_backend,
)
from token_throttle._limiter_backends._redis._backend import RedisBackend
from token_throttle._limiter_backends._redis._bucket import (
    MaxCapacityOverrideParseError,
    RedisBucket,
    RedisPipelineResultError,
    _normalize_bucket_state_pair,
)
from token_throttle._limiter_backends._redis._sync_backend import SyncRedisBackend
from token_throttle._limiter_backends._redis._sync_bucket import (
    RedisPipelineResultError as SyncRedisPipelineResultError,
)
from token_throttle._limiter_backends._redis._sync_bucket import (
    SyncRedisBucket,
)


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="test/model",
        quotas=UsageQuotas([Quota(metric="requests", limit=20, per_seconds=60)]),
    )


def _quota() -> Quota:
    return next(iter(_config().quotas))


def _multi_config() -> PerModelConfig:
    return PerModelConfig(
        model_family="test/model",
        quotas=UsageQuotas(
            [
                Quota(metric="alpha", limit=10, per_seconds=60),
                Quota(metric="beta", limit=20, per_seconds=60),
            ]
        ),
    )


class _AsyncPipeline:
    def __init__(self, result: object) -> None:
        self.result = result

    def get(self, _key: str) -> None:
        return None

    def expire(self, _key: str, _seconds: int) -> None:
        return None

    def set(self, *_args: object, **_kwargs: object) -> None:
        return None

    def delete(self, _key: str) -> None:
        return None

    async def execute(self) -> object:
        return self.result


class _SyncPipeline(_AsyncPipeline):
    def execute(self) -> object:
        return self.result


def _async_backend() -> tuple[RedisBackend, RedisBucket]:
    redis_client = MagicMock()
    redis_client.get = AsyncMock(return_value=None)
    bucket = RedisBucket(_quota(), _config(), redis_client, key_prefix="test")
    return RedisBackend([bucket], redis_client, _config(), key_prefix="test"), bucket


def _sync_backend_pair() -> tuple[SyncRedisBackend, SyncRedisBucket]:
    redis_client = MagicMock()
    redis_client.get.return_value = None
    bucket = SyncRedisBucket(_quota(), _config(), redis_client, key_prefix="test")
    return SyncRedisBackend(
        [bucket], redis_client, _config(), key_prefix="test"
    ), bucket


def _async_multi_backend() -> tuple[RedisBackend, tuple[RedisBucket, ...]]:
    config = _multi_config()
    redis_client = MagicMock()
    redis_client.get = AsyncMock(return_value=None)
    redis_client.expire = AsyncMock(return_value=True)
    buckets = [
        RedisBucket(quota, config, redis_client, key_prefix="test")
        for quota in config.quotas
    ]
    backend = RedisBackend(buckets, redis_client, config, key_prefix="test")
    return backend, tuple(backend.sorted_buckets)


def _sync_multi_backend() -> tuple[SyncRedisBackend, tuple[SyncRedisBucket, ...]]:
    config = _multi_config()
    redis_client = MagicMock()
    redis_client.get.return_value = None
    redis_client.expire.return_value = True
    buckets = [
        SyncRedisBucket(quota, config, redis_client, key_prefix="test")
        for quota in config.quotas
    ]
    backend = SyncRedisBackend(buckets, redis_client, config, key_prefix="test")
    return backend, tuple(backend.sorted_buckets)


def _override_payload(
    bucket: RedisBucket | SyncRedisBucket,
    value: float,
) -> bytes:
    return json.dumps(
        {
            "configured_max_capacity": bucket.configured_max_capacity,
            "override_max_capacity": value,
        }
    ).encode()


async def test_async_read_pipeline_rejects_embedded_expire_error() -> None:
    backend, _bucket = _async_backend()
    pipeline = _AsyncPipeline(
        [b"1000.0", b"4.0", redis.exceptions.ResponseError("EXPIRE failed"), True, None]
    )

    with pytest.raises(RedisPipelineResultError, match="slot 2"):
        await backend._get_capacities_unsafe(pipeline=pipeline, current_time=1001.0)


async def test_async_write_pipeline_rejects_embedded_set_error_without_dedup() -> None:
    backend, _bucket = _async_backend()
    pipeline = _AsyncPipeline(
        [True, redis.exceptions.ResponseError("SET capacity failed")]
    )

    with pytest.raises(RedisPipelineResultError, match="slot 1"):
        await backend._set_capacities_unsafe(
            frozendict({("requests", 60): 5.0}),
            pipeline=pipeline,
            current_time=1001.0,
        )


def test_sync_write_pipeline_rejects_embedded_set_error_without_dedup() -> None:
    backend, _bucket = _sync_backend_pair()
    pipeline = _SyncPipeline([True, redis.exceptions.ResponseError("SET failed")])

    with pytest.raises(RuntimeError, match="slot 1"):
        backend._set_capacities_unsafe(
            frozendict({("requests", 60): 5.0}),
            pipeline=pipeline,
            current_time=1001.0,
        )


async def test_async_get_slots_reject_non_redis_get_python_shapes() -> None:
    backend, _bucket = _async_backend()
    pipeline = _AsyncPipeline([1000, b"4.0", True, True, None])

    with pytest.raises(RedisPipelineResultError, match="unexpected Redis GET"):
        await backend._get_capacities_unsafe(pipeline=pipeline, current_time=1001.0)


@pytest.mark.parametrize(
    ("bad_capacity", "expected_error", "match"),
    [
        (123, RedisPipelineResultError, "unexpected Redis GET"),
        (b"bad", ValueError, "Invalid last_checked"),
    ],
)
async def test_async_multi_bucket_read_validates_all_slots_before_override_cache_update(
    bad_capacity: object,
    expected_error: type[Exception],
    match: str,
) -> None:
    backend, buckets = _async_multi_backend()
    first, _second = buckets
    pipeline = _AsyncPipeline(
        [
            b"1000.0",
            b"5.0",
            True,
            True,
            b"1000.0",
            bad_capacity,
            True,
            True,
            _override_payload(first, 7.0),
            None,
        ]
    )

    with pytest.raises(expected_error, match=match):
        await backend._get_capacities_unsafe(pipeline=pipeline, current_time=1001.0)

    assert first.max_capacity == pytest.approx(first.configured_max_capacity)
    assert first._max_capacity_cached is None
    assert first._max_capacity_cache_populated is False


@pytest.mark.parametrize(
    ("bad_capacity", "expected_error", "match"),
    [
        (123, SyncRedisPipelineResultError, "unexpected Redis GET"),
        (b"bad", ValueError, "Invalid last_checked"),
    ],
)
def test_sync_multi_bucket_read_validates_all_slots_before_override_cache_update(
    bad_capacity: object,
    expected_error: type[Exception],
    match: str,
) -> None:
    backend, buckets = _sync_multi_backend()
    first, _second = buckets
    pipeline = _SyncPipeline(
        [
            b"1000.0",
            b"5.0",
            True,
            True,
            b"1000.0",
            bad_capacity,
            True,
            True,
            _override_payload(first, 7.0),
            None,
        ]
    )

    with pytest.raises(expected_error, match=match):
        backend._get_capacities_unsafe(pipeline=pipeline, current_time=1001.0)

    assert first.max_capacity == pytest.approx(first.configured_max_capacity)
    assert first._max_capacity_cached is None
    assert first._max_capacity_cache_populated is False


def test_bucket_state_pair_rejects_numeric_get_shape() -> None:
    with pytest.raises(RedisPipelineResultError, match="unexpected Redis GET"):
        _normalize_bucket_state_pair(1000, b"4.0", context="unit")


def test_override_parser_rejects_direct_dict_shape() -> None:
    _backend, bucket = _async_backend()

    with pytest.raises(MaxCapacityOverrideParseError, match="bytes, str, or None"):
        bucket.update_max_capacity_from_result(
            {"configured_max_capacity": 20.0, "override_max_capacity": 99.0}
        )


def test_oversized_override_error_is_bounded() -> None:
    _backend, bucket = _async_backend()
    raw = b"x" * (32 * 1024)

    with pytest.raises(MaxCapacityOverrideParseError) as exc_info:
        bucket.update_max_capacity_from_result(raw)

    message = str(exc_info.value)
    assert "32768 bytes" in message
    assert len(message) < 700


def test_async_script_status_decode_normalizes_bad_utf8_and_controls() -> None:
    with pytest.raises(_backend.RedisScriptResultError, match="not valid UTF-8"):
        _backend._decode_redis_script_status(b"\xff", context="unit script")

    with pytest.raises(_backend.RedisScriptResultError, match="invalid status"):
        _backend._decode_redis_script_status(b"\x00ABC", context="unit script")


def test_sync_script_status_decode_normalizes_bad_utf8_and_controls() -> None:
    with pytest.raises(_sync_backend.RedisScriptResultError, match="not valid UTF-8"):
        _sync_backend._decode_redis_script_status(b"\xff", context="unit script")

    with pytest.raises(_sync_backend.RedisScriptResultError, match="invalid status"):
        _sync_backend._decode_redis_script_status(b"\x00ABC", context="unit script")


def test_time_parser_rejects_bytes_and_huge_seconds() -> None:
    with pytest.raises(TypeError, match="integer-coercible"):
        _server_time._parse_time_response((b"123", b"456"))

    with pytest.raises(ValueError, match="out of range"):
        _server_time._parse_time_response((10**400, 0))
