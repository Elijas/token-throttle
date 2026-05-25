"""Regression coverage for Redis override format durability."""

import json
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackend
from token_throttle._limiter_backends._redis._bucket import (
    MaxCapacityOverrideParseError,
    RedisBucket,
)


@pytest.fixture
def quota() -> Quota:
    return Quota(metric="requests", limit=20, per_seconds=60)


@pytest.fixture
def limit_config(quota: Quota) -> PerModelConfig:
    return PerModelConfig(
        model_family="test/model",
        quotas=UsageQuotas([quota]),
    )


def _bucket(
    quota: Quota,
    limit_config: PerModelConfig,
    redis_client: AsyncMock | None = None,
    *,
    override_ttl_seconds: int | None = None,
) -> RedisBucket:
    return RedisBucket(
        quota=quota,
        limit_config=limit_config,
        redis_client=redis_client or AsyncMock(),
        key_prefix="test",
        override_ttl_seconds=override_ttl_seconds,
    )


async def test_d01_backend_probe_warns_on_era1_legacy_key(
    quota: Quota,
    limit_config: PerModelConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    redis_client = AsyncMock()
    bucket = _bucket(quota, limit_config, redis_client)
    backend = RedisBackend([bucket], redis_client, limit_config, key_prefix="test")
    redis_client.get.return_value = b"85.0"

    with caplog.at_level("WARNING"):
        await backend._probe_legacy_override_keys_once()

    redis_client.get.assert_awaited_once_with(bucket._legacy_max_capacity_key)
    assert "Era 1" in caplog.text
    assert "old :max_capacity key path" in caplog.text


@pytest.mark.parametrize(
    ("payload", "expected_warning"),
    [
        (b"85.0", "JSON must decode to an object"),
        (
            json.dumps({"override_max_capacity": 85.0}).encode(),
            "invalid configured_max_capacity",
        ),
    ],
)
def test_d02_era2_legacy_shapes_now_raise(
    quota: Quota,
    limit_config: PerModelConfig,
    caplog: pytest.LogCaptureFixture,
    payload: bytes,
    expected_warning: str,
) -> None:
    bucket = _bucket(quota, limit_config)

    with (
        caplog.at_level("WARNING"),
        pytest.raises(MaxCapacityOverrideParseError),
    ):
        bucket.update_max_capacity_from_result(payload)

    assert bucket.max_capacity == pytest.approx(float(quota.limit))
    assert expected_warning in caplog.text


async def test_d03_non_utf8_override_warns_and_raises(
    quota: Quota,
    limit_config: PerModelConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    redis_client = AsyncMock()
    redis_client.get.return_value = b"\xff"
    bucket = _bucket(quota, limit_config, redis_client)

    with (
        caplog.at_level("WARNING"),
        pytest.raises(MaxCapacityOverrideParseError),
    ):
        await bucket.get_max_capacity()

    assert "not valid UTF-8" in caplog.text
    assert "Refusing to ignore" in caplog.text


def test_d05_float_anchor_uses_isclose() -> None:
    quota = Quota(metric="requests", limit=0.1 + 0.2, per_seconds=60)
    limit_config = PerModelConfig(
        model_family="test/model",
        quotas=UsageQuotas([quota]),
    )
    bucket = _bucket(quota, limit_config)
    payload = json.dumps(
        {"configured_max_capacity": 0.3, "override_max_capacity": 0.2}
    ).encode()

    bucket.update_max_capacity_from_result(payload)

    assert bucket.max_capacity == pytest.approx(0.2)


async def test_d04_override_ttl_is_applied_when_configured(
    quota: Quota,
    limit_config: PerModelConfig,
) -> None:
    redis_client = AsyncMock()
    bucket = _bucket(
        quota,
        limit_config,
        redis_client,
        override_ttl_seconds=30 * 24 * 60 * 60,
    )

    await bucket.set_max_capacity(10.0)

    override_call = redis_client.set.await_args_list[1]
    assert override_call.args[0] == bucket._max_capacity_key
    assert override_call.kwargs == {"ex": 30 * 24 * 60 * 60}


async def test_d06_schema_version_key_written_on_first_override_write(
    quota: Quota,
    limit_config: PerModelConfig,
) -> None:
    redis_client = AsyncMock()
    bucket = _bucket(quota, limit_config, redis_client)

    await bucket.set_max_capacity(10.0)

    schema_call = redis_client.set.await_args_list[0]
    assert schema_call.args == (bucket._schema_version_key, bucket._SCHEMA_VERSION)
    assert schema_call.kwargs == {"nx": True}
