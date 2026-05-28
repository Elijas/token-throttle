"""Tests for SyncRedisBucket max_capacity validation."""

import json
from unittest.mock import MagicMock

import pytest

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._sync_backend import SyncRedisBackend
from token_throttle._limiter_backends._redis._sync_bucket import (
    MaxCapacityOverrideParseError,
    SyncRedisBucket,
)


@pytest.fixture
def quota() -> Quota:
    return Quota(metric="requests", limit=20, per_seconds=1)


@pytest.fixture
def limit_config(quota: Quota) -> PerModelConfig:
    return PerModelConfig(
        model_family="test/model",
        quotas=UsageQuotas([quota]),
    )


@pytest.fixture
def mock_redis() -> MagicMock:
    mock = MagicMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.pipeline.return_value = MagicMock()
    return mock


@pytest.fixture
def bucket(
    quota: Quota,
    limit_config: PerModelConfig,
    mock_redis: MagicMock,
) -> SyncRedisBucket:
    return SyncRedisBucket(
        quota=quota,
        limit_config=limit_config,
        redis_client=mock_redis,
        key_prefix="test",
    )


def test_set_max_capacity_rejects_boolean(bucket):
    with pytest.raises(ValueError, match="max_capacity must not be a boolean"):
        bucket.set_max_capacity(True)


def test_get_max_capacity_ignores_legacy_max_capacity_key(bucket, mock_redis):
    def get_side_effect(key: str):
        if key == bucket._legacy_max_capacity_key:
            return b"5.0"
        if key == bucket._max_capacity_key:
            return None
        return None

    mock_redis.get.side_effect = get_side_effect

    assert bucket.get_max_capacity() == 20.0
    mock_redis.get.assert_called_once_with(bucket._max_capacity_key)


def test_backend_probe_warns_on_era1_legacy_key(
    bucket: SyncRedisBucket,
    limit_config: PerModelConfig,
    mock_redis: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    backend = SyncRedisBackend([bucket], mock_redis, limit_config, key_prefix="test")
    mock_redis.get.return_value = b"85.0"

    with caplog.at_level("WARNING"):
        backend._probe_legacy_override_keys_once()

    mock_redis.get.assert_called_once_with(bucket._legacy_max_capacity_key)
    assert "Era 1" in caplog.text
    assert "old :max_capacity key path" in caplog.text

    caplog.clear()
    mock_redis.get.reset_mock()

    backend._probe_legacy_override_keys_once()

    mock_redis.get.assert_not_called()
    assert "Era 1" not in caplog.text


def test_set_max_capacity_writes_baseline_metadata(bucket, mock_redis):
    bucket.set_max_capacity(5.0)

    assert mock_redis.set.call_count == 2
    schema_call = mock_redis.set.call_args_list[0]
    assert schema_call.args == (bucket._schema_version_key, bucket._SCHEMA_VERSION)
    assert schema_call.kwargs == {"nx": True}
    key, payload = mock_redis.set.call_args_list[1].args
    assert key == bucket._max_capacity_key
    assert mock_redis.set.call_args_list[1].kwargs == {
        "ex": bucket._override_ttl_seconds
    }
    assert json.loads(payload) == {
        "configured_max_capacity": 20.0,
        "override_max_capacity": 5.0,
    }


def test_update_max_capacity_from_result_warns_on_stale_metadata(
    bucket,
    caplog: pytest.LogCaptureFixture,
):
    payload = json.dumps(
        {
            "configured_max_capacity": 10.0,
            "override_max_capacity": 5.0,
        }
    ).encode()

    with caplog.at_level("WARNING"):
        bucket.update_max_capacity_from_result(payload)

    assert bucket._max_capacity_cached is None
    assert bucket.max_capacity == 20.0
    assert "configured_max_capacity anchor 10.0" in caplog.text
    assert "does not match current configured limit 20.0" in caplog.text


def test_bare_numeric_string_rejected(bucket):
    """Bare numeric strings are invalid unanchored overrides."""
    with pytest.raises(MaxCapacityOverrideParseError, match="object"):
        bucket.update_max_capacity_from_result(b"15.0")


def test_dict_missing_configured_max_capacity_rejected(bucket):
    """Dict without configured_max_capacity surfaces to the caller."""
    payload = json.dumps({"override_max_capacity": 5.0}).encode()

    with pytest.raises(MaxCapacityOverrideParseError, match="configured"):
        bucket.update_max_capacity_from_result(payload)
