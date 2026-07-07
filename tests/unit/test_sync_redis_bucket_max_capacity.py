"""Tests for SyncRedisBucket max_capacity validation."""

import json
from unittest.mock import MagicMock

import pytest

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
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


def test_set_max_capacity_writes_override_payload(bucket, mock_redis):
    bucket.set_max_capacity(5.0)

    assert mock_redis.set.call_count == 1
    key, payload = mock_redis.set.call_args_list[0].args
    assert key == bucket._max_capacity_key
    assert mock_redis.set.call_args_list[0].kwargs == {
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
