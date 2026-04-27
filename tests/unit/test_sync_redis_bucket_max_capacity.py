"""Tests for SyncRedisBucket max_capacity validation."""

import json
from unittest.mock import MagicMock

import pytest

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket


@pytest.fixture
def mock_redis():
    mock = MagicMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.pipeline.return_value = MagicMock()
    return mock


@pytest.fixture
def bucket(mock_redis):
    quota = Quota(metric="requests", limit=20, per_seconds=1)
    limit_config = PerModelConfig(
        model_family="test/model",
        quotas=UsageQuotas([quota]),
    )
    return SyncRedisBucket(
        quota=quota,
        limit_config=limit_config,
        redis_client=mock_redis,
    )


def test_set_max_capacity_rejects_boolean(bucket):
    with pytest.raises(ValueError, match="max_capacity must not be a boolean"):
        bucket.set_max_capacity(True)


def test_get_max_capacity_ignores_legacy_max_capacity_key(bucket, mock_redis):
    legacy_key = f"{bucket.full_redis_key}:max_capacity"

    def get_side_effect(key: str):
        if key == legacy_key:
            return b"5.0"
        if key == bucket._max_capacity_key:
            return None
        return None

    mock_redis.get.side_effect = get_side_effect

    assert bucket.get_max_capacity() == 20.0
    mock_redis.get.assert_called_once_with(bucket._max_capacity_key)


def test_set_max_capacity_writes_baseline_metadata(bucket, mock_redis):
    bucket.set_max_capacity(5.0)

    mock_redis.set.assert_called_once()
    key, payload = mock_redis.set.call_args.args
    assert key == bucket._max_capacity_key
    assert json.loads(payload) == {
        "configured_max_capacity": 20.0,
        "override_max_capacity": 5.0,
    }


def test_update_max_capacity_from_result_ignores_stale_metadata(bucket):
    bucket.update_max_capacity_from_result(
        json.dumps(
            {
                "configured_max_capacity": 10.0,
                "override_max_capacity": 5.0,
            }
        ).encode()
    )

    assert bucket._max_capacity_cached is None
    assert bucket.max_capacity == 20.0


def test_bare_numeric_string_rejected(bucket):
    """Bare numeric strings are rejected — all overrides must use anchored JSON."""
    bucket.update_max_capacity_from_result(b"15.0")

    assert bucket._max_capacity_cached is None
    assert bucket.max_capacity == 20.0


def test_dict_missing_configured_max_capacity_rejected(bucket):
    """Dict without configured_max_capacity is rejected to prevent anchor bypass."""
    payload = json.dumps({"override_max_capacity": 5.0}).encode()
    bucket.update_max_capacity_from_result(payload)

    assert bucket._max_capacity_cached is None
    assert bucket.max_capacity == 20.0
