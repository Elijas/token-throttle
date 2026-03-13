"""Tests for SyncRedisBucket max_capacity validation."""

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
