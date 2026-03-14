"""Bug 2: _compute_sleep bare next() gives cryptic StopIteration.

All 4 backend _compute_sleep() methods use `next(generator)` without a
default.  If the bucket is missing (shouldn't happen normally, but can via
internal bugs or race conditions), Python raises StopIteration — a cryptic
error.  This should be a descriptive ValueError instead.
"""

import pytest
from frozendict import frozendict

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, SecondsIn, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._bucket import MemoryBucket
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._limiter_backends._redis._backend import RedisBackend
from token_throttle._limiter_backends._redis._sync_backend import SyncRedisBackend


def _make_config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric="tokens", limit=1000, per_seconds=SecondsIn.MINUTE),
                Quota(metric="requests", limit=10, per_seconds=SecondsIn.MINUTE),
            ]
        ),
        model_family="test-family",
    )


USAGE = frozendict({"tokens": 100.0})
PRECONSUMPTION = frozendict({("tokens", 60): 50.0})


# ---------------------------------------------------------------------------
# Async memory backend
# ---------------------------------------------------------------------------


class TestAsyncMemoryComputeSleepMissingBucket:
    def test_missing_bucket_raises_value_error(self):
        backend = MemoryBackendBuilder().build(_make_config())
        # Remove the tokens bucket so _compute_sleep can't find it
        backend._buckets = [b for b in backend._buckets if b.usage_metric != "tokens"]

        with pytest.raises(ValueError, match="No bucket found"):
            backend._compute_sleep(USAGE, PRECONSUMPTION)


# ---------------------------------------------------------------------------
# Sync memory backend
# ---------------------------------------------------------------------------


class TestSyncMemoryComputeSleepMissingBucket:
    def test_missing_bucket_raises_value_error(self):
        backend = SyncMemoryBackendBuilder().build(_make_config())
        backend._buckets = [b for b in backend._buckets if b.usage_metric != "tokens"]

        with pytest.raises(ValueError, match="No bucket found"):
            backend._compute_sleep(USAGE, PRECONSUMPTION)


# ---------------------------------------------------------------------------
# Async Redis backend
# ---------------------------------------------------------------------------


class TestAsyncRedisComputeSleepMissingBucket:
    def test_missing_bucket_raises_value_error(self):
        backend = object.__new__(RedisBackend)
        backend._sleep_interval = 0.1
        # Only a "requests" bucket — no "tokens" bucket
        dummy = MemoryBucket(
            metric="requests", per_seconds=60, limit=10.0, model_family="test"
        )
        backend.sorted_buckets = [dummy]

        with pytest.raises(ValueError, match="No bucket found"):
            backend._compute_sleep(USAGE, PRECONSUMPTION)


# ---------------------------------------------------------------------------
# Sync Redis backend
# ---------------------------------------------------------------------------


class TestSyncRedisComputeSleepMissingBucket:
    def test_missing_bucket_raises_value_error(self):
        backend = object.__new__(SyncRedisBackend)
        backend._sleep_interval = 0.1
        dummy = MemoryBucket(
            metric="requests", per_seconds=60, limit=10.0, model_family="test"
        )
        backend.sorted_buckets = [dummy]

        with pytest.raises(ValueError, match="No bucket found"):
            backend._compute_sleep(USAGE, PRECONSUMPTION)
