import json

import pytest
from frozendict import frozendict

pytest.importorskip("redis")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder
from token_throttle._limiter_backends._redis._bucket import (
    MaxCapacityOverrideParseError,
)
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)

MODEL_FAMILY = "redis-override-errors-family"


def _config(*, limit: float = 10.0, per_seconds: int = 3600) -> PerModelConfig:
    return PerModelConfig(
        model_family=MODEL_FAMILY,
        quotas=UsageQuotas(
            [Quota(metric="requests", limit=limit, per_seconds=per_seconds)]
        ),
    )


@pytest.mark.redis
async def test_p4_redis_02_malformed_async_override_raises(redis_client):
    backend = RedisBackendBuilder(redis_client, key_prefix="v8-p4-02").build(
        _config(limit=10.0)
    )
    bucket = backend.sorted_buckets[0]
    await redis_client.set(bucket._max_capacity_key, b"not-json")

    with pytest.raises(MaxCapacityOverrideParseError, match="not valid JSON"):
        await backend.await_for_capacity(frozendict({"requests": 1.0}), timeout=0)


@pytest.mark.redis
def test_p4_redis_02_wrong_shape_sync_override_raises(sync_redis_client):
    backend = SyncRedisBackendBuilder(
        sync_redis_client, key_prefix="v8-p4-02-sync"
    ).build(_config(limit=10.0))
    bucket = backend.sorted_buckets[0]
    sync_redis_client.set(
        bucket._max_capacity_key,
        json.dumps({"override_max_capacity": 5.0}).encode(),
    )

    with pytest.raises(ValueError, match="configured_max_capacity"):
        backend.wait_for_capacity(frozendict({"requests": 1.0}), timeout=0)
