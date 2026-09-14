from __future__ import annotations

import asyncio
import uuid

import pytest

from token_throttle import (
    MemoryBackendBuilder,
    PerModelConfig,
    Quota,
    RateLimiter,
    SqliteBackendBuilder,
    SyncMemoryBackendBuilder,
    SyncRateLimiter,
    SyncSqliteBackendBuilder,
    UsageQuotas,
)


@pytest.fixture
def scoped_client(request, kind, synchronous):
    if kind == "redis":
        return request.getfixturevalue(
            "sync_redis_client" if synchronous else "redis_client"
        )
    return None


@pytest.mark.parametrize("kind", ["memory", "redis", "sqlite"])
@pytest.mark.parametrize("synchronous", [False, True], ids=["async", "sync"])
@pytest.mark.parametrize("same_metric", [False, True], ids=["new-metric", "new-window"])
async def test_refund_after_rebuild_preserves_unrelated_bucket_overflow(
    kind, synchronous, same_metric, scoped_client, tmp_path
):
    old_quota = Quota(metric="a", limit=100, per_seconds=10000)
    new_quota = Quota(metric="a" if same_metric else "b", limit=100, per_seconds=20000)
    quotas = [old_quota]

    def config(_model):
        return PerModelConfig(model_family="scope", quotas=UsageQuotas(quotas))

    if kind == "memory":
        builder = (SyncMemoryBackendBuilder if synchronous else MemoryBackendBuilder)()
    elif kind == "sqlite":
        builder = (SyncSqliteBackendBuilder if synchronous else SqliteBackendBuilder)(
            tmp_path / "scope.db", key_prefix="scope"
        )
    elif synchronous:
        from token_throttle import SyncRedisBackendBuilder  # noqa: PLC0415

        builder = SyncRedisBackendBuilder(
            scoped_client, key_prefix=f"scope-{uuid.uuid4().hex}"
        )
    else:
        from token_throttle import RedisBackendBuilder  # noqa: PLC0415

        builder = RedisBackendBuilder(
            scoped_client, key_prefix=f"scope-{uuid.uuid4().hex}"
        )
    limiter = (SyncRateLimiter if synchronous else RateLimiter)(config, backend=builder)

    async def call(method, *args, **kwargs):
        if synchronous:
            return await asyncio.to_thread(getattr(limiter, method), *args, **kwargs)
        return await getattr(limiter, method)(*args, **kwargs)

    try:
        old = await call("acquire_capacity", {"a": 40}, "model", timeout=0)
        quotas.append(new_quota)
        with pytest.warns(UserWarning, match="changed metric set"):
            initialized = await call(
                "acquire_capacity",
                {quota.metric: 0 for quota in quotas},
                "model",
                timeout=0,
            )
        await call(
            "set_max_capacity", "model", new_quota.metric, new_quota.per_seconds, 10
        )
        await call("refund_capacity", {"a": 0}, old)
        await call(
            "set_max_capacity", "model", new_quota.metric, new_quota.per_seconds, 100
        )
        diagnostic = await call("diagnose")
        untouched = next(
            bucket for bucket in diagnostic.buckets if bucket.per_seconds == 20000
        )
        assert untouched.current_capacity == pytest.approx(100)
        await call(
            "refund_capacity", {quota.metric: 0 for quota in quotas}, initialized
        )
    finally:
        await call("close" if synchronous else "aclose")
