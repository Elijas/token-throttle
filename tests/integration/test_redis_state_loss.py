"""State loss is detected only by observers with recent store-clock evidence."""

import inspect
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackend
from token_throttle._limiter_backends._redis._bucket import RedisBucket
from token_throttle._limiter_backends._redis._sync_backend import SyncRedisBackend
from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket

pytestmark = pytest.mark.redis


async def call(result):
    return await result if inspect.isawaitable(result) else result


@pytest.fixture(params=[False, True], ids=["async", "sync"])
def lane(request):
    sync = request.param
    client = request.getfixturevalue("sync_redis_client" if sync else "redis_client")
    bucket_type = SyncRedisBucket if sync else RedisBucket
    backend_type = SyncRedisBackend if sync else RedisBackend
    quota = Quota(metric="requests", limit=100, per_seconds=60)
    config = PerModelConfig(model_family="state-loss", quotas=UsageQuotas([quota]))

    def build():
        bucket = bucket_type(
            quota, config, client, key_prefix="state-loss", bucket_ttl_seconds=200
        )
        backend = backend_type(
            buckets=[bucket], redis=client, limit_config=config, key_prefix="state-loss"
        )
        return bucket, backend

    return client, build


@pytest.mark.parametrize("missing", ["both", "capacity", "last_checked"])
async def test_warm_loss_drains_and_repairs_durably(lane, missing):
    client, build = lane
    bucket, backend = build()
    await call(
        backend._set_capacities_unsafe({("requests", 60): 12}, current_time=1000)
    )
    keys = {"capacity": bucket._capacity_key, "last_checked": bucket._last_checked_key}
    await call(
        client.delete(*(keys.values() if missing == "both" else [keys[missing]]))
    )
    result = await call(backend._get_capacities_unsafe(current_time=1001))
    assert result.capacities[("requests", 60)] == 0
    assert bucket._missing_consumption_data_reason == "state_loss_drained"
    assert float(await call(client.get(bucket._capacity_key))) == 0
    _, observer = build()
    result = await call(observer._get_capacities_unsafe(current_time=1001))
    assert result.capacities[("requests", 60)] == 0


@pytest.mark.parametrize(
    "age, expected", [(179.999, 0), (180, 100), (181, 100), (-10, 0)]
)
async def test_total_loss_recent_confirmation_boundary(lane, age, expected):
    client, build = lane
    bucket, backend = build()
    await call(bucket.set_capacity(12, current_time=1000))
    await call(client.delete(bucket._capacity_key, bucket._last_checked_key))
    result = await call(backend._get_capacities_unsafe(current_time=1000 + age))
    assert result.capacities[("requests", 60)] == expected


async def test_new_observer_is_fresh_and_cold_first_recreation_hides_loss(lane):
    client, build = lane
    bucket, warm = build()
    await call(bucket.set_capacity(12, current_time=1000))
    await call(client.delete(bucket._capacity_key, bucket._last_checked_key))
    _, cold = build()
    result = await call(cold._get_capacities_unsafe(current_time=1001))
    assert result.capacities[("requests", 60)] == 100
    await call(cold._set_capacities_unsafe({("requests", 60): 99}, current_time=1001))
    result = await call(warm._get_capacities_unsafe(current_time=1001))
    assert result.capacities[("requests", 60)] == 99


@pytest.mark.parametrize("missing", ["both", "capacity"])
async def test_diagnostics_predict_loss_without_writes_or_proof_changes(
    lane, monkeypatch, missing
):
    client, build = lane
    bucket, backend = build()
    await call(bucket.set_capacity(12, current_time=1000))
    await call(client.delete(bucket._capacity_key))
    if missing == "both":
        await call(client.delete(bucket._last_checked_key))
    module = inspect.getmodule(type(backend))
    clock_name = (
        "async_server_time" if isinstance(backend, RedisBackend) else "sync_server_time"
    )
    clock = (
        AsyncMock(return_value=1001)
        if isinstance(backend, RedisBackend)
        else Mock(return_value=1001)
    )
    monkeypatch.setattr(module, clock_name, clock)
    proof = bucket._state_confirmed_at_server_time
    ttl_before = await call(client.pttl(bucket._last_checked_key))
    diagnostic = await call(backend.introspect())
    assert diagnostic.buckets[0].current_capacity == 0
    assert diagnostic.buckets[0].status == (
        "state_loss" if missing == "both" else "partial_missing"
    )
    assert await call(client.get(bucket._capacity_key)) is None
    assert await call(client.pttl(bucket._last_checked_key)) <= ttl_before
    assert bucket._state_confirmed_at_server_time == proof
    result = await call(backend._get_capacities_unsafe(current_time=1001))
    assert result.capacities[("requests", 60)] == 0


@pytest.mark.parametrize("wipe_before", [False, True])
async def test_rebuild_preserves_survivor_proof_and_snapshot_repairs_loss(
    lane, monkeypatch, wipe_before
):
    client, build = lane
    bucket, backend = build()
    await call(bucket.set_capacity(12, current_time=1000))
    replacement, new_backend = build()
    module = inspect.getmodule(type(backend))
    clock_name = (
        "async_server_time" if isinstance(backend, RedisBackend) else "sync_server_time"
    )
    clock = (
        AsyncMock(return_value=1001)
        if isinstance(backend, RedisBackend)
        else Mock(return_value=1001)
    )
    monkeypatch.setattr(module, clock_name, clock)
    if wipe_before:
        await call(client.delete(bucket._capacity_key, bucket._last_checked_key))
    await call(
        backend.prepare_reconfigured_backend(new_backend, new_backend._limit_config)
    )
    assert backend.sorted_buckets[0] is replacement
    assert replacement._state_confirmed_at_server_time == 1001
    if wipe_before:
        assert float(await call(client.get(bucket._capacity_key))) == 0
    await call(client.delete(bucket._capacity_key, bucket._last_checked_key))
    result = await call(backend._get_capacities_unsafe(current_time=1002))
    assert result.capacities[("requests", 60)] == 0


async def test_standalone_bucket_read_repairs_loss_and_proves_complete_state(lane):
    client, build = lane
    bucket, _ = build()
    await call(client.set(bucket._capacity_key, 12))
    await call(client.set(bucket._last_checked_key, 1000))
    result = await call(bucket.get_capacity(current_time=1000))
    assert result.amount == 12
    await call(client.delete(bucket._capacity_key, bucket._last_checked_key))
    result = await call(bucket.get_capacity(current_time=1001))
    assert result.amount == 0
    assert float(await call(client.get(bucket._capacity_key))) == 0


@pytest.mark.parametrize("operation", ["read", "write"])
async def test_malformed_pipeline_never_confirms_state(lane, monkeypatch, operation):
    _, build = lane
    bucket, backend = build()
    pipeline = Mock()
    pipeline.execute = (
        AsyncMock(return_value=[b"1000", b"12", 1, "invalid", None])
        if isinstance(backend, RedisBackend)
        else Mock(return_value=[b"1000", b"12", 1, "invalid", None])
    )
    if operation == "write":
        pipeline.execute.return_value = [True, None]
    if operation == "read":
        with pytest.raises(RuntimeError):
            await call(
                backend._get_capacities_unsafe(pipeline=pipeline, current_time=1000)
            )
    else:
        with pytest.raises(RuntimeError):
            await call(
                backend._set_capacities_unsafe(
                    {("requests", 60): 12}, pipeline=pipeline, current_time=1000
                )
            )
    assert bucket._state_confirmed_at_server_time is None


async def test_later_corrupt_bucket_cannot_confirm_earlier_valid_bucket(lane):
    client, build = lane
    bucket, backend = build()
    quota = Quota(metric="tokens", limit=100, per_seconds=60)
    second = type(bucket)(
        quota,
        backend._limit_config,
        client,
        key_prefix="state-loss",
        bucket_ttl_seconds=200,
    )
    backend.add_bucket(second)
    pipeline = Mock()
    payload = [b"1000", b"12", 1, 1, b"1000", b"bad", 1, 1, None, None]
    pipeline.execute = (
        AsyncMock(return_value=payload)
        if isinstance(backend, RedisBackend)
        else Mock(return_value=payload)
    )
    with pytest.raises(ValueError, match="Invalid last_checked or capacity"):
        await call(backend._get_capacities_unsafe(pipeline=pipeline, current_time=1000))
    assert bucket._state_confirmed_at_server_time is None
    assert second._state_confirmed_at_server_time is None
    assert await call(client.dbsize()) == 0


async def test_removal_drops_proof_for_reintroduced_identity(lane):
    client, build = lane
    bucket, backend = build()
    await call(bucket.set_capacity(12, current_time=1000))
    backend.install_reconfigured_state(buckets=[], cfg=backend._limit_config)
    replacement, _ = build()
    backend.install_reconfigured_state(buckets=[replacement], cfg=backend._limit_config)
    await call(client.delete(bucket._last_checked_key, bucket._capacity_key))
    result = await call(backend._get_capacities_unsafe(current_time=1001))
    assert result.capacities[("requests", 60)] == 100


async def test_read_only_complete_state_cannot_create_proof(lane, monkeypatch):
    client, build = lane
    bucket, backend = build()
    await call(client.set(bucket._capacity_key, 12, ex=10))
    await call(client.set(bucket._last_checked_key, 1000, ex=10))
    module = inspect.getmodule(type(backend))
    clock_name = (
        "async_server_time" if isinstance(backend, RedisBackend) else "sync_server_time"
    )
    clock = (
        AsyncMock(return_value=1000)
        if isinstance(backend, RedisBackend)
        else Mock(return_value=1000)
    )
    monkeypatch.setattr(module, clock_name, clock)
    diagnostic = await call(backend.introspect())
    assert diagnostic.buckets[0].current_capacity == 12
    assert bucket._state_confirmed_at_server_time is None
    assert await call(client.ttl(bucket._capacity_key)) <= 10
    await call(client.delete(bucket._last_checked_key, bucket._capacity_key))
    diagnostic = await call(backend.introspect())
    assert diagnostic.buckets[0].status == "fresh_start"
    assert diagnostic.buckets[0].current_capacity == 100


async def test_store_clock_drives_confirmation(lane, monkeypatch):
    client, build = lane
    bucket, backend = build()
    module = inspect.getmodule(type(backend))
    clock_name = (
        "async_server_time" if isinstance(backend, RedisBackend) else "sync_server_time"
    )
    clock = (
        AsyncMock(return_value=1000)
        if isinstance(backend, RedisBackend)
        else Mock(return_value=1000)
    )
    monkeypatch.setattr(module, clock_name, clock)
    await call(backend._set_capacities_unsafe({("requests", 60): 12}))
    assert bucket._state_confirmed_at_server_time == 1000
    await call(client.delete(bucket._last_checked_key, bucket._capacity_key))
    clock.return_value = 1001
    result = await call(backend._get_capacities_unsafe())
    assert result.capacities[("requests", 60)] == 0


async def test_backend_read_and_repair_refresh_confirmation(lane):
    client, build = lane
    bucket, backend = build()
    await call(client.set(bucket._capacity_key, 12))
    await call(client.set(bucket._last_checked_key, 1000))
    await call(backend._get_capacities_unsafe(current_time=1000))
    assert bucket._state_confirmed_at_server_time == 1000
    await call(client.delete(bucket._last_checked_key, bucket._capacity_key))
    await call(backend._get_capacities_unsafe(current_time=1170))
    assert bucket._state_confirmed_at_server_time == 1170
    await call(client.delete(bucket._last_checked_key, bucket._capacity_key))
    result = await call(backend._get_capacities_unsafe(current_time=1340))
    assert result.capacities[("requests", 60)] == 0


@pytest.mark.parametrize("age, expected", [(179, 0), (180, 100)])
async def test_rebuild_with_longer_ttl_preserves_original_proof_window(
    lane, age, expected
):
    client, build = lane
    bucket, backend = build()
    await call(bucket.set_capacity(12, current_time=1000))
    replacement, _ = build()
    replacement._bucket_ttl_seconds = 1000
    backend.install_reconfigured_state(buckets=[replacement], cfg=backend._limit_config)
    await call(client.delete(bucket._capacity_key, bucket._last_checked_key))
    result = await call(backend._get_capacities_unsafe(current_time=1000 + age))
    assert result.capacities[("requests", 60)] == expected


@pytest.mark.parametrize("missing", ["both", "capacity", "last_checked"])
async def test_blocked_acquire_emits_loss_once_with_exact_fields(
    lane, monkeypatch, missing
):
    client, build = lane
    bucket, backend = build()
    events = []

    async def async_callback(**kwargs):
        events.append(kwargs)

    def sync_callback(**kwargs):
        events.append(kwargs)

    backend._callbacks = (
        RateLimiterCallbacks(on_missing_consumption_data=async_callback)
        if isinstance(backend, RedisBackend)
        else SyncRateLimiterCallbacks(on_missing_consumption_data=sync_callback)
    )
    module = inspect.getmodule(type(backend))
    clock_name = (
        "async_server_time" if isinstance(backend, RedisBackend) else "sync_server_time"
    )
    clock = (
        AsyncMock(return_value=1001)
        if isinstance(backend, RedisBackend)
        else Mock(return_value=1001)
    )
    monkeypatch.setattr(module, clock_name, clock)
    await call(bucket.set_capacity(12, current_time=1000))
    keys = {"last_checked": bucket._last_checked_key, "capacity": bucket._capacity_key}
    await call(
        client.delete(*(keys.values() if missing == "both" else [keys[missing]]))
    )
    result = await call(backend._check_and_consume_capacity({"requests": 1}))
    assert result[0] is False
    assert len(events) == 1
    assert events[0]["missing_state_reason"] == "state_loss_drained"
    assert events[0]["missing_state_keys"] == tuple(
        k for k in keys if missing in ("both", k)
    )
    assert events[0]["present_state_keys"] == tuple(
        k for k in keys if missing not in ("both", k)
    )
    result = await call(backend._check_and_consume_capacity({"requests": 1}))
    assert result[0] is False
    assert len(events) == 1
