"""Regression tests for Redis backend metric-set reconfiguration snapshots."""

import asyncio
import threading

import pytest
from frozendict import frozendict

pytest.importorskip("redis", reason="redis package not installed")

import token_throttle._limiter_backends._redis._backend as async_backend_module
import token_throttle._limiter_backends._redis._sync_backend as sync_backend_module
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackend
from token_throttle._limiter_backends._redis._sync_backend import SyncRedisBackend


class FakeAsyncRedis:
    def pipeline(self):
        return object()


class FakeSyncRedis:
    def pipeline(self):
        return object()


class FakeAsyncBucket:
    def __init__(self, key: str, metric: str, per_seconds: int, max_capacity: float):
        self.full_redis_key = key
        self.usage_metric = metric
        self.per_seconds = per_seconds
        self.max_capacity = max_capacity
        self._max_capacity_default = max_capacity
        self._rate_per_sec = max_capacity / per_seconds

    async def set_max_capacity(self, value: float) -> None:
        self.max_capacity = value
        self._rate_per_sec = value / self.per_seconds

    def set_configured_max_capacity(self, value: float) -> None:
        self._max_capacity_default = value
        self.max_capacity = value
        self._rate_per_sec = value / self.per_seconds

    async def clear_max_capacity_override(self) -> None:
        return None


class FakeSyncBucket:
    def __init__(self, key: str, metric: str, per_seconds: int, max_capacity: float):
        self.full_redis_key = key
        self.usage_metric = metric
        self.per_seconds = per_seconds
        self.max_capacity = max_capacity
        self._max_capacity_default = max_capacity
        self._rate_per_sec = max_capacity / per_seconds

    def set_max_capacity(self, value: float) -> None:
        self.max_capacity = value
        self._rate_per_sec = value / self.per_seconds

    def set_configured_max_capacity(self, value: float) -> None:
        self._max_capacity_default = value
        self.max_capacity = value
        self._rate_per_sec = value / self.per_seconds

    def clear_max_capacity_override(self) -> None:
        return None


class AsyncNoopContextManager:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class SyncNoopContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def make_config(*quotas: Quota, family: str = "test-family") -> PerModelConfig:
    return PerModelConfig(quotas=UsageQuotas(list(quotas)), model_family=family)


async def test_async_check_and_consume_uses_one_bucket_snapshot_during_reconfigure(
    monkeypatch,
):
    cfg_old = make_config(Quota(metric="tokens", limit=100, per_seconds=60))
    cfg_new = make_config(
        Quota(metric="tokens", limit=100, per_seconds=60),
        Quota(metric="requests", limit=10, per_seconds=60),
    )

    backend = RedisBackend(
        buckets=[FakeAsyncBucket("old:tokens", "tokens", 60, 100.0)],
        redis=FakeAsyncRedis(),
        limit_config=cfg_old,
    )
    new_backend = RedisBackend(
        buckets=[
            FakeAsyncBucket("new:requests", "requests", 60, 10.0),
            FakeAsyncBucket("old:tokens", "tokens", 60, 100.0),
        ],
        redis=FakeAsyncRedis(),
        limit_config=cfg_new,
    )

    lock_snapshots: list[list[str]] = []
    capacity_snapshots: list[list[str]] = []
    entered_get_capacities = asyncio.Event()
    allow_get_capacities = asyncio.Event()

    async def fake_lock(*, timeout, blocking_timeout=None, buckets=None):
        effective_buckets = backend.sorted_buckets if buckets is None else buckets
        lock_snapshots.append(
            [bucket.full_redis_key for bucket in effective_buckets],
        )
        return AsyncNoopContextManager()

    async def fake_get_capacities_unsafe(
        *, pipeline=None, current_time=None, buckets=None
    ):
        entered_get_capacities.set()
        await allow_get_capacities.wait()
        effective_buckets = backend.sorted_buckets if buckets is None else buckets
        capacity_snapshots.append(
            [bucket.full_redis_key for bucket in effective_buckets],
        )
        capacities = frozendict(
            {
                (bucket.usage_metric, int(bucket.per_seconds)): bucket.max_capacity
                for bucket in effective_buckets
            }
        )
        return async_backend_module.CapacitiesGetterResult(capacities, [])

    async def fake_set_capacities_unsafe(
        new_capacities,
        pipeline=None,
        current_time=None,
        *,
        allow_negative=False,
        buckets=None,
    ) -> None:
        return None

    async def fake_server_time(_redis) -> float:
        return 0.0

    monkeypatch.setattr(backend, "_lock", fake_lock)
    monkeypatch.setattr(backend, "_get_capacities_unsafe", fake_get_capacities_unsafe)
    monkeypatch.setattr(backend, "_set_capacities_unsafe", fake_set_capacities_unsafe)
    monkeypatch.setattr(async_backend_module, "async_server_time", fake_server_time)

    task = asyncio.create_task(
        backend._check_and_consume_capacity(frozendict({"tokens": 1.0}))
    )
    await entered_get_capacities.wait()
    await backend.prepare_reconfigured_backend(new_backend, cfg_new)
    allow_get_capacities.set()
    await task

    assert lock_snapshots[0] == ["old:tokens"]
    assert capacity_snapshots == [["old:tokens"]]
    assert [bucket.full_redis_key for bucket in backend.sorted_buckets] == [
        "new:requests",
        "old:tokens",
    ]


def test_sync_check_and_consume_uses_one_bucket_snapshot_during_reconfigure(
    monkeypatch,
):
    cfg_old = make_config(Quota(metric="tokens", limit=100, per_seconds=60))
    cfg_new = make_config(
        Quota(metric="tokens", limit=100, per_seconds=60),
        Quota(metric="requests", limit=10, per_seconds=60),
    )

    backend = SyncRedisBackend(
        buckets=[FakeSyncBucket("old:tokens", "tokens", 60, 100.0)],
        redis=FakeSyncRedis(),
        limit_config=cfg_old,
    )
    new_backend = SyncRedisBackend(
        buckets=[
            FakeSyncBucket("new:requests", "requests", 60, 10.0),
            FakeSyncBucket("old:tokens", "tokens", 60, 100.0),
        ],
        redis=FakeSyncRedis(),
        limit_config=cfg_new,
    )

    lock_snapshots: list[list[str]] = []
    capacity_snapshots: list[list[str]] = []
    entered_get_capacities = threading.Event()
    allow_get_capacities = threading.Event()
    errors: list[BaseException] = []

    def fake_lock(*, timeout, blocking_timeout=None, buckets=None):
        effective_buckets = backend.sorted_buckets if buckets is None else buckets
        lock_snapshots.append(
            [bucket.full_redis_key for bucket in effective_buckets],
        )
        return SyncNoopContextManager()

    def fake_get_capacities_unsafe(*, pipeline=None, current_time=None, buckets=None):
        entered_get_capacities.set()
        assert allow_get_capacities.wait(timeout=2.0)
        effective_buckets = backend.sorted_buckets if buckets is None else buckets
        capacity_snapshots.append(
            [bucket.full_redis_key for bucket in effective_buckets],
        )
        capacities = frozendict(
            {
                (bucket.usage_metric, int(bucket.per_seconds)): bucket.max_capacity
                for bucket in effective_buckets
            }
        )
        return sync_backend_module.SyncCapacitiesGetterResult(capacities, [])

    def fake_set_capacities_unsafe(
        new_capacities,
        pipeline=None,
        current_time=None,
        *,
        allow_negative=False,
        buckets=None,
    ) -> None:
        return None

    monkeypatch.setattr(backend, "_lock", fake_lock)
    monkeypatch.setattr(backend, "_get_capacities_unsafe", fake_get_capacities_unsafe)
    monkeypatch.setattr(backend, "_set_capacities_unsafe", fake_set_capacities_unsafe)
    monkeypatch.setattr(sync_backend_module, "sync_server_time", lambda _redis: 0.0)

    def worker() -> None:
        try:
            backend._check_and_consume_capacity(frozendict({"tokens": 1.0}))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    assert entered_get_capacities.wait(timeout=2.0)
    backend.prepare_reconfigured_backend(new_backend, cfg_new)
    allow_get_capacities.set()
    thread.join(timeout=2.0)

    assert not errors
    assert lock_snapshots[0] == ["old:tokens"]
    assert capacity_snapshots == [["old:tokens"]]
    assert [bucket.full_redis_key for bucket in backend.sorted_buckets] == [
        "new:requests",
        "old:tokens",
    ]


async def test_async_waiter_fails_cleanly_when_metric_is_removed(monkeypatch):
    cfg_old = make_config(Quota(metric="tokens", limit=10, per_seconds=60))
    cfg_new = make_config(Quota(metric="requests", limit=10, per_seconds=60))

    backend = RedisBackend(
        buckets=[FakeAsyncBucket("old:tokens", "tokens", 60, 10.0)],
        redis=FakeAsyncRedis(),
        limit_config=cfg_old,
        sleep_interval=5.0,
    )
    new_backend = RedisBackend(
        buckets=[FakeAsyncBucket("new:requests", "requests", 60, 10.0)],
        redis=FakeAsyncRedis(),
        limit_config=cfg_new,
        sleep_interval=5.0,
    )

    call_count = 0
    entered_wait = asyncio.Event()

    async def fake_lock(*, timeout, blocking_timeout=None, buckets=None):
        return AsyncNoopContextManager()

    async def fake_get_capacities_unsafe(
        *, pipeline=None, current_time=None, buckets=None
    ):
        nonlocal call_count
        call_count += 1
        effective_buckets = backend.sorted_buckets if buckets is None else buckets
        current_capacity = 0.0 if call_count == 1 else 10.0
        capacities = frozendict(
            {
                (bucket.usage_metric, int(bucket.per_seconds)): current_capacity
                for bucket in effective_buckets
            }
        )
        return async_backend_module.CapacitiesGetterResult(capacities, [])

    async def fake_set_capacities_unsafe(
        new_capacities,
        pipeline=None,
        current_time=None,
        *,
        allow_negative=False,
        buckets=None,
    ) -> None:
        return None

    async def fake_server_time(_redis) -> float:
        return 0.0

    monkeypatch.setattr(backend, "_lock", fake_lock)
    monkeypatch.setattr(backend, "_get_capacities_unsafe", fake_get_capacities_unsafe)
    monkeypatch.setattr(backend, "_set_capacities_unsafe", fake_set_capacities_unsafe)
    monkeypatch.setattr(async_backend_module, "async_server_time", fake_server_time)

    original_wait = backend._local_condition.wait

    async def wrapped_wait():
        entered_wait.set()
        return await original_wait()

    monkeypatch.setattr(backend._local_condition, "wait", wrapped_wait)

    task = asyncio.create_task(
        backend.await_for_capacity(frozendict({"tokens": 1.0}), timeout=1.0)
    )
    await entered_wait.wait()
    await backend.prepare_reconfigured_backend(new_backend, cfg_new)
    async with backend._local_condition:
        backend._local_condition.notify_all()

    with pytest.raises(ValueError, match="no longer active"):
        await task


def test_sync_waiter_fails_cleanly_when_metric_is_removed(monkeypatch):
    cfg_old = make_config(Quota(metric="tokens", limit=10, per_seconds=60))
    cfg_new = make_config(Quota(metric="requests", limit=10, per_seconds=60))

    backend = SyncRedisBackend(
        buckets=[FakeSyncBucket("old:tokens", "tokens", 60, 10.0)],
        redis=FakeSyncRedis(),
        limit_config=cfg_old,
        sleep_interval=5.0,
    )
    new_backend = SyncRedisBackend(
        buckets=[FakeSyncBucket("new:requests", "requests", 60, 10.0)],
        redis=FakeSyncRedis(),
        limit_config=cfg_new,
        sleep_interval=5.0,
    )

    call_count = 0
    entered_wait = threading.Event()
    result: dict[str, BaseException] = {}

    def fake_lock(*, timeout, blocking_timeout=None, buckets=None):
        return SyncNoopContextManager()

    def fake_get_capacities_unsafe(*, pipeline=None, current_time=None, buckets=None):
        nonlocal call_count
        call_count += 1
        effective_buckets = backend.sorted_buckets if buckets is None else buckets
        current_capacity = 0.0 if call_count == 1 else 10.0
        capacities = frozendict(
            {
                (bucket.usage_metric, int(bucket.per_seconds)): current_capacity
                for bucket in effective_buckets
            }
        )
        return sync_backend_module.SyncCapacitiesGetterResult(capacities, [])

    def fake_set_capacities_unsafe(
        new_capacities,
        pipeline=None,
        current_time=None,
        *,
        allow_negative=False,
        buckets=None,
    ) -> None:
        return None

    monkeypatch.setattr(backend, "_lock", fake_lock)
    monkeypatch.setattr(backend, "_get_capacities_unsafe", fake_get_capacities_unsafe)
    monkeypatch.setattr(backend, "_set_capacities_unsafe", fake_set_capacities_unsafe)
    monkeypatch.setattr(sync_backend_module, "sync_server_time", lambda _redis: 0.0)

    original_wait = backend._local_condition.wait

    def wrapped_wait(timeout=None):
        entered_wait.set()
        return original_wait(timeout)

    monkeypatch.setattr(backend._local_condition, "wait", wrapped_wait)

    def worker() -> None:
        try:
            backend.wait_for_capacity(frozendict({"tokens": 1.0}), timeout=1.0)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    assert entered_wait.wait(timeout=2.0)
    backend.prepare_reconfigured_backend(new_backend, cfg_new)
    with backend._local_condition:
        backend._local_condition.notify_all()
    thread.join(timeout=2.0)

    assert isinstance(result.get("error"), ValueError)
    assert "no longer active" in str(result["error"])
