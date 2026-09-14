"""SQLite rebuilds must retain database override authority and static quotas."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from token_throttle import (
    PerModelConfig,
    Quota,
    RateLimiter,
    SqliteBackendBuilder,
    SyncRateLimiter,
    SyncSqliteBackendBuilder,
    UsageQuotas,
)
from token_throttle._limiter_backends._sqlite._engine import SqliteEngine

if TYPE_CHECKING:
    from pathlib import Path


def _config(**limits: float) -> PerModelConfig:
    return PerModelConfig(
        model_family="override-rebuild",
        quotas=UsageQuotas(
            [
                Quota(metric=metric, limit=limit, per_seconds=1)
                for metric, limit in limits.items()
            ]
        ),
    )


@dataclass
class _Clock:
    now: float = 100.0

    def __call__(self) -> float:
        return self.now


@dataclass
class _Harness:
    limiter: RateLimiter | SyncRateLimiter
    builder: SqliteBackendBuilder | SyncSqliteBackendBuilder
    configs: list[PerModelConfig]
    db_path: Path
    clock: _Clock

    async def call(self, method: str, *args, **kwargs):
        operation = getattr(self.limiter, method)
        if isinstance(self.limiter, RateLimiter):
            return await operation(*args, **kwargs)
        return await asyncio.to_thread(operation, *args, **kwargs)

    def override(self) -> tuple[float | None, float | None]:
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT override_value, override_expires_at FROM buckets "
                "WHERE key_prefix = 'rebuild' AND model_family = 'override-rebuild' "
                "AND metric = 'requests' AND per_seconds = 1"
            ).fetchone()
        assert row is not None
        return tuple(row)

    def configured_max(self) -> float:
        backend = self.limiter._model_family_to_backend["override-rebuild"]
        return backend._engine.configured_max_capacity("requests", 1)

    async def rebuild(self, **limits: float) -> None:
        self.configs[0] = _config(**limits)
        with pytest.warns(UserWarning, match="changed metric set"):
            await self.call("record_usage", dict.fromkeys(limits, 0), "model")

    async def acquire_requests(self, requests: int):
        usage = {quota.metric: 0 for quota in self.configs[0].quotas}
        usage["requests"] = requests
        return await self.call("acquire_capacity", usage, "model", timeout=0)


@pytest.fixture(params=["sync", "async"])
async def harness(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = _Clock()

    def use_clock(original):
        def transaction(engine, **kwargs):
            return original(engine, **{**kwargs, "clock": clock})

        return transaction

    for method in ("_transaction", "_read_transaction"):
        monkeypatch.setattr(
            SqliteEngine, method, use_clock(getattr(SqliteEngine, method))
        )
    db_path = tmp_path / "override-rebuild.sqlite3"
    configs = [_config(requests=10)]
    if request.param == "async":
        builder = SqliteBackendBuilder(
            db_path, key_prefix="rebuild", override_ttl_seconds=1
        )
        limiter = RateLimiter(lambda _model: configs[0], backend=builder)
    else:
        builder = SyncSqliteBackendBuilder(
            db_path, key_prefix="rebuild", override_ttl_seconds=1
        )
        limiter = SyncRateLimiter(lambda _model: configs[0], backend=builder)
    instance = _Harness(limiter, builder, configs, db_path, clock)
    try:
        await instance.call("record_usage", {"requests": 0}, "model")
        yield instance
    finally:
        await instance.call("aclose" if request.param == "async" else "close")


async def test_expired_override_cannot_become_static_quota(harness: _Harness):
    await harness.call("set_max_capacity", "model", "requests", 1, 20)
    assert harness.override() == (20, 101)
    harness.clock.now = 101.1
    await harness.rebuild(requests=10, tokens=100)
    harness.clock.now = 102.2

    with pytest.raises(ValueError, match="exceeds bucket max"):
        await harness.acquire_requests(11)
    assert harness.configured_max() == 10
    assert harness.override() == (None, None)


@pytest.mark.parametrize("value", [5, 20])
async def test_live_override_keeps_original_expiry(harness: _Harness, value: int):
    await harness.call("set_max_capacity", "model", "requests", 1, value)
    original_override = harness.override()
    assert original_override == (value, 101)
    harness.clock.now = 100.4
    await harness.rebuild(requests=10, tokens=100)

    assert harness.override() == original_override
    assert harness.configured_max() == 10
    harness.clock.now = 100.8
    await harness.rebuild(requests=10, tokens=100, images=100)
    assert harness.override() == original_override
    assert harness.configured_max() == 10
    harness.clock.now = 101.1
    with pytest.raises(ValueError, match="exceeds bucket max"):
        await harness.acquire_requests(11)


@pytest.mark.parametrize("local_override", [None, 20])
async def test_peer_override_written_during_rebuild_remains_authoritative(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    local_override: int | None,
):
    if local_override is not None:
        await harness.call("set_max_capacity", "model", "requests", 1, local_override)
    harness.clock.now = 100.4
    replacement_built = threading.Event()
    peer_finished = threading.Event()
    original_build = harness.builder.build

    def pause_after_build(*args, **kwargs):
        replacement = original_build(*args, **kwargs)
        replacement_built.set()
        if not peer_finished.wait(timeout=5):
            raise RuntimeError("peer did not finish its override write")
        return replacement

    def write_peer_override():
        try:
            if not replacement_built.wait(timeout=5):
                raise RuntimeError("replacement backend was not built")
            builder = SyncSqliteBackendBuilder(
                harness.db_path, key_prefix="rebuild", override_ttl_seconds=1
            )
            with SyncRateLimiter(_config(requests=10), backend=builder) as peer:
                peer.record_usage({"requests": 0}, "model")
                peer.set_max_capacity("model", "requests", 1, 30)
        finally:
            peer_finished.set()

    monkeypatch.setattr(harness.builder, "build", pause_after_build)
    with ThreadPoolExecutor(max_workers=1) as executor:
        peer_write = executor.submit(write_peer_override)
        try:
            await harness.rebuild(requests=10, tokens=100)
        finally:
            replacement_built.set()
            peer_write.result(timeout=5)

    assert harness.override() == (30, 101.4)
    assert harness.configured_max() == 10
    harness.clock.now = 101
    await harness.acquire_requests(25)


async def test_static_change_and_metric_removal_clear_overrides(harness: _Harness):
    await harness.call("set_max_capacity", "model", "requests", 1, 20)
    harness.clock.now = 100.2
    await harness.rebuild(requests=12, tokens=100)
    assert harness.configured_max() == 12
    assert harness.override() == (None, None)
    await harness.call("set_max_capacity", "model", "requests", 1, 30)
    harness.clock.now = 100.4
    await harness.rebuild(tokens=100)
    assert harness.override() == (None, None)
    await harness.rebuild(requests=10, tokens=100)
    assert harness.configured_max() == 10
    assert harness.override() == (None, None)
