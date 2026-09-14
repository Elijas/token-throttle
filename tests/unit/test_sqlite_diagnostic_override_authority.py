from __future__ import annotations

import asyncio
import inspect
import threading
from functools import partialmethod

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
from token_throttle._diagnostic import _reconcile_bucket, make_bucket_diagnostic
from token_throttle._limiter_backends._sqlite._engine import SqliteEngine


async def _call(method, *args, **kwargs):
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    return await asyncio.to_thread(method, *args, **kwargs)


@pytest.fixture(params=["sync", "async"])
async def sqlite_limiter(request, tmp_path, monkeypatch):
    now = [100.0]
    for method_name in (
        "initialize_buckets",
        "consume",
        "set_max_capacity",
        "inspect_snapshot",
    ):
        monkeypatch.setattr(
            SqliteEngine,
            method_name,
            partialmethod(getattr(SqliteEngine, method_name), clock=lambda: now[0]),
        )
    config = PerModelConfig(
        model_family="diagnostic-overrides",
        quotas=UsageQuotas([Quota(metric="requests", limit=10, per_seconds=60)]),
    )
    builder_class, limiter_class = (
        (SyncSqliteBackendBuilder, SyncRateLimiter)
        if request.param == "sync"
        else (SqliteBackendBuilder, RateLimiter)
    )
    builder = builder_class(
        tmp_path / "diagnostic-overrides.sqlite3",
        key_prefix="diagnostic-overrides",
        bucket_ttl_seconds=120,
        override_ttl_seconds=1,
    )
    limiter = limiter_class(config, backend=builder)
    try:
        await _call(limiter.record_usage, {"requests": 0}, model="model")
        await _call(limiter.set_max_capacity, "model", "requests", 60, 1)
        backend = limiter._model_family_to_backend[config.model_family]
        yield limiter, backend, builder, config, now
    finally:
        close = limiter.close if request.param == "sync" else limiter.aclose
        await _call(close)


@pytest.mark.parametrize(
    ("elapsed", "expected_status"), [(1.1, "ok"), (121.0, "fresh_start")]
)
async def test_sqlite_diagnose_drops_expired_local_override(
    sqlite_limiter, monkeypatch, elapsed, expected_status
):
    limiter, backend, _builder, config, now = sqlite_limiter
    before = await _call(limiter.diagnose)
    assert before.buckets[0].runtime_override == 1
    assert before.buckets[0].override_source == "both"

    read_started = threading.Event()
    clock_advanced = threading.Event()
    inspect_snapshot = backend._engine.inspect_snapshot

    def inspect_after_clock_advance():
        read_started.set()
        assert clock_advanced.wait(5), "test did not release the diagnostic read"
        return inspect_snapshot()

    monkeypatch.setattr(
        backend._engine, "inspect_snapshot", inspect_after_clock_advance
    )
    pending = asyncio.create_task(_call(limiter.diagnose))
    try:
        assert await asyncio.to_thread(read_started.wait, 5)
        now[0] += elapsed
    finally:
        clock_advanced.set()
        diagnostic = await asyncio.wait_for(pending, 5)

    authoritative = (await _call(backend.introspect)).buckets[0]
    assert authoritative.runtime_override is None
    assert authoritative.effective_max_capacity == 10
    assert authoritative.status == expected_status
    assert limiter._model_family_to_runtime_max_capacity[config.model_family] == {
        ("requests", 60): 1
    }

    bucket = diagnostic.buckets[0]
    assert bucket.effective_max_capacity == authoritative.effective_max_capacity
    assert bucket.refill_rate_per_second == pytest.approx(10 / 60)
    assert bucket.runtime_override is None
    assert bucket.override_source == "none"
    assert bucket.configured_limit == 10
    assert bucket.configured_to_effective_gap == 0
    assert bucket.current_capacity == authoritative.current_capacity
    assert bucket.status == expected_status
    assert diagnostic.runtime_overrides == ()
    assert diagnostic.issues == ()


@pytest.mark.parametrize("replacement", [7, 10])
async def test_sqlite_diagnose_trusts_active_shared_override(
    sqlite_limiter, replacement
):
    limiter, _backend, builder, config, _now = sqlite_limiter
    other_backend = await _call(builder.build, config)
    await _call(other_backend.set_max_capacity, "requests", 60, replacement)

    diagnostic = await _call(limiter.diagnose)

    bucket = diagnostic.buckets[0]
    assert bucket.runtime_override == replacement
    assert bucket.effective_max_capacity == replacement
    assert bucket.refill_rate_per_second == pytest.approx(replacement / 60)
    assert bucket.override_source == "both"
    assert diagnostic.runtime_overrides[0].override_capacity == replacement
    assert any("overrides differ" in issue.message for issue in diagnostic.issues)
    assert limiter._model_family_to_runtime_max_capacity[config.model_family] == {
        ("requests", 60): 1
    }


@pytest.mark.parametrize("failure", ["unsupported", "raises", "omitted"])
async def test_sqlite_diagnose_keeps_local_fallback_when_introspection_fails(
    sqlite_limiter, monkeypatch, failure
):
    limiter, backend, _builder, _config, now = sqlite_limiter
    now[0] += 1.1
    authoritative = await _call(backend.introspect)
    assert authoritative.buckets[0].runtime_override is None

    def unavailable_introspection():
        if failure == "raises":
            raise RuntimeError("diagnostic read unavailable")
        return authoritative.model_copy(update={"buckets": ()})

    monkeypatch.setattr(
        backend,
        "introspect",
        None if failure == "unsupported" else unavailable_introspection,
    )
    diagnostic = await _call(limiter.diagnose)

    bucket = diagnostic.buckets[0]
    assert bucket.runtime_override == 1
    assert bucket.effective_max_capacity == 1
    assert bucket.refill_rate_per_second == pytest.approx(1 / 60)
    assert bucket.override_source == "limiter"
    assert bucket.status == "missing"
    assert bucket.current_capacity is None
    assert diagnostic.runtime_overrides[0].override_capacity == 1
    assert any("not returned" in issue.message for issue in diagnostic.issues)
    if failure == "raises":
        assert any(
            "introspect() failed" in issue.message for issue in diagnostic.issues
        )
    elif failure == "unsupported":
        assert any("does not implement" in issue.message for issue in diagnostic.issues)


@pytest.mark.parametrize(
    "backend_type", ["sqlite", "memory", "redis", "custom", "unknown"]
)
@pytest.mark.parametrize(
    "status",
    [
        "ok",
        "fresh_start",
        "missing",
        "partial_missing",
        "state_loss",
        "corrupt",
        "unavailable",
    ],
)
def test_diagnostic_override_absence_is_authoritative_for_readable_sqlite(
    backend_type, status
):
    backend_bucket = make_bucket_diagnostic(
        model_family="diagnostic-overrides",
        metric="requests",
        per_seconds=60,
        backend_type=backend_type,
        current_capacity=(
            0
            if status in {"partial_missing", "state_loss"}
            else 10
            if status in {"ok", "fresh_start"}
            else None
        ),
        configured_limit=10,
        effective_max_capacity=10,
        override_source="none",
        status=status,
        as_of_monotonic=100,
    )
    issues = []
    bucket = _reconcile_bucket(
        backend_bucket, configured_limit=10, local_override=1, issues=issues
    )

    authoritative = backend_type == "sqlite" and status in {
        "ok",
        "fresh_start",
        "partial_missing",
        "state_loss",
    }
    assert bucket.effective_max_capacity == (10 if authoritative else 1)
    assert bucket.runtime_override == (None if authoritative else 1)
    assert bucket.override_source == ("none" if authoritative else "limiter")
    assert bucket.current_capacity == backend_bucket.current_capacity
    assert bucket.status == status
    assert issues == []
