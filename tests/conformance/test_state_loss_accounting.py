"""Shared missing-state metadata and repair invariants for built-in backends."""

from __future__ import annotations

import inspect
import sqlite3

import pytest

from tests.conformance import test_cross_backend_accounting_traces as accounting
from token_throttle import RateLimiterCallbacks, SyncRateLimiterCallbacks

trace = accounting.trace


def _capture_events(trace):
    events = []

    async def asynchronous(**kwargs):
        events.append(kwargs)

    def synchronous(**kwargs):
        events.append(kwargs)

    cls = RateLimiterCallbacks if trace.asynchronous else SyncRateLimiterCallbacks
    trace.backend._callbacks = cls(
        on_missing_consumption_data=(
            asynchronous if trace.asynchronous else synchronous
        )
    )
    return events


async def _delete_requests_state(trace, fields):
    backend = trace.backend
    if hasattr(backend, "_engine"):
        with sqlite3.connect(backend._engine.db_path) as connection:
            if len(fields) == 2:
                connection.execute("DELETE FROM buckets WHERE metric = 'requests'")
            else:
                column = fields[0]
                assert column in {"capacity", "last_checked"}
                connection.execute(
                    f"UPDATE buckets SET {column} = NULL WHERE metric = 'requests'"  # noqa: S608
                )
    else:
        bucket = next(b for b in backend.sorted_buckets if b.usage_metric == "requests")
        keys = [getattr(bucket, f"_{field}_key") for field in fields]
        outcome = bucket._redis.delete(*keys)
        if inspect.isawaitable(outcome):
            await outcome


async def test_missing_state_fresh_start_metadata_matches_all_backends(trace):
    events = _capture_events(trace)
    await trace.acquire(6, 2)
    assert len(events) == 2
    assert {event["usage_metric"] for event in events} == {"requests", "tokens"}
    for event in events:
        assert event["missing_state_reason"] == "fresh_start"
        assert event["missing_state_keys"] == ("last_checked", "capacity")
        assert event["present_state_keys"] == ()


@pytest.mark.parametrize(
    "fields", [("last_checked", "capacity"), ("last_checked",), ("capacity",)]
)
async def test_detected_state_loss_drains_only_affected_bucket(trace, fields):
    if type(trace.backend).__name__ in {"MemoryBackend", "SyncMemoryBackend"}:
        pytest.skip("Memory has no external persistent bucket store to delete")
    events = _capture_events(trace)
    await trace.acquire(6, 2)
    events.clear()
    await _delete_requests_state(trace, fields)

    for _ in range(2):
        diagnostic = await trace.call("introspect")
        buckets = {bucket.metric: bucket for bucket in diagnostic.buckets}
        assert buckets["requests"].current_capacity == 0
        assert buckets["requests"].status == (
            "state_loss" if len(fields) == 2 else "partial_missing"
        )
        assert buckets["tokens"].current_capacity == pytest.approx(18)
    assert events == [], "read-only diagnostics must not emit repair callbacks"

    with pytest.raises(TimeoutError):
        await trace.acquire(1, 0)
    assert len(events) == 1
    assert events[0]["missing_state_reason"] == "state_loss_drained"
    assert events[0]["missing_state_keys"] == fields
    assert events[0]["present_state_keys"] == tuple(
        field for field in ("last_checked", "capacity") if field not in fields
    )
    await trace.capacities(0, 18)
    with pytest.raises(TimeoutError):
        await trace.acquire(1, 0)
    assert len(events) == 1, "persisted repair must not report the same loss again"
    trace.clock[0] += 1
    await trace.acquire(1, 0)
    await trace.capacities(0, 19)
