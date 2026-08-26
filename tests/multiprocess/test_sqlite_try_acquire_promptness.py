"""SQLite cross-process nonblocking-acquire promptness regression."""

from __future__ import annotations

import multiprocessing

import pytest

from tests.multiprocess._harness import (
    ChildHandle,
    LimiterSpec,
    finish_child,
    receive_event,
    scaled,
    spawn_child,
    terminate_children,
)

PROBE_COUNT = 3


@pytest.mark.xfail(
    strict=True,
    reason=(
        "pending SQLite per-operation busy_timeout fix for spec-review M1; "
        "remove this mark after re-merging the engine fix"
    ),
)
def test_sqlite_try_acquire_returns_promptly_while_writer_holds_lock(
    sqlite_backend_case,
) -> None:
    """Require timeout=0 calls to return without inheriting the 5s busy timeout."""
    limiter_spec = LimiterSpec(limit=4, per_seconds=60)
    context = multiprocessing.get_context("spawn")
    release_gate = context.Event()
    probe_gate = context.Event()
    probes = [
        spawn_child(
            "timed_try_acquire",
            sqlite_backend_case.spec,
            limiter_spec,
            amount=1,
            start_gate=probe_gate,
        )
        for _ in range(PROBE_COUNT)
    ]
    children: list[ChildHandle] = list(probes)
    for probe in probes:
        receive_event(probe, "ready", timeout=scaled(15))

    holder = spawn_child(
        "hold_sqlite_write_transaction",
        sqlite_backend_case.spec,
        limiter_spec,
        release_gate=release_gate,
    )
    children.append(holder)
    results: list[dict[str, object]] = []
    try:
        receive_event(holder, "write_locked", timeout=scaled(15))
        probe_gate.set()
        results = [finish_child(probe, timeout=scaled(10)) for probe in probes]
    finally:
        probe_gate.set()
        release_gate.set()
        if holder.process.is_alive():
            finish_child(holder, timeout=scaled(5))
        terminate_children(children)

    prompt_bound = scaled(1)
    assert all(
        result["outcome"] in {"acquired", "TimeoutError"} for result in results
    ), results
    assert all(float(result["elapsed"]) < prompt_bound for result in results), results
