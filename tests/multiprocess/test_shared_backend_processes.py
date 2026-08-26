"""
Real-process shared-backend scenarios.

Workers use ``multiprocessing.get_context("spawn")`` rather than ``fork``.
Spawn is portable and imports this module in a fresh interpreter, while still
supporting deterministic start gates. More importantly, every worker constructs
its own limiter, backend builder, and backend client after the process boundary;
none of those documented process-affine objects are inherited or pickled.

Results are small dictionaries sent over receive-only parent pipes. Every poll,
gate wait, and join has a deadline, and failed tests terminate only their owned
children. Assertions use deadline polling and quota-derived refill tolerances,
not exact scheduler sleeps.

CI intent: run this directory in the Linux Redis ``test-integration`` job with
``--redis-url redis://localhost:6379/13``. Allow roughly twice the ordinary
integration-test wall time. Slow runners may set
``TOKEN_THROTTLE_MULTIPROCESS_TIMING_SCALE=2`` to scale process/join deadlines;
quota/refill arithmetic remains unchanged. CI workflow edits are intentionally
outside this lane.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import signal
import time

import pytest

from tests.multiprocess._harness import (
    MODEL_FAMILY,
    ChildHandle,
    LimiterSpec,
    finish_child,
    receive_event,
    scaled,
    spawn_child,
    terminate_children,
    timing_scale,
)


def _finish_all(
    children: list[ChildHandle], *, timeout: float
) -> list[dict[str, object]]:
    deadline = time.monotonic() + timeout
    results = []
    for child in children:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(
                f"children did not all finish within shared {timeout:g}s deadline"
            )
        results.append(finish_child(child, timeout=remaining))
    return results


def test_cross_process_budget_enforcement(shared_backend_case) -> None:
    process_count = 4
    limit = 8
    per_seconds = 60
    limiter_spec = LimiterSpec(limit=limit, per_seconds=per_seconds)
    context = multiprocessing.get_context("spawn")
    start_gate = context.Event()
    children = [
        spawn_child(
            "greedy_acquire",
            shared_backend_case.spec,
            limiter_spec,
            start_gate=start_gate,
        )
        for _ in range(process_count)
    ]
    try:
        for child in children:
            receive_event(child, "ready", timeout=scaled(15))
        released_at = time.monotonic()
        start_gate.set()
        results = _finish_all(children, timeout=scaled(20))
    finally:
        terminate_children(children)

    finished_at = max(float(result["finished_monotonic"]) for result in results)
    elapsed = finished_at - released_at
    refill_tolerance = math.ceil(elapsed * limit / per_seconds)
    total_granted = sum(int(result["granted"]) for result in results)
    assert total_granted >= limit
    assert total_granted <= limit + refill_tolerance


def test_cross_process_refund_unblocks_waiter(shared_backend_case) -> None:
    limit = 8
    wait_timeout = scaled(10)
    limiter_spec = LimiterSpec(limit=limit, per_seconds=60)
    context = multiprocessing.get_context("spawn")
    refund_gate = context.Event()
    holder = spawn_child(
        "hold_then_refund",
        shared_backend_case.spec,
        limiter_spec,
        release_gate=refund_gate,
    )
    children = [holder]
    try:
        receive_event(holder, "acquired", timeout=scaled(15))
        waiter = spawn_child(
            "wait_for_refund",
            shared_backend_case.spec,
            limiter_spec,
            wait_timeout=wait_timeout,
        )
        children.append(waiter)
        receive_event(waiter, "waiting", timeout=scaled(15))
        refund_gate.set()
        holder_result = finish_child(holder, timeout=scaled(15))
        waiter_result = finish_child(waiter, timeout=scaled(15))
    finally:
        refund_gate.set()
        terminate_children(children)

    assert float(waiter_result["finished_monotonic"]) >= float(
        holder_result["refunded_monotonic"]
    )
    assert float(waiter_result["elapsed"]) < wait_timeout * 0.75


@pytest.mark.skipif(os.name != "posix", reason="requires exact POSIX SIGKILL semantics")
def test_killed_reservation_recovers_only_by_refill(shared_backend_case) -> None:
    limit = 4
    per_seconds = max(12, math.ceil(12 * timing_scale()))
    acquiring_spec = LimiterSpec(
        limit=limit,
        per_seconds=per_seconds,
        max_reservation_lifetime_seconds=0.75,
    )
    replay_spec = LimiterSpec(
        limit=limit,
        per_seconds=per_seconds,
        max_reservation_lifetime_seconds=5.0 * timing_scale(),
    )
    holder = spawn_child(
        "hold_until_killed",
        shared_backend_case.spec,
        acquiring_spec,
    )
    children = [holder]
    try:
        acquired = receive_event(holder, "acquired", timeout=scaled(15))
        reservation_id = str(acquired["reservation_id"])
        reservation_json = str(acquired["reservation_json"])
        acquired_at = float(acquired["acquired_monotonic"])

        assert holder.process.pid is not None
        os.kill(holder.process.pid, signal.SIGKILL)
        holder.process.join(timeout=scaled(5))
        assert not holder.process.is_alive()
        assert holder.process.exitcode == -signal.SIGKILL
        holder.connection.close()

        shared_backend_case.wait_until_marker_expires(reservation_id, timeout=scaled(5))

        full_probe = spawn_child(
            "try_acquire",
            shared_backend_case.spec,
            replay_spec,
            amount=limit,
            timeout=0,
        )
        children.append(full_probe)
        full_result = finish_child(full_probe, timeout=scaled(15))
        assert float(full_result["finished_monotonic"]) - acquired_at < per_seconds
        assert full_result["acquired"] is False

        replay = spawn_child(
            "replay_refund",
            shared_backend_case.spec,
            replay_spec,
            reservation_json=reservation_json,
        )
        children.append(replay)
        replay_result = finish_child(replay, timeout=scaled(15))
        assert replay_result["exception_type"] == "UnknownReservationError"
        assert replay_result["reason"] == "unknown_reservation"

        refill_probe = spawn_child(
            "try_acquire",
            shared_backend_case.spec,
            replay_spec,
            amount=1,
            timeout=scaled(6),
        )
        children.append(refill_probe)
        refill_result = finish_child(refill_probe, timeout=scaled(15))
        assert refill_result["acquired"] is True
    finally:
        terminate_children(children)


def test_contention_acquire_refund_cycles_preserve_capacity(
    shared_backend_case,
) -> None:
    process_count = 6
    cycles = 10
    limit = 4
    limiter_spec = LimiterSpec(limit=limit, per_seconds=60)
    context = multiprocessing.get_context("spawn")
    start_gate = context.Event()
    active = context.Value("i", 0)
    max_active = context.Value("i", 0)
    children = [
        spawn_child(
            "contention_cycles",
            shared_backend_case.spec,
            limiter_spec,
            start_gate=start_gate,
            active=active,
            max_active=max_active,
            cycles=cycles,
            hold_seconds=scaled(0.01),
            wait_timeout=scaled(10),
        )
        for _ in range(process_count)
    ]
    try:
        for child in children:
            receive_event(child, "ready", timeout=scaled(15))
        start_gate.set()
        results = _finish_all(children, timeout=scaled(30))
    finally:
        start_gate.set()
        terminate_children(children)

    assert active.value == 0
    assert 1 <= max_active.value <= limit
    assert sum(int(result["cycles"]) for result in results) == process_count * cycles
    assert all(int(result["distinct_reservations"]) == cycles for result in results)

    probe_gate = context.Event()
    final_probe = spawn_child(
        "greedy_acquire",
        shared_backend_case.spec,
        limiter_spec,
        start_gate=probe_gate,
    )
    probe_children = [final_probe]
    try:
        receive_event(final_probe, "ready", timeout=scaled(15))
        probe_gate.set()
        final_result = finish_child(final_probe, timeout=scaled(15))
    finally:
        terminate_children(probe_children)
    assert int(final_result["granted"]) == limit


def test_fresh_interpreter_observes_prior_consumption(
    fresh_interpreter_backend_case,
) -> None:
    limiter_spec = LimiterSpec(limit=2, per_seconds=60)
    first = spawn_child(
        "try_acquire",
        fresh_interpreter_backend_case.spec,
        limiter_spec,
        amount=2,
        timeout=0,
    )
    children = [first]
    try:
        first_result = finish_child(first, timeout=scaled(15))
        assert first_result["acquired"] is True

        second = spawn_child(
            "try_acquire",
            fresh_interpreter_backend_case.spec,
            limiter_spec,
            amount=1,
            timeout=0,
        )
        children.append(second)
        second_result = finish_child(second, timeout=scaled(15))
    finally:
        terminate_children(children)

    assert second_result["acquired"] is False, (
        f"{fresh_interpreter_backend_case.spec.kind} did not preserve shared "
        f"state for {MODEL_FAMILY} across fresh interpreters"
    )
