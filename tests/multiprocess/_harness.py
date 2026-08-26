"""Spawn-based, backend-agnostic child runner for integration scenarios."""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from token_throttle import (
    CapacityReservation,
    DuplicateRefundError,
    PerModelConfig,
    Quota,
    SyncMemoryBackendBuilder,
    SyncRateLimiter,
    SyncRateLimiterCallbacks,
    UnknownReservationError,
    UsageQuotas,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from multiprocessing.connection import Connection

MODEL = "multiprocess-model"
MODEL_FAMILY = "multiprocess-family"
RESULT_EVENT = "result"
ERROR_EVENT = "error"
MAX_TRACEBACK_CHARS = 8_000


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """Serializable instructions for constructing a backend inside a child."""

    kind: Literal["memory", "redis", "sqlite"]
    locator: str | None
    key_prefix: str
    options: tuple[tuple[str, int | float | bool], ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class LimiterSpec:
    """Serializable quota and limiter-lifetime inputs for a child."""

    limit: float
    per_seconds: int
    max_reservation_lifetime_seconds: float | None = None


@dataclass(slots=True)
class ChildHandle:
    """Parent-owned process and receive-only result channel."""

    process: multiprocessing.Process
    connection: Connection


def timing_scale() -> float:
    """Return the opt-in multiplier used for process and polling deadlines."""
    raw = os.environ.get("TOKEN_THROTTLE_MULTIPROCESS_TIMING_SCALE", "1")
    try:
        scale = float(raw)
    except ValueError as exc:
        raise ValueError(
            "TOKEN_THROTTLE_MULTIPROCESS_TIMING_SCALE must be a positive number"
        ) from exc
    if scale <= 0:
        raise ValueError(
            "TOKEN_THROTTLE_MULTIPROCESS_TIMING_SCALE must be a positive number"
        )
    return scale


def scaled(seconds: float) -> float:
    """Scale a process/polling deadline without changing quota arithmetic."""
    return seconds * timing_scale()


def _config(spec: LimiterSpec) -> PerModelConfig:
    return PerModelConfig(
        model_family=MODEL_FAMILY,
        quotas=UsageQuotas(
            [
                Quota(
                    metric="requests",
                    limit=spec.limit,
                    per_seconds=spec.per_seconds,
                )
            ]
        ),
    )


@contextlib.contextmanager
def _build_limiter(
    backend_spec: BackendSpec,
    limiter_spec: LimiterSpec,
    *,
    callbacks: SyncRateLimiterCallbacks | None = None,
) -> Iterator[SyncRateLimiter]:
    """Build all process-affine objects after the child has spawned."""
    options = dict(backend_spec.options)
    if backend_spec.kind == "redis":
        if backend_spec.locator is None:
            raise ValueError("Redis backend spec requires a URL locator")
        import redis  # noqa: PLC0415

        from token_throttle import SyncRedisBackendBuilder  # noqa: PLC0415

        client = redis.from_url(backend_spec.locator)
        builder = SyncRedisBackendBuilder(
            client,
            key_prefix=backend_spec.key_prefix,
            owns_redis_client=True,
            **options,
        )
    elif backend_spec.kind == "memory":
        builder = SyncMemoryBackendBuilder()
    elif backend_spec.kind == "sqlite":
        # This branch is the only harness edit the parallel SQLite lane needs.
        # Its public builder name is intentionally not guessed before that API lands.
        raise NotImplementedError(
            "SQLite backend builder is not available in this lane"
        )
    else:  # pragma: no cover - Literal plus spawn input validation makes this defensive.
        raise ValueError(f"Unknown backend kind: {backend_spec.kind}")

    with SyncRateLimiter(
        _config(limiter_spec),
        backend=builder,
        callbacks=callbacks,
        max_reservation_lifetime_seconds=(
            limiter_spec.max_reservation_lifetime_seconds
        ),
    ) as limiter:
        yield limiter


def _send(connection: Connection, event: str, **payload: object) -> None:
    connection.send({"event": event, **payload})


def _wait_for_gate(gate: object, timeout: float, *, name: str) -> None:
    wait = getattr(gate, "wait", None)
    if not callable(wait) or not wait(timeout=timeout):
        raise TimeoutError(f"timed out waiting for parent {name} gate")


def _greedy_acquire(
    connection: Connection,
    backend_spec: BackendSpec,
    limiter_spec: LimiterSpec,
    payload: Mapping[str, object],
) -> dict[str, object]:
    start_gate = payload["start_gate"]
    _send(connection, "ready", pid=os.getpid())
    _wait_for_gate(start_gate, scaled(15), name="start")
    started = time.monotonic()
    granted = 0
    with _build_limiter(backend_spec, limiter_spec) as limiter:
        while True:
            try:
                limiter.acquire_capacity({"requests": 1}, MODEL, timeout=0)
            except TimeoutError:
                break
            granted += 1
    return {
        "granted": granted,
        "started_monotonic": started,
        "finished_monotonic": time.monotonic(),
    }


def _hold_then_refund(
    connection: Connection,
    backend_spec: BackendSpec,
    limiter_spec: LimiterSpec,
    payload: Mapping[str, object],
) -> dict[str, object]:
    release_gate = payload["release_gate"]
    with _build_limiter(backend_spec, limiter_spec) as limiter:
        reservation = limiter.acquire_capacity(
            {"requests": limiter_spec.limit}, MODEL, timeout=0
        )
        _send(
            connection,
            "acquired",
            reservation_id=reservation.reservation_id,
            acquired_monotonic=time.monotonic(),
        )
        _wait_for_gate(release_gate, scaled(15), name="refund")
        limiter.refund_capacity({"requests": 0}, reservation)
        refunded_monotonic = time.monotonic()
    return {"refunded_monotonic": refunded_monotonic}


def _wait_for_refund(
    connection: Connection,
    backend_spec: BackendSpec,
    limiter_spec: LimiterSpec,
    payload: Mapping[str, object],
) -> dict[str, object]:
    wait_timeout = float(payload["wait_timeout"])
    waiting_sent = False

    def on_wait_start(**_kwargs: object) -> None:
        nonlocal waiting_sent
        if not waiting_sent:
            waiting_sent = True
            _send(connection, "waiting", waiting_monotonic=time.monotonic())

    callbacks = SyncRateLimiterCallbacks(on_wait_start=on_wait_start)
    started = time.monotonic()
    with _build_limiter(backend_spec, limiter_spec, callbacks=callbacks) as limiter:
        reservation = limiter.acquire_capacity(
            {"requests": 1}, MODEL, timeout=wait_timeout
        )
        reservation_id = reservation.reservation_id
    return {
        "reservation_id": reservation_id,
        "elapsed": time.monotonic() - started,
        "finished_monotonic": time.monotonic(),
    }


def _hold_until_killed(
    connection: Connection,
    backend_spec: BackendSpec,
    limiter_spec: LimiterSpec,
    payload: Mapping[str, object],
) -> dict[str, object]:
    with _build_limiter(backend_spec, limiter_spec) as limiter:
        reservation = limiter.acquire_capacity(
            {"requests": limiter_spec.limit}, MODEL, timeout=0
        )
        _send(
            connection,
            "acquired",
            acquired_monotonic=time.monotonic(),
            reservation_json=reservation.model_dump_json(),
            reservation_id=reservation.reservation_id,
        )
        # This event is deliberately process-local. A shared multiprocessing
        # Condition can be left permanently wedged if SIGKILL lands while the
        # child is inside Condition.wait(), making parent cleanup deadlock.
        threading.Event().wait(timeout=scaled(3_600))
    return {"unexpected_release": True}


def _try_acquire(
    connection: Connection,
    backend_spec: BackendSpec,
    limiter_spec: LimiterSpec,
    payload: Mapping[str, object],
) -> dict[str, object]:
    amount = float(payload["amount"])
    timeout = float(payload.get("timeout", 0.0))
    started = time.monotonic()
    with _build_limiter(backend_spec, limiter_spec) as limiter:
        try:
            reservation = limiter.acquire_capacity(
                {"requests": amount}, MODEL, timeout=timeout
            )
        except TimeoutError:
            return {
                "acquired": False,
                "elapsed": time.monotonic() - started,
                "finished_monotonic": time.monotonic(),
            }
        return {
            "acquired": True,
            "elapsed": time.monotonic() - started,
            "finished_monotonic": time.monotonic(),
            "reservation_id": reservation.reservation_id,
        }


def _replay_refund(
    connection: Connection,
    backend_spec: BackendSpec,
    limiter_spec: LimiterSpec,
    payload: Mapping[str, object],
) -> dict[str, object]:
    reservation = CapacityReservation.model_validate_json(
        str(payload["reservation_json"])
    )
    with _build_limiter(backend_spec, limiter_spec) as limiter:
        # Deliberate fault-injection seam: ordinary public use must not move a
        # reservation between limiter instances. Matching the trusted issuing ID
        # lets this scenario reach the shared backend's expired-marker decision.
        limiter._limiter_instance_id = reservation.limiter_instance_id
        try:
            limiter.refund_capacity({"requests": 0}, reservation)
        except (UnknownReservationError, DuplicateRefundError) as exc:
            return {
                "exception_type": type(exc).__name__,
                "reason": exc.reason,
            }
    return {"exception_type": None, "reason": None}


def _contention_cycles(
    connection: Connection,
    backend_spec: BackendSpec,
    limiter_spec: LimiterSpec,
    payload: Mapping[str, object],
) -> dict[str, object]:
    start_gate = payload["start_gate"]
    active = payload["active"]
    max_active = payload["max_active"]
    cycles = int(payload["cycles"])
    hold_seconds = float(payload["hold_seconds"])
    wait_timeout = float(payload["wait_timeout"])
    _send(connection, "ready", pid=os.getpid())
    _wait_for_gate(start_gate, scaled(15), name="start")
    reservation_ids: set[str] = set()
    with _build_limiter(backend_spec, limiter_spec) as limiter:
        for _ in range(cycles):
            reservation = limiter.acquire_capacity(
                {"requests": 1}, MODEL, timeout=wait_timeout
            )
            reservation_ids.add(reservation.reservation_id)
            active_lock = active.get_lock()
            with active_lock:
                active.value += 1
                max_active.value = max(max_active.value, active.value)
            try:
                time.sleep(hold_seconds)
                limiter.refund_capacity({"requests": 0}, reservation)
            finally:
                with active_lock:
                    active.value -= 1
    return {
        "cycles": cycles,
        "distinct_reservations": len(reservation_ids),
    }


_ACTIONS = {
    "contention_cycles": _contention_cycles,
    "greedy_acquire": _greedy_acquire,
    "hold_then_refund": _hold_then_refund,
    "hold_until_killed": _hold_until_killed,
    "replay_refund": _replay_refund,
    "try_acquire": _try_acquire,
    "wait_for_refund": _wait_for_refund,
}


def _child_main(
    connection: Connection,
    action: str,
    backend_spec: BackendSpec,
    limiter_spec: LimiterSpec,
    payload: Mapping[str, object],
) -> None:
    try:
        handler = _ACTIONS[action]
        result = handler(connection, backend_spec, limiter_spec, payload)
        _send(connection, RESULT_EVENT, **result)
    except BaseException as exc:
        _send(
            connection,
            ERROR_EVENT,
            exception_type=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(limit=12)[-MAX_TRACEBACK_CHARS:],
        )
    finally:
        connection.close()


def spawn_child(
    action: str,
    backend_spec: BackendSpec,
    limiter_spec: LimiterSpec,
    **payload: object,
) -> ChildHandle:
    """Start a fresh-interpreter worker with a bounded one-way result pipe."""
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_main,
        args=(send_connection, action, backend_spec, limiter_spec, payload),
        name=f"token-throttle-mp-{action}",
    )
    process.start()
    send_connection.close()
    return ChildHandle(process=process, connection=receive_connection)


def receive_event(
    child: ChildHandle,
    expected_event: str,
    *,
    timeout: float,
) -> dict[str, object]:
    """Receive one child message before a deadline and validate its event type."""
    if not child.connection.poll(timeout):
        raise AssertionError(
            f"child {child.process.name!r} did not report {expected_event!r} "
            f"within {timeout:g}s (alive={child.process.is_alive()}, "
            f"exitcode={child.process.exitcode})"
        )
    try:
        message = child.connection.recv()
    except EOFError as exc:
        raise AssertionError(
            f"child {child.process.name!r} closed its result pipe before "
            f"reporting {expected_event!r} (exitcode={child.process.exitcode})"
        ) from exc
    if not isinstance(message, dict):
        raise AssertionError(  # noqa: TRY004
            f"child sent non-dictionary result: {message!r}"
        )
    event = message.get("event")
    if event == ERROR_EVENT:
        raise AssertionError(
            f"child {child.process.name!r} failed with "
            f"{message.get('exception_type')}: {message.get('message')}\n"
            f"{message.get('traceback')}"
        )
    if event != expected_event:
        raise AssertionError(
            f"child {child.process.name!r} reported event {event!r}; "
            f"expected {expected_event!r}: {message!r}"
        )
    return message


def finish_child(child: ChildHandle, *, timeout: float) -> dict[str, object]:
    """Receive the terminal result, then join without permitting a suite hang."""
    result = receive_event(child, RESULT_EVENT, timeout=timeout)
    child.process.join(timeout=timeout)
    if child.process.is_alive():
        child.process.terminate()
        child.process.join(timeout=scaled(3))
        raise AssertionError(
            f"child {child.process.name!r} reported a result but did not exit "
            f"within {timeout:g}s"
        )
    child.connection.close()
    if child.process.exitcode != 0:
        raise AssertionError(
            f"child {child.process.name!r} exited with {child.process.exitcode}"
        )
    return result


def terminate_children(children: list[ChildHandle]) -> None:
    """Best-effort cleanup for children left alive by a failed assertion."""
    for child in children:
        if child.process.is_alive():
            child.process.terminate()
    for child in children:
        child.process.join(timeout=scaled(3))
        child.connection.close()
