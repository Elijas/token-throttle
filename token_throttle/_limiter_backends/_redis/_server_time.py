"""
Redis server-time helpers - anchor refill math to the Redis server's clock.

All Redis backend code that writes timestamps to shared state uses the TIME
command instead of the local ``time.time()``. With a single Redis primary this
guarantees workers see the same "now" regardless of host-clock skew.

Round-trip cost
---------------
TIME runs as its own round-trip BEFORE any pipeline of writes - every
Redis-backed acquire, set_capacity, and refund pays 2x the Redis network
latency. This is intentional: pipelined commands like ``SET _last_checked_key
<current_time>`` need the timestamp before they can be queued. Folding TIME
into the same pipeline would require executing the pipeline first to learn
the timestamp, then queueing the follow-ups, then executing again - same
total round-trip count.

Forward-jump detection
----------------------
A server-side clock that jumps forward silently inflates ``time_passed`` in
``calculate_capacity`` and over-grants bucket capacity. We detect a genuine
jump by comparing consecutive TIME readings against locally-elapsed
*monotonic* time (per Redis client): the rail trips only when the server
clock advanced more than ``MAX_FORWARD_JUMP_SECONDS`` beyond the monotonic
interval between two readings. Each reading's own round-trip is bounded into
that interval (the anchor stores the pre-call monotonic reading; elapsed time
is measured to the post-reply reading), so one slow TIME reply widens the
tolerance instead of tripping the rail. The first reading for a client
establishes the baseline and never raises.

An anomalous reading has two possible causes, and the local *wall* clock
discriminates them. A suspended host (paused VM, laptop sleep, live
migration) stalls the monotonic clock while real time keeps passing, so on
resume the wall clock corroborates the server's advance: we re-anchor, log a
warning, and continue, because refill math uses Redis server time exclusively
and a stalled local clock is harmless to correctness. A genuine server-side
jump (realistically a Sentinel/managed failover to a clock-skewed primary)
leaves wall and monotonic in agreement while the server leaps: we raise,
exactly once per event, because the anchor advances to the jumped reading
before raising and concurrent callers holding readings from around the same
event compare against the new baseline instead of re-raising. Readings older than the stored server anchor
(out-of-order arrivals under concurrency, or a backward server jump, which
never over-grants) never move the anchor backward.

A *lagging local wall clock* (NTP outage, container drift) never hard-fails
an operation. When server-vs-wall skew exceeds the threshold we emit a
one-time warning per client, because it often points at NTP problems worth
investigating, but we never raise on it.

Bare client only
----------------
``client`` MUST be a non-pipelined ``Redis`` instance. ``Pipeline`` is a
``Redis`` subclass, so the static type accepts it; runtime rejection
prevents the silent failure where TIME would be queued (not executed) and
the unpack would receive an empty queued-command list.

TIME response shape contract
----------------------------
``async_server_time`` / ``sync_server_time`` are called by the Redis backends
and buckets with the ``redis_client`` the user supplied - a
``redis.asyncio.Redis`` / ``redis.Redis`` instance or a compatible substitute
(e.g. fakeredis, test doubles). Whatever that client is, its ``time()`` must
return TIME as a 2-element tuple/list of integer values with ``seconds >= 0``
and ``0 <= microseconds < 1_000_000``. Off-shape responses raise ``TypeError``
or ``ValueError`` instead of producing wrong-but-not-erroring math.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from typing import Any

try:
    import redis.asyncio.client
    import redis.client
except ImportError as exc:
    raise ImportError(
        'The "redis" package is required for the Redis backend. '
        'Install it with: pip install "token-throttle[redis]"'
    ) from exc


_logger = logging.getLogger("token_throttle")


MAX_FORWARD_JUMP_SECONDS = 10.0
"""Tolerated forward divergence of the Redis clock between consecutive readings.

Catches a Sentinel/managed failover to a clock-skewed primary: a forward jump
silently inflates ``time_passed`` in ``calculate_capacity`` and over-grants
bucket capacity. Compared against locally-elapsed monotonic time (with each
reading's round-trip bounded into the interval), so neither a lagging local
wall clock nor one slow TIME reply can trip it; a suspended/resumed host is
recognized via the wall clock and re-anchored instead of raising.
NTP-disciplined primaries diverge by < 100 ms in practice; 10 s is a
permissive sanity rail, not a tightness guarantee.
"""

# KNOWN UNKNOWN: 10s threshold is set by intuition, not measured P99.9
# inter-primary skew under NTP partition. Tune if real-world data emerges.

_TIME_RESPONSE_LEN = 2  # Redis TIME returns (seconds, microseconds)
_MICROSECONDS_PER_SECOND = 1_000_000
_MAX_REDIS_TIME_SECONDS = 253_402_300_799  # 9999-12-31T23:59:59Z


class _ClientClockState:
    """
    Per-client anchor for consecutive-reading forward-jump detection.

    ``last_monotonic`` / ``last_wall`` hold the local clock readings taken just
    *before* the anchored TIME command was issued, so the interval measured to
    a later reading's post-reply clocks is an upper bound on the real time
    between the two server samplings (both round-trips fall inside it).
    """

    __slots__ = ("last_monotonic", "last_server_time", "last_wall", "wall_skew_warned")

    def __init__(self, server_time: float, monotonic: float, wall: float) -> None:
        self.last_server_time = server_time
        self.last_monotonic = monotonic
        self.last_wall = wall
        self.wall_skew_warned = False


# State is keyed by the Redis client so it lives exactly as long as the client
# (real redis-py clients and the test mocks are weakref-able). The lock guards
# both the sync and async paths; the critical section is pure CPU work with no
# await/blocking I/O, so holding it briefly on the event loop is fine.
_client_states: weakref.WeakKeyDictionary[Any, _ClientClockState] = (
    weakref.WeakKeyDictionary()
)
_state_lock = threading.Lock()


def _reject_pipeline(client: object) -> None:
    if isinstance(client, (redis.client.Pipeline, redis.asyncio.client.Pipeline)):
        raise TypeError(
            "server_time helpers require a bare Redis client, not a Pipeline. "
            "On a Pipeline, TIME would be queued instead of executed and the "
            "unpack would fail. Pass the underlying client instead."
        )


def _parse_time_response(raw: Any) -> tuple[int, int]:
    if not isinstance(raw, (tuple, list)) or len(raw) != _TIME_RESPONSE_LEN:
        raise TypeError(
            f"Redis TIME returned unexpected shape: {raw!r} "
            f"(expected a 2-element tuple/list of integers)"
        )
    raw_seconds, raw_microseconds = raw
    if type(raw_seconds) is not int or type(raw_microseconds) is not int:
        raise TypeError(
            "Redis TIME components must be integer-coercible Redis integer "
            f"components, got {raw!r}"
        )
    seconds = raw_seconds
    microseconds = raw_microseconds
    if (
        seconds < 0
        or seconds > _MAX_REDIS_TIME_SECONDS
        or not (0 <= microseconds < _MICROSECONDS_PER_SECOND)
    ):
        raise ValueError(
            f"Redis TIME out of range: seconds={seconds}, "
            f"microseconds={microseconds} (need 0 <= seconds <= "
            f"{_MAX_REDIS_TIME_SECONDS} and "
            f"0 <= microseconds < {_MICROSECONDS_PER_SECOND})"
        )
    return seconds, microseconds


def _to_float(seconds: int, microseconds: int) -> float:
    try:
        return seconds + microseconds / _MICROSECONDS_PER_SECOND
    except OverflowError as exc:
        raise ValueError(f"Redis TIME out of range: seconds={seconds}") from exc


def _maybe_warn_wall_skew(
    state: _ClientClockState, server_time: float, wall_now: float
) -> None:
    """Log once per client when server-vs-wall skew is large (never raises)."""
    if state.wall_skew_warned:
        return
    skew = server_time - wall_now
    if abs(skew) <= MAX_FORWARD_JUMP_SECONDS:
        return
    state.wall_skew_warned = True
    _logger.warning(
        "Redis server TIME (%.3f) diverges from the local wall clock (%.3f) "
        "by %.1fs, exceeding MAX_FORWARD_JUMP_SECONDS=%ss. Refill math uses "
        "Redis server time, so correctness is unaffected, but a skew this "
        "large often signals NTP trouble on this host (paused/resumed VM, "
        "container clock drift, NTP outage) worth investigating. Logged once "
        "per Redis client.",
        server_time,
        wall_now,
        skew,
        MAX_FORWARD_JUMP_SECONDS,
    )


def _check_forward_jump(
    client: object,
    server_time: float,
    monotonic_before: float,
    wall_before: float,
) -> None:
    """
    Raise on a genuine server-side forward jump between consecutive readings.

    ``monotonic_before`` / ``wall_before`` were read just before the TIME
    command was issued; the post-reply clocks are read here, inside the lock.
    Anchors keep the *before* readings while elapsed time is measured to the
    *after* readings, so each measured interval is an upper bound on the real
    time between the two server samplings - a slow TIME reply widens the
    tolerance instead of tripping the rail. The first reading for a client
    only establishes the baseline.
    """
    with _state_lock:
        # Post-reply clocks are read inside the lock, so they are ordered
        # after the anchored reading's pre-call clocks even under concurrency
        # (guarantees non-negative monotonic elapsed).
        now_monotonic = time.monotonic()
        wall_now = time.time()

        state = _client_states.get(client)
        if state is None:
            state = _ClientClockState(server_time, monotonic_before, wall_before)
            _client_states[client] = state
            _maybe_warn_wall_skew(state, server_time, wall_now)
            return

        _maybe_warn_wall_skew(state, server_time, wall_now)

        if server_time < state.last_server_time:
            # Stale out-of-order reading (a concurrent caller already anchored
            # a newer one) or a backward server jump, which never over-grants.
            # Regressing the anchor would make the next in-order reading look
            # like a fresh forward jump, so keep the anchor where it is.
            return

        server_elapsed = server_time - state.last_server_time
        monotonic_elapsed = now_monotonic - state.last_monotonic
        wall_elapsed = wall_now - state.last_wall

        # Advance the anchor before deciding the outcome so a detected anomaly
        # is consumed exactly once: concurrent callers holding readings from
        # around the same event compare against this new baseline instead of
        # re-reporting the same jump.
        state.last_server_time = server_time
        state.last_monotonic = monotonic_before
        state.last_wall = wall_before

        excess = server_elapsed - monotonic_elapsed
        if excess <= MAX_FORWARD_JUMP_SECONDS:
            return

        # The server advanced far beyond locally-elapsed monotonic time. The
        # local wall clock discriminates the two causes: a suspended/resumed
        # host stalls the monotonic clock while real time keeps passing, so
        # the wall clock corroborates the server; a genuine server-side jump
        # leaves wall and monotonic in agreement while the server leaps.
        if server_elapsed - wall_elapsed <= MAX_FORWARD_JUMP_SECONDS:
            _logger.warning(
                "Redis server TIME advanced %.1fs between consecutive readings "
                "while only %.1fs elapsed on the local monotonic clock, but the "
                "local wall clock advanced %.1fs and corroborates the server. "
                "This is the signature of a suspended/resumed host (paused VM, "
                "laptop sleep, live migration): the monotonic clock stalls "
                "while real time keeps passing. Re-anchored the clock baseline "
                "and continuing; refill math uses Redis server time, so "
                "correctness is unaffected.",
                server_elapsed,
                monotonic_elapsed,
                wall_elapsed,
            )
            return

        raise RuntimeError(
            f"Redis server TIME jumped forward {server_elapsed:.1f}s between "
            f"consecutive readings while only {monotonic_elapsed:.1f}s elapsed "
            f"on the local monotonic clock and {wall_elapsed:.1f}s on the local "
            f"wall clock (excess {excess:.1f}s > MAX_FORWARD_JUMP_SECONDS="
            f"{MAX_FORWARD_JUMP_SECONDS}s). A forward jump silently inflates "
            f"token-bucket refill and over-grants capacity. This typically "
            f"signals a Sentinel or managed failover to a clock-skewed primary; "
            f"investigate NTP discipline across your Redis primaries."
        )


async def async_server_time(client: redis.asyncio.Redis) -> float:
    """Return the Redis server's current time as a ``time.time()``-compatible float."""
    _reject_pipeline(client)
    monotonic_before = time.monotonic()
    wall_before = time.time()
    seconds, microseconds = _parse_time_response(await client.time())
    server_time = _to_float(seconds, microseconds)
    _check_forward_jump(client, server_time, monotonic_before, wall_before)
    return server_time


def sync_server_time(client: redis.Redis) -> float:
    """Return the Redis server's current time as a ``time.time()``-compatible float."""
    _reject_pipeline(client)
    monotonic_before = time.monotonic()
    wall_before = time.time()
    seconds, microseconds = _parse_time_response(client.time())
    server_time = _to_float(seconds, microseconds)
    _check_forward_jump(client, server_time, monotonic_before, wall_before)
    return server_time
