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
*monotonic* time (per Redis client): a raise only fires when the server clock
advanced more than ``MAX_FORWARD_JUMP_SECONDS`` beyond the monotonic interval
between two readings. The realistic trigger is a Sentinel/managed failover to
a clock-skewed primary. The first reading for a client establishes the
baseline and never raises.

A *lagging local wall clock* (NTP outage, paused/resumed VM, container drift)
is harmless to correctness because refill math uses Redis server time
exclusively - it must never hard-fail an operation. When server-vs-wall skew
exceeds the threshold we emit a one-time warning per client, because it often
points at NTP problems worth investigating, but we never raise on it.

Bare client only
----------------
``client`` MUST be a non-pipelined ``Redis`` instance. ``Pipeline`` is a
``Redis`` subclass, so the static type accepts it; runtime rejection
prevents the silent failure where TIME would be queued (not executed) and
the unpack would receive an empty queued-command list.

Custom-gateway shape contract
-----------------------------
Implementations of ``AbstractRedisGateway`` must return TIME as a
2-element tuple/list of integer values with ``seconds >= 0`` and
``0 <= microseconds < 1_000_000``. Off-shape responses raise ``TypeError``
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
bucket capacity. Compared against locally-elapsed monotonic time, so a lagging
local wall clock cannot trip it. NTP-disciplined primaries diverge by < 100 ms
in practice; 10 s is a permissive sanity rail, not a tightness guarantee.
"""

# KNOWN UNKNOWN: 10s threshold is set by intuition, not measured P99.9
# inter-primary skew under NTP partition. Tune if real-world data emerges.

_TIME_RESPONSE_LEN = 2  # Redis TIME returns (seconds, microseconds)
_MICROSECONDS_PER_SECOND = 1_000_000
_MAX_REDIS_TIME_SECONDS = 253_402_300_799  # 9999-12-31T23:59:59Z


class _ClientClockState:
    """Per-client anchor for consecutive-reading forward-jump detection."""

    __slots__ = ("last_monotonic", "last_server_time", "wall_skew_warned")

    def __init__(self, server_time: float, monotonic: float) -> None:
        self.last_server_time = server_time
        self.last_monotonic = monotonic
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


def _check_forward_jump(client: object, server_time: float) -> None:
    """
    Raise on a genuine server-side forward jump between consecutive readings.

    Detection is relative to locally-elapsed monotonic time, not the wall clock,
    so a lagging local clock never trips it. The first reading for a client only
    establishes the baseline.
    """
    with _state_lock:
        # Read the clocks inside the lock so the monotonic reading is ordered
        # with respect to the stored anchor (guarantees non-negative elapsed).
        now_monotonic = time.monotonic()
        wall_now = time.time()

        state = _client_states.get(client)
        if state is None:
            state = _ClientClockState(server_time, now_monotonic)
            _client_states[client] = state
            _maybe_warn_wall_skew(state, server_time, wall_now)
            return

        server_elapsed = server_time - state.last_server_time
        monotonic_elapsed = now_monotonic - state.last_monotonic
        excess = server_elapsed - monotonic_elapsed

        # Re-anchor to the latest reading regardless of the outcome, so a
        # one-off jump is reported once and the next reading compares against
        # the new baseline instead of hard-failing every future operation.
        state.last_server_time = server_time
        state.last_monotonic = now_monotonic

        _maybe_warn_wall_skew(state, server_time, wall_now)

        if excess > MAX_FORWARD_JUMP_SECONDS:
            raise RuntimeError(
                f"Redis server TIME jumped forward {server_elapsed:.1f}s "
                f"between consecutive readings while only {monotonic_elapsed:.1f}s "
                f"elapsed on the local monotonic clock (excess {excess:.1f}s > "
                f"MAX_FORWARD_JUMP_SECONDS={MAX_FORWARD_JUMP_SECONDS}s). A forward "
                f"jump silently inflates token-bucket refill and over-grants "
                f"capacity. This typically signals a Sentinel or managed failover "
                f"to a clock-skewed primary; investigate NTP discipline across "
                f"your Redis primaries."
            )


async def async_server_time(client: redis.asyncio.Redis) -> float:
    """Return the Redis server's current time as a ``time.time()``-compatible float."""
    _reject_pipeline(client)
    seconds, microseconds = _parse_time_response(await client.time())
    server_time = _to_float(seconds, microseconds)
    _check_forward_jump(client, server_time)
    return server_time


def sync_server_time(client: redis.Redis) -> float:
    """Return the Redis server's current time as a ``time.time()``-compatible float."""
    _reject_pipeline(client)
    seconds, microseconds = _parse_time_response(client.time())
    server_time = _to_float(seconds, microseconds)
    _check_forward_jump(client, server_time)
    return server_time
