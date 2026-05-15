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

Cluster mode (RedisCluster)
---------------------------
``redis-py`` routes TIME to a single "default" node. As long as that same
node serves TIME, refill math stays internally consistent. On default-node
failover the new node's clock can be ahead; this module sanity-checks by
raising on forward jumps larger than ``MAX_FORWARD_JUMP_SECONDS`` relative
to the local wall clock.

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

import time
from typing import Any

try:
    import redis.asyncio.client
    import redis.client
except ImportError as exc:
    raise ImportError(
        'The "redis" package is required for the Redis backend. '
        'Install it with: pip install "token-throttle[redis]"'
    ) from exc


MAX_FORWARD_JUMP_SECONDS = 10.0
"""Tolerated forward divergence between Redis TIME and local wall clock.

Catches RedisCluster default-node failover to a clock-skewed primary - the
hazard described in R4 L21 T01: a forward jump silently inflates
``time_passed`` in ``calculate_capacity`` and over-grants bucket capacity.
NTP-disciplined cluster primaries diverge by < 100 ms in practice; 10 s is
a permissive sanity rail, not a tightness guarantee.
"""

# KNOWN UNKNOWN: 10s threshold is set by intuition, not measured P99.9
# inter-primary skew under NTP partition. Tune if real-world data emerges.

_TIME_RESPONSE_LEN = 2  # Redis TIME returns (seconds, microseconds)
_MICROSECONDS_PER_SECOND = 1_000_000
_MAX_REDIS_TIME_SECONDS = 253_402_300_799  # 9999-12-31T23:59:59Z


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


def _to_float_with_jump_check(seconds: int, microseconds: int) -> float:
    try:
        server_time = seconds + microseconds / _MICROSECONDS_PER_SECOND
    except OverflowError as exc:
        raise ValueError(f"Redis TIME out of range: seconds={seconds}") from exc
    local_time = time.time()
    forward_jump = server_time - local_time
    if forward_jump > MAX_FORWARD_JUMP_SECONDS:
        raise RuntimeError(
            f"Redis TIME ({server_time}) is {forward_jump:.1f}s ahead of "
            f"local wall clock ({local_time}). This typically signals a "
            f"RedisCluster default-node failover to a clock-skewed node, "
            f"which would silently over-grant token-bucket capacity. "
            f"Investigate NTP discipline across primaries; threshold is "
            f"MAX_FORWARD_JUMP_SECONDS={MAX_FORWARD_JUMP_SECONDS}."
        )
    return server_time


async def async_server_time(client: redis.asyncio.Redis) -> float:
    """Return the Redis server's current time as a ``time.time()``-compatible float."""
    _reject_pipeline(client)
    seconds, microseconds = _parse_time_response(await client.time())
    return _to_float_with_jump_check(seconds, microseconds)


def sync_server_time(client: redis.Redis) -> float:
    """Return the Redis server's current time as a ``time.time()``-compatible float."""
    _reject_pipeline(client)
    seconds, microseconds = _parse_time_response(client.time())
    return _to_float_with_jump_check(seconds, microseconds)
