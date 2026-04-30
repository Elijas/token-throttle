"""
Redis server-time helpers — eliminates clock-skew in distributed rate limiting.

All Redis backend code that writes timestamps to shared state must use the Redis
server's clock (via the TIME command) instead of the local ``time.time()``.
This ensures all workers — regardless of their host clock — agree on "now" when
computing token-bucket refill math.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import redis
    import redis.asyncio


async def async_server_time(client: redis.asyncio.Redis) -> float:
    """Return the Redis server's current time as a ``time.time()``-compatible float."""
    seconds, microseconds = cast("tuple[int, int]", await client.time())
    return float(seconds) + float(microseconds) / 1_000_000


def sync_server_time(client: redis.Redis) -> float:
    """Return the Redis server's current time as a ``time.time()``-compatible float."""
    seconds, microseconds = cast("tuple[int, int]", client.time())
    return float(seconds) + float(microseconds) / 1_000_000
