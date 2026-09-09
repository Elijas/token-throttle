# ruff: noqa: T201, INP001
"""
Repro (documented design divergence): a reservation refunded exactly at its
lifetime.

  memory : ignores reservation_lifetime_seconds -> refund ok.
  sqlite : marker expires when expires_at <= now (inclusive) -> at exactly
           lifetime the refund raises UnknownReservationError.
  redis  : marker expires by Redis PX in real time; at exactly lifetime the
           outcome depends on server timing (shown with a 0.5 s lifetime and a
           real sleep of 0.6 s).

Run from the repository root:
    TT_AUDIT_REDIS_URL=redis://localhost:6379/13 .venv/bin/python repros/marker_expiry_boundary.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from pathlib import Path

import redis.asyncio as async_redis

from tests.differential._clock import FakeClock, bind_sqlite_engine_clock, patched_clock
from token_throttle import (
    MemoryBackendBuilder,
    Quota,
    RedisBackendBuilder,
    SqliteBackendBuilder,
    UsageQuotas,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import frozen_usage

REDIS_URL = os.environ.get("TT_AUDIT_REDIS_URL", "redis://localhost:6379/13")
FAMILY = "repro"
CFG = PerModelConfig(
    model_family=FAMILY,
    quotas=UsageQuotas([Quota(metric="requests", limit=10.0, per_seconds=60)]),
)
BUCKETS = frozenset({("requests", 60)})
USAGE = frozen_usage({"requests": 2.0})


async def refund(backend, rid: str) -> str:
    try:
        await backend.refund_capacity_for_buckets(
            USAGE,
            frozen_usage({"requests": 0.0}),
            bucket_ids=BUCKETS,
            reservation_id=rid,
            reservation_model_family=FAMILY,
            reservation_bucket_ids=BUCKETS,
            reservation_reserved_usage=USAGE,
        )
        return "refund ok"
    except Exception as exc:  # noqa: BLE001 - repro prints every outcome kind
        return f"refund {type(exc).__name__}"


async def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="tt-repro-"))
    clock = FakeClock()
    with patched_clock(clock):
        memory = MemoryBackendBuilder().build(CFG)
        sqlite_builder = SqliteBackendBuilder(tmp / "repro.sqlite3", key_prefix="repro")
        sqlite = sqlite_builder.build(CFG)
        bind_sqlite_engine_clock(sqlite._engine, clock)  # noqa: SLF001
        for name, backend in (("memory", memory), ("sqlite", sqlite)):
            rid = f"exp-{uuid.uuid4().hex}"
            await backend.await_for_capacity(
                USAGE,
                timeout=0,
                reservation_id=rid,
                reservation_lifetime_seconds=3600.0,
            )
            clock.advance(3600.0)  # exactly the lifetime
            print(
                f"{name:<7} fake clock, +lifetime exactly: {await refund(backend, rid)}"
            )
        await sqlite_builder.aclose()

    client = async_redis.from_url(REDIS_URL)
    prefix = f"repro-{uuid.uuid4().hex}"
    redis_builder = RedisBackendBuilder(
        client, key_prefix=prefix, bucket_ttl_seconds=60, refund_dedup_ttl_seconds=60
    )
    backend = redis_builder.build(CFG)
    try:
        rid = f"exp-{uuid.uuid4().hex}"
        await backend.await_for_capacity(
            USAGE, timeout=0, reservation_id=rid, reservation_lifetime_seconds=0.5
        )
        await asyncio.sleep(0.6)
        print(
            f"redis   real clock, lifetime 0.5s + 0.6s sleep: {await refund(backend, rid)}"
        )
    finally:
        await redis_builder.aclose()
        cursor = 0
        while True:
            cursor, keys = await client.scan(
                cursor=cursor, match=f"{prefix}:*", count=500
            )
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
