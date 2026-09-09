# ruff: noqa: T201, INP001
"""
Repro: bucket_ttl_seconds == per_seconds (allowed by the validator) with a
bucket in debt. Real clock.

Documented claim (docs/sqlite-backend.md, TTL table; same wording in the Redis
TTL validator docstring): "After expiry the next use is a fresh bucket; by the
validated window bound it has already had enough time to refill fully."
That holds only for capacity >= 0. consume_capacity floors capacity at
-max_capacity, from which a full refill takes TWO windows.

  sqlite : the bucket row's expires_at is refreshed only by WRITES, so after
           1 x TTL of read-only polling the row expires and the next read
           re-grants FULL capacity (the debt is forgiven).
  redis  : every read refreshes the key TTL, so the debt keeps refilling
           linearly: at t=1.0s capacity is ~0, full only at t=2.0s.
  memory : never expires; same curve as Redis.

Run from the repository root:
    TT_AUDIT_REDIS_URL=redis://localhost:6379/13 .venv/bin/python repros/bucket_ttl_expiry_under_debt.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import uuid
import warnings
from pathlib import Path

import redis.asyncio as async_redis

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
WINDOW = 1  # seconds
LIMIT = 10.0
CFG = PerModelConfig(
    model_family="repro",
    quotas=UsageQuotas([Quota(metric="requests", limit=LIMIT, per_seconds=WINDOW)]),
)


async def capacity(backend) -> float:
    diag = await backend.introspect()
    return float(
        next(b for b in diag.buckets if b.metric == "requests").current_capacity
    )


async def run(name: str, backend) -> None:
    # Drain to the floor: 10 acquired, then 10 more consumed -> -10.
    await backend.await_for_capacity(frozen_usage({"requests": LIMIT}), timeout=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        await backend.consume_capacity(frozen_usage({"requests": LIMIT}))
    start = time.monotonic()
    samples = []
    # Poll read-only (failed try-acquires + introspect) for 1.4 windows.
    while time.monotonic() - start < 1.4 * WINDOW:
        try:
            await backend.await_for_capacity(frozen_usage({"requests": 1.0}), timeout=0)
            outcome = "GRANTED"
        except TimeoutError:
            outcome = "blocked"
        samples.append(
            (
                round(time.monotonic() - start, 2),
                round(await capacity(backend), 2),
                outcome,
            )
        )
        if outcome == "GRANTED":
            break
        await asyncio.sleep(0.1)
    print(f"{name:<7} " + "  ".join(f"t={t}s cap={c} {o}" for t, c, o in samples))


async def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="tt-repro-"))
    memory = MemoryBackendBuilder().build(CFG)
    sqlite_builder = SqliteBackendBuilder(
        tmp / "repro.sqlite3",
        key_prefix="repro",
        bucket_ttl_seconds=WINDOW,
        refund_dedup_ttl_seconds=WINDOW,
        override_ttl_seconds=WINDOW,
    )
    sqlite = sqlite_builder.build(CFG)
    client = async_redis.from_url(REDIS_URL)
    prefix = f"repro-{uuid.uuid4().hex}"
    redis_builder = RedisBackendBuilder(
        client,
        key_prefix=prefix,
        bucket_ttl_seconds=WINDOW,
        refund_dedup_ttl_seconds=WINDOW,
        override_ttl_seconds=WINDOW,
    )
    redis_backend = redis_builder.build(CFG)
    try:
        await run("memory", memory)
        await run("sqlite", sqlite)
        await run("redis", redis_backend)
    finally:
        await sqlite_builder.aclose()
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
