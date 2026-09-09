# ruff: noqa: T201, INP001
"""
Repro: acquire -> refund -> acquire AGAIN with the same reservation_id -> refund.

Three-way divergence at the backend protocol:
  memory : second acquire accepted; second refund accepted (capacity credited).
  sqlite : second acquire raises DuplicateRefundError(reason="duplicate_acquire")
           because the refund tombstone is still live.
  redis  : second acquire accepted (Lua script only checks the acquire marker),
           second refund raises RedisScriptResultError("incoherent marker/
           tombstone state") -> the re-acquired capacity can never be refunded.

Run from the repository root:
    TT_AUDIT_REDIS_URL=redis://localhost:6379/13 .venv/bin/python repros/reacquire_refunded_reservation_id.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
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
FAMILY = "repro"
CFG = PerModelConfig(
    model_family=FAMILY,
    quotas=UsageQuotas([Quota(metric="requests", limit=10.0, per_seconds=60)]),
)
BUCKETS = frozenset({("requests", 60)})
USAGE = frozen_usage({"requests": 4.0})
LIFETIME = 3600.0


async def capacity(backend) -> float:
    diag = await backend.introspect()
    return float(
        next(b for b in diag.buckets if b.metric == "requests").current_capacity
    )


async def step(label: str, coro) -> str:
    try:
        await coro
        return f"{label}: ok"
    except Exception as exc:  # noqa: BLE001 - the repro prints every outcome kind
        reason = getattr(exc, "reason", None)
        return f"{label}: {type(exc).__name__}" + (
            f"(reason={reason})" if reason else ""
        )


async def run(name: str, backend) -> None:
    rid = f"recycled-{uuid.uuid4().hex}"

    def refund():
        return backend.refund_capacity_for_buckets(
            USAGE,
            frozen_usage({"requests": 0.0}),
            bucket_ids=BUCKETS,
            reservation_id=rid,
            reservation_model_family=FAMILY,
            reservation_bucket_ids=BUCKETS,
            reservation_reserved_usage=USAGE,
        )

    def acquire():
        return backend.await_for_capacity(
            USAGE, timeout=0, reservation_id=rid, reservation_lifetime_seconds=LIFETIME
        )

    lines = [
        await step("acquire#1", acquire()),
        await step("refund#1 ", refund()),
        await step("acquire#2", acquire()),
        await step("refund#2 ", refund()),
    ]
    print(f"{name:<7} capacity={await capacity(backend):.1f}/10 | " + " | ".join(lines))


async def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="tt-repro-"))
    memory = MemoryBackendBuilder().build(CFG)
    sqlite_builder = SqliteBackendBuilder(tmp / "repro.sqlite3", key_prefix="repro")
    sqlite = sqlite_builder.build(CFG)
    client = async_redis.from_url(REDIS_URL)
    prefix = f"repro-{uuid.uuid4().hex}"
    redis_builder = RedisBackendBuilder(client, key_prefix=prefix)
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
