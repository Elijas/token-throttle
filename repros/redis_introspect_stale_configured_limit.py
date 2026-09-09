# ruff: noqa: T201, INP001
"""
Repro: after ``apply_configured_max_capacity``, the Redis backend's
``introspect()`` reports the ORIGINAL quota limit while its decision path
uses the NEW configured limit. Memory and SQLite report the new limit.

Run from the repository root:
    TT_AUDIT_REDIS_URL=redis://127.0.0.1:6399/13 .venv/bin/python repros/redis_introspect_stale_configured_limit.py
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

REDIS_URL = os.environ.get("TT_AUDIT_REDIS_URL", "redis://127.0.0.1:6399/13")
CFG = PerModelConfig(
    model_family="repro",
    quotas=UsageQuotas([Quota(metric="requests", limit=10.0, per_seconds=60)]),
)


async def probe(name: str, backend) -> None:
    await backend.apply_configured_max_capacity("requests", 60, 0.5)
    diag = await backend.introspect()
    bucket = next(b for b in diag.buckets if b.metric == "requests")
    try:
        await backend.await_for_capacity(frozen_usage({"requests": 1.0}), timeout=0)
        decision = "acquire(requests=1) ACCEPTED -> decision path max >= 1"
    except ValueError as exc:
        decision = f"acquire(requests=1) ValueError -> decision path max < 1 ({exc})"
    except TimeoutError:
        decision = "acquire(requests=1) TimeoutError"
    print(
        f"{name:<8} introspect: current={bucket.current_capacity} "
        f"effective_max={bucket.effective_max_capacity} "
        f"configured_limit={bucket.configured_limit} | {decision}"
    )


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
        await probe("memory", memory)
        await probe("sqlite", sqlite)
        await probe("redis", redis_backend)
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
