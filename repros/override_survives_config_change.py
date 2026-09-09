# ruff: noqa: T201, INP001
"""
Repro: a runtime override written under configured limit 100 is applied by a
NEW process whose configured limit is 200.

  sqlite : the persisted override (50) has no configured-limit anchor, so the
           new process runs with max 50 -> acquire(150) raises ValueError.
  redis  : the override payload carries configured_max_capacity=100, which does
           not match 200, so it is ignored (warning logged) -> acquire(150) ok.
  memory : not applicable (process-local), shown for completeness with two
           independent backends.

Run from the repository root:
    TT_AUDIT_REDIS_URL=redis://localhost:6379/13 .venv/bin/python repros/override_survives_config_change.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from pathlib import Path

import redis.asyncio as async_redis

from token_throttle import (
    Quota,
    RedisBackendBuilder,
    SqliteBackendBuilder,
    UsageQuotas,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import frozen_usage

REDIS_URL = os.environ.get("TT_AUDIT_REDIS_URL", "redis://localhost:6379/13")
FAMILY = "repro"


def cfg(limit: float) -> PerModelConfig:
    return PerModelConfig(
        model_family=FAMILY,
        quotas=UsageQuotas([Quota(metric="requests", limit=limit, per_seconds=60)]),
    )


async def probe(name: str, backend) -> None:
    diag = await backend.introspect()
    bucket = next(b for b in diag.buckets if b.metric == "requests")
    try:
        await backend.await_for_capacity(frozen_usage({"requests": 150.0}), timeout=0)
        decision = "acquire(150) ok"
    except ValueError as exc:
        decision = f"acquire(150) ValueError: {exc}"
    except TimeoutError:
        decision = "acquire(150) TimeoutError"
    print(
        f"{name:<7} new process (configured 200): effective_max={bucket.effective_max_capacity} "
        f"override_source={bucket.override_source} | {decision}"
    )


async def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="tt-repro-"))
    db = tmp / "repro.sqlite3"
    client = async_redis.from_url(REDIS_URL)
    prefix = f"repro-{uuid.uuid4().hex}"

    # Deployment v1: configured limit 100, operator override 50.
    sqlite_v1 = SqliteBackendBuilder(db, key_prefix="repro")
    redis_v1 = RedisBackendBuilder(client, key_prefix=prefix)
    s1 = sqlite_v1.build(cfg(100.0))
    r1 = redis_v1.build(cfg(100.0))
    await s1.set_max_capacity("requests", 60, 50.0)
    await r1.set_max_capacity("requests", 60, 50.0)
    await sqlite_v1.aclose()
    await redis_v1.aclose()

    # Deployment v2: configured limit 200, same store.
    sqlite_v2 = SqliteBackendBuilder(db, key_prefix="repro")
    redis_v2 = RedisBackendBuilder(client, key_prefix=prefix)
    try:
        await probe("sqlite", sqlite_v2.build(cfg(200.0)))
        await probe("redis", redis_v2.build(cfg(200.0)))
    finally:
        await sqlite_v2.aclose()
        await redis_v2.aclose()
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
