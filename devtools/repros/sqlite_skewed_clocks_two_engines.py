# ruff: noqa: INP001
"""
Repro (documented as unsupported, quantified here): two SQLite engines on the
same database and prefix whose wall clocks differ by SKEW seconds -- the model
of two processes/containers that do not share the host clock.

Each engine samples its own clock inside the write transaction. The engine
that is AHEAD writes future timestamps; the engine that is BEHIND sees them,
clamps refill to zero, and (if the gap exceeds 1 s) repairs the row's
last_checked to its own "now". When the ahead engine reads next, it sees a
timestamp SKEW seconds in its past and refills for time that never elapsed.

Redis is immune by design (server TIME). Memory has one clock per process.

Run from the repository root:
    .venv/bin/python devtools/repros/sqlite_skewed_clocks_two_engines.py
"""

from __future__ import annotations

import asyncio
import tempfile
import warnings
from pathlib import Path

from tests.differential._clock import FakeClock, bind_sqlite_engine_clock
from token_throttle import Quota, SqliteBackendBuilder, UsageQuotas
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import frozen_usage

LIMIT = 60.0
WINDOW = 60  # 1 unit / s
CFG = PerModelConfig(
    model_family="repro",
    quotas=UsageQuotas([Quota(metric="requests", limit=LIMIT, per_seconds=WINDOW)]),
)


async def run(skew: float, steps: int, step_seconds: float) -> tuple[int, float]:
    tmp = Path(tempfile.mkdtemp(prefix="tt-repro-"))
    clock_a = FakeClock(1_800_000_000.0 + skew)  # ahead
    clock_b = FakeClock(1_800_000_000.0)  # behind
    builder_a = SqliteBackendBuilder(tmp / "shared.sqlite3", key_prefix="repro")
    builder_b = SqliteBackendBuilder(tmp / "shared.sqlite3", key_prefix="repro")
    a = builder_a.build(CFG)
    b = builder_b.build(CFG)
    bind_sqlite_engine_clock(a._engine, clock_a)  # noqa: SLF001
    bind_sqlite_engine_clock(b._engine, clock_b)  # noqa: SLF001
    admitted = 0
    try:
        # Drain once so every later admission must come from refill.
        await a.await_for_capacity(frozen_usage({"requests": LIMIT}), timeout=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in range(steps):
                clock_a.advance(step_seconds)
                clock_b.advance(step_seconds)
                backend = a if i % 2 == 0 else b
                while True:
                    try:
                        await backend.await_for_capacity(
                            frozen_usage({"requests": 1.0}), timeout=0
                        )
                        admitted += 1
                    except TimeoutError:
                        break
    finally:
        await builder_a.aclose()
        await builder_b.aclose()
    true_refill = steps * step_seconds * (LIMIT / WINDOW)
    return admitted, true_refill


async def main() -> None:
    for skew in (0.0, 0.5, 2.0, 5.0):
        admitted, true_refill = await run(skew, steps=40, step_seconds=1.0)
        print(
            f"skew={skew:>4}s  admitted={admitted:>4}  true refill={true_refill:.0f}  "
            f"over-grant={admitted - true_refill:+.0f} ({(admitted / true_refill - 1) * 100:+.0f}%)"
        )


if __name__ == "__main__":
    asyncio.run(main())
