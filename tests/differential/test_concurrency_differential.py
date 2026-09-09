# ruff: noqa: PLR0913, PLC0415
"""
Concurrency differential: N concurrent callers, one bucket, identical usage.

Invariants compared across backends:
  * never MORE than floor(capacity / usage) admissions (no over-admission),
    for both try-acquire (timeout=0) and bounded wait (timeout=2 s);
  * with a bounded wait, EXACTLY floor(capacity / usage) admissions;
  * with try-acquire, the number of admissions is recorded per backend: a
    shortfall is a contention false negative (the caller was told "no
    capacity" while capacity existed). memory/SQLite serialise in-process
    callers under a lock; Redis uses a distributed lock whose blocking wait
    is bounded by the caller's timeout -- see tt-audit.REPORT.md D6.

Threads and async tasks run under a frozen fake clock (exact arithmetic).
Processes cannot share the fake clock, so they use a one-hour window: refill
during the run is < 0.1 unit and cannot admit an extra caller.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import multiprocessing
import threading
import uuid
from pathlib import Path

import pytest

from tests.differential._backends import (
    build_one,
    make_config,
    purge_redis_prefix,
    require_redis,
)
from tests.differential._clock import FakeClock, RealClock, patched_clock
from tests.differential._driver import Driver
from token_throttle._interfaces._models import frozen_usage

FAMILY = "conc"
LIMIT = 100.0
USAGE = 3.0
EXPECTED_ADMITTED = math.floor(LIMIT / USAGE)  # 33
CALLERS = 48
BOUNDED_WAIT = 2.0
ADMISSIONS: dict[str, int] = {}  # test id -> admitted, printed at the end of the module


@pytest.fixture
def loop():
    loop_ = asyncio.new_event_loop()
    yield loop_
    loop_.close()


def _record(request, admitted: int, timeout: float, *, in_process: bool) -> None:
    ADMISSIONS[request.node.nodeid] = admitted
    print(
        f"\n[concurrency] {request.node.nodeid}: admitted {admitted}/{EXPECTED_ADMITTED} (timeout={timeout})"
    )
    assert admitted <= EXPECTED_ADMITTED, "over-admission"
    if timeout > 0 or in_process:
        # A bounded wait must admit everyone capacity allows; so must an
        # in-process try-acquire, where the only contention is between callers
        # of the same backend object and the backend serialises them itself.
        assert admitted == EXPECTED_ADMITTED, (
            f"admitted {admitted}, expected {EXPECTED_ADMITTED} (timeout={timeout})"
        )


KINDS = ["memory", "sqlite", pytest.param("redis", marks=pytest.mark.redis)]
TIMEOUTS = [0.0, BOUNDED_WAIT]


@pytest.mark.parametrize("timeout", TIMEOUTS, ids=["try", "wait2s"])
@pytest.mark.parametrize("kind", KINDS)
def test_async_tasks(kind: str, timeout: float, loop, tmp_path: Path, request) -> None:
    if kind == "redis":
        require_redis()
    clock = FakeClock()
    with patched_clock(clock):
        target = build_one(
            kind,
            "async",
            make_config(FAMILY, (("requests", 60, LIMIT),)),
            clock,
            loop=loop,
            tmp_path=tmp_path,
        )
        backend = target.backend
        try:
            gate = asyncio.Event()

            async def one(i: int) -> bool:
                await gate.wait()
                try:
                    await backend.await_for_capacity(
                        frozen_usage({"requests": USAGE}), timeout=timeout
                    )
                except TimeoutError:
                    return False
                return True

            async def run_all() -> list[bool]:
                tasks = [asyncio.ensure_future(one(i)) for i in range(CALLERS)]
                await asyncio.sleep(0)
                gate.set()
                return await asyncio.gather(*tasks)

            admitted = sum(loop.run_until_complete(run_all()))
            assert Driver(loop).capacities(target)[("requests", 60)][
                0
            ] == pytest.approx(LIMIT - admitted * USAGE)
            _record(request, admitted, timeout, in_process=True)
        finally:
            target.cleanup()


@pytest.mark.parametrize("timeout", TIMEOUTS, ids=["try", "wait2s"])
@pytest.mark.parametrize("kind", KINDS)
def test_threads(kind: str, timeout: float, loop, tmp_path: Path, request) -> None:
    if kind == "redis":
        require_redis()
    clock = FakeClock()
    with patched_clock(clock):
        target = build_one(
            kind,
            "sync",
            make_config(FAMILY, (("requests", 60, LIMIT),)),
            clock,
            loop=loop,
            tmp_path=tmp_path,
        )
        backend = target.backend
        try:
            barrier = threading.Barrier(CALLERS)

            def one(i: int) -> bool:
                barrier.wait()
                try:
                    backend.wait_for_capacity(
                        frozen_usage({"requests": USAGE}), timeout=timeout
                    )
                except TimeoutError:
                    return False
                return True

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=CALLERS, thread_name_prefix="tt-diff"
            ) as pool:
                admitted = sum(pool.map(one, range(CALLERS)))
            assert Driver(loop).capacities(target)[("requests", 60)][
                0
            ] == pytest.approx(LIMIT - admitted * USAGE)
            _record(request, admitted, timeout, in_process=True)
        finally:
            target.cleanup()


# ----------------------------------------------------------------- processes

WINDOW_LONG = 3600
PROCESSES = 8
ATTEMPTS_PER_PROCESS = 8  # 64 attempts > 33 admissions


def _child(
    kind: str, locator: str, prefix: str, attempts: int, timeout: float, barrier, queue
) -> None:
    from token_throttle import Quota, UsageQuotas
    from token_throttle._interfaces._interfaces import PerModelConfig
    from token_throttle._interfaces._models import frozen_usage

    cfg = PerModelConfig(
        model_family=FAMILY,
        quotas=UsageQuotas(
            [Quota(metric="requests", limit=LIMIT, per_seconds=WINDOW_LONG)]
        ),
    )
    if kind == "sqlite":
        from token_throttle import SyncSqliteBackendBuilder

        builder = SyncSqliteBackendBuilder(
            locator, key_prefix=prefix, sleep_interval=0.01
        )
    else:
        import redis as sync_redis

        from token_throttle._limiter_backends._redis._sync_backend import (
            SyncRedisBackendBuilder,
        )

        builder = SyncRedisBackendBuilder(
            sync_redis.from_url(locator), key_prefix=prefix, sleep_interval=0.01
        )
    backend = builder.build(cfg)
    admitted = 0
    try:
        barrier.wait(timeout=60)
        for _ in range(attempts):
            try:
                backend.wait_for_capacity(
                    frozen_usage({"requests": USAGE}), timeout=timeout
                )
                admitted += 1
            except TimeoutError:
                pass
    finally:
        builder.close()
    queue.put(admitted)


@pytest.mark.parametrize("timeout", TIMEOUTS, ids=["try", "wait2s"])
@pytest.mark.parametrize(
    "kind", ["sqlite", pytest.param("redis", marks=pytest.mark.redis)]
)
def test_processes(kind: str, timeout: float, loop, tmp_path: Path, request) -> None:
    prefix = f"proc-{uuid.uuid4().hex}"
    locator = str(tmp_path / "proc.sqlite3") if kind == "sqlite" else require_redis()
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    barrier = ctx.Barrier(PROCESSES)
    children = [
        ctx.Process(
            target=_child,
            args=(kind, locator, prefix, ATTEMPTS_PER_PROCESS, timeout, barrier, queue),
        )
        for _ in range(PROCESSES)
    ]
    try:
        for child in children:
            child.start()
        results = [queue.get(timeout=180) for _ in children]
        for child in children:
            child.join(timeout=30)
    except BaseException:
        if kind == "redis":
            purge_redis_prefix(locator, prefix)
        raise
    admitted = sum(results)
    target = build_one(
        kind,
        "sync",
        make_config(FAMILY, (("requests", WINDOW_LONG, LIMIT),)),
        RealClock(),
        loop=loop,
        tmp_path=tmp_path,
        key_prefix=prefix,
        db_path=Path(locator) if kind == "sqlite" else None,
    )
    try:
        capacity, _ = Driver(loop).capacities(target)[("requests", WINDOW_LONG)]
        # Refill during a slow run (100 units per hour) must stay below one
        # USAGE, or an extra admission would have been possible.
        assert capacity == pytest.approx(LIMIT - admitted * USAGE, abs=USAGE - 0.1)
    finally:
        target.cleanup()
    print(f"\n[concurrency] per-process admissions: {results}")
    _record(request, admitted, timeout, in_process=False)


# ----------------------------------------- cross-process contention rate

CONTENTION_LIMIT = 1e9  # capacity is never the reason for a TimeoutError here
CONTENTION_TRIES = 200


def _contention_child(
    kind: str, locator: str, prefix: str, tries: int, barrier, queue
) -> None:
    from token_throttle import Quota, UsageQuotas
    from token_throttle._interfaces._interfaces import PerModelConfig
    from token_throttle._interfaces._models import frozen_usage

    cfg = PerModelConfig(
        model_family=FAMILY,
        quotas=UsageQuotas(
            [Quota(metric="requests", limit=CONTENTION_LIMIT, per_seconds=WINDOW_LONG)]
        ),
    )
    if kind == "sqlite":
        from token_throttle import SyncSqliteBackendBuilder

        builder = SyncSqliteBackendBuilder(
            locator, key_prefix=prefix, sleep_interval=0.01
        )
    else:
        import redis as sync_redis

        from token_throttle._limiter_backends._redis._sync_backend import (
            SyncRedisBackendBuilder,
        )

        builder = SyncRedisBackendBuilder(
            sync_redis.from_url(locator), key_prefix=prefix, sleep_interval=0.01
        )
    backend = builder.build(cfg)
    admitted = refused = 0
    try:
        barrier.wait(timeout=60)
        for _ in range(tries):
            try:
                backend.wait_for_capacity(frozen_usage({"requests": 1.0}), timeout=0)
                admitted += 1
            except TimeoutError:
                refused += 1
    finally:
        builder.close()
    queue.put((admitted, refused))


@pytest.mark.parametrize(
    "kind", ["sqlite", pytest.param("redis", marks=pytest.mark.redis)]
)
def test_processes_try_acquire_contention_false_negatives(
    kind: str, tmp_path: Path, request
) -> None:
    """Capacity is effectively infinite, so every TimeoutError is a contention
    false negative. Measured, not asserted (both backends document it).
    """
    prefix = f"contend-{uuid.uuid4().hex}"
    locator = str(tmp_path / "contend.sqlite3") if kind == "sqlite" else require_redis()
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    barrier = ctx.Barrier(PROCESSES)
    children = [
        ctx.Process(
            target=_contention_child,
            args=(kind, locator, prefix, CONTENTION_TRIES, barrier, queue),
        )
        for _ in range(PROCESSES)
    ]
    try:
        for child in children:
            child.start()
        results = [queue.get(timeout=300) for _ in children]
        for child in children:
            child.join(timeout=30)
    finally:
        if kind == "redis":
            purge_redis_prefix(locator, prefix)
    admitted = sum(a for a, _ in results)
    refused = sum(r for _, r in results)
    total = PROCESSES * CONTENTION_TRIES
    print(
        f"\n[contention] {kind}: {PROCESSES} processes x {CONTENTION_TRIES} try-acquires, capacity unlimited: admitted {admitted}, refused {refused} ({refused / total:.1%} false negatives); per process {results}"
    )
    assert admitted + refused == total
