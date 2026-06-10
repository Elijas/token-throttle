"""
Concrete benchmark scenarios for the token-throttle acquire path.

Two measurement altitudes are covered:

* **Backend protocol level** -- builds a single backend via the public builder
  (``MemoryBackendBuilder`` / ``RedisBackendBuilder`` and sync variants) and
  times ``await_for_capacity`` / ``wait_for_capacity`` paired with
  ``refund_capacity``. This is the lowest-overhead view and, for Redis, captures
  the per-bucket lock plus Lua round-trips.
* **RateLimiter level** -- times the higher-level ``acquire_capacity`` /
  ``refund_capacity`` flow that applications actually call, including reservation
  bookkeeping.

All scenarios are sized so capacity is never the bottleneck (large quotas, tiny
per-op usage, immediate refund of the full reservation): we are characterizing
the acquire-path *overhead*, not the wait-for-refill behavior. Each operation is
acquire+refund so the bucket returns to full and later iterations are not
throttled.

Honesty rules enforced here:

* warmup iterations run before any timed iteration and are discarded;
* setup (builder/backend/limiter construction, Redis client creation) happens
  outside the timed region;
* async scenarios use a single event loop per scenario;
* the no-op baseline measures the harness's own per-iteration overhead so a
  reader can subtract it.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmarks._stats import summarize
from token_throttle import (
    MemoryBackendBuilder,
    PerModelConfig,
    Quota,
    RateLimiter,
    SecondsIn,
    SyncMemoryBackendBuilder,
    SyncRateLimiter,
    UsageQuotas,
    frozen_usage,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from benchmarks._stats import ScenarioResult
    from token_throttle._interfaces._interfaces import (
        RateLimiterBackend,
        SyncRateLimiterBackend,
    )

# A single model family / one bucket pair, quotas large enough that the
# acquire path never blocks on capacity for the smoke and profile sizes we run.
_MODEL = "bench-model"
_RESERVE = {"requests": 1, "tokens": 10}
_ACTUAL = {"requests": 1, "tokens": 5}
_RESERVE_FROZEN = frozen_usage(_RESERVE)
_ACTUAL_FROZEN = frozen_usage(_ACTUAL)
# A long lifetime so backend-level reservation markers (Redis) are accepted; the
# Redis bucket/refund TTLs default well above this.
_RESERVATION_LIFETIME_SECONDS = 300.0


def _bench_config(*, model_family: str | None = None) -> PerModelConfig:
    """
    Build the shared benchmark config.

    Backend-level scenarios build a backend directly and must pass
    ``model_family`` because ``builder.build()`` requires it. Limiter-level
    scenarios leave it ``None``; the limiter injects the family from the model
    name passed to ``acquire_capacity``.
    """
    return PerModelConfig(
        model_family=model_family,
        quotas=UsageQuotas(
            [
                Quota(
                    metric="requests", limit=1_000_000_000, per_seconds=SecondsIn.MINUTE
                ),
                Quota(
                    metric="tokens", limit=1_000_000_000, per_seconds=SecondsIn.MINUTE
                ),
            ]
        ),
    )


@dataclass(frozen=True)
class ScenarioSpec:
    """A named, selectable benchmark and whether it needs Redis."""

    name: str
    backend: str
    api: str
    needs_redis: bool
    run: Callable[[RunContext], ScenarioResult]


@dataclass(frozen=True)
class RunContext:
    """Knobs passed to every scenario for one run."""

    iterations: int
    warmup: int
    concurrency: int
    redis_url: str | None


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


def _time_sync_loop(
    operation: Callable[[], None],
    *,
    iterations: int,
    warmup: int,
) -> tuple[list[float], float]:
    """
    Run ``operation`` ``warmup`` times (discarded) then ``iterations`` timed.

    Returns ``(per_op_latencies_seconds, wall_seconds)``.
    """
    for _ in range(warmup):
        operation()
    latencies: list[float] = []
    wall_start = time.perf_counter()
    for _ in range(iterations):
        op_start = time.perf_counter()
        operation()
        latencies.append(time.perf_counter() - op_start)
    wall_seconds = time.perf_counter() - wall_start
    return latencies, wall_seconds


async def _time_async_loop(
    operation: Callable[[], object],
    *,
    iterations: int,
    warmup: int,
) -> tuple[list[float], float]:
    """Async counterpart of :func:`_time_sync_loop`; ``operation`` returns an awaitable."""
    for _ in range(warmup):
        await operation()
    latencies: list[float] = []
    wall_start = time.perf_counter()
    for _ in range(iterations):
        op_start = time.perf_counter()
        await operation()
        latencies.append(time.perf_counter() - op_start)
    wall_seconds = time.perf_counter() - wall_start
    return latencies, wall_seconds


def _run_sync_concurrent(
    make_operation: Callable[[], Callable[[], None]],
    *,
    iterations: int,
    warmup: int,
    concurrency: int,
) -> tuple[list[float], float]:
    """
    Drive ``concurrency`` threads, each running its own operation closure.

    ``iterations`` is divided across workers (each does ``iterations //
    concurrency``, with the remainder dropped so the split is even). Every
    worker warms up independently. Wall time spans the whole timed region so
    ``ops_per_second`` reflects real aggregate throughput.
    """
    per_worker = iterations // concurrency
    if per_worker == 0:
        raise ValueError(
            f"iterations={iterations} too small for concurrency={concurrency}"
        )
    barrier = threading.Barrier(concurrency)
    results: list[tuple[list[float], float]] = [([], 0.0)] * concurrency

    def worker(slot: int) -> None:
        operation = make_operation()
        for _ in range(warmup):
            operation()
        barrier.wait()  # release all workers together for a fair contended window
        latencies: list[float] = []
        for _ in range(per_worker):
            op_start = time.perf_counter()
            operation()
            latencies.append(time.perf_counter() - op_start)
        results[slot] = (latencies, 0.0)

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(worker, range(concurrency)))
    wall_seconds = time.perf_counter() - wall_start

    latencies = [sample for worker_samples, _ in results for sample in worker_samples]
    return latencies, wall_seconds


async def _run_async_concurrent(
    make_operation: Callable[[], Callable[[], object]],
    *,
    iterations: int,
    warmup: int,
    concurrency: int,
) -> tuple[list[float], float]:
    """Async counterpart of :func:`_run_sync_concurrent` on one event loop."""
    per_worker = iterations // concurrency
    if per_worker == 0:
        raise ValueError(
            f"iterations={iterations} too small for concurrency={concurrency}"
        )
    start_event = asyncio.Event()

    async def worker() -> list[float]:
        operation = make_operation()
        for _ in range(warmup):
            await operation()
        await start_event.wait()
        latencies: list[float] = []
        for _ in range(per_worker):
            op_start = time.perf_counter()
            await operation()
            latencies.append(time.perf_counter() - op_start)
        return latencies

    tasks = [asyncio.ensure_future(worker()) for _ in range(concurrency)]
    # Let every worker reach its warmup+wait point before the timed window opens.
    await asyncio.sleep(0)
    wall_start = time.perf_counter()
    start_event.set()
    worker_latencies = await asyncio.gather(*tasks)
    wall_seconds = time.perf_counter() - wall_start

    latencies = [sample for samples in worker_latencies for sample in samples]
    return latencies, wall_seconds


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


def _unique_key_prefix() -> str:
    # Namespaced and unique per run so we only ever touch our own keys.
    return f"benchmark-{uuid.uuid4().hex[:12]}"


def _require_empty_or_skip_sync(redis_url: str):
    """
    Connect a sync client, skip-by-raising if Redis is down, refuse non-empty DB.

    Returns the connected ``redis.Redis`` client. Mirrors the test suite's
    safety convention: we never flush, and we refuse to run against a non-empty
    database so a shared/production host is never silently disturbed.
    """
    try:
        import redis as sync_redis
        from redis.exceptions import RedisError
    except ImportError as exc:
        raise _RedisUnavailableError("redis package not installed") from exc

    client = sync_redis.from_url(redis_url)
    try:
        client.ping()
    except RedisError as exc:
        client.close()
        raise _RedisUnavailableError(
            f"Redis unreachable at {redis_url}: {exc}"
        ) from exc

    dbsize = client.dbsize()
    if dbsize != 0:
        client.close()
        raise _RedisNonEmptyError(
            f"refusing to run Redis benchmarks: {redis_url!r} is NOT empty "
            f"(DBSIZE={dbsize}). Point --redis-url at a dedicated empty DB index, "
            f"e.g. redis://localhost:6379/13."
        )
    return client


class _RedisUnavailableError(RuntimeError):
    """Redis is not reachable / not installed -- the scenario is skipped."""


class _RedisNonEmptyError(RuntimeError):
    """The target Redis DB is non-empty -- refuse rather than risk shared data."""


def _delete_prefixed_keys_sync(client, key_prefix: str) -> None:
    """
    Delete only the keys this run created under ``{key_prefix}:``.

    Uses SCAN (not KEYS/FLUSHDB) and only removes our own namespace, so a DB that
    started empty is left empty without touching anything outside our prefix.
    """
    pattern = f"{key_prefix}:*"
    cursor = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, match=pattern, count=512)
        if keys:
            client.delete(*keys)
        if cursor == 0:
            break


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def _baseline(ctx: RunContext) -> ScenarioResult:
    """
    No-op loop: measures the harness's own per-iteration timing overhead.

    Subtract this scenario's latency from any other to approximate the
    token-throttle-attributable cost.
    """

    def operation() -> None:
        return None

    latencies, wall = _time_sync_loop(
        operation, iterations=ctx.iterations, warmup=ctx.warmup
    )
    return summarize(
        name="baseline_noop",
        backend="none",
        api="none",
        concurrency=1,
        latencies_seconds=latencies,
        wall_seconds=wall,
    )


# ---------------------------------------------------------------------------
# Memory backend -- backend protocol level
# ---------------------------------------------------------------------------


def _memory_sync_backend(ctx: RunContext) -> ScenarioResult:
    builder = SyncMemoryBackendBuilder()
    backend: SyncRateLimiterBackend = builder.build(_bench_config(model_family=_MODEL))

    def operation() -> None:
        backend.wait_for_capacity(
            _RESERVE_FROZEN,
            reservation_id=uuid.uuid4().hex,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        )
        backend.refund_capacity(_RESERVE_FROZEN, _ACTUAL_FROZEN)

    latencies, wall = _time_sync_loop(
        operation, iterations=ctx.iterations, warmup=ctx.warmup
    )
    builder.close()
    return summarize(
        name="memory_sync_backend_uncontended",
        backend="memory",
        api="sync",
        concurrency=1,
        latencies_seconds=latencies,
        wall_seconds=wall,
    )


def _memory_sync_backend_contended(ctx: RunContext) -> ScenarioResult:
    builder = SyncMemoryBackendBuilder()
    backend: SyncRateLimiterBackend = builder.build(_bench_config(model_family=_MODEL))

    def make_operation() -> Callable[[], None]:
        def operation() -> None:
            backend.wait_for_capacity(
                _RESERVE_FROZEN,
                reservation_id=uuid.uuid4().hex,
                reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
            )
            backend.refund_capacity(_RESERVE_FROZEN, _ACTUAL_FROZEN)

        return operation

    latencies, wall = _run_sync_concurrent(
        make_operation,
        iterations=ctx.iterations,
        warmup=ctx.warmup,
        concurrency=ctx.concurrency,
    )
    builder.close()
    return summarize(
        name="memory_sync_backend_contended",
        backend="memory",
        api="sync",
        concurrency=ctx.concurrency,
        latencies_seconds=latencies,
        wall_seconds=wall,
    )


def _memory_async_backend(ctx: RunContext) -> ScenarioResult:
    async def run() -> tuple[list[float], float]:
        builder = MemoryBackendBuilder()
        backend: RateLimiterBackend = builder.build(_bench_config(model_family=_MODEL))

        async def operation() -> None:
            await backend.await_for_capacity(
                _RESERVE_FROZEN,
                reservation_id=uuid.uuid4().hex,
                reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
            )
            await backend.refund_capacity(_RESERVE_FROZEN, _ACTUAL_FROZEN)

        result = await _time_async_loop(
            operation, iterations=ctx.iterations, warmup=ctx.warmup
        )
        await builder.aclose()
        return result

    latencies, wall = asyncio.run(run())
    return summarize(
        name="memory_async_backend_uncontended",
        backend="memory",
        api="async",
        concurrency=1,
        latencies_seconds=latencies,
        wall_seconds=wall,
    )


def _memory_async_backend_contended(ctx: RunContext) -> ScenarioResult:
    async def run() -> tuple[list[float], float]:
        builder = MemoryBackendBuilder()
        backend: RateLimiterBackend = builder.build(_bench_config(model_family=_MODEL))

        def make_operation() -> Callable[[], object]:
            async def operation() -> None:
                await backend.await_for_capacity(
                    _RESERVE_FROZEN,
                    reservation_id=uuid.uuid4().hex,
                    reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
                )
                await backend.refund_capacity(_RESERVE_FROZEN, _ACTUAL_FROZEN)

            return operation

        result = await _run_async_concurrent(
            make_operation,
            iterations=ctx.iterations,
            warmup=ctx.warmup,
            concurrency=ctx.concurrency,
        )
        await builder.aclose()
        return result

    latencies, wall = asyncio.run(run())
    return summarize(
        name="memory_async_backend_contended",
        backend="memory",
        api="async",
        concurrency=ctx.concurrency,
        latencies_seconds=latencies,
        wall_seconds=wall,
    )


# ---------------------------------------------------------------------------
# Memory backend -- RateLimiter level
# ---------------------------------------------------------------------------


def _memory_sync_limiter(ctx: RunContext) -> ScenarioResult:
    limiter = SyncRateLimiter(_bench_config(), backend=SyncMemoryBackendBuilder())

    def operation() -> None:
        reservation = limiter.acquire_capacity(_RESERVE, _MODEL)
        limiter.refund_capacity(_ACTUAL, reservation)

    latencies, wall = _time_sync_loop(
        operation, iterations=ctx.iterations, warmup=ctx.warmup
    )
    limiter.close()
    return summarize(
        name="memory_sync_limiter_uncontended",
        backend="memory",
        api="sync",
        concurrency=1,
        latencies_seconds=latencies,
        wall_seconds=wall,
    )


def _memory_async_limiter(ctx: RunContext) -> ScenarioResult:
    async def run() -> tuple[list[float], float]:
        limiter = RateLimiter(_bench_config(), backend=MemoryBackendBuilder())

        async def operation() -> None:
            reservation = await limiter.acquire_capacity(_RESERVE, _MODEL)
            await limiter.refund_capacity(_ACTUAL, reservation)

        result = await _time_async_loop(
            operation, iterations=ctx.iterations, warmup=ctx.warmup
        )
        await limiter.aclose()
        return result

    latencies, wall = asyncio.run(run())
    return summarize(
        name="memory_async_limiter_uncontended",
        backend="memory",
        api="async",
        concurrency=1,
        latencies_seconds=latencies,
        wall_seconds=wall,
    )


# ---------------------------------------------------------------------------
# Redis backend -- backend protocol level (captures lock + Lua round-trips)
# ---------------------------------------------------------------------------


def _redis_sync_backend(ctx: RunContext) -> ScenarioResult:
    if ctx.redis_url is None:
        raise _RedisUnavailableError("no --redis-url configured")
    from token_throttle import SyncRedisBackendBuilder

    client = _require_empty_or_skip_sync(ctx.redis_url)
    key_prefix = _unique_key_prefix()
    try:
        builder = SyncRedisBackendBuilder(client, key_prefix=key_prefix)
        backend: SyncRateLimiterBackend = builder.build(
            _bench_config(model_family=_MODEL)
        )

        def operation() -> None:
            backend.wait_for_capacity(
                _RESERVE_FROZEN,
                reservation_id=uuid.uuid4().hex,
                reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
            )
            backend.refund_capacity(_RESERVE_FROZEN, _ACTUAL_FROZEN)

        latencies, wall = _time_sync_loop(
            operation, iterations=ctx.iterations, warmup=ctx.warmup
        )
        builder.close()
        return summarize(
            name="redis_sync_backend_uncontended",
            backend="redis",
            api="sync",
            concurrency=1,
            latencies_seconds=latencies,
            wall_seconds=wall,
        )
    finally:
        _delete_prefixed_keys_sync(client, key_prefix)
        client.close()


def _redis_sync_backend_contended(ctx: RunContext) -> ScenarioResult:
    if ctx.redis_url is None:
        raise _RedisUnavailableError("no --redis-url configured")
    from token_throttle import SyncRedisBackendBuilder

    client = _require_empty_or_skip_sync(ctx.redis_url)
    key_prefix = _unique_key_prefix()
    try:
        builder = SyncRedisBackendBuilder(client, key_prefix=key_prefix)
        backend: SyncRateLimiterBackend = builder.build(
            _bench_config(model_family=_MODEL)
        )

        def make_operation() -> Callable[[], None]:
            def operation() -> None:
                backend.wait_for_capacity(
                    _RESERVE_FROZEN,
                    reservation_id=uuid.uuid4().hex,
                    reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
                )
                backend.refund_capacity(_RESERVE_FROZEN, _ACTUAL_FROZEN)

            return operation

        latencies, wall = _run_sync_concurrent(
            make_operation,
            iterations=ctx.iterations,
            warmup=ctx.warmup,
            concurrency=ctx.concurrency,
        )
        builder.close()
        return summarize(
            name="redis_sync_backend_contended",
            backend="redis",
            api="sync",
            concurrency=ctx.concurrency,
            latencies_seconds=latencies,
            wall_seconds=wall,
        )
    finally:
        _delete_prefixed_keys_sync(client, key_prefix)
        client.close()


def _redis_async_backend(ctx: RunContext) -> ScenarioResult:
    if ctx.redis_url is None:
        raise _RedisUnavailableError("no --redis-url configured")
    # Pre-flight + cleanup via a sync client (no event loop needed for SCAN/DEL).
    preflight = _require_empty_or_skip_sync(ctx.redis_url)
    key_prefix = _unique_key_prefix()

    async def run() -> tuple[list[float], float]:
        import redis.asyncio as aioredis

        from token_throttle import RedisBackendBuilder

        async_client = aioredis.from_url(ctx.redis_url)
        builder = RedisBackendBuilder(
            async_client, key_prefix=key_prefix, owns_redis_client=True
        )
        backend: RateLimiterBackend = builder.build(_bench_config(model_family=_MODEL))

        async def operation() -> None:
            await backend.await_for_capacity(
                _RESERVE_FROZEN,
                reservation_id=uuid.uuid4().hex,
                reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
            )
            await backend.refund_capacity(_RESERVE_FROZEN, _ACTUAL_FROZEN)

        try:
            return await _time_async_loop(
                operation, iterations=ctx.iterations, warmup=ctx.warmup
            )
        finally:
            await builder.aclose()

    try:
        latencies, wall = asyncio.run(run())
    finally:
        _delete_prefixed_keys_sync(preflight, key_prefix)
        preflight.close()
    return summarize(
        name="redis_async_backend_uncontended",
        backend="redis",
        api="async",
        concurrency=1,
        latencies_seconds=latencies,
        wall_seconds=wall,
    )


def _redis_async_backend_contended(ctx: RunContext) -> ScenarioResult:
    if ctx.redis_url is None:
        raise _RedisUnavailableError("no --redis-url configured")
    preflight = _require_empty_or_skip_sync(ctx.redis_url)
    key_prefix = _unique_key_prefix()

    async def run() -> tuple[list[float], float]:
        import redis.asyncio as aioredis

        from token_throttle import RedisBackendBuilder

        async_client = aioredis.from_url(ctx.redis_url)
        builder = RedisBackendBuilder(
            async_client, key_prefix=key_prefix, owns_redis_client=True
        )
        backend: RateLimiterBackend = builder.build(_bench_config(model_family=_MODEL))

        def make_operation() -> Callable[[], object]:
            async def operation() -> None:
                await backend.await_for_capacity(
                    _RESERVE_FROZEN,
                    reservation_id=uuid.uuid4().hex,
                    reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
                )
                await backend.refund_capacity(_RESERVE_FROZEN, _ACTUAL_FROZEN)

            return operation

        try:
            return await _run_async_concurrent(
                make_operation,
                iterations=ctx.iterations,
                warmup=ctx.warmup,
                concurrency=ctx.concurrency,
            )
        finally:
            await builder.aclose()

    try:
        latencies, wall = asyncio.run(run())
    finally:
        _delete_prefixed_keys_sync(preflight, key_prefix)
        preflight.close()
    return summarize(
        name="redis_async_backend_contended",
        backend="redis",
        api="async",
        concurrency=ctx.concurrency,
        latencies_seconds=latencies,
        wall_seconds=wall,
    )


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("baseline_noop", "none", "none", needs_redis=False, run=_baseline),
    ScenarioSpec(
        "memory_sync_backend_uncontended",
        "memory",
        "sync",
        needs_redis=False,
        run=_memory_sync_backend,
    ),
    ScenarioSpec(
        "memory_sync_backend_contended",
        "memory",
        "sync",
        needs_redis=False,
        run=_memory_sync_backend_contended,
    ),
    ScenarioSpec(
        "memory_async_backend_uncontended",
        "memory",
        "async",
        needs_redis=False,
        run=_memory_async_backend,
    ),
    ScenarioSpec(
        "memory_async_backend_contended",
        "memory",
        "async",
        needs_redis=False,
        run=_memory_async_backend_contended,
    ),
    ScenarioSpec(
        "memory_sync_limiter_uncontended",
        "memory",
        "sync",
        needs_redis=False,
        run=_memory_sync_limiter,
    ),
    ScenarioSpec(
        "memory_async_limiter_uncontended",
        "memory",
        "async",
        needs_redis=False,
        run=_memory_async_limiter,
    ),
    ScenarioSpec(
        "redis_sync_backend_uncontended",
        "redis",
        "sync",
        needs_redis=True,
        run=_redis_sync_backend,
    ),
    ScenarioSpec(
        "redis_sync_backend_contended",
        "redis",
        "sync",
        needs_redis=True,
        run=_redis_sync_backend_contended,
    ),
    ScenarioSpec(
        "redis_async_backend_uncontended",
        "redis",
        "async",
        needs_redis=True,
        run=_redis_async_backend,
    ),
    ScenarioSpec(
        "redis_async_backend_contended",
        "redis",
        "async",
        needs_redis=True,
        run=_redis_async_backend_contended,
    ),
)


def scenario_names() -> list[str]:
    return [spec.name for spec in SCENARIOS]


def select_scenarios(selectors: Iterable[str] | None) -> list[ScenarioSpec]:
    """
    Resolve user selectors to scenario specs.

    A selector matches by exact name or by substring (so ``memory`` selects all
    memory scenarios, ``redis`` all Redis ones, ``contended`` all contended).
    ``None`` selects everything.
    """
    if selectors is None:
        return list(SCENARIOS)
    chosen: list[ScenarioSpec] = []
    seen: set[str] = set()
    for selector in selectors:
        matches = [
            spec for spec in SCENARIOS if spec.name == selector or selector in spec.name
        ]
        if not matches:
            available = ", ".join(scenario_names())
            raise ValueError(
                f"no scenario matches {selector!r}; available: {available}"
            )
        for spec in matches:
            if spec.name not in seen:
                seen.add(spec.name)
                chosen.append(spec)
    return chosen
