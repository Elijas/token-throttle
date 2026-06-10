# token-throttle benchmarks

A small, stdlib-only harness that characterizes the **overhead of the
acquire path** — the work token-throttle does on every LLM call to reserve and
refund capacity. It uses `time.perf_counter` and computes percentiles in pure
Python; there is no `pytest-benchmark` or other benchmark dependency, and it is
not collected by `pytest` (it lives outside `tests/` and has no `test_` names).

## What it measures

Each timed operation is an **acquire + immediate full refund**. Quotas are sized
so large that capacity is never the bottleneck, so the numbers reflect
acquire-path *overhead*, not the time spent waiting for a bucket to refill.

Two altitudes are covered:

- **Backend protocol level** — builds a backend through the public builder and
  times `wait_for_capacity` / `await_for_capacity` paired with
  `refund_capacity`. This is the lowest-overhead view. For Redis it captures the
  per-bucket lock acquisition plus the Lua round-trips.
- **RateLimiter level** — times the higher-level `acquire_capacity` /
  `refund_capacity` flow that applications actually call, including reservation
  bookkeeping.

There is also a `baseline_noop` scenario: an empty loop measuring the harness's
own per-iteration timing cost. Subtract it from any other row to approximate the
token-throttle-attributable cost.

## Scenarios

| Scenario | Backend | API | Contention |
|----------|---------|-----|------------|
| `baseline_noop` | none | none | — |
| `memory_sync_backend_uncontended` | memory | sync | single caller |
| `memory_sync_backend_contended` | memory | sync | N threads |
| `memory_async_backend_uncontended` | memory | async | single task |
| `memory_async_backend_contended` | memory | async | N tasks |
| `memory_sync_limiter_uncontended` | memory | sync | single caller |
| `memory_async_limiter_uncontended` | memory | async | single task |
| `redis_sync_backend_uncontended` | redis | sync | single caller |
| `redis_sync_backend_contended` | redis | sync | N threads |
| `redis_async_backend_uncontended` | redis | async | single task |
| `redis_async_backend_contended` | redis | async | N tasks |

## Running

From the repo root, with dev dependencies installed (`uv sync --all-extras --group dev`):

    # All memory scenarios (no external services needed)
    uv run python -m benchmarks.run

    # Memory only / Redis only / all contended variants (substring match)
    uv run python -m benchmarks.run --scenario memory
    uv run python -m benchmarks.run --scenario contended

    # Redis scenarios against a dedicated, EMPTY database
    uv run python -m benchmarks.run --scenario redis --redis-url redis://localhost:6379/13

    # Tune iteration count and contended concurrency
    uv run python -m benchmarks.run -n 5000 -c 8

    # Also emit machine-readable JSON
    uv run python -m benchmarks.run --json results.json

`task bench` runs a quick memory-only profile.

Redis scenarios are **skipped with a clear message** when `--redis-url` is
omitted or Redis is unreachable. They never run against a non-empty database:
the harness refuses to start a Redis scenario unless the target DB is empty
(mirroring the test suite's safety gate), it never flushes, and it deletes only
the keys it created under a unique per-run prefix. Point `--redis-url` at a
dedicated empty database, e.g. `redis://localhost:6379/13`.

## Output

A readable table per scenario with **p50 / p90 / p99 / mean latency** (in
microseconds) and **ops/sec**, under a header recording the Python version,
platform, and timestamp. The optional `--json PATH` writes the same data plus
environment and config metadata for tracking over time.

## Interpreting the numbers

**Absolute numbers are not authoritative.** They depend heavily on the machine,
current system load, the Python build, and — for Redis — network round-trip time
and how local the Redis server is. The same code on a loaded laptop, a quiet CI
runner, and a cloud box with a remote Redis will report very different figures.

Read the results **relatively**, not as published constants:

- Compare backends and APIs against each other on the *same* run.
- Subtract `baseline_noop` to isolate token-throttle's own cost.
- Treat the Redis rows as dominated by lock + network round-trips, so they track
  your Redis locality far more than CPU.
- For your own capacity planning, run this on hardware and against a Redis
  topology that resembles production, and re-measure under representative load.
