# Changelog

Notable changes for token-throttle releases. For operator upgrade steps, see
[`MIGRATION.md`](MIGRATION.md).

## Unreleased

These changes are on `main` and not yet in a tagged release. The lock-contention
change below is breaking, so the next release is expected to be a major version.

- **Breaking:** adds the public `BackendLockContentionError` exception and stops
  leaking raw `redis.exceptions.LockError`. Redis per-bucket lock contention now
  surfaces as this library exception: `await_for_capacity` / `wait_for_capacity`
  with no caller timeout retry through contention instead of raising (logging a
  throttled warning), and `consume_capacity`, `refund_capacity`,
  `set_max_capacity`, and reconfiguration raise `BackendLockContentionError`
  (chained from the underlying redis error) on lock starvation or mid-operation
  lock loss. Handlers that caught `redis.exceptions.LockError` must catch
  `BackendLockContentionError` instead; see [`MIGRATION.md`](MIGRATION.md) and
  the per-bucket locking section in [`docs/operations.md`](docs/operations.md).
- Adds a test-suite safety gate that refuses to run when `--redis-url` points at
  a non-empty Redis database. The suite flushes that database around every test,
  so it now aborts with an actionable message instead of silently wiping data;
  set `TOKEN_THROTTLE_TESTS_ALLOW_FLUSH=1` to opt in to running against a
  non-empty database.
- Adds a test-suite thread-leak detector that fails the session if a test leaves
  a non-daemon thread or a thread-pool worker alive after a short grace period,
  catching cross-test interference that previously surfaced only as full-suite
  flakiness. Set `TOKEN_THROTTLE_THREAD_LEAK_MODE=report` to investigate a leak
  without failing the run.
- Adds a stdlib-only acquire-path benchmark harness under `benchmarks/`
  (`uv run python -m benchmarks.run`, or `task bench`) that reports p50/p90/p99
  and ops/sec for the memory and Redis backends across sync/async and
  uncontended/contended workloads, with optional JSON output. It is not part of
  the test suite and adds no runtime dependency; absolute numbers are
  machine- and Redis-locality-dependent and meant to be read relatively. See
  `benchmarks/README.md`.
- Adds a weekly scheduled soak/stress workflow (`.github/workflows/soak.yml`,
  also runnable on demand) that repeats the concurrency stress suites many times
  back to back, runs the property-based accounting suite, and runs a
  tightened-timing conformance pass. It exists to catch load- and soak-class
  regressions (contention and accounting bugs that only appear under sustained,
  repeated load) that the single-pass PR CI does not exercise. It changes no
  library behavior.
- Widens the recommended pip install version bounds in the README from a
  next-minor cap to a next-major cap (for example `>=8.0.8,<9.0.0` instead of
  `>=8.0.8,<8.1.0`), so installs can pick up minor and patch releases within the
  same major version without re-pinning. This reflects the project's semantic
  versioning guarantee that no breaking changes ship within a major.
- Removes a stray empty `__init__.py` from the repository root that was never
  part of the published `token_throttle` package; it changes no library
  behavior.

## v8.0.1 – v8.0.8 - 2026-05-28 to 2026-06-06

- Patch releases with internal hardening, portability and test-coverage
  improvements, and release-tooling fixes — including making the README
  install-line lint release-agnostic so tagged-release CI passes. No intended
  public API changes; see the git tags for per-release details. Note: v8.0.7
  was tagged but never published to PyPI because that lint failure gated the
  publish step; v8.0.8 supersedes it.

## v8.0.0 - 2026-05-25

- Enforces strict `limiter_instance_id` binding for Redis refunds; reservations
  must be refunded by the limiter lifetime that issued them.
- Treats partial Redis bucket state as drained instead of fresh capacity, firing
  missing-consumption callbacks and avoiding silent overgrant.
- Removes implicit loguru routing from generic logging callback factories;
  stdlib logging is the default path.
- Raises dependency floors to `pydantic>=2.12.0` and `tiktoken>=0.10.0`.
- Retains Python 3.14 support after fixing the conformance harness behavior.
- Refreshes README, migration, custom-backend, and public docstring coverage.

## v7.0.1 - 2026-05-22

- Hardened conformance harness critical-exception handling by deriving tuples
  from the canonical lifecycle-critical set.
- Corrected `CancelledError` normalization rationale and lock behavior around
  `concurrent.futures.CancelledError`.
- Updated release-flow documentation and lockfile formatting.

## v7.0.0 - 2026-05-21

- Breaking: backend-method lifecycle-critical exceptions now propagate raw
  instead of being wrapped by acquire-cleanup recovery paths.
- Closed the critical exception-propagation contracts.
- Added conformance AST guards for cancellation-composition reachability.
- Skipped Redis-specific critical-propagation tests when the Redis optional dependency is absent.

## v6.0.0 - 2026-05-20

- Breaking: user callbacks now propagate `MemoryError` and `RecursionError`
  instead of treating them as ordinary warning-only failures.
- Updated callback critical-exception handling and release metadata.

## v5.2.2 - 2026-05-20

- Skipped built-in Redis Cluster tests when the Redis optional dependency is
  unavailable.
- Closed async cancellation composition gaps across cleanup helpers.
- Tightened Redis Cluster rejection to built-in Redis builders only.

## v5.2.1 - 2026-05-19

- Consolidated critical-exception callback dispatch.
- Extended cancellation cleanup across lifecycle emission, limited-acquire, and
  fallback-refund paths.
- Added shared cleanup-on-raise helpers for in-flight callback paths.

## v5.2.0 - 2026-05-17

- Hardened conformance timing validation and public-path probes.
- Expanded marker-authority negative probes, side-effect checks, and refund
  forgery coverage.
- Improved run-step wrapper cancellation propagation, deadline handling, and
  `ExceptionGroup` taxonomy.

## v5.1.0 - 2026-05-16

- Expanded custom-backend conformance coverage for timing, callback payloads,
  marker authority, constructor cleanup, public limiter round trips, and fault
  injection for interrupted acquire delivery with failed fallback refunds.

## v5.0.1 - 2026-05-16

- Stabilized the late-exception reporter test and removed a non-Linux skip.

## v5.0.0 - 2026-05-16

- Breaking: custom backend interfaces moved to structural protocols with a full
  conformance suite.
- Added package-wide mypy enforcement and platform unit coverage.
- Documented deferred production items around proxy validation, topology,
  capacity sizing, OpenAI counter accuracy, tenant isolation, and Redis Cluster
  redesign.
