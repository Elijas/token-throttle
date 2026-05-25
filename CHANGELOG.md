# Changelog

Notable changes for token-throttle releases. For operator upgrade steps, see
[`MIGRATION.md`](MIGRATION.md).

## v8.0.0 - Unreleased

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
- Closed TC2-001/002/003 critical propagation contracts.
- Added conformance AST guards for cancellation-composition reachability.
- Skipped Redis-specific TC2 tests when the Redis optional dependency is absent.

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
  marker authority, constructor cleanup, public limiter round trips, and FIX-50
  fault injection.

## v5.0.1 - 2026-05-16

- Stabilized the late-exception reporter test and removed a non-Linux skip.

## v5.0.0 - 2026-05-16

- Breaking: custom backend interfaces moved to structural protocols with a full
  conformance suite.
- Added package-wide mypy enforcement and platform unit coverage.
- Documented deferred production items around proxy validation, topology,
  capacity sizing, OpenAI counter accuracy, tenant isolation, and Redis Cluster
  redesign.
