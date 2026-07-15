# Changelog

Notable changes for token-throttle releases. Each major version's breaking
changes and upgrade steps are recorded in its entry below.

## Unreleased

- Adds a runnable Anthropic Messages example with independent RPM, ITPM, and
  OTPM buckets; server-side pre-flight token counting; prompt-cache prewarming;
  cache-aware input refunds; observed-p99 output reservations; raw rate-limit
  header logging; and conservative cleanup on every response/error path.
- Clarifies in the README that the high major-version sequence reflects strict
  semver during rapid beta development, rather than implying ten generations
  of product maturity.
- Restores the missing v2.0.0-v4.0.0 history from the archived migration guide
  and adds explicit upgrade actions to the terse v5.0.0-v8.0.0 entries, so the
  changelog's major-version migration claim is complete and auditable.

## 10.0.0 - 2026-07-07

- **Breaking:** this library no longer carries pre-v9 upgrade tooling. The
  `token_throttle.migration` module and its four public names —
  `validate_config_for_v2_0`, `cleanup_legacy_buckets`,
  `async_cleanup_legacy_buckets`, and `ConfigMigrationIssue` — have been
  removed, along with the standalone migration guide. Those helpers existed to
  pre-flight a configuration migration for the v1.4.x-to-v2.0.0 upgrade and are
  not part of the package any longer. Current documentation describes current
  behavior only; each major version's breaking changes and upgrade steps remain
  recorded in the entries below. The Redis ACL command list, the `+@scripting`
  rationale, and the `SCRIPT FLUSH` operational hazard that the guide carried
  now live in [`docs/operations.md`](docs/operations.md).
- **Breaking:** the logging-callback factories (`create_logging_callbacks` and
  `create_sync_logging_callbacks`) no longer accept the `TRACE` or `SUCCESS`
  log-level names. Passing either now raises `ValueError` listing the supported
  standard level names (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) instead
  of quietly remapping them to `DEBUG`/`INFO`.
- **Breaking:** `refund_capacity_from_response` (async and sync) now rejects
  unknown keyword arguments with `TypeError`. A mistyped keyword — for example
  `usaage=` instead of `usage=` — used to be silently ignored, which could drop
  the refund amount you intended. The supported `response=` and `usage=`
  arguments are unchanged.
- Removed internal legacy-upgrade code paths that had no effect on current
  usage: the rejection branch for reservations issued without a limiter
  instance id, a Redis probe for an old max-capacity key that was only ever
  logged and never applied, and a write-only Redis schema-version marker key.
  These changes do not alter behavior for reservations and Redis state produced
  by this version.
- Reworded the warning emitted when a `usage_counter` is defined without
  `**kwargs`. It now presents accepting `**kwargs` as the recommended
  convention and a fixed signature as a supported convenience, rather than
  labeling the fixed-signature path "deprecated". Behavior is unchanged: fixed-
  signature counters still work and still warn that request fields not named in
  the signature are filtered out before the counter is called. The warning text
  still contains "without \*\*kwargs" if you match on it.

## v9.1.1 - 2026-07-07

- Fixes a Redis lock-loss error that surfaced the wrong exception. When a
  per-bucket lock was lost mid-operation (its TTL lapsed or another worker stole
  it), non-waiting ops such as `consume_capacity`, `refund_capacity`, and
  `set_max_capacity` were documented to raise `BackendLockContentionError`, but
  the lock-release cleanup on the way out hit the same lost lock and leaked a raw
  `redis.exceptions.LockNotOwnedError` that replaced it. The release cleanup now
  tolerates an already-lost lock, so callers reliably see
  `BackendLockContentionError` (safe to retry) as promised in
  [`docs/operations.md`](docs/operations.md). Applies to both the async and sync
  Redis backends.
- Fixes the Redis backend's server-clock sanity rail raising a spurious
  `RuntimeError` after a host suspend/resume (paused VM, laptop sleep, live
  migration) or after a single slow Redis `TIME` reply. The rail compares
  consecutive `TIME` readings against locally-elapsed monotonic time, and a
  suspended host stalls the monotonic clock while real time keeps passing -
  which previously looked identical to a server-side clock jump. The local
  wall clock now discriminates the two cases: when it corroborates the
  server's advance, the backend re-anchors its baseline and logs a warning
  instead of raising. Each reading's own round-trip is also now bounded into
  the detection tolerance, so one delayed reply cannot trip the rail. A
  genuine server-side forward jump (for example a Sentinel or managed
  failover to a clock-skewed primary) still raises, and is now reported
  exactly once per event: out-of-order readings from concurrent callers no
  longer regress the detection baseline and re-raise for the same jump.
- Hardens the `reserve()` context manager against three edge cases that could
  leak an in-flight reservation or fabricate capacity accounting:
  - The scope returned by `reserve()` is now single-use. Re-entering the same
    scope — whether after it has exited or while it is still active — raises
    `RuntimeError` instead of silently acquiring a second reservation while
    reusing the first block's recorded actual usage. Call `reserve()` again for
    each attempt.
  - When a `reserve()` block exits without calling `set_actual_usage()`, the
    reservation is refunded before the "forgot to report actual usage"
    `RuntimeWarning` is emitted, so running under `-W error` (warnings promoted
    to exceptions) can no longer skip the refund and leak the reservation.
  - When a `reserve()` block exits normally but the recorded actual usage is
    malformed (for example its metric keys do not match the reservation), the
    reservation is conservatively closed before the resulting `ValueError` is
    re-raised, so the bad-usage error still surfaces to the caller without
    leaking the in-flight reservation.
- Fixes caller cancellation being silently lost when an async callback's
  cancellation cleanup raised an ordinary exception: previously that exception
  replaced the `CancelledError`, was logged and swallowed like any callback
  error, and the cancelled `acquire_capacity()` returned normally, defeating
  `asyncio.timeout()` and `TaskGroup` aborts. The callback error is now logged
  and `CancelledError` is re-raised, so cancellation propagates and reserved
  capacity is refunded. Applies both with `callback_timeout` wrapping and with
  `callback_timeout=None`.
- Fixes a callback that itself raises `TimeoutError` being misreported as
  exceeding `callback_timeout`, on both the async and sync paths. It is now
  handled as an ordinary callback error (warning logged, acquire/refund call
  unaffected) instead of deadline expiry.
- Fixes an unbounded internal accumulation of timed-out async callbacks whose
  event loop closed before they finished; stale entries are now pruned, so
  short-lived event loops are no longer pinned in memory by abandoned
  callbacks.
- Fixes "Exception ignored" noise when an in-flight callback invocation is
  torn down via coroutine `close()` (for example during garbage collection or
  event-loop shutdown); teardown now propagates plain `GeneratorExit` instead
  of an internal wrapper exception.
- Documents in [`docs/observability.md`](docs/observability.md) that an
  abandoned timed-out async callback that also swallows cancellation can block
  `asyncio.run()` shutdown, unlike abandoned sync callbacks, which run in
  daemon helper threads.
- Fixes the two `OpenAIUsageCounter` regressions disclosed in the v9.1.0 entry
  below: a stored-prompt-only Responses request (`prompt={"id": ...}` with no
  `input`) is counted instead of rejected, and `prompt.variables` values that
  are images or files are accepted again (counted as 0 tokens, with a
  once-per-process warning naming the variable and the best-effort tradeoff),
  restoring v9.0.0 behavior.
- Fixes `OpenAIUsageCounter` raising an error while counting request text that
  contains a `tiktoken` special-token literal (for example `<|endoftext|>`,
  which can show up verbatim in text copied from LLM documentation or output).
  Such text is now counted as the ordinary request text it is on the wire
  instead of crashing the acquire.
- Fixes `cleanup_legacy_buckets` / `async_cleanup_legacy_buckets` in
  `token_throttle/migration.py` not escaping
  Redis glob metacharacters (`*`, `?`, `[`, `]`, `\`) in a configured
  `key_prefix` before building its cleanup scan pattern, so a prefix
  containing one of those characters could match and delete a sibling
  deployment's keys instead of only its own.
- Fixes `validate_config_for_v2_0` in
  `token_throttle/migration.py` reporting a
  false-positive "Redis builders require key_prefix" issue for configs where
  the Redis builder options, including `key_prefix`, live in a nested `redis`
  section rather than at the top level.
- Fixes `RateLimiter.aclose()` / `SyncRateLimiter.close()` raising
  `AttributeError` when closing a custom backend builder that omits the
  documented-optional `aclose()` / `close()` cleanup hook; the hook is now
  called only when the backend defines it.
- Fixes the limiter-close warning misreporting "1 reservations still in
  flight" (instead of "1 reservation") when exactly one reservation is
  outstanding at close. No behavior change.
- Documents `diagnose()`'s `RateLimiterDiagnostic` return type in
  [`docs/observability.md`](docs/observability.md), including when to reach
  for it over the lighter `snapshot_state()`; this API surface previously
  shipped without documentation. No behavior change.

## v9.1.0 - 2026-07-07

- Fixes `OpenAIUsageCounter` under-reserving for two request fields that carry
  real, billed prompt text but were not yet counted: Chat Completions'
  `prediction` (Predicted Outputs content, which shows up as accepted/rejected
  prediction tokens in usage) and the Responses API's `prompt.variables`
  (the client-supplied values substituted into a stored prompt template).
  Requests that use either feature now get a larger, more accurate token
  reservation instead of one that silently under-counts; a stored prompt's
  server-side template body itself remains unknowable from the client and is
  documented as a best-effort blind spot. **Known issues as released:** this
  change also made `OpenAIUsageCounter` raise `ValueError` at acquire time
  for `prompt.variables` values containing non-text content parts (images or
  files) — requests that v9.0.0 accepted — and a stored-prompt-only request
  (`prompt={"id": ...}` with no `input`), a valid Responses API shape, was
  still rejected rather than counted. Both are fixed in the next release:
  non-text variable parts count as 0 tokens with a warning, and
  stored-prompt-only requests are accepted.
- Clarifies the Redis lock-contention warning log message and its docstrings
  to say "a waiter" instead of "the no-timeout waiter", since deadline-bounded
  callers also retry through contention as of v9.0.0, not just callers with no
  timeout. No behavior change.
- Restructures the README to lead the quickstart with `reserve()` and move
  operational depth (concurrency model, lifecycle events, bucket-state loss)
  into `docs/operations.md` and `docs/observability.md`; adds a Requirements
  line and a strict-semver/migration note. No behavior change.

## v9.0.0 - 2026-07-06

- **Breaking:** adds the public `BackendLockContentionError` exception and stops
  leaking raw `redis.exceptions.LockError`. Redis per-bucket lock contention now
  surfaces as this library exception: `await_for_capacity` / `wait_for_capacity`
  with no caller timeout retry through contention instead of raising (logging a
  throttled warning), and `consume_capacity`, `refund_capacity`,
  `set_max_capacity`, and reconfiguration raise `BackendLockContentionError`
  (chained from the underlying redis error) on lock starvation or mid-operation
  lock loss. Handlers that caught `redis.exceptions.LockError` must catch
  `BackendLockContentionError` instead; see `MIGRATION.md` and
  the per-bucket locking section in [`docs/operations.md`](docs/operations.md).
- **Breaking:** `RedisBackendBuilder.build()` / `SyncRedisBackendBuilder.build()`
  now raise `ValueError` at build time when any configured quota's
  `per_seconds` window is longer than `bucket_ttl_seconds`. That combination
  previously built without error but silently reset a drained long-window
  quota back to full capacity once an idle gap outlived the TTL. Widen
  `bucket_ttl_seconds`, or shorten the offending quota's `per_seconds`, for any
  configuration the check now rejects; see `MIGRATION.md` and
  the key-TTL guidance in [`docs/operations.md`](docs/operations.md).
- **Breaking:** `OpenAIUsageCounter` / `get_encoding` no longer guess a
  tokenizer from a hardcoded model-family fallback table for models the
  installed `tiktoken` cannot resolve on its own (for example a very new model
  release). They now raise a `ValueError` with upgrade/workaround guidance
  instead of either a possibly-wrong guessed encoding or a raw `KeyError`
  escaping from `tiktoken`. Code that specifically caught `KeyError` around
  token counting must catch `ValueError` instead; upgrade `tiktoken` or pass an
  explicit `get_encoding_func` to `OpenAIUsageCounter` for models it does not
  yet recognize. See `MIGRATION.md`.
- **Breaking:** `UsageQuotas` no longer accepts the private
  `_allow_empty_quotas` constructor keyword; passing it now raises `TypeError`
  (unknown keyword argument) instead of silently building an empty quota set.
  `UsageQuotas([])` still raises the same `ValueError` pointing you to
  `UsageQuotas.unlimited()`, which remains the supported way to build an
  explicit no-limit quota set. See `MIGRATION.md`.
- Fixes Redis `await_for_capacity` / `wait_for_capacity` with a caller
  `timeout`: lock contention now retries acquisition until the caller's
  deadline instead of raising `TimeoutError` after
  `lock_blocking_timeout_seconds` (default 5s). `timeout=0` still fails fast,
  and the timeout message now names lock contention as the cause instead of
  misleading capacity fields. See the per-bucket locking section in
  [`docs/operations.md`](docs/operations.md).
- Fixes async `callback_timeout` so it returns at the deadline even when a
  callback swallows cancellation, including when it is torn down via
  `GeneratorExit` (for example an async generator that uses the limiter being
  closed early). Previously such a callback could block `acquire_capacity` /
  `refund_capacity` for its full runtime and, on a swallowed cancellation,
  without ever logging the documented "callback exceeded timeout" warning; the
  async path now abandons the callback the same way the synchronous path
  already does, logging any error the callback raises afterward. See
  [docs/observability.md](docs/observability.md#callback-timeouts).
- Fixes a `SyncRateLimiter` deadlock when a `PerModelConfigGetter` calls back
  into the limiter (for example `clear_unused_model_families`) while shared
  model-family validation is in progress; the internal validation lock is now
  reentrant. `acquire_capacity_for_request` also now emits the same
  `RuntimeWarning` as `acquire_capacity` when called from inside a running
  event loop.
- Fixes a spurious shutdown warning: closing a limiter with zero in-flight
  reservations no longer logs a "reservations still outstanding" warning.
- Fixes the Redis backend hard-failing every rate-limit operation whenever the
  host's local clock lags behind the Redis server clock (for example an NTP
  outage, a paused/resumed VM, or container clock drift). Refill math already
  uses Redis server time exclusively, so a lagging local clock is harmless to
  correctness; the library now detects a genuine server-side clock jump by
  comparing consecutive Redis `TIME` readings against locally-elapsed
  monotonic time instead of the local wall clock, and raises only on a real
  forward jump between readings (the realistic trigger is a Sentinel/managed
  failover to a clock-skewed primary). A large divergence between the Redis
  server clock and the local wall clock now logs a one-time warning about
  possible NTP trouble instead of raising.
- Fixes two error messages: the `ValueError` raised when usage exceeds a
  bucket's max capacity during acquire now names the failing quota window
  (for example "for the 60s window"), disambiguating cases where two windows
  on the same metric share a limit value; and `set_max_capacity`'s validation
  now reports a dedicated "must be an int or float" message for wrong-typed
  inputs instead of misleadingly reusing the finite/positive-value message.
- Fixes cancellation-path capacity refunds that fail: they now log a warning
  identifying the affected reservation instead of failing silently; the
  original cancellation error still propagates and the reserved capacity
  still recovers through normal refill.
- Adds `RateLimiter.reserve()` / `SyncRateLimiter.reserve()`: a context
  manager over the acquire -> call -> refund cycle. It yields a handle with
  `.reservation` and `.set_actual_usage()`, refunds the unused remainder on
  normal exit (warning and conservatively refunding the full reserved usage if
  `set_actual_usage` was never called), and on an exception refunds with an
  optional `usage_on_error` (or conservatively) before re-raising the original
  exception. If a non-critical `usage_on_error` refund itself fails (for
  example its metric keys do not match the reservation), the reservation
  still falls back to the conservative refund instead of leaking as
  in-flight; the failure is logged, and the caller's original exception
  still propagates. See the README's "Reserve capacity around a call"
  example.
- Fixes `OpenAIUsageCounter` undercounting Responses API requests that use
  `text={"format": {...}}` for structured output: that config is now counted
  by JSON-serializing it like `response_format`/`tools`/`functions`, instead of
  being walked as plain text fragments that dropped the JSON structural
  tokens (previously undercounting affected requests by roughly 62%).
- Adds a weekly `tokenizer-drift` CI canary (no API key required) that checks
  the OpenAI token counter against the latest unpinned `openai`/`tiktoken`
  releases for newly-unresolvable models or untriaged request parameters.
- Fixes the Redis ACL command list in `MIGRATION.md` and
  [`docs/operations.md`](docs/operations.md): it was missing `PEXPIRE` (used
  by redis-py's lock extend/reacquire script) and `MULTI` / `EXEC` /
  `DISCARD` (used by redis-py's transaction pipelines), so a user provisioned
  strictly per the old list could pass an initial smoke test but fail under
  ordinary multi-quota usage.
- Expands documentation coverage: the Redis ACL command list in
  `MIGRATION.md` now includes `PTTL`; its validation-error
  guidance more precisely distinguishes pydantic `ValidationError` from
  `CardinalityLimitExceededError`; the README's OpenAI example sets an
  explicit output-token budget and notes the zero-token refund on error as an
  approximation; [`docs/configuration.md`](docs/configuration.md) gains a
  "Choosing reservation sizes" subsection; and
  [`docs/operations.md`](docs/operations.md) gains an "Application-facing
  errors" reference section covering `DuplicateRefundError`,
  `UnknownReservationError`, `AcquireRefundFailedError`, and
  `CardinalityLimitExceededError`.
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
- **Upgrade:** drain Redis-backed in-flight reservations before deploying v8,
  route refunds through the limiter lifetime that issued each reservation,
  replace implicit loguru routing with explicit stdlib/application callbacks,
  and update the dependency floors above.

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
- **Upgrade:** custom backends and callers must allow lifecycle-critical
  exceptions to propagate raw instead of wrapping them in ordinary cleanup
  errors; rerun the appropriate backend conformance helper before deployment.

## v6.0.0 - 2026-05-20

- Breaking: user callbacks now propagate `MemoryError` and `RecursionError`
  instead of treating them as ordinary warning-only failures.
- Updated callback critical-exception handling and release metadata.
- **Upgrade:** audit callbacks and tests that assumed every callback failure was
  warning-only; ordinary exceptions remain best-effort, but process-health
  exceptions must escape.

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
- **Upgrade:** third-party backends must implement the complete protocol surface
  and pass `conformance_test_for(...)`, `run_conformance_test_for(...)`, or
  `sync_conformance_test_for(...)` before moving from v4.

## v4.0.0 - 2026-05-16

- **Breaking:** `CapacityReservation` became an opaque handle backed by the
  issuing limiter's internal snapshot. Copying or mutating a returned DTO no
  longer redirects or resizes a refund; pass actual usage to the refund API.
- **Breaking:** `AcquireRefundFailedError` stopped inheriting from
  `asyncio.CancelledError` and instead exposes the original interruption through
  `.interrupted_by`. Catch `AcquireRefundFailedError` directly when recovering
  a delivered reservation whose automatic cleanup failed.
- **Breaking:** Redis builders reject Redis Cluster, and public DTO/key inputs
  gained stricter exact-type, character, and length validation.
- **Upgrade:** v3 and v4 reservation/refund authority is intentionally not
  wire-compatible. Stop new v3 work, drain or expire in-flight reservations,
  then deploy v4; do not run a mixed fleet on one Redis prefix. Canary with an
  isolated prefix if a drained cutover is not possible.

## v3.0.0 - 2026-05-15

- **Breaking:** Redis acquires now write a durable acquire marker, and refunds
  must consume that marker before capacity is credited. This prevents forged or
  pre-v3 reservations from fabricating capacity.
- Duplicate refunds continue to fail closed with `DuplicateRefundError`; an
  unprovable reservation fails with `UnknownReservationError` and is not
  credited.
- **Upgrade:** drain or refund every v2 reservation before deploying v3. Mixed
  v2/v3 Redis fleets on the same prefix are unsupported because v2 does not
  write the markers v3 requires.

## v2.1.0 - 2026-05-15

- Adds optional `max_reservation_lifetime_seconds`; Redis backends derive a
  bounded default from their TTL configuration when it is omitted.
- Hardens durable refund outcomes, capacity-setting reconciliation, close
  transactions, cardinality accounting, DTO trust boundaries, and Redis key
  prefix invariants without a further public migration break.

## v2.0.0 - 2026-05-13

- **Breaking:** Redis builders require an explicit tenant `key_prefix`, legacy
  v1.4 reservations without modern limiter identity are rejected, and bucket
  state gains expiration plus refund de-duplication.
- Configuration, DTO, key-segment, callback, close, and in-process cardinality
  boundaries became fail-closed; OpenAI factory callback defaults are merged by
  slot rather than replacing an entire caller bundle.
- **Upgrade:** preflight configuration against the stricter shapes, drain v1.4
  reservations, assign stable per-tenant Redis prefixes, clean up legacy Redis
  bucket keys only after the old fleet is stopped, and audit DTO subclasses and
  partial callback construction before deploying v2.
