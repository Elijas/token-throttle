# Migration Guide

## Migrating from v7.x to v8.0.0

### What changed

v8.0.0 is the greenfield hardening release for the production-readiness sweep.
It closes Redis refund authority gaps, removes implicit loguru routing, updates
dependency floors, and changes one Redis failure-mode default to fail safe.

- **Strict limiter binding for refunds (PD-1):** Redis refunds now require the
  `CapacityReservation.limiter_instance_id` to match the limiter instance that
  is processing the refund. A reservation issued by one limiter instance is no
  longer refundable through another limiter even when both share the same Redis
  `key_prefix`. Those refunds raise `UnknownReservationError` with
  `reservation issued by a different limiter instance`.
- **Partial Redis bucket state is treated as drained (PD-2):** if Redis returns
  only one half of a bucket state pair, token-throttle must not infer fresh full
  capacity. It fires the missing-consumption callback and treats the bucket as
  unavailable until refill/state repair rather than silently overgranting.
- **loguru auto-routing removed (PD-4):** `create_logging_callbacks()` and
  `create_sync_logging_callbacks()` are stdlib logging factories. loguru is no
  longer selected merely because it is importable, and any loguru-specific
  callback helpers should be replaced with stdlib logging callbacks or
  application-owned callbacks.
- **Dependency floors move up:** v8 requires `pydantic>=2.12.0` because runtime
  validation relies on APIs added after 2.11, and the OpenAI tokenizer extra
  requires `tiktoken>=0.10.0` for current model encodings.
- **Python 3.14 support retained:** v8 keeps the Python 3.14 classifier after
  fixing the conformance harness path that previously failed on 3.14 asyncio
  shield behavior.

Before v8, this pattern could appear to work when two Redis-backed limiters
shared a prefix:

```python
reservation = await limiter_a.acquire_capacity(
    {"requests": 1, "tokens": 1000},
    model="gpt-4o",
)
await limiter_b.refund_capacity({"requests": 1, "tokens": 650}, reservation)
```

In v8, refund through the issuing limiter lifetime:

```python
reservation = await limiter.acquire_capacity(
    {"requests": 1, "tokens": 1000},
    model="gpt-4o",
)
try:
    response = await call_provider()
finally:
    await limiter.refund_capacity({"requests": 1, "tokens": 650}, reservation)
```

Before v8, logging callback routing could change when `loguru` happened to be
installed:

```python
from token_throttle import create_logging_callbacks

callbacks = create_logging_callbacks()  # v7: loguru if importable, else stdlib
```

In v8, configure stdlib logging for `token_throttle`, or use an explicit
application-owned loguru adapter if your service standardizes on loguru:

```python
import logging

from token_throttle import create_logging_callbacks

logging.getLogger("token_throttle").setLevel(logging.INFO)
callbacks = create_logging_callbacks(capacity_refunded="INFO")
```

### What you must do

Drain or bound in-flight reservations before upgrading Redis-backed fleets.
Do not roll v7 and v8 workers together on the same Redis prefix while old
reservations are still refundable; v7 workers may have issued reservations that
v8 will reject when presented through a different limiter instance.

Audit code that serialized, queued, or handed `CapacityReservation` objects to
another process for refund. Replace it with a request lifecycle where the same
limiter instance that acquired capacity also performs the refund, or drain the
queue before moving to v8.

Code that relied on cross-limiter refunds must route each reservation back to
its issuing limiter instance, or centralize acquisition and refund through a
single shared limiter object in the process.

Configure stdlib logging handlers for the `token_throttle` logger before
deploying if you previously relied on automatic loguru routing. If you used
loguru-specific sinks, attach a small application callback that sends the
structured callback payloads to loguru explicitly.

Applications that want Loguru output should configure a stdlib logging handler
that forwards records to Loguru, or provide custom callback bundles.

Update deployment constraints to include `pydantic>=2.12.0` and, when using the
OpenAI helper extra, `tiktoken>=0.10.0`. Do not rely on the v7 lower bounds.

For Redis deployments with eviction enabled, monitor
`on_missing_consumption_data` callbacks. After v8, partial bucket-state loss
causes temporary undergranting instead of silent overgranting; investigate Redis
memory pressure, eviction policy, TTL settings, and persistence before raising
traffic.

## Migrating from v6.x to v7.0.0

### What changed

v7.0.0 tightened the severe-exception contract for custom backend methods.
Backend methods that raise lifecycle-critical exceptions now propagate those
exceptions raw to callers. This applies to async and sync capacity wait,
consume, refund, bucket-specific refund, max-capacity update, and configured
max-capacity hooks.

Interrupted-acquire cleanup follows the same rule. If capacity was committed
but delivery was interrupted, token-throttle attempts a cleanup refund. Ordinary
`Exception` failures still surface as `AcquireRefundFailedError`, but severe
exceptions such as cancellation, process-exit signals, `MemoryError`, and
`RecursionError` escape raw instead of being wrapped.

Before v7, some cleanup failures could be recovered only through the ordinary
envelope:

```python
from token_throttle import AcquireRefundFailedError

try:
    reservation = await limiter.acquire_capacity({"tokens": 1000}, model="demo")
except AcquireRefundFailedError as exc:
    recover_or_alert(exc.reservation, exc.refund_error)
```

In v7, keep that handler for ordinary cleanup failures, but allow severe
exceptions to propagate to the runtime or supervisor:

```python
from token_throttle import AcquireRefundFailedError

try:
    reservation = await limiter.acquire_capacity({"tokens": 1000}, model="demo")
except AcquireRefundFailedError as exc:
    recover_or_alert(exc.reservation, exc.refund_error)
except (KeyboardInterrupt, SystemExit, MemoryError, RecursionError):
    raise
```

### What you must do

If you maintain a custom backend, update tests so every backend method either
completes normally, raises documented ordinary exceptions, or lets lifecycle-
critical exceptions escape raw. Do not wrap `BaseException` broadly, and do not
convert `MemoryError`, `RecursionError`, `KeyboardInterrupt`, `SystemExit`,
`GeneratorExit`, `asyncio.CancelledError`, or
`concurrent.futures.CancelledError` into `AcquireRefundFailedError`.

Run `conformance_test_for(...)`, `run_conformance_test_for(...)`, or
`sync_conformance_test_for(...)` from `token_throttle` after updating the
backend. Add backend-specific fault-injection tests for severe exceptions
because the public helper does not inject every severe exception into every
backend method.

## Migrating from v5.x to v6.0.0

### What changed

v6.0.0 changed callback failure handling for severe process-health exceptions.
`MemoryError` and `RecursionError` now propagate from user callbacks instead of
being treated as ordinary callback failures with a warning. This matches the
existing lifecycle-critical handling for cancellation, interpreter shutdown,
and process-exit signals.

Before v6, callback failures in this category could be suppressed by the
best-effort callback wrapper:

```python
async def on_capacity_refunded(**kwargs) -> None:
    raise MemoryError("simulated callback failure")

# v5 could warn and continue after the callback failure.
```

In v6, treat these exceptions as fatal to the current operation:

```python
async def on_capacity_refunded(**kwargs) -> None:
    raise MemoryError("simulated callback failure")

try:
    await limiter.refund_capacity({"tokens": 10}, reservation)
except (MemoryError, RecursionError):
    raise
```

### What you must do

Audit callback code for broad `except Exception` or `except BaseException`
handlers that hide out-of-memory or runaway-recursion failures. Let
`MemoryError` and `RecursionError` propagate, and move non-critical telemetry
errors behind ordinary `Exception` handling inside your callback.

If your tests asserted that callback failures are always warning-only, split
them into ordinary exception tests and severe exception tests. Ordinary
`Exception` subclasses remain best-effort; `MemoryError` and `RecursionError`
now escape.

## Migrating from v4.x to v5.0.0

### What changed

Custom backend interfaces are now structural `Protocol` classes. Backends that
subclass `RateLimiterBackend`, `SyncRateLimiterBackend`, or the builder
interfaces can keep inheriting the conservative default hooks, but custom
backends that rely on structural typing must implement the full protocol
surface, including cleanup hooks and optional authority hooks.

Before v5, custom backends commonly depended on nominal inheritance and a
smaller checked surface:

```python
from token_throttle import RateLimiterBackend


class MyBackend(RateLimiterBackend):
    ...
```

In v5, structural implementations are accepted, but the full protocol must be
present and should be verified by the conformance helper:

```python
from token_throttle import run_conformance_test_for

run_conformance_test_for(my_async_backend_builder)
```

### What you must do

Run `conformance_test_for(...)` or `sync_conformance_test_for(...)` against
third-party backend builders before upgrading. See
[`docs/custom-backends.md`](docs/custom-backends.md) for the full contract.

## Migrating from v3.x to v4.0.0

v4.0.0 is intentionally not reservation-wire-compatible with v3.x during
rolling deploys. Drain or bound in-flight work before upgrading fleets that
serialize reservations, use Redis cross-process refunds, or run canaries.

### Reservation snapshot authority

Refund authority now comes from the limiter's internal reservation snapshot
captured at acquire time. Mutating or copying a returned
`CapacityReservation` no longer changes what the limiter refunds. In
particular, `reservation.model_copy(update={...})` is not a way to redirect or
resize a refund; the issuing limiter refunds from its stored snapshot.

This is a breaking hardening change for code that intentionally edited
reservations before refund. Refund the original reservation object and pass the
actual API usage through `refund_capacity(...)` /
`refund_capacity_from_response(...)` instead.

### AcquireRefundFailedError base class

`AcquireRefundFailedError` is no longer an `asyncio.CancelledError` subclass.
Code that recovered failed acquire-cleanup reservations with
`except asyncio.CancelledError` must catch `AcquireRefundFailedError` directly:

```python
from token_throttle import AcquireRefundFailedError

try:
    reservation = await limiter.acquire_capacity({"tokens": 1000}, model)
except AcquireRefundFailedError as exc:
    reservation = exc.reservation
    original_interrupt = exc.interrupted_by
    # Refund manually or continue with the delivered reservation.
```

The exception still exposes `.reservation`; v4 adds `.interrupted_by` for the
original cancellation or control-flow exception that interrupted acquire
delivery. This keeps the critical cleanup payload visible through
`asyncio.wait_for`, `asyncio.shield`, `asyncio.gather(return_exceptions=True)`,
`TaskGroup`, pickle, and cross-process exception propagation.

`DuplicateRefundError` now includes a `.reason` string for monitoring and
branching: `"already_refunded"`, `"in_progress"`, or `"duplicate_acquire"`.
The existing messages are unchanged.

### Redis Cluster rejection

Redis builders reject Redis Cluster deployments in v4.0.0. token-throttle uses
multi-key Lua transactions and redis-py locks that require all touched keys to
share the same single-node execution context. Do not rely on Cluster hash tags
or partial slot pinning as a workaround; use standalone Redis, Sentinel, or a
managed single-primary Redis-compatible deployment until Cluster support is
explicitly documented.

Full Redis Cluster support is deferred because it would require coordinated
changes to the public key shape, hash-tagged keys, per-shard Lua execution,
Cluster-aware client handling, and cross-shard transaction semantics. If you
need Cluster, fork and rework key hashing and multi-key Lua routing as one
design; PRs are welcome, but partial hash-tag changes are not enough.

R7 validation covered `fakeredis` plus local vanilla Redis 7.x. token-throttle
targets vanilla Redis 6.2 or newer; Redis 6.0/6.1, Sentinel failover behavior,
Redis Cluster, KeyDB, Dragonfly, and low `maxmemory` / low `maxclients`
configurations were not part of that validation matrix. Treat those as
operator-side validation before production use.

### Stricter public input validation

v4.0.0 narrows several public inputs to exact, predictable shapes:

- integer fields such as `per_seconds` and bucket ids reject `bool` and integer
  subclasses; pass plain `int` values
- Redis key segments, model families, model aliases, metrics, key prefixes,
  and reservation ids reject whitespace, control characters, `:`, `{`, and `}`
- bounded string fields enforce public length caps before key construction
- DTO subclasses remain unsupported at security-sensitive boundaries

These checks are fail-closed and raise `ValueError` or
`CardinalityLimitExceededError`. Normalize external configuration before
constructing token-throttle DTOs.

### Rolling deploy v3 -> v4

Do not run a rolling half-fleet with v3 and v4 processes sharing one Redis
prefix while reservations are in flight. A v3 process can issue reservations
whose runtime shape and refund assumptions are not authoritative to v4, and a
v4 process can issue reservations whose hardening fields are ignored or
misinterpreted by v3.

Recommended rollout:

1. Stop admitting new acquire work on v3 workers.
2. Wait for in-flight v3 reservations to refund or expire according to your
   request deadline and `max_reservation_lifetime_seconds`.
3. Deploy v4 workers with the same Redis prefix.
4. Watch `UnknownReservationError`, `DuplicateRefundError.reason`, and
   `AcquireRefundFailedError.reason` counters during the cutover.

### Mixed v3/v4 canary

A mixed v3/v4 canary is not a safe half-fleet mode when both versions share a
Redis prefix and can refund each other's reservations. Canary v4 with a
separate `key_prefix`, isolated traffic, and separate quotas, or canary a full
drained deployment slice. If you cannot isolate the prefix, treat the canary as
a full migration and drain first.

### Rollback v4 -> v3

Rollback is not recommended once v4 has issued reservations. v4 reservation
authority does not round-trip to v3, and v3 may not preserve v4's stricter
refund assumptions. If rollback is unavoidable, stop v4 traffic, drain v4
in-flight reservations, then start v3. Expect a short window of failed refunds
for any reservation that crosses the version boundary; failed refunds are safer
than double-crediting capacity.

## Migrating from v2.x to v3.0.0

v3.0.0 requires Redis-backed refunds to prove that the backend previously issued
the reservation. Acquires now write a durable marker in Redis, and refunds
consume that marker before crediting capacity.

Drain or refund all in-flight v2.x reservations before upgrading Redis-backed
fleets. Reservations created by v2.x processes do not have acquire markers, so a
v3 process cannot distinguish them from manually forged reservations. Refunding
one after the upgrade fails closed with `UnknownReservationError` and does not
credit capacity.

Duplicate refunds still fail as duplicates: once a legitimate v3 refund consumes
the acquire marker and writes the refund tombstone, a retry raises
`DuplicateRefundError`. Mixed v2/v3 Redis fleets are not supported because v2
processes do not write acquire markers for v3 processes to consume.

### Rolling deploy v2 -> v3

The v2-reservation -> v3-refund contract is intentionally fail-closed:
refunding a Redis-backed v2 reservation through a v3 process raises
`UnknownReservationError`. That error means v3 could not prove the reservation
was acquired, so it refuses to credit capacity.

Recommended rollout:

1. Stop or drain v2 traffic.
2. Wait for normal request deadlines and queue retries to finish refunding v2
   reservations.
3. Deploy v3.
4. Accept a brief refund-error window for stragglers and monitor
   `UnknownReservationError`.

Do not run v2/v3 as a durable mixed fleet on the same Redis prefix.

## Migrating from v1.4.x to v2.0.0

v2.0.0 keeps the strict runtime validation introduced before this release.
Do not rely on construction-time coercion during the upgrade. Run the
migration helper against your stored configuration dictionaries first, fix all
reported issues, then deploy the new version.

## 1. Preflight Config Dictionaries

```python
from token_throttle.migration import validate_config_for_v2_0

errors = validate_config_for_v2_0(your_config)
if errors:
    for error in errors:
        print(
            f"{error.field_path}: {error.value!r} -> "
            f"{error.reason}; {error.suggested_fix}"
        )
```

The helper is read-only: it does not mutate input and does not coerce values.
It reports values that v1.4.x may have accepted but v2.0.0 rejects, including:

- quoted numeric limits such as `"1000"`; use `1000`
- float time windows such as `60.0`; use `60`
- whitespace, `:`, `{`, or `}` in metrics, model families, and Redis prefixes
- bytes values where plain strings are required

## 2. Drain Reservations

Drain or refund in-flight reservations before upgrading. Legacy serialized
reservations may have `limiter_instance_id=None`; v2.0.0 reports this as a
migration issue because those reservations cannot provide the same ownership
signal as new reservations.

At runtime, refunding a legacy v1.4.x reservation without
`limiter_instance_id` fails closed. The canonical operator-facing error is:

```text
legacy v1.4.x reservations no longer supported in v2.0.0; drain v1.4.x before upgrade
```

Depending on the entry path, this may surface as either a Pydantic validation
error while loading or constructing the `CapacityReservation`, or as a
`ValueError` from `RateLimiter.refund_capacity(...)` /
`SyncRateLimiter.refund_capacity(...)` when a previously serialized object is
presented for refund. Logs use the shorter wording
`legacy v1.4.x reservations are rejected in v2.0.0`. Treat both shapes as the
same migration signal: drain or refund in-flight reservations before moving
traffic to v2.0.0+ processes.

v2.1.0 adds an optional `max_reservation_lifetime_seconds` constructor
argument on `RateLimiter` and `SyncRateLimiter`. Memory backends preserve the
v2.0.0 unbounded lifetime behavior when this is omitted. Redis backends derive a
default bounded lifetime from `bucket_ttl_seconds` and
`refund_dedup_ttl_seconds`; if you pass an explicit bound, both TTLs must be
greater than `max_reservation_lifetime_seconds * 2`; construction raises
`ValueError` otherwise. Drain old serialized reservations before enabling the
bound because v2.0.0 reservations do not carry the `created_at_seconds`
timestamp required to enforce it.

## 3. Add Redis Key Prefixes

Redis backend builders and OpenAI Redis factories require a deployment-scoped
`key_prefix`. Pick a stable prefix per deployment or tenant, for example
`"prod-api"` or `"tenant-a"`. The same prefix must be used by every process
that should share rate-limit state.

`key_prefix` is namespace isolation only, not fairness or hostile-tenant
resource isolation. Tenants sharing one Redis server still share CPU, memory,
`maxclients`, Lua scheduling, eviction policy, and network capacity. Hostile
tenants require deployment-layer isolation: use separate Redis instances per
tenant, or put Redis behind a quota-aware proxy or infrastructure layer that
enforces per-tenant resource limits.

## 4. Review Callback Construction

`RateLimiterCallbacks(...)` and `SyncRateLimiterCallbacks(...)` now merge
user-provided slots with factory defaults. Update code that assumed a partially
specified callback bundle disabled every default callback.

## 5. Refactor DTO Subclasses

`Quota`, `PerModelConfig`, and `CapacityReservation` are strict DTOs, not
extension points. Replace subclass-based customization with composition,
factory functions, or explicit `PerModelConfig` construction before upgrading.

## 6. Redis ACL Requirements

Redis backends require Redis server 6.2 or newer. token-throttle uses `GET`,
`EXISTS`, `SET`, `DEL`, `TIME`, `EXPIRE`, and pipeline operations. No `KEYS`,
`FLUSHDB`, `FLUSHALL`, `CONFIG`, or Pub/Sub commands are issued by the library.

Redis acquire-marker and refund transactions use Lua `EVAL`. Redis lock release
and extension (via redis-py) also require `EVALSHA` and `SCRIPT LOAD`. These are
typically covered by the `+@scripting` ACL category. If your managed Redis
restricts scripting, ensure that category is allowed for the token-throttle
connection user.

**`SCRIPT FLUSH` operational hazard**: avoid running `SCRIPT FLUSH` on a
Redis instance shared with token-throttle. It evicts the cached Lua SHA for
redis-py's lock release/extend scripts; the next lock operation reloads the
script via `SCRIPT LOAD`, which adds a round-trip and, if that command is also
blocked by an ACL, permanently breaks lock release until the process restarts.
Schedule `SCRIPT FLUSH` only during planned maintenance windows when token-throttle
is not running, or use a dedicated Redis DB that is not shared with other
services that flush the script cache.

## 6a. Clean Up Pre-FIX-38 Redis Bucket Keys

FIX-38 added expiries to bucket state when keys are touched. Idle keys written
before that fix may still have no TTL. After draining in-flight reservations
and during a maintenance window, run the cleanup helper for each deployment
prefix:

```python
from token_throttle.migration import cleanup_legacy_buckets

deleted = cleanup_legacy_buckets(redis_client, key_prefix="prod-api")
print(f"deleted {deleted} legacy bucket state keys")
```

For `redis.asyncio.Redis`, use `async_cleanup_legacy_buckets(...)`. Both helpers
are prefix-scoped: they scan only keys under
`{key_prefix}:rate_limiting:bucket:*`, then delete only `:last_checked` and
`:capacity` keys whose Redis `TTL` is `-1`. They do not touch acquire markers,
refund-dedup keys, schema registry keys, max-capacity overrides, or keys for
other prefixes.

The cleanup is idempotent. Re-running it after a successful live run should
delete `0` keys because only no-expiry legacy bucket-state keys are eligible.
Run it separately for each deployment prefix that used token-throttle before
FIX-38.

Recommended dry-run workflow:

1. Run the same Redis `SCAN` pattern for the target `key_prefix`, or use broad
   inventory patterns such as `{key_prefix}:rate_limiting:*:last_checked` and
   `{key_prefix}:rate_limiting:*:capacity`, and count candidate bucket-state
   keys whose `TTL` is `-1`.
2. Confirm the prefix is the intended deployment and that in-flight
   reservations have drained.
3. Run `cleanup_legacy_buckets(...)` or `async_cleanup_legacy_buckets(...)`
   during a maintenance window.
4. Repeat the scan and verify the no-TTL candidate count is `0`.

## 7. Reservation Serialization Notes

### 7a. Future reservation fields

`CapacityReservation` uses strict Pydantic DTO validation. Unknown fields are
not ignored on load; they raise `ValidationError`. Forward compatibility is
limited to fields already known to the installed version and defined with a
safe default. If a future field carries an authorization signal, older
processes will not understand or enforce it. The only safe upgrade path for
security-sensitive field additions is to drain in-flight reservations before
exposing old processes to new ones.

### 7b. Redis key format and Lua compatibility

The Redis bucket key format is stable across v1.4.1, v1.5.0, v2.0.0, and
v3.0.0:

```text
{key_prefix}:rate_limiting:bucket:{model_family}:{metric}:{per_seconds}:{suffix}
```

v3.0.0 also uses acquire-marker keys:

```text
{key_prefix}:rate_limiting:acquired:{reservation_id}
```

token-throttle runs Lua for atomic acquire-marker writes and refunds, and
redis-py's lock implementation runs its standard lock release, extend, and
reacquire scripts. If you write custom scripts that interact with token-throttle
bucket keys, version them or use redis-py's `Script` object (which retries
`EVALSHA` with `SCRIPT LOAD` on `NOSCRIPT` errors) so that script-cache eviction
does not silently break your writes.

### 7c. Callback slot compatibility

The five callback slots (`on_wait_start`, `after_wait_end_consumption`,
`on_capacity_consumed`, `on_capacity_refunded`, `on_missing_consumption_data`)
are unchanged in v2.0.0. Partial callback construction such as
`RateLimiterCallbacks(on_wait_start=my_fn)` remains valid. The behavioral
change in v2.0.0 (FIX-27) is that factory-provided defaults are now *merged*
into partial bundles rather than replaced; see section 4 above.
