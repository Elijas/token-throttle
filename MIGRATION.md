# Migration Guide

## Migrating from v4.x to v5.0.0

Custom backend interfaces are now structural `Protocol` classes. Backends that
subclass `RateLimiterBackend`, `SyncRateLimiterBackend`, or the builder
interfaces can keep inheriting the conservative default hooks, but custom
backends that rely on structural typing must implement the full protocol
surface, including cleanup hooks and optional authority hooks.

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
`{key_prefix}:rate_limiting:*:last_checked` and
`{key_prefix}:rate_limiting:*:capacity`, then delete only keys whose Redis
`TTL` is `-1`. They do not touch acquire markers, refund-dedup keys, schema
registry keys, max-capacity overrides, or keys for other prefixes.

The cleanup is idempotent. Re-running it after a successful live run should
delete `0` keys because only no-expiry legacy bucket-state keys are eligible.
Run it separately for each deployment prefix that used token-throttle before
FIX-38.

Recommended dry-run workflow:

1. Run the same Redis `SCAN` patterns for the target `key_prefix` and count
   candidate `:last_checked` / `:capacity` keys whose `TTL` is `-1`.
2. Confirm the prefix is the intended deployment and that in-flight
   reservations have drained.
3. Run `cleanup_legacy_buckets(...)` or `async_cleanup_legacy_buckets(...)`
   during a maintenance window.
4. Repeat the scan and verify the no-TTL candidate count is `0`.

## 7. Reservation Serialization Notes

### 7a. Future reservation fields

`CapacityReservation` uses Pydantic with `populate_by_name=True` and ignores
unknown fields on load. New fields added in future versions always have a
default value so that older-version pickles load without error (the field will
be `None` or a safe default). However, do not assume old readers will
*enforce* new security-sensitive fields. If a future field carries an
authorization signal, older processes that ignore it can still bypass that
signal. The only safe upgrade path for security-sensitive field additions is to
drain in-flight reservations before exposing old processes to new ones.

### 7b. Redis key format and Lua compatibility

The Redis bucket key format is stable across v1.4.1, v1.5.0, v2.0.0, and
v3.0.0:

```
{key_prefix}:rate_limiting:{model_family}:{metric}:{per_seconds}:{suffix}
```

v3.0.0 also uses acquire-marker keys:

```
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
