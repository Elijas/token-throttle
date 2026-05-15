# Migrating from v1.4.x to v2.0.0

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

v2.1.0 adds an optional `max_reservation_lifetime_seconds` constructor
argument on `RateLimiter` and `SyncRateLimiter`. The default is `None`, which
preserves the v2.0.0 unbounded lifetime behavior. If you enable a bounded
lifetime with Redis backends, `bucket_ttl_seconds` and
`refund_dedup_ttl_seconds` must both be greater than
`max_reservation_lifetime_seconds * 2`; construction raises `ValueError`
otherwise. Drain old serialized reservations before enabling the bound because
v2.0.0 reservations do not carry the `created_at_seconds` timestamp required to
enforce it.

## 3. Add Redis Key Prefixes

Redis backend builders and OpenAI Redis factories require a deployment-scoped
`key_prefix`. Pick a stable prefix per deployment or tenant, for example
`"prod-api"` or `"tenant-a"`. The same prefix must be used by every process
that should share rate-limit state.

## 4. Review Callback Construction

`RateLimiterCallbacks(...)` and `SyncRateLimiterCallbacks(...)` now merge
user-provided slots with factory defaults. Update code that assumed a partially
specified callback bundle disabled every default callback.

## 5. Refactor DTO Subclasses

`Quota`, `PerModelConfig`, and `CapacityReservation` are strict DTOs, not
extension points. Replace subclass-based customization with composition,
factory functions, or explicit `PerModelConfig` construction before upgrading.

## 6. Redis ACL Requirements

token-throttle uses `GET`, `SET`, `DEL`, `TIME`, `EXPIRE`, and pipeline
operations. No `KEYS`, `FLUSHDB`, `FLUSHALL`, `CONFIG`, or Pub/Sub commands
are issued by the library.

Redis lock acquire and release (via redis-py) also require scripting commands:
`EVALSHA` and `SCRIPT LOAD`. These are typically covered by the
`+@scripting` ACL category. If your managed Redis restricts scripting, ensure
that category is allowed for the token-throttle connection user.

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

For `redis.asyncio.Redis`, use `async_cleanup_legacy_buckets(...)`. The helper
uses `SCAN`, checks only token-throttle bucket `:last_checked` and `:capacity`
keys, and deletes only keys whose Redis `TTL` is `-1`.

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

The Redis key format is stable across v1.4.1, v1.5.0, and v2.0.0:

```
{key_prefix}:rate_limiting:{model_family}:{metric}:{per_seconds}:{suffix}
```

token-throttle does not register custom Lua scripts. The only Lua in normal
operation is redis-py's standard lock release, extend, and reacquire scripts.
If you write custom scripts that interact with token-throttle's bucket keys,
version them or use redis-py's `Script` object (which retries `EVALSHA` with
`SCRIPT LOAD` on `NOSCRIPT` errors) so that script-cache eviction does not
silently break your writes.

### 7c. Callback slot compatibility

The five callback slots (`on_wait_start`, `after_wait_end_consumption`,
`on_capacity_consumed`, `on_capacity_refunded`, `on_missing_consumption_data`)
are unchanged in v2.0.0. Partial callback construction such as
`RateLimiterCallbacks(on_wait_start=my_fn)` remains valid. The behavioral
change in v2.0.0 (FIX-27) is that factory-provided defaults are now *merged*
into partial bundles rather than replaced; see section 4 above.
