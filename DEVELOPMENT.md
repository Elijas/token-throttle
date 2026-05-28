# Development

## Setup

```bash
uv sync --all-extras --group dev
```

## Running tests

```bash
# Unit tests only (no Redis required)
uv run pytest tests/unit -v

# Full suite (requires Redis on localhost:6379)
uv run pytest tests/ -v --redis-url redis://localhost:6379
```

## Type checking

```bash
uv sync --all-extras --group dev
uv run mypy
```

The mypy gate checks the complete `token_throttle/` package with normal import
following. Install all extras before running it because the checked package
includes Redis, OpenAI, and tokenizer integration modules.

## CI structure

CI runs six jobs (see `.github/workflows/ci.yml`):

| Job | What it does | Extras installed |
|-----|-------------|-----------------|
| `lint` | ruff check + format on Linux / Python 3.12 | dev only |
| `type-check` | package-wide mypy on Linux / Python 3.12 | all extras + dev |
| `test-unit-core` | Unit tests without optional deps on Linux / Python 3.12, 3.13, 3.14 | dev only |
| `test-unit-full` | Unit tests with all optional deps on Linux / Python 3.12, 3.13, 3.14 | all extras + dev |
| `test-unit-platform` | Unit tests with all optional deps on macOS and Windows / Python 3.13 | all extras + dev |
| `test-integration` | Integration tests against Redis on Linux / Python 3.12, 3.13, 3.14 | all extras + dev |
| `coverage` | Full suite + Codecov upload on Linux / Python 3.13 | all extras + dev |

Supported Python matrix: Python 3.12, 3.13, and 3.14, matching
`requires-python = ">=3.12"` and the package classifiers. Python 3.10 and 3.11
are not supported or tested. `type-check` runs on Python 3.12 only, matching
`[tool.mypy].python_version`.

Platform matrix: Linux remains the full test and Redis integration target.
macOS and Windows run the all-extras unit suite on Python 3.13 to catch
platform-specific unit bugs without multiplying every Python version across
every operating system. Redis integration tests stay Linux-only because GitHub
Actions service containers are Linux-oriented and this project targets Redis
behavior rather than OS-specific Redis packaging. Redis 7 (alpine) is used as
the GitHub service container for integration and coverage jobs.

## Known constraints and assumptions

### Redis fault-injection testing scope

FIX-45 (commit `cc651dd`) verifies Redis acquire-marker reconciliation with
deterministic fake-client tests that simulate an `EVAL` reply being lost after
Redis has committed the Lua transaction. Those tests cover token-throttle's
retry/reconciliation behavior without relying on network timing.

R7 deliberately deferred a real TCP proxy fault-injection harness. The current
test suite does not drop TCP ACKs after a real server commit, simulate a real
Redis server crash mid-`EVAL`, or validate every managed-Redis failure mode.
Correctness here relies on Redis Lua atomicity plus fake-client coverage;
operators should validate end-to-end behavior in their own deployment.

### API validation raises `ValueError`

At public API boundaries, token-throttle raises `ValueError` for bad-value
and bad-shape input, including cases that Python's narrower convention might
classify as `TypeError`. This gives callers one user-facing validation class
to catch. Ruff's `TRY004` is suppressed at the specific sites that encode this
choice.

Use `RuntimeError` for broken internal invariants that users cannot correct by
changing their request arguments. Reserve `TypeError` for places where the code
is intentionally matching Python's own call-signature convention.

### Logging uses stdlib only

`create_logging_callbacks` and `create_sync_logging_callbacks` emit through
stdlib `logging.getLogger("token_throttle")`. There is no `loguru` optional
extra or loguru-specific callback factory in v8.

`TRACE` and `SUCCESS` are accepted only as compatibility level-name aliases in
`_STDLIB_LEVEL_MAP`; they map to stdlib `DEBUG` and `INFO`, respectively.
`_models.py` uses `warnings.warn()` for the empty-quota warning.

### Reservation and serialization trust boundary

`CapacityReservation` is trusted in-process state. It is safe to pass between
your own functions, but it is not a signed or durable authorization token.
Do not accept pickled, JSON, cloudpickle, dill, or other serialized
reservations from an untrusted process as proof that capacity was acquired.
Arrow IPC is less direct because callers must convert back to plain Python
types before `model_validate()`, but the trust-boundary rule is the same:
only refund reservations produced inside the trusted limiter workflow.

Reservations should be refunded while the issuing limiter and backend are still
alive. After callable-config changes, refunds are scoped to bucket ids captured
at acquire time; removed buckets are not credited back into unrelated future
metrics.

v2.0.0 deliberately breaks v1.4.x reservation compatibility. Drain in-flight
reservations before upgrading; mixed v1.4.x/v2.0.0 fleets are not supported.
`CapacityReservation` requires a non-empty `limiter_instance_id`, and refund
raises `ValueError("legacy v1.4.x reservations no longer supported in v2.0.0; drain v1.4.x before upgrade")`
if a legacy object with `limiter_instance_id is None` reaches the refund path.
For Redis backends, successful refunds claim
`{key_prefix}:rate_limiting:refund_dedup:{reservation_id}` with `SET NX EX`;
the TTL defaults to 7 days and is configurable via
`refund_dedup_ttl_seconds`. Memory backends have no durable refund dedup and
reject refunds for reservations missing from local in-flight state.

v3.0.0 deliberately breaks Redis refund compatibility with v2.x reservations.
Redis acquires write durable acquire markers, and refunds must consume the
marker before crediting capacity. A Redis-backed v2 reservation presented to a
v3 process raises `UnknownReservationError`; this is the intended fail-closed
contract, not a transient Redis miss. Drain v2 traffic before deploying v3 and
accept only a brief straggler refund-error window.

v4.0.0 makes the limiter's internal reservation snapshot authoritative for
refunds. Caller mutations or `model_copy()` changes to a returned reservation
do not change refund scope or reserved usage. This preserves the acquire-time
authority captured by the limiter and prevents user-space reservation edits
from redirecting capacity credits during rollback or mixed-version windows.

`max_reservation_lifetime_seconds` is optional on both public limiters. Memory
backends leave it unbounded when omitted. Redis builders derive a default from
the shorter of `bucket_ttl_seconds` and `refund_dedup_ttl_seconds` so every
Redis reservation has a finite lifetime: just below
`min(bucket_ttl_seconds, refund_dedup_ttl_seconds) / 2` with the default safety
margin. Every new reservation records `created_at_seconds`, and refund rejects
reservations older than the configured or derived bound. Redis builders validate
the bound at limiter construction:
`bucket_ttl_seconds` and `refund_dedup_ttl_seconds` must both be greater than
`max_reservation_lifetime_seconds * 2`. This keeps bucket state, acquire
markers, and refund dedup tombstones alive for the expected reservation
lifecycle.

Redis bucket-state TTLs are inactivity TTLs on `:last_checked`, `:capacity`,
and max-capacity override state. They are refreshed when bucket state is read or
written, not on unrelated acquire-marker or refund-dedup operations. The schema
version registry key is intentionally long-lived and has no TTL. Acquire-marker
and refund-dedup keys have their own TTL/lifetime budgets and should be sized
from request latency, retry delay, and traffic rate.

### Unlimited configs

`UsageQuotas.unlimited()` is the public way to disable limits for a model.
`PerModelConfig.is_unlimited` is derived from the quota set and is part of the
model-family signature: models sharing a `model_family` must all be limited or
all be unlimited with identical quota structure.

Unlimited direct acquires validate numeric values but intentionally do not
shape-check metric keys because there is no quota key set. Request acquires may
still invoke `usage_counter` for telemetry consistency; the resulting usage and
any `extra_usage` are discarded in the returned unlimited reservation.

### Custom backend reconfiguration contract

Audited in R4 L18 H03/H04. A backend that returns `True` from
`supports_metric_set_change()` must preserve live state for surviving buckets
when callable configs add or remove metrics. It can do that by storing state in
external shared storage such as Redis, or by overriding
`prepare_reconfigured_backend()` to migrate local state into the rebuilt
backend. In-process custom backends that return `True` while inheriting the
no-op migration method can silently reset consumption state.

Configured-cap changes from callable configs are per process. Runtime overrides
from `set_max_capacity()` are distributed by Redis; `apply_configured_max_capacity`
is the internal config-rebuild path and is not a public `RateLimiter` method.

### Redis integration tests are not parallel-safe

- Fixtures call `flushdb()` on teardown (`tests/integration/conftest.py`).
- Tests use fixed model families like `"test"`.
- This is fine for serial test runs (current CI), but will break under `pytest-xdist` or a shared Redis instance.

**If you adopt parallel test execution**, you'll need per-worker key prefixes or separate Redis DB numbers.

### Sync concurrency test thread is joined

`test_excess_acquires_must_wait` in `tests/integration/test_sync_concurrency.py` starts a background thread but joins it with a 2-second timeout and asserts it exited cleanly. No daemon thread leak.

### Reservation lifecycle

`CapacityReservation` is a refund token for the backend that issued it. Reservations require `limiter_instance_id`, a per-`RateLimiter` / `SyncRateLimiter` UUID generated at construction time, so v1.4.x legacy objects remain distinguishable. Redis refund authority comes from the durable acquire marker written atomically with bucket consumption. Memory refund authority comes from the in-process acquire table.

This is a v2.0.0 contract change. Reservations serialized by v1.4.x through pickle, JSON, job queues, or caches do not have `limiter_instance_id`; those are rejected. Drain in-flight reservations before upgrading and do not run mixed v1.4.x/v2.0.0 fleets. New serialized reservations must preserve the field. Starting in v3.0.0, Redis-backed cross-process refunds are allowed when the refunding process uses the same Redis deployment and key prefix, because the backend verifies and consumes the acquire marker before crediting capacity.

Memory reservations have no TTL by default. Redis reservations always have a finite lifetime: either the caller's `max_reservation_lifetime_seconds` or the backend-derived default from Redis TTLs. A reservation remains eligible until it is refunded, rejected because the model now routes differently, rejected because its lifetime has elapsed, rejected because its acquire marker is missing, or the limiter is closed. Redis backends write durable acquire markers and refund dedup keys, so an in-flight reservation can be refunded by another process sharing the same Redis deployment and key prefix while the marker exists. Memory backends reject cold-restart and cross-process cases because they cannot prove whether capacity was acquired or already credited. `close()` / `aclose()` mark the limiter closed, log the number of reservations still in flight, and block subsequent acquire/refund operations.

Callable config changes are checked at refund time for held limited reservations. A limited-to-unlimited flip raises instead of crediting an obsolete cached backend, and a `model_family` reroute raises instead of crediting the old family backend. Metric-set changes still project refunds onto surviving bucket ids; if the projection is empty, the refund id is committed to local dedup and, for Redis backends, durable Redis dedup before returning so queue retries cannot double-credit after a rebuild.

### `per_seconds` is constrained to integers

`Quota.per_seconds` is typed as `int`. Pydantic v2 coerces whole floats (`60.0` -> `60`) but rejects fractional values (`0.5`). This is intentional — fractional rate-limiting windows don't make practical sense, and the Redis key format and capacity dict keys rely on integer values.

### `sleep_interval=0` is a valid configuration

Backend constructors accept `sleep_interval=0` for busy-wait polling. The default (`0.1s`) only applies when `sleep_interval` is `None` (not passed). This uses `is None` checks, not truthiness, so `0` is not treated as "use default."

### Negative capacity is preserved, not clamped to zero

The speedometer pattern (`record_usage` / `consume_capacity`) and `refund_capacity` both allow capacity to go negative. This is intentional — clamping to zero would erase debt and let the bucket refill from zero instead of recovering naturally.

Example: bucket at 50, actual usage 130 → capacity becomes −80. The token-bucket refill adds `rate_per_sec × elapsed` on each check, so the bucket gradually recovers to positive. If we clamped to 0, the 80-unit overuse would vanish.

The `allow_negative` flag on `set_capacity` / `_set_capacities_unsafe` controls this. The blocking path (`await_for_capacity` / `_check_and_consume_capacity`) uses `allow_negative=False` because it guarantees capacity ≥ usage before consuming.

### `set_max_capacity` anchors capacity before swapping the refill rate

When `set_max_capacity` changes a bucket's limit, it performs an anchor-and-swap in three steps:

1. **Anchor at old rate:** capacity is updated using the *old* `_rate_per_sec` for the time elapsed since `last_checked` (`capacity += elapsed × old_rate`).
2. **Reset timestamp:** `last_checked` is set to `now`, so no elapsed time carries over.
3. **Swap rate:** `max_capacity` and `_rate_per_sec` are updated to reflect the new limit.

Because `last_checked` is reset to `now` before the new rate takes effect, the new rate applies only to *future* time — it is not applied retroactively to time that elapsed under the old rate. The Redis backend achieves the same result via `_snapshot_bucket_state` (which anchors and writes back to Redis) before calling `bucket.set_max_capacity`.

### `_log()` fails fast on unrecognized level strings

`_STDLIB_LEVEL_MAP` uses `[]` lookup (not `.get()`). Unknown levels raise
`KeyError` immediately. This is intentional — `_log()` is a private function
only called from the `create_*_callbacks()` factories, which only pass
standard level strings (DEBUG, INFO, WARNING, ERROR, CRITICAL) plus the
compatibility aliases (TRACE, SUCCESS). Adding a `.get()` fallback would
silently mask typos in level names.

### Capacity matching loop and the postconsumption invariant

`_check_and_consume_capacity` and `consume_capacity` build a `postconsumption_dict` via a nested loop that matches capacity entries against usage entries by metric name. This O(n×m) pattern appears in all four backends (async/sync × memory/Redis).

**Invariant:** `postconsumption_dict` must cover ALL entries in `preconsumption_capacities`. If any bucket is missed, `_set_capacities` won't update its timestamp, causing incorrect refill calculations on the next read. This is guaranteed by `validate_acquire_usage()` (called in `_acquire_capacity`) which enforces `set(usage.keys()) == set(quotas.names)`. Each backend asserts this post-hoc with `assert len(postconsumption_dict) == len(preconsumption_capacities)`.

The O(n×m) cost is irrelevant in practice — n (buckets) and m (usage metrics) are typically 2–5.

### `_PIPELINE_CMDS_PER_BUCKET` in Redis backends

Redis `_get_capacities_unsafe` batches pipeline commands in a fixed layout:
each bucket enqueues `GET last_checked`, `GET capacity`, then `EXPIRE` for
both bucket-state keys to refresh the mandatory bucket TTL. The backend then
queues `GET max_capacity_override` plus `EXPIRE max_capacity_override` for each
bucket. `_PIPELINE_CMDS_PER_BUCKET`, `_PIPELINE_CMDS_PER_OVERRIDE`, and result
count assertions enforce this layout. If `get_capacity()` or the pipeline
structure changes, the assertion catches the mismatch immediately rather than
silently reading wrong values. The `{key_prefix}:rate_limiting:schema_version`
key is intentionally exempt from TTL because it is a long-lived registry.

### Over-limit validation lives in the backend, not `validate_acquire_usage`

`validate_acquire_usage` checks key-match, finiteness, and non-negativity — but does **not** check `usage > quota.limit`. That check was removed intentionally because `set_max_capacity` can change a bucket's limit at runtime, making the static `quota.limit` stale.

Instead, each backend performs the over-limit check inside its lock against the live `bucket.max_capacity`. This means over-limit requests acquire the lock (and, for Redis, a pipeline round-trip) before failing. That cost is acceptable because over-limit requests are programming errors, not normal traffic, and checking outside the lock would require reading a potentially-stale cached value then re-checking under the lock anyway.

### `on_missing_consumption_data` callback is delayed until first successful acquire

When `_check_and_consume_capacity` returns `False` (insufficient capacity), it exits before calling `_fresh_start_buckets_callback`. The `on_missing_consumption_data` callback won't fire until the first *successful* capacity acquisition. This is by design — firing it on every 100ms poll iteration would be noisy. Since `last_checked` is never written on the insufficient-capacity path, the fresh-start condition persists and the callback fires exactly once when capacity is first successfully consumed.

### Redis `consume_capacity` callbacks are best-effort after cancellation

In the async Redis backend, `consume_capacity` shields the Redis write so that a task cancellation cannot leave the write half-applied. After `suppress_current_task_cancellation()` returns, callbacks (`on_capacity_consumed`, `on_missing_consumption_data`) still fire. If a *second* cancellation arrives during those callbacks, they are skipped. This is acceptable: the speedometer consumption is already durably recorded in Redis, and callbacks are best-effort via `_invoke_callback_safe`.

### `SyncRateLimiter` has no abstract base class

`RateLimiter` extends `BaseRateLimiter` (ABC); `SyncRateLimiter` does not extend any abstract base. Adding a `BaseSyncRateLimiter` would be a public API change. The sync interface is documented by its method signatures and mirrors the async API.

### Model-family caches are bounded and explicitly evictable

`RateLimiter` and `SyncRateLimiter` maintain per-model-family dicts
(`_model_family_to_backend`, `_model_family_to_model_name`, etc.) and a
model-alias reverse map. These are guarded by mandatory fail-closed caps:
`max_model_families`, `max_metrics_per_family`, `max_aliases`, and
`max_in_flight_reservations`. Length caps are enforced for model families,
metrics, and aliases. Applications that generate dynamic user-controlled model
names should still prefer allowlists where possible and should run
`clear_unused_model_families(unused_for_seconds)` from an operator-controlled
maintenance path to evict idle in-process rows. The cleanup API skips families
with in-flight reservations; Redis bucket state uses its own inactivity TTL.

### Redis `max_capacity_override` self-heals on config mismatch

When `_deserialize_max_capacity_override` reads a stored override from Redis, it compares the `configured_max_capacity` field in the JSON payload against the current process's `_max_capacity_default` (from `Quota.limit`). If they differ — e.g. after a deployment changes the static quota — the override is silently discarded (returns `None`), causing the bucket to fall back to the new static limit.

This is intentional self-healing: an override created under a previous quota configuration should not pin the new deployment to a stale limit. The override was set relative to the old config; applying it under a different config would produce an unexpected effective limit. Discarding it lets the new static config take effect cleanly, and operators can re-apply an override if needed.

### Redis connection pool sizing

The Redis backend accepts a user-provided `redis.asyncio.Redis` (or `redis.Redis`) client and uses its connection pool as-is. By default, `redis-py` creates a pool with `max_connections=2**31` (effectively unlimited). In high-fanout applications (many concurrent `await_for_capacity` calls), the actual connection count may spike because each pipeline/lock acquisition can consume a connection.

If you need to bound connection usage, use `BlockingConnectionPool` before passing the client to the rate limiter. Size `max_connections` to at least your expected `max_concurrent_acquires` plus headroom for Redis lock acquire/release, the `TIME` command, and pipeline reads/writes. As a starting point, use `max_connections >= max_concurrent_acquires + 10` and tune from Redis pool wait time and server connection metrics. The Redis backend emits a `RuntimeWarning` when it sees `max_connections < 10`, because that is usually too small for production traffic.

The README sizing numbers come from R7 short local runs and Redis object-size
estimates, not a maintained sustained-load production benchmark suite. Treat
them as planning guidance only and validate throughput, p99 latency, key
cardinality, memory, and `maxclients` pressure under the workload and Redis
configuration you will run in production.

```python
import redis.asyncio as aioredis
from token_throttle import RedisBackendBuilder

pool = aioredis.BlockingConnectionPool.from_url(
    "redis://localhost",
    max_connections=110,
    timeout=5,
)
client = aioredis.Redis(connection_pool=pool)
backend = RedisBackendBuilder(client, key_prefix="test")
limiter = RateLimiter(get_config, backend=backend)
```

Redis lock polling is also configurable. `lock_sleep_seconds` controls the redis-py lock polling interval and defaults to `0.05` seconds instead of redis-py's `0.1` second default. `lock_blocking_timeout_seconds` defaults to `5.0` seconds and bounds a Redis lock acquisition even when the caller waits for rate-limit capacity without an overall timeout. The sync Redis builder also accepts `lock_blocking_thread_sleep_seconds` for redis-py's sync `Lock.acquire(sleep=...)` override.

### Fork safety (Redis backend)

`RateLimiter` and `SyncRateLimiter` capture the user-supplied Redis client by reference in the builder, backend, and every bucket. If a limiter is built before `os.fork()` (common with gunicorn preload or `multiprocessing.Pool`), the parent and child share the same connection pool, leading to interleaved I/O and silent data corruption.

- Do not reuse a `RateLimiter` instance across fork boundaries.
- Build the limiter lazily inside each worker process.
- If fork cannot be avoided, call `redis_client.close()` and re-create both the client and the limiter in the child's post-fork hook.

### Float precision at extreme capacity limits

All capacity values are Python `float` (IEEE 754 double). At limits above ~2^53, integer precision is lost: consecutive integers are indistinguishable, so `capacity - usage` may not change the stored value. This is a known limitation of the float64 representation and is acceptable for all real-world rate-limiting scenarios (token quotas are orders of magnitude below 2^53).

### Redis ACL requirements and `SCRIPT FLUSH` hazard

The minimum Redis ACL permission set for token-throttle:

| Command category | Commands used | Why |
|---|---|---|
| `+@read` | `GET`, `EXISTS` | Read bucket capacity, marker, and refund-dedup state |
| `+@write` | `SET`, `DEL`, `EXPIRE` | Write bucket state, acquire markers, refund dedup keys, and TTL |
| `+@string` | included in read/write above | Plain string key/value operations |
| `+@scripting` | `EVAL`, `EVALSHA`, `SCRIPT LOAD` | Atomic acquire/refund Lua plus redis-py lock release and extend Lua scripts |
| `+TIME` | `TIME` | Server-side clock for elapsed-time calculations |

Redis backends require Redis server 6.2 or newer.

token-throttle does **not** use `KEYS`, `FLUSHDB`, `FLUSHALL`, `CONFIG`, or any
Pub/Sub command. A restrictive ACL can safely deny those categories.

The migration helper `cleanup_legacy_buckets()` is an operator-run maintenance
tool, not part of limiter hot paths. It uses `SCAN`, `TTL`, and `DEL` to remove
pre-FIX-38 bucket `:last_checked` / `:capacity` keys that have no expiry.

**`SCRIPT FLUSH` operational hazard**: `SCRIPT FLUSH` evicts the Lua SHA cache
used by redis-py's lock release and extend scripts. The next lock operation
reloads the script, but if scripting commands are denied by ACL, the reload
fails and lock release silently errors. Schedule `SCRIPT FLUSH` only during
planned maintenance when token-throttle is not running. Do not issue it on a
shared Redis DB that token-throttle shares with other services unless all
Lua-using clients have been stopped.

### Sync `config_getter` reentrancy under `_validation_lock`

`SyncRateLimiter._validate_shared_model_family_config` calls the user-supplied
`config_getter` (the `cfg` argument to `SyncRateLimiter.__init__`) while
holding `_validation_lock`. `threading.Lock` is **not reentrant**.

If your `config_getter` calls back into the same `SyncRateLimiter` instance
(for example, to check current capacity before deciding what config to return),
the reentrant acquire path will also call `_validate_shared_model_family_config`,
which will block trying to acquire `_validation_lock` that the outer call
already holds — deadlock.

This is a user-code contract, not a library bug. The library never calls
`config_getter` recursively. Safe `config_getter` implementations:

- Return a static or externally-cached `PerModelConfig` without touching the limiter.
- Call out to an external service or config store.
- Call into a *different* `SyncRateLimiter` instance.

### R4 documentation audit cross-references

FIX-21 checked and closed the documentation-only R4 audit gaps across L01-L22:
F03/F05/F11-F14/F24/F31/F32/F34/F41/F43/F45; E08; P02-P06 where still
applicable; I03/I04/I06/I07/I10/I12; U02/U05/U10/U12-U15; S03/S05/S07;
X05/X10/X12/X14; Y05/Y08-Y10/Y13/Y14; N03/N07/N09/N11/N14; J06/J09;
T03/T04/T06; O04. Some lanes were already closed by earlier fix bundles in
this branch; the status report for FIX-21 records which surfaces were verified
rather than re-edited.

### R5 documentation audit cross-references

FIX-39 closed the R5 informational findings D16/D18/D19/D20/D32/D33/D34:
Redis ACL and SCRIPT FLUSH hazard (D16); reservation future-field contract
(D18); wire-format and Lua continuity v1.4.1–v2.0.0 (D19); callback slot
compatibility (D20); runtime-override map `_lock` invariant (D32);
`config_getter` reentrancy under `_validation_lock` (D33); Redis
`_extend_locks` coverage confirmed clean, lint test added (D34).
