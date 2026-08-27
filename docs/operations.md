# Operations guide

Running token-throttle in production across multiple workers and processes:
reservation durability, supported Redis topologies, multi-tenant isolation,
connection/TTL sizing, and capacity planning for high-RPS fleets.

Start with the [README](../README.md) for installation, the mental model, and
quickstarts. Reach for this guide once you are deploying on Redis and need to
reason about durability, scaling ceilings, and resource budgets.
For single-host, multi-process deployments, use the dedicated
[SQLite backend guide](sqlite-backend.md).

## Reservation lifecycle and durability

A `CapacityReservation` is an internal accounting token, not a durable portable
credential. Refund it on the same limiter lifetime that issued it, after the API
call finishes. If your config changes before refund, token-throttle refunds only
the surviving buckets that still correspond to the reservation; buckets removed
by a callable-config rebuild are skipped to avoid crediting unrelated capacity.

Unlimited reservations are no-ops on refund. They are trusted in-process
objects, so do not deserialize, pickle, or accept reservations across trust
boundaries as proof that a caller was rate-limited. For queue-and-retry
workflows, reserve immediately before dispatching the external request rather
than storing reservations in a long-lived queue.

Every `CapacityReservation` requires a non-empty `limiter_instance_id`.

### If a worker crashes before refunding

There is no crash-recovery path that gives reserved-but-unrefunded capacity
back early. A reservation that is never refunded — because the worker process
died, was killed, or lost its network connection before calling
`refund_capacity()` — behaves exactly like fully-used capacity: it stays
consumed against the bucket. Capacity is not returned in a burst when the
crash is detected; it recovers the same way any other consumed capacity does,
through the bucket's normal linear refill over the quota window.

The TTL knobs (`bucket_ttl_seconds`, `refund_dedup_ttl_seconds`,
`max_reservation_lifetime_seconds`) govern how long Redis keeps the
bookkeeping keys that make a reservation *refundable* and how long refund
idempotency is remembered — they do not govern capacity crediting. TTL expiry
never credits capacity back. A public refund after
`max_reservation_lifetime_seconds` fails with `ValueError` before backend I/O;
an otherwise missing backend marker raises `UnknownReservationError`.

`snapshot_state()["in_flight_reservations"]` is a process-local, in-memory
count on the limiter instance that issued the reservations. It cannot detect
reservations orphaned by a crashed process — that process's counter is gone
with it. Use application-level request tracing or deadline enforcement
(bound `max_reservation_lifetime_seconds` to a value close to your real
request timeout) if you need to detect or bound crash-orphaned reservations.

### If Redis loses bucket state (restart, FLUSHALL, eviction)

Each bucket's state lives in two Redis keys. If exactly one of them is missing,
token-throttle treats the bucket as corrupt and drains it to zero — it fails
**closed**, and the `on_missing_consumption_data` callback reports the reason
`"partial_state_drained"`. But if **both** keys are missing, token-throttle
cannot distinguish a model family it has simply never seen from a bucket whose
state was lost, so it resets the bucket to full capacity — it fails **open**,
with the reason `"fresh_start"`.

The practical consequence is the inverse of the fail-closed behavior described
elsewhere in this guide: a Redis restart without RDB/AOF persistence, a
`FLUSHALL`, or an eviction wave that drops both keys of a bucket silently resets
the affected quotas to full. If that reset is unacceptable, run Redis with
persistence and a `maxmemory-policy` that protects bucket keys. Monitor
`on_missing_consumption_data`: a single `"fresh_start"` is expected the first
time a model family is used, but a burst of `"fresh_start"` reasons across many
already-active model families at once is the signal that bucket-state loss just
reset live quotas.

## Concurrency model

Create one limiter per process and, for the async API, one limiter per event
loop. `RateLimiter` and `SyncRateLimiter` own in-process locks, lifecycle
state, and backend builders; they are not pickleable and should be constructed
inside each worker process after `fork()` or `spawn()`. For distributed
deployments construct the `redis_client` (with its connection pool) the same
way — build it after the fork/spawn point in each child, never once in a
parent/preload process and shared into children. By default the limiters check
process affinity on every public method and raise if a limiter is reused in a
different PID. Pass `pid_check=False` only when you deliberately accept the
unsupported risk of divergent in-memory state.

Use `RateLimiter` from async code and `SyncRateLimiter` from synchronous code.
Calling `SyncRateLimiter.acquire_capacity()` while an event loop is running
blocks that loop; token-throttle emits a `RuntimeWarning` once per process.

Both limiter types can own their close lifecycle through context managers:

```python
# (fragment — see the README Memory quickstart for standalone context)
async with RateLimiter(get_config, backend=MemoryBackendBuilder()) as limiter:
    reservation = await limiter.acquire_capacity({"requests": 1, "tokens": 500}, model="gpt-4.1")
    await limiter.refund_capacity({"requests": 1, "tokens": 320}, reservation)
```

```python
# (fragment — see the README Sync API example for standalone context)
with SyncRateLimiter(get_config, backend=SyncMemoryBackendBuilder()) as limiter:
    reservation = limiter.acquire_capacity({"requests": 1, "tokens": 500}, model="gpt-4.1")
    limiter.refund_capacity({"requests": 1, "tokens": 320}, reservation)
```

`close()` / `aclose()` drain pending acquires and in-flight refunds before
releasing backend resources, bounded by `close_drain_timeout_seconds` (default
5.0s). Size your process's shutdown grace period (for example Kubernetes
`terminationGracePeriodSeconds`) at or above this value so a rolling deploy does
not cut off reservations mid-refund.

## Redis topology support

token-throttle supports standalone Redis and Sentinel-aware clients connected
to the current Redis primary. It does not support Redis Cluster or client-side
sharded Redis.
The Redis backend uses multi-key Lua scripts for atomic acquire-marker and
refund updates; in Redis Cluster those keys can span hash slots and fail at
runtime. `RateLimiter` and `SyncRateLimiter` reject redis-py `RedisCluster`
clients during construction with a `ValueError` instead of failing later during
`EVAL`.

Do not add caller-controlled Redis hash tags to key prefixes, model families,
metrics, reservation ids, or other key segments. Public validators reject `{`
and `}` in Redis key segments, and Cluster support is intentionally unsupported.

### Supported redis-py client versions

The `redis` extra declares `redis>=5.2.1,!=8.0.0,!=8.0.1` — deliberately with
**no upper bound**. A ceiling in a library's metadata cannot be relaxed without
publishing a release, and it propagates into dependency-resolution conflicts for
anyone who depends on both this package and a newer client. So a new redis-py
major is allowed by default rather than blocked on the assumption it will break.

The two exclusions are specific defects, not caution:

| Excluded | Why |
|---|---|
| `8.0.0` | unix-socket clients fail the RESP3 maintenance-notification handshake ([redis-py#4086](https://github.com/redis/redis-py/issues/4086)) |
| `8.0.0`, `8.0.1` | the sync Sentinel pool permanently loses a slot on every failover until all commands raise `MaxConnectionsError` ([redis-py#4187](https://github.com/redis/redis-py/issues/4187), fixed in 8.1.0) — and Sentinel is a supported topology here |

An open-ended range is a promise about versions that do not exist yet, so it is
backed by a scheduled job that installs the newest redis-py and runs the Redis
suites against it. If a future release breaks the backend, that job goes red
before your deployment does; the fix is then either a code change or a new
exclusion with the same kind of evidence as the two above.

If you pin a client version yourself, note the pool-sizing change in
[Connection pooling](#connection-pooling-and-key-ttls) — redis-py 8 caps the
default pool where earlier versions did not.

Redis backends require Redis server 6.2 or newer and a Redis user that can run
`GET`, `EXISTS`, `SET`, `DEL`, `EXPIRE`, `PEXPIRE`, `PTTL`, `TIME`, `MULTI`,
`EXEC`, `DISCARD`, and Lua scripting (`EVAL`, `EVALSHA`, `SCRIPT LOAD`).
`PEXPIRE` is used by redis-py's lock extend/reacquire script, and `MULTI` /
`EXEC` / `DISCARD` are used by redis-py's transaction pipelines that
token-throttle uses for atomic acquire/refund transactions. token-throttle's own
code never issues `PEXPIRE` or `MULTI` / `EXEC` / `DISCARD` by name, so they are
easy to omit from a hand-written ACL: a user provisioned with only
`GET` / `EXISTS` / `SET` / `DEL` / `EXPIRE` / `PTTL` / `TIME` plus scripting can
pass an initial smoke test but fail during ordinary multi-quota usage as soon as
a lock needs to extend or a pipelined transaction is discarded. Lua scripting
access (`EVAL`, `EVALSHA`, `SCRIPT LOAD`) is typically covered by the
`+@scripting` ACL category; if your managed Redis restricts scripting, ensure
that category — or the three individual commands — is allowed for the
token-throttle connection user. No `KEYS`, `FLUSHDB`, `FLUSHALL`, `CONFIG`, or
Pub/Sub commands are issued by the library.

**`SCRIPT FLUSH` operational hazard**: avoid running `SCRIPT FLUSH` on a Redis
instance shared with token-throttle. It evicts the cached Lua SHA for redis-py's
lock release/extend scripts; the next lock operation reloads the script via
`SCRIPT LOAD`, which adds a round-trip and, if that command is also blocked by an
ACL, permanently breaks lock release until the process restarts. Schedule
`SCRIPT FLUSH` only during planned maintenance windows when token-throttle is not
running, or use a dedicated Redis DB that is not shared with other services that
flush the script cache.

Compatibility testing used `fakeredis` for unit tests plus local standalone
Redis (7.x for the test matrix, 8.4.0 for the benchmarks below). Redis 6.2 is
the floor for the commands token-throttle issues, and 6.0/6.1 are outside the
range — but treat 6.2 through 7.1 as best-effort rather than covered: neither
this project's test matrix nor redis-py's own supported-version table reaches
below 7.2. If you run a server older than 7.2, validate it in your own
environment. The test matrix did not cover
Sentinel failover behavior, KeyDB, Dragonfly, client-side sharding, or low
`maxmemory` / low `maxclients` configurations. KeyDB and Dragonfly may work as
Redis-compatible servers, but they are untested and not officially supported;
validate topology and resource limits in your environment.

## Multi-tenant deployments

`key_prefix` provides namespace isolation only. It keeps one tenant's Redis
keys from colliding with another tenant's keys, but it is not resource
isolation. Tenants that share a Redis server still share Redis CPU, memory,
`maxclients`, command scheduling, Lua script execution time, network bandwidth,
and eviction policy.

Do not rely on `key_prefix` for hostile-tenant fairness. A hostile or runaway
tenant on shared Redis can starve benign tenants by exhausting connections,
memory, CPU, or Lua scheduling. For hostile-tenant scenarios, use separate
Redis instances per tenant, or place Redis behind infrastructure that enforces
hardware-level CPU, memory, connection, and network quotas per tenant.

## Connection pooling and key TTLs

For bounded Redis deployments, prefer `redis.asyncio.BlockingConnectionPool`
or `redis.BlockingConnectionPool` and size `max_connections` to at least
`max_concurrent_acquires` plus headroom for Redis lock acquire/release, `TIME`,
and pipeline commands. A pool below 10 connections triggers a runtime warning
because it is usually too small for production traffic.

**Size the pool explicitly on redis-py 8 and newer.** redis-py 8 lowered the
default `ConnectionPool` cap from effectively unbounded to 100 connections, and
raises `MaxConnectionsError` once they are all checked out. A client built the
short way — `redis.from_url(...)`, as the README quickstarts show — silently
inherits that cap, while token-throttle's own default ceiling on in-flight
reservations is 100,000 and it imposes no Redis-side concurrency limit of its
own. Below-10 pools warn; a 100-connection default does not, because it is not
obviously wrong. If your workload can have more than 100 acquires in flight at
once, pass `max_connections` yourself rather than inheriting the default.

Redis bucket state expires by default after 7 days of inactivity. Configure
`bucket_ttl_seconds` on Redis builders or Redis OpenAI factories to choose a
different positive TTL. The TTL is refreshed whenever bucket state is read or
written. `bucket_ttl_seconds` must be at least as long as your longest
configured quota window (`per_seconds`); backend build time rejects
configurations where it is shorter, since an idle gap longer than the TTL
would silently reset a quota that has not actually refilled.

Redis refunds also write a cross-process idempotency key:
`{key_prefix}:rate_limiting:refund_dedup:{reservation_id}`. The TTL defaults to
7 days and can be changed with `refund_dedup_ttl_seconds` on Redis backend
builders or Redis OpenAI factories. Memory backends keep only process-local
refund dedup state and cannot safely refund reservations after a cold restart.

## Shared-storage locking and contention

Redis serializes each bucket with a distributed lock. SQLite instead serializes
transactions through the database-wide writer lock and uses
`busy_timeout_ms` as the analogue of Redis's
`lock_blocking_timeout_seconds`. Both backends keep multi-bucket writes atomic;
their tuning and throughput ceilings differ.

The Redis backend serializes every mutation of a given bucket through a
short-lived per-bucket lock, so concurrent workers never race on the same
capacity counter. Two knobs on the Redis backend builders (and the Redis OpenAI
factories) tune that lock:

- `lock_blocking_timeout_seconds` (default `5.0`): how long a single attempt to
  acquire a bucket lock will poll before giving up. This bounds one acquire
  attempt; it does not bound how long `await_for_capacity` /
  `wait_for_capacity` will wait overall.
- `lock_sleep_seconds` (default `0.05`): the poll interval while waiting for a
  contended lock.

Because the lock is poll-based, it is not strictly fair: under heavy contention
(many workers on one hot bucket) an individual waiter can be repeatedly outraced
and fail to acquire within `lock_blocking_timeout_seconds`. Behavior under that
contention depends on the call:

- `await_for_capacity` / `wait_for_capacity` **with no `timeout`** treat lock
  contention as part of waiting: they keep retrying the acquire indefinitely and
  never raise because of lock starvation. Contention is reported through a
  throttled warning on the `token_throttle.lock` logger (logged once, then
  suppressed for a cooldown) so a hot bucket is still visible in logs.
- `await_for_capacity` / `wait_for_capacity` **with a `timeout`** convert lock
  starvation into the same `TimeoutError` you already handle for "no capacity in
  time" — the caller-supplied deadline is the single source of truth.
- `consume_capacity`, `refund_capacity` / `refund_capacity_for_buckets`,
  `set_max_capacity`, and reconfiguration have no internal wait loop. If they
  cannot acquire the lock within `lock_blocking_timeout_seconds`, or if the lock
  is lost mid-operation (it expired or was stolen by another worker, in which
  case the write is aborted and makes no change), they raise
  `BackendLockContentionError`.

### Backend contention retry contract

`BackendLockContentionError` is exported from the top-level `token_throttle`
package and is shared by the Redis and SQLite backends. When you see it, the
operation did not modify state and is safe to retry.

For Redis, repeated errors mean a bucket is genuinely hot or its lock deadline
is too short. Reduce concurrency on that bucket, spread traffic across more
model families/windows, provision Redis CPU headroom, or raise
`lock_blocking_timeout_seconds`. SQLite raises the same exception when a
non-waiting write cannot obtain the database writer lock within
`busy_timeout_ms`; see [Write contention and try-acquire behavior](sqlite-backend.md#write-contention-and-try-acquire-behavior).
Memory backends have no shared-storage lock and do not raise this exception.

## Application-facing errors

Beyond `BackendLockContentionError` (backend write/lock contention, covered
above) and
`TimeoutError` (capacity-wait deadline exceeded, see the
[README Timeout section](../README.md#timeout)), these are the
token-throttle-specific exceptions an application should expect at the
acquire/refund call site. This is not an exhaustive list of everything a call
can raise — raw backend errors can surface too (see
[Raw Redis connection errors](#raw-redis-connection-errors) below).

### `DuplicateRefundError`

Despite the name, this is raised in two different places:

- From `refund_capacity()` / `refund_capacity_from_response()`, with
  `.reason` `"already_refunded"` or `"in_progress"`, when the same
  reservation is refunded a second time, or refunded concurrently by two
  callers. The refund-dedup tombstone (or in-progress marker) already
  prevented a second credit, so **no additional capacity state changed** —
  the legitimate (or racing) refund already happened. Retrying the same
  refund call keeps raising; it is not a transient error. Treat it as a
  signal that your call site refunds the same reservation more than once
  (for example both a `finally` block and an earlier explicit refund) and
  fix the call site rather than retry.
- From `acquire_capacity()` / `record_usage()`, with `.reason`
  `"duplicate_acquire"`, only if a `reservation_id` is reused across two
  acquire attempts with different usage or buckets. token-throttle generates
  a fresh id per reservation, so this should not happen from normal use of
  the public API; it indicates a reservation object was manually reused or
  constructed by hand.

`.reservation_id` and `.model_family` identify the reservation for all three
reasons.

### `UnknownReservationError`

Raised when a refund reaches the backend but it has no record that the
reservation was acquired, or earlier when the reservation belongs to a
different limiter instance. This includes lost backend state and forged or
deserialized reservations. A reservation older than
`max_reservation_lifetime_seconds` instead raises `ValueError` before backend
lookup. Both failures fail closed: **capacity is not credited**. Retrying the
identical refund keeps failing — there is no transient condition to wait out.
Refund through the same limiter instance, before its lifetime/TTL window
elapses, and do not serialize or queue reservations across processes.

### `AcquireRefundFailedError`

Raised only from `acquire_capacity()` / `acquire_capacity_for_request()` (and
their sync equivalents), never from `refund_capacity()`. It means acquire
delivery was interrupted after capacity was already reserved (for example by
cancellation), and token-throttle's own best-effort cleanup refund then also
failed. Unlike the errors above, **state did change**: real capacity is
reserved and outstanding, and it was not automatically returned. Recovery
uses three attributes:

- `.reservation` — the delivered `CapacityReservation`. Refund it yourself
  (`await limiter.refund_capacity(actual_usage, exc.reservation)`) once your
  cleanup path can reach the backend again.
- `.interrupted_by` — the exception that interrupted acquire delivery (for
  example `asyncio.CancelledError` or a caller timeout), for diagnosing *why*
  delivery was interrupted.
- `.refund_error` — the exception raised by the automatic cleanup refund
  attempt itself (for example a `BackendLockContentionError` or a backend
  connection error), for diagnosing *why cleanup failed*.

Manually refunding `.reservation` is the correct recovery action. If you do
nothing, the reservation stays outstanding until it is refunded or its
`max_reservation_lifetime_seconds` window elapses (see
[If a worker crashes before refunding](#if-a-worker-crashes-before-refunding)
above for what "outstanding" means for bucket capacity).

### `CardinalityLimitExceededError`

Raised when a mandatory in-process cap is exceeded — model families, metrics
per family, model aliases, or in-flight reservations, see
[docs/configuration.md](configuration.md#per-model-configuration) — or when a
DTO length cap is exceeded while constructing `Quota`, `CapacityReservation`,
or `PerModelConfig` directly (surfaced there as a pydantic `ValidationError`
instead). Every one of these checks runs before any backend capacity is
consumed, so
**no capacity state changed**. Retrying the identical call keeps failing; it
is a structural limit, not contention. Raise the relevant constructor cap,
reduce the cardinality your application generates (fix a `model_family` or
alias typo, cap distinct metric names), or reduce in-flight concurrency.

### Raw Redis connection errors

token-throttle does not wrap or retry genuine Redis connectivity failures.
`acquire_capacity()`, `refund_capacity()`, `consume_capacity()`, and
`set_max_capacity()` (and their sync equivalents) can raise
`redis.exceptions.ConnectionError`, `redis.exceptions.TimeoutError`, or the base
`redis.exceptions.RedisError` directly during a Redis outage; these are not
translated into token-throttle exceptions. Callers building retry or
circuit-breaker logic should catch `redis.exceptions.RedisError` alongside the
token-throttle exceptions above and choose their own fail-open or fail-closed
policy for the outage.

## Performance and capacity planning

Performance testing identified two important ceilings for 10k RPS-class
deployments:

- A single hot Redis bucket is the throughput ceiling well before 10k RPS,
  because all callers serialize on the same Redis lock and Lua-scripted state.
  In a single-machine benchmark — 1,000 workers contending on one bucket,
  target 10k RPS, local Redis 8.4.0 — throughput collapsed to ~180 ops/s with
  p99 acquire latency near 4.8s and frequent lock timeouts at the 5s lock-wait
  boundary.
- The async memory backend is process-local and also falls short of 10k RPS on
  a single process: the same benchmark topped out around 4.4k-5.3k ops/s, and
  raising workers from 100 to 1,000 only inflated tail latency (p99 acquire
  ~16 ms to ~290 ms) without adding throughput. It is not a horizontal scaling
  substitute for Redis.

Exact throughput and p99 latency depend on Redis CPU, network RTT, Python
runtime, concurrency, quota shape, and how concentrated traffic is on one
model family. Treat 10k RPS as a workload that requires staging benchmarks with
your real quota mix. These numbers are based on short local runs and sizing
estimates, not a maintained sustained-load production benchmark suite; use them
as planning signals, not guarantees. As a starting planning table:

| Sustained acquire/refund rate | Expected p99 driver | Operational guidance |
| ---: | --- | --- |
| 100 RPS | Redis RTT plus Python scheduling | Default Redis client pools usually work; still set timeouts and monitor waits. |
| 1k RPS | Lock contention, Redis CPU, pool wait | Use `BlockingConnectionPool`, provision Redis CPU headroom, and benchmark p99 under peak concurrency. |
| 10k RPS | Hot buckets, Lua scheduling, key churn | Avoid concentrating all traffic in one model family/window; use dedicated Redis capacity and load-test before production. |

Redis backends create per-reservation keys in addition to bucket
state. Size Redis from traffic rate and TTLs, not only from the number of model
families:

- Acquire marker keys, rough count:
  `acquires_per_second * max_reservation_lifetime_seconds`.
- Refund dedup keys, rough count:
  `refunds_per_second * refund_dedup_ttl_seconds`.

The Redis default TTLs are intentionally conservative for correctness and
compatibility: `bucket_ttl_seconds=604800` and
`refund_dedup_ttl_seconds=604800` (7 days). With no explicit
`max_reservation_lifetime_seconds`, Redis derives the reservation lifetime from
the shorter Redis TTL: just below
`min(bucket_ttl_seconds, refund_dedup_ttl_seconds) / 2`. At 10k acquires/sec,
the default derived marker lifetime is about 302,400 seconds, which can imply
roughly 3.0 billion marker keys. Tune these values for sustained high-RPS
deployments.

A practical starting point is to choose the smallest reservation lifetime that
covers normal request latency, retry delay, and shutdown drain time, then choose
`bucket_ttl_seconds` and `refund_dedup_ttl_seconds` longer than twice that
lifetime. Redis enforces this invariant:
`bucket_ttl_seconds > max_reservation_lifetime_seconds * 2` and
`refund_dedup_ttl_seconds > max_reservation_lifetime_seconds * 2`. The margin
keeps bucket state, acquire markers, and refund tombstones alive for the full
window in which a reservation can still be refunded.

Example budgets for a tuned deployment, assuming about 0.5-1.0 KB per marker or
dedup key after Redis object overhead and leaving operational headroom:

| Traffic | Example knobs | Approx keys | Suggested Redis memory budget |
| --- | --- | ---: | ---: |
| 1k acquire/refund RPS | `max_reservation_lifetime_seconds=300`, `refund_dedup_ttl_seconds=900`, `bucket_ttl_seconds=900` | 300k markers + 900k dedup keys | 1-2 GB |
| 10k acquire/refund RPS | `max_reservation_lifetime_seconds=300`, `refund_dedup_ttl_seconds=900`, `bucket_ttl_seconds=900` | 3M markers + 9M dedup keys | 10-20 GB |

Validate with `INFO memory`, `DBSIZE` or keyspace scans in staging because Redis
memory per key depends on key-prefix length, allocator behavior, and value size.

Sample Redis monitoring points:

```text
INFO commandstats   # eval/evalsha, set, get, del latency and call volume
INFO clients        # connected_clients, blocked_clients, maxclients pressure
INFO memory         # used_memory, mem_fragmentation_ratio, evicted_keys
INFO stats          # instantaneous_ops_per_sec, rejected_connections
LATENCY LATEST      # server-side latency spikes
SLOWLOG GET 128     # slow Lua scripts or lock commands
```

Application-side monitoring should track acquire wait duration, timeout count,
callback errors, `snapshot_state()["in_flight_reservations"]`, Redis pool wait
time, and p50/p95/p99 latency for acquire and refund calls.
