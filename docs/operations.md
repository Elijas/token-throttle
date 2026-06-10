# Operations guide

Running token-throttle in production across multiple workers and processes:
reservation durability, supported Redis topologies, multi-tenant isolation,
connection/TTL sizing, and capacity planning for high-RPS fleets.

Start with the [README](../README.md) for installation, the mental model, and
quickstarts. Reach for this guide once you are deploying on Redis and need to
reason about durability, scaling ceilings, and resource budgets.

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

v2.0.0 is a clean break from v1.4.x reservation compatibility. Every
`CapacityReservation` requires a non-empty `limiter_instance_id`; legacy
v1.4.x reservations without it are rejected. Drain in-flight reservations
before upgrading and do not run mixed v1.4.x/v2.0.0 fleets. See
[`MIGRATION.md`](../MIGRATION.md#migrating-from-v14x-to-v200) for the full
upgrade procedure.

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

Redis backends require Redis server 6.2 or newer and a Redis user that can run
`GET`, `EXISTS`, `SET`, `DEL`, `EXPIRE`, `PTTL`, `TIME`, and Lua scripting commands
used by redis-py locks and token-throttle acquire/refund transactions.

Compatibility testing used `fakeredis` for unit tests plus local standalone
Redis (7.x for the test matrix, 8.4.0 for the benchmarks below); 6.2 or newer is
supported and 6.0/6.1 are outside the range. The test matrix did not cover
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
If you are upgrading from a pre-v4 deployment, see [`MIGRATION.md`](../MIGRATION.md)
for the multi-tenant isolation change.

## Connection pooling and key TTLs

For bounded Redis deployments, prefer `redis.asyncio.BlockingConnectionPool`
or `redis.BlockingConnectionPool` and size `max_connections` to at least
`max_concurrent_acquires` plus headroom for Redis lock acquire/release, `TIME`,
and pipeline commands. A pool below 10 connections triggers a runtime warning
because it is usually too small for production traffic.

Redis bucket state expires by default after 7 days of inactivity. Configure
`bucket_ttl_seconds` on Redis builders or Redis OpenAI factories to choose a
different positive TTL. The TTL is refreshed whenever bucket state is read or
written; Redis schema-version registry keys are intentionally long-lived and do
not expire.

Redis refunds also write a cross-process idempotency key:
`{key_prefix}:rate_limiting:refund_dedup:{reservation_id}`. The TTL defaults to
7 days and can be changed with `refund_dedup_ttl_seconds` on Redis backend
builders or Redis OpenAI factories. Memory backends keep only process-local
refund dedup state and cannot safely refund reservations after a cold restart.

## Per-bucket locking and contention

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

`BackendLockContentionError` is exported from the top-level `token_throttle`
package. When you see it, the operation did not modify state and is safe to
retry. Seeing it repeatedly means a bucket is genuinely hot; the durable fixes
are to reduce concurrency on that bucket, spread traffic across more model
families/windows, provision Redis CPU headroom, or raise
`lock_blocking_timeout_seconds` so attempts wait longer before giving up. The
memory backends have no Redis lock and never raise `BackendLockContentionError`.

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
