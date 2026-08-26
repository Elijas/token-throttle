# SQLite backend guide

The SQLite backend shares one rate-limit budget between processes on a single
host and keeps bucket state across ordinary process restarts. It is a good fit
for CLI tools, cron jobs, local worker pools, and small services that need more
than process-local memory but do not want to operate Redis.

SQLite support needs no token-throttle install extra and no external service.
It uses Python's standard-library `sqlite3` module. Both `SqliteBackendBuilder`
and `SyncSqliteBackendBuilder` implement the same accounting contract as the
memory and Redis backends, including atomic multi-bucket acquisition, durable
acquire markers, and refund-deduplication tombstones.

```python
# (fragment — use this builder with the README async quickstart)
from token_throttle import SqliteBackendBuilder

backend = SqliteBackendBuilder(
    "/var/lib/my-service/rate-limits.sqlite3",
    key_prefix="my-service-prod",
)
```

Use `SyncSqliteBackendBuilder` with `SyncRateLimiter`. Close the limiter when it
shuts down so its SQLite connection and, for the async backend, worker thread
are released.

## Scope boundary

SQLite coordination is deliberately limited to one host and one local
filesystem:

- Every cooperating process must use the same stable database path and the
  same `key_prefix`. The builder resolves the path with `realpath`; prefer one
  absolute path rather than relying on different processes' working
  directories.
- Do not place the database on NFS, SMB, overlayfs, a distributed filesystem,
  or any other network/overlay-mounted filesystem. Those filesystems are
  unsupported for this backend and are the most likely way to break its WAL
  locking assumptions. token-throttle does not try to detect the mount type.
- Containers work only when they run on the same host, share the same local
  volume, and share the host's clock. A container-local path that is not
  mounted into every cooperating container creates separate budgets.
- SQLite does not coordinate multiple machines. Use the
  [Redis backend](operations.md#redis-topology-support) when callers on
  different hosts must share a budget.

The `key_prefix` isolates unrelated budgets inside one database file. It is a
namespace boundary, not a CPU or disk-I/O isolation boundary; use separate
database files when independent workloads should not contend on the same
SQLite writer lock.

## Clock authority

The host wall clock is authoritative. Every process on the host reads the same
kernel clock, so there is no separate server-clock protocol.

- If the clock moves backward, refill clamps elapsed time to zero. It emits
  process-wide throttled `RuntimeWarning` and structured warning notifications
  with the affected model family and metric. A stored timestamp more than one
  second in the future is repaired in the same transaction while preserving
  stored capacity, so it cannot pin a drained persistent bucket indefinitely.
- If the clock moves forward, the bucket refills for the apparent elapsed time,
  capped at its maximum capacity. There is no forward-jump detection rail.
- Time spent suspended counts as elapsed wall time, so a laptop or VM may
  resume with refilled buckets.

This contract does not promise agreement between different hosts, time
namespaces, or a database opened from machines with different clocks. Keep NTP
or the host's equivalent time synchronization healthy.

## Persistence and path identity

Bucket capacity, refill timestamps, active runtime overrides, acquire markers,
and refund tombstones live in the database. SQLite uses WAL mode and atomic
transactions, so incomplete process transactions are rolled back and committed
state is available when a new process opens the same path and prefix.

Static `PerModelConfig` quota definitions do not live in SQLite. Every process
must deploy compatible configuration for each shared `model_family`; the
database is coordination state, not configuration distribution. A new process
combines its own configured quotas with the persisted capacity and any active
runtime override.

`CapacityReservation` remains trusted in-process state, not a portable durable
credential. Do not serialize a reservation and present it to a newly created
limiter after a restart. Persistence protects shared accounting and duplicate
refund handling between live cooperating processes; it does not change the
[reservation trust boundary](operations.md#reservation-lifecycle-and-durability).

The connection uses `synchronous=NORMAL`. SQLite preserves database consistency
after a process or operating-system crash, but the most recently committed
transactions can be lost after abrupt power or storage failure. If losing the
latest accounting update is unacceptable, use infrastructure and a backend
whose durability settings match that requirement.

## TTL and reservation-lifetime knobs

SQLite prunes expired rows lazily during write operations, in bounded batches
controlled by `prune_batch_size` (default `256`). There is no cleanup daemon, so
an expired row can remain physically present until later activity. Expiry is
still enforced when a specific marker or tombstone is addressed.

| Builder option | Default | Contract |
| --- | ---: | --- |
| `bucket_ttl_seconds` | 604800 (7 days) | Inactivity lifetime for bucket rows. It must be at least the longest configured quota window. After expiry the next use is a fresh bucket; by the validated window bound it has already had enough time to refill fully. |
| `refund_dedup_ttl_seconds` | 604800 (7 days) | How long a completed refund remains recognizable as a duplicate. |
| `max_reservation_lifetime_seconds` | Derived | Maximum age at which an acquire marker remains refundable. When omitted, it is just below half of the shorter bucket/refund TTL. |
| `override_ttl_seconds` | `bucket_ttl_seconds` | Fixed lifetime of a shared `set_max_capacity()` override, measured from the call that writes it. Ordinary bucket activity does not extend it. |
| `busy_timeout_ms` | 5000 | Maximum SQLite writer-lock wait for ordinary operations. A finite acquire deadline can reduce it, and a try-acquire uses zero. |
| `prune_batch_size` | 256 | Maximum expired rows pruned from each durable table by one cleanup pass. |

All three SQLite TTL options require a positive, finite integer number of
seconds (at most `2**31 - 1`). `None` is not a supported "never expire" value,
including for bucket or refund-dedup state. This release intentionally keeps
all durable bookkeeping bounded; choose a sufficiently long finite TTL when
you need a long idle or refund window.

The builder enforces both safety inequalities:

```text
bucket_ttl_seconds > max_reservation_lifetime_seconds * 2
refund_dedup_ttl_seconds > max_reservation_lifetime_seconds * 2
```

Choose the reservation lifetime from the longest real request latency, retry
delay, and shutdown-drain period you intend to support. Then make both durable
TTLs more than twice that lifetime. Shorter values reduce persistent marker and
tombstone rows but also shorten the valid late-refund window.

## Runtime overrides and configuration

`set_max_capacity()` writes an expiring override into SQLite after anchoring
capacity accrued at the previous rate. Other processes using the same database
and prefix observe it on their next backend operation; a process already
waiting for capacity polls shared state at least about once per second.

The override is a shared control layer, while static configuration remains
process-local. When `override_ttl_seconds` elapses, each process returns to its
own configured quota on its next operation. Calling `set_max_capacity()` again
replaces the value and starts a new TTL. Keep static configuration consistent
across the fleet so override expiry cannot reveal conflicting base limits.

A static quota change detected by a process clears the shared override for that
bucket and applies the new configured limit locally. Removing a metric and
later re-adding it also drops its prior runtime override. Coordinate config
rollouts with override writers, and reapply an override after the rollout when
that is the intended final state. The complete limiter-level contract is in
[Dynamic rate limits](configuration.md#dynamic-rate-limits).

## Write contention and try-acquire behavior

SQLite permits one writer at a time. `busy_timeout_ms` (default `5000`) lets a
normal write wait through short lock contention. Non-waiting operations such as
refund, direct consumption, runtime-limit changes, and reconfiguration raise
`BackendLockContentionError` if the writer remains locked past that timeout. A
failed transaction makes no state change, so the operation is safe to retry.

`acquire_capacity(..., timeout=0)` is a real try-acquire: its write-lock wait is
set to zero for that attempt, and either insufficient capacity or a busy writer
produces `TimeoutError` promptly. A finite positive acquire timeout bounds both
capacity waiting and SQLite write-lock waiting. With no acquire timeout,
temporary lock contention is retried as part of the ordinary wait loop.

Blocked capacity waiters poll because SQLite has no cross-process notification
channel. Poll intervals follow the refill estimate, are capped at one second,
and include jitter to reduce synchronized wakeups. Scheduling is not FIFO.

## Worker crashes and late refunds

Killing a worker does not return its reservation in a burst. The consumed
capacity recovers only through normal linear refill, exactly as if the request
had used the full reservation. The orphaned acquire marker remains until its
reservation lifetime expires and lazy pruning removes it.

Once a reservation exceeds `max_reservation_lifetime_seconds`, the public refund
methods fail closed with `ValueError` before backend I/O. If the SQLite backend
is instead reached without a live acquire marker, it raises
`UnknownReservationError`. Neither path creates capacity. A successful refund
consumes the marker and writes a tombstone, so a duplicate refund fails without
double-crediting capacity. After the tombstone TTL expires, another attempt is
unknown rather than accepted.

## Performance envelope

SQLite is intended for CLI/cron scale and modest local worker pools: tens of
processes and roughly up to 100 acquire/refund operations per second per
database file. This is a planning envelope, not a throughput guarantee; disk,
filesystem, quota concentration, and transaction duration determine the real
limit.

If write-lock waits are common, tail latency matters, or traffic is growing
beyond this envelope, move the shared budget to Redis. Do not split one logical
budget across multiple SQLite files to gain throughput; that creates
independent limits.

## Observability

SQLite snapshot fields and exact namespace-scoped diagnostics are documented in
the [observability reference](observability.md#health-snapshot).

## Troubleshooting

### `database is locked` or repeated `BackendLockContentionError`

Confirm every process uses a writable local filesystem and that no external
tool is holding a long write transaction. Leave the database's `-wal` and
`-shm` sidecar files under SQLite's control while processes are running.
Increasing `busy_timeout_ms` can absorb short bursts, but repeated contention
usually means the workload needs less concurrency, separate unrelated database
files, or Redis. For `timeout=0` acquires, immediate `TimeoutError` under a
writer lock is expected and should be handled like unavailable capacity.

### Clock warnings

Check the host's current time, synchronization service, suspend/resume history,
and VM snapshot history. A backward-clock warning means refill was clamped
rather than calculated from negative elapsed time; a sufficiently future stored
timestamp is repaired without adding capacity. A forward clock jump cannot be
distinguished from real elapsed time and may refill up to the bucket maximum.
Do not delete the database as a clock repair: deleting it resets shared bucket
state to fresh capacity.

### Processes do not appear to share a budget

Log and compare each builder's resolved `db_path` and `key_prefix`. Relative
paths from different working directories, container-local filesystems, and
different prefixes all create independent state. Move every process to one
stable absolute path on a shared local volume; do not solve the mismatch with a
network filesystem.
