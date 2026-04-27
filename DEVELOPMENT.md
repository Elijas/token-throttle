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

## CI structure

CI runs five jobs (see `.github/workflows/ci.yml`):

| Job | What it does | Extras installed |
|-----|-------------|-----------------|
| `lint` | ruff check + format | dev only |
| `test-unit-core` | Unit tests without optional deps | dev only |
| `test-unit-full` | Unit tests with all optional deps | all extras + dev |
| `test-integration` | Integration tests against Redis | all extras + dev |
| `coverage` | Full suite + Codecov upload | all extras + dev |

Matrix: Python 3.12 and 3.13. Redis 7 (alpine) as a GitHub service container.

## Known constraints and assumptions

### `loguru` is optional — stdlib logging is the default

`loguru` is listed under `[project.optional-dependencies]` (`token-throttle[loguru]`), not in runtime deps. The logging layer auto-detects it:

- `_callbacks._log()` uses loguru if installed, otherwise stdlib `logging.getLogger("token_throttle")`.
- `create_logging_callbacks` / `create_sync_logging_callbacks` use this auto-detection (default for new code).
- `create_loguru_callbacks` / `create_sync_loguru_callbacks` require loguru explicitly and raise `ImportError` if missing.
- `_models.py` uses `warnings.warn()` for the empty-quota warning — no loguru dependency.
- `_probe_loguru()` caches its result on first call. If loguru is not installed at import time, the `None` result is cached for the process lifetime. Installing loguru after the first probe will not be detected — restart the process to pick it up.

### Redis integration tests are not parallel-safe

- Fixtures call `flushdb()` on teardown (`tests/integration/conftest.py`).
- Tests use fixed model families like `"test"`.
- This is fine for serial test runs (current CI), but will break under `pytest-xdist` or a shared Redis instance.

**If you adopt parallel test execution**, you'll need per-worker key prefixes or separate Redis DB numbers.

### Sync concurrency test thread is joined

`test_excess_acquires_must_wait` in `tests/integration/test_sync_concurrency.py` starts a background thread but joins it with a 2-second timeout and asserts it exited cleanly. No daemon thread leak.

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
loguru extensions (TRACE, SUCCESS). Adding a `.get()` fallback would silently
mask typos in level names.

### Capacity matching loop and the postconsumption invariant

`_check_and_consume_capacity` and `consume_capacity` build a `postconsumption_dict` via a nested loop that matches capacity entries against usage entries by metric name. This O(n×m) pattern appears in all four backends (async/sync × memory/Redis).

**Invariant:** `postconsumption_dict` must cover ALL entries in `preconsumption_capacities`. If any bucket is missed, `_set_capacities` won't update its timestamp, causing incorrect refill calculations on the next read. This is guaranteed by `validate_acquire_usage()` (called in `_acquire_capacity`) which enforces `set(usage.keys()) == set(quotas.names)`. Each backend asserts this post-hoc with `assert len(postconsumption_dict) == len(preconsumption_capacities)`.

The O(n×m) cost is irrelevant in practice — n (buckets) and m (usage metrics) are typically 2–5.

### `_PIPELINE_CMDS_PER_BUCKET` in Redis backends

Redis `_get_capacities_unsafe` batches pipeline commands in a fixed layout: each bucket enqueues exactly 2 GETs (`last_checked`, `capacity`), followed by 1 `max_capacity` GET per bucket. The constant `_PIPELINE_CMDS_PER_BUCKET = 2` and an assertion on result count enforce this layout. If `get_capacity()` or the pipeline structure changes, the assertion catches the mismatch immediately rather than silently reading wrong values.

### Over-limit validation lives in the backend, not `validate_acquire_usage`

`validate_acquire_usage` checks key-match, finiteness, and non-negativity — but does **not** check `usage > quota.limit`. That check was removed intentionally because `set_max_capacity` can change a bucket's limit at runtime, making the static `quota.limit` stale.

Instead, each backend performs the over-limit check inside its lock against the live `bucket.max_capacity`. This means over-limit requests acquire the lock (and, for Redis, a pipeline round-trip) before failing. That cost is acceptable because over-limit requests are programming errors, not normal traffic, and checking outside the lock would require reading a potentially-stale cached value then re-checking under the lock anyway.

### `on_missing_consumption_data` callback is delayed until first successful acquire

When `_check_and_consume_capacity` returns `False` (insufficient capacity), it exits before calling `_fresh_start_buckets_callback`. The `on_missing_consumption_data` callback won't fire until the first *successful* capacity acquisition. This is by design — firing it on every 100ms poll iteration would be noisy. Since `last_checked` is never written on the insufficient-capacity path, the fresh-start condition persists and the callback fires exactly once when capacity is first successfully consumed.

### Redis `consume_capacity` callbacks are best-effort after cancellation

In the async Redis backend, `consume_capacity` shields the Redis write so that a task cancellation cannot leave the write half-applied. After `suppress_current_task_cancellation()` returns, callbacks (`on_capacity_consumed`, `on_missing_consumption_data`) still fire. If a *second* cancellation arrives during those callbacks, they are skipped. This is acceptable: the speedometer consumption is already durably recorded in Redis, and callbacks are best-effort via `_invoke_callback_safe`.

### `SyncRateLimiter` has no abstract base class

`RateLimiter` extends `BaseRateLimiter` (ABC); `SyncRateLimiter` does not extend any abstract base. Adding a `BaseSyncRateLimiter` would be a public API change. The sync interface is documented by its method signatures and mirrors the async API.

### Model-family caches grow without eviction

`RateLimiter` and `SyncRateLimiter` maintain six per-model-family dicts (`_model_family_to_backend`, `_model_family_to_model_name`, etc.) that grow with each distinct model name seen. For bounded deployments (a handful of model families), this is fine. For applications that generate unbounded unique model names (e.g. per-user model aliases), consider using a single rate limiter per model family or periodically creating fresh limiter instances.

### Redis `max_capacity_override` self-heals on config mismatch

When `_deserialize_max_capacity_override` reads a stored override from Redis, it compares the `configured_max_capacity` field in the JSON payload against the current process's `_max_capacity_default` (from `Quota.limit`). If they differ — e.g. after a deployment changes the static quota — the override is silently discarded (returns `None`), causing the bucket to fall back to the new static limit.

This is intentional self-healing: an override created under a previous quota configuration should not pin the new deployment to a stale limit. The override was set relative to the old config; applying it under a different config would produce an unexpected effective limit. Discarding it lets the new static config take effect cleanly, and operators can re-apply an override if needed.

### Redis connection pool sizing

The Redis backend accepts a user-provided `redis.asyncio.Redis` (or `redis.Redis`) client and uses its connection pool as-is. By default, `redis-py` creates a pool with `max_connections=2**31` (effectively unlimited). In high-fanout applications (many concurrent `await_for_capacity` calls), the actual connection count may spike because each pipeline/lock acquisition can consume a connection.

If you need to bound connection usage, configure `max_connections` on the Redis client before passing it to the rate limiter:

```python
import redis.asyncio as aioredis

pool = aioredis.ConnectionPool.from_url("redis://localhost", max_connections=50)
client = aioredis.Redis(connection_pool=pool)
limiter = RateLimiter(client=client, ...)
```

### Float precision at extreme capacity limits

All capacity values are Python `float` (IEEE 754 double). At limits above ~2^53, integer precision is lost: consecutive integers are indistinguishable, so `capacity - usage` may not change the stored value. This is a known limitation of the float64 representation and is acceptable for all real-world rate-limiting scenarios (token quotas are orders of magnitude below 2^53).
