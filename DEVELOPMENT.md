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

### Redis integration tests are not parallel-safe

- Fixtures call `flushdb()` on teardown (`tests/integration/conftest.py`).
- Tests use fixed model families like `"test"`.
- This is fine for serial test runs (current CI), but will break under `pytest-xdist` or a shared Redis instance.

**If you adopt parallel test execution**, you'll need per-worker key prefixes or separate Redis DB numbers.

### Sync concurrency test leaks a daemon thread

`test_excess_acquires_must_wait` in `tests/integration/test_sync_concurrency.py` starts a daemon thread that is never joined. This is benign in isolation but can cause nondeterministic behavior in longer test runs.

### `per_seconds` is constrained to integers

`Quota.per_seconds` is typed as `int`. Pydantic v2 coerces whole floats (`60.0` -> `60`) but rejects fractional values (`0.5`). This is intentional — fractional rate-limiting windows don't make practical sense, and the Redis key format and capacity dict keys rely on integer values.

### `sleep_interval=0` is a valid configuration

Backend constructors accept `sleep_interval=0` for busy-wait polling. The default (`0.1s`) only applies when `sleep_interval` is `None` (not passed). This uses `is None` checks, not truthiness, so `0` is not treated as "use default."

### Negative capacity is preserved, not clamped to zero

The speedometer pattern (`record_usage` / `consume_capacity`) and `refund_capacity` both allow capacity to go negative. This is intentional — clamping to zero would erase debt and let the bucket refill from zero instead of recovering naturally.

Example: bucket at 50, actual usage 130 → capacity becomes −80. The token-bucket refill adds `rate_per_sec × elapsed` on each check, so the bucket gradually recovers to positive. If we clamped to 0, the 80-unit overuse would vanish.

The `allow_negative` flag on `set_capacity` / `_set_capacities_unsafe` controls this. The blocking path (`await_for_capacity` / `_check_and_consume_capacity`) uses `allow_negative=False` because it guarantees capacity ≥ usage before consuming.

### `set_max_capacity` applies the new refill rate retroactively

When `set_max_capacity` changes a bucket's limit, `_rate_per_sec` is recalculated immediately. The next `calculate_capacity` call uses the new rate for the *entire* elapsed time since the last check — not just the time since the rate changed.

If the last capacity check was 5 seconds ago and the rate doubles, the refill is `5 × new_rate` instead of `4 × old_rate + 1 × new_rate`. The error is bounded by `|rate_diff| × sleep_interval` (~0.1s typically), so it's negligible in practice. Tracking rate-change timestamps would add significant complexity for minimal benefit.

### `_log()` fails fast on unrecognized level strings

`_STDLIB_LEVEL_MAP` uses `[]` lookup (not `.get()`). Unknown levels raise
`KeyError` immediately. This is intentional — `_log()` is a private function
only called from the `create_*_callbacks()` factories, which only pass
standard level strings (DEBUG, INFO, WARNING, ERROR, CRITICAL) plus the
loguru extensions (TRACE, SUCCESS). Adding a `.get()` fallback would silently
mask typos in level names.

### `on_missing_consumption_data` callback is delayed until first successful acquire

When `_check_and_consume_capacity` returns `False` (insufficient capacity), it exits before calling `_fresh_start_buckets_callback`. The `on_missing_consumption_data` callback won't fire until the first *successful* capacity acquisition. This is by design — firing it on every 100ms poll iteration would be noisy. Since `last_checked` is never written on the insufficient-capacity path, the fresh-start condition persists and the callback fires exactly once when capacity is first successfully consumed.
