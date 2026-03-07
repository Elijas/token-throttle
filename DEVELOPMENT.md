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

### `loguru` is a dev-only dependency

`loguru` is used in two places (`_models.py` top-level import, `_callbacks.py` lazy import) but is listed under `[dependency-groups] dev`, not under `[project] dependencies`. This means:

- The callback logging factory (`create_loguru_callbacks`) is intentionally optional — it's a convenience for users who already use loguru.
- The `_models.py` import is just for a `logger.warning()` on empty quota lists. This will fail at import time if loguru is not installed.

**Status:** Known issue. Fixing properly requires either adding loguru to runtime deps (adds a dependency for one warning call) or making the `_models.py` import conditional.

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
