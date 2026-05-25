# W3 Memory Bucket Validation Fix Report

## Summary

- Closed P3-API-004 by validating direct `MemoryBucket` `metric` and `model_family` inputs with the same key-segment validator family used by `Quota` / `PerModelConfig`.
- Added a `validate_model_family()` wrapper in `token_throttle/_validation.py`; reused existing `validate_metric()`.
- Added `MemoryBucket.bucket_id` and preserved the existing internal ID shape for valid inputs.
- Added `logging.WARNING` records on the `token_throttle` logger alongside existing `RuntimeWarning`s for memory overuse, negative refund, and backward-clock anomalies. Logs include `metric`, `model_family`, `value`, and `bucket_id`.
- No Redis backend files or limiter core files were touched.

## Sync Bucket Note

The prompt references a `_sync_bucket.py` mirror, but this worktree has no sync bucket file. `SyncMemoryBackend` uses the shared `MemoryBucket`, so the constructor validation covers both async and sync memory paths.

## Tests Added

- `tests/unit/test_memory_bucket.py`
  - invalid direct `MemoryBucket` key segments now raise `ValueError`;
  - valid direct construction still preserves bucket ID shape;
  - memory bucket backward-clock `set_max_capacity()` warnings now log context.
- `tests/unit/test_memory_backend.py`
  - async memory overuse and negative-refund warnings now also log context.
- `tests/unit/test_sync_memory_backend.py`
  - sync memory overuse and negative-refund warnings now also log context.
- `tests/unit/test_capacity.py`
  - shared backward-clock warning path now also logs context.

## Gates

- PASS: `uv run pytest tests/unit -q`  
  `1678 passed, 5 skipped`
- FAIL, pre-existing W4-owned issue: `uv run pytest tests/unit tests/conformance -q` under CPython 3.14.3  
  `1763 passed, 5 skipped, 1 failed` at `tests/conformance/test_run_step_wrappers.py::test_external_cancellation_cancels_and_consumes_backend_task`. This matches `SURFACE.md` P7-003 and is outside this worker's ownership.
- PASS: `uv run mypy token_throttle/`
- PASS: `uv run ruff check token_throttle/ tests/`
- PASS: `uv run ruff format --check token_throttle/ tests/`

Note: I ran `uv sync --all-extras --dev` before the exact mypy gate so optional `redis` and `tiktoken` imports were available in the fresh local `.venv`.
