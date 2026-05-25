# W5-deps fix report

## Scope closed

- P7-004: bumped `pydantic` floor to `>=2.12.0,<3`.
- P7-005: bumped `tiktoken` floor to `>=0.10.0,<1`.
- PD-4: removed the `loguru` optional extra and dev dependency from `pyproject.toml`; `uv.lock` now drops `loguru` and `win32-setctime`.
- P7-001: removed Redis imports from `tests/integration/conftest.py` module load and moved them behind fixture-local `pytest.importorskip(...)` calls.
- P7-001/P7-002: added module-level Redis skips before Redis-only integration/property module imports.
- P7-002: extended `tests/conformance/test_redis_import_portability.py` to scan `tests/unit`, `tests/integration`, and `tests/property`, including `conftest.py`.

## Regression coverage

- Added `tests/unit/test_strict_dto_floor.py` to assert `StrictDTO.model_validate(..., extra="allow")` works. This is the pydantic floor regression.
- The expanded Redis import conformance guard is the structural regression test for no-Redis collection portability across unit/integration/property tests.

## Verification

- `uv lock`
- `uv lock --check`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run pytest tests/conformance/test_redis_import_portability.py tests/unit/test_strict_dto_floor.py -q`
- No-Redis simulation: `pytest tests/integration --collect-only -qq --tb=short -p no:cacheprovider` with a `sys.meta_path` Redis blocker exited 0.
- No-Redis simulation: `pytest tests/property --collect-only -q --tb=short -p no:cacheprovider` with the same Redis blocker exited 0.
- Lowest-direct verification on Python 3.12: `UV_PROJECT_ENVIRONMENT=/tmp/tt-w5-deps-lowest312 uv run --python 3.12 --resolution lowest-direct --all-extras --group dev pytest tests/unit tests/conformance --collect-only -qq` exited 0.
- Full unit/conformance on Python 3.12: `UV_PROJECT_ENVIRONMENT=/tmp/tt-w5-deps-py312 uv run --python 3.12 --all-extras --group dev pytest tests/unit tests/conformance -q` -> `1713 passed, 45 skipped, 28 warnings`.

## Notes

- A lowest-direct run under the checkout's default Python 3.14 tried to build `tiktoken==0.10.0` from source and failed because its bundled PyO3 supports up to Python 3.13. That lines up with P7-003 being a separate Python 3.14 support issue, so the dependency-floor verification above was rerun under Python 3.12.
