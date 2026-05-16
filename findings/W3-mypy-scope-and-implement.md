# W3 mypy scope and implementation

## Survey

Commands:

```bash
uv sync --all-extras --group dev
uv run mypy token_throttle/ 2>&1 | tee /tmp/r8-mypy-full.txt
uv run mypy --follow-imports=normal token_throttle/ 2>&1 | tee /tmp/r8-mypy-full-normal.txt
```

Baseline with the existing config (`follow_imports = "skip"`): 94 errors in 12
files, 95 output lines.

Normal-import baseline: 113 errors in 12 files, checked 32 source files.

Normal-import error categories:

| Error code | Count |
| --- | ---: |
| `assignment` | 43 |
| `arg-type` | 42 |
| `union-attr` | 9 |
| `misc` | 6 |
| `return-value` | 5 |
| `var-annotated` | 2 |
| `index` | 2 |
| `attr-defined` | 2 |
| `return` | 1 |
| `override` | 1 |

Largest file clusters were Redis implementation typing (47 errors across async
and sync backends), callback log-level narrowing (11), validation/model
normalization (15), limiter refund kwargs and usage narrowing (14), Redis bucket
helpers (14), memory backend `float | None` narrowing (8), and strict DTO/Pydantic
override typing (4).

## Decision

Option 1: strict everywhere.

Rationale: the normal-import error count was above the `<50` automatic threshold
but still concentrated in repeated, mechanical patterns. The Redis-heavy errors
were mostly local narrowing and redis-py stub precision, not design issues or API
changes. No v5/API-breaking change was needed.

## Changes

- `[tool.mypy]` now targets `token_throttle/` and no longer skips followed imports.
- Added `task lint-types` as a local `uv run mypy` shortcut.
- Added a `type-check` CI job that installs all extras and runs `uv run mypy`.
- Documented the package-wide type gate in `DEVELOPMENT.md`.
- Added type narrowing/annotations across DTOs, models, validation-facing helpers,
  limiter refund paths, memory backends, Redis bucket helpers, and Redis backends.
- Fixed two internal type-signature bugs: `SyncRedisBackendBuilder.__init__`
  now has a valid `-> None` return annotation, and
  `SyncRedisBackend.consume_capacity()` now matches the backend interface return
  type (`float | None`) instead of claiming `None`.

## Final mypy output

After implementation, both the explicit normal-import check and the configured
gate are clean:

```text
Success: no issues found in 32 source files
```

Before/after:

- Before: 113 errors in 12 files with `--follow-imports=normal`.
- After: 0 errors in 32 checked source files.

## KNOWN UNKNOWNs

No KNOWN UNKNOWNs remain for W3.
