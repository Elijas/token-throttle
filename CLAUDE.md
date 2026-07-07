# token-throttle

## Releasing a New Version

Releases are fully automated via GitHub Actions. Do NOT bump versions or publish locally.

### Trigger a release

```bash
gh workflow run release.yml --ref main -f bump=patch   # 0.5.0 -> 0.5.1
gh workflow run release.yml --ref main -f bump=minor   # 0.5.0 -> 0.6.0
gh workflow run release.yml --ref main -f bump=major   # 0.6.0 -> 1.0.0
```

### What the workflow does

1. Verifies manual dispatch is running from `refs/heads/main` with a non-empty `bump` input, then runs full CI test suite (`.github/workflows/ci.yml`)
2. `bump-my-version bump <patch|minor|major>` — updates version in `pyproject.toml` and `token_throttle/__init__.py`, creates a commit
3. `devtools/bump_readme_version.py` — updates pip install version bounds and badge in README.md
4. `uv lock` — syncs `uv.lock` with the new version
5. `ruff check --fix`, `ruff format`, then `ruff check` — applies safe lint autofixes, formats, and fails if issues remain
6. Creates a second commit with lockfile/formatting changes (if any)
7. Porcelain check — fails the release if the working tree is still dirty after all fixes
8. Creates an annotated `vX.Y.Z` tag and atomically pushes `main` plus the tag
9. Internally re-dispatches `release.yml` on the immutable tag
10. Re-runs full CI test suite on the tag
11. Runs `uv build` before tagging, then builds again from the tag and publishes to PyPI via OIDC trusted publishing

### Version is tracked in two places

- `pyproject.toml` (`version` field + `[tool.bumpversion] current_version`)
- `token_throttle/__init__.py` (`__version__`)

Both are updated automatically by `bump-my-version`. Do not edit these manually.

## Development

```bash
uv sync --group dev
uv run pytest tests/unit
uv run ruff check .
```

`tests/unit` doesn't require Redis — but its default `--redis-url` is
`redis://localhost:6379`, so tests that talk to a real Redis will use one if
it's reachable there, and some of those flush the database around every test.
The suite refuses to run (aborts the whole session) against a non-empty
database unless you opt in. Point at a dedicated, empty DB index instead of a
shared one:

```bash
uv run pytest --redis-url redis://localhost:6379/13
```

Type checking is a hard release gate and needs the optional extras installed,
because the checked package includes the Redis, OpenAI, and tokenizer
integration modules:

```bash
uv sync --all-extras --group dev
uv run mypy
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full test/CI breakdown, the
pre-commit hook setup, doc-lint fixture maintenance, and test-naming
conventions.

## Documentation conventions

Public docs — `README.md`, `docs/*.md`, `CHANGELOG.md` — must read in
**user-facing register**: describe behavior and changes in terms a user can act on, never
internal development codenames (audit-round IDs like `R7`, lane IDs like `L38`, finding IDs
like `PF03`/`AD-31`, `FIX-NN` tracker IDs, or capsule date-codenames). A user cannot look
those up. This is enforced by `tests/lint/test_public_docs_no_internal_codenames.py`, which
fails CI on any leak. `DEVELOPMENT.md` and this file are contributor/maintainer docs and may
reference internal IDs.

The `README.md` is a **bounded front door** (pitch, quickstarts, mental model, pointers);
reference and operational depth lives in tiered `docs/*.md` reached by one-line pointers,
not inlined into the README.
