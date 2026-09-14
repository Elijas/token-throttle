# token-throttle

## Releasing a New Version

Releases are fully automated via GitHub Actions. Do NOT bump versions or publish locally.

Releasing is **two dispatches** with a reviewed PR between them. `main` is
protected by strict required CI with no bypass, so the version bump cannot be
pushed straight to `main` — it lands through a normal PR like any other change.

### Step 1 — prepare the release PR

```bash
gh workflow run release.yml --ref main -f bump=patch   # 0.5.0 -> 0.5.1
gh workflow run release.yml --ref main -f bump=minor   # 0.5.0 -> 0.6.0
gh workflow run release.yml --ref main -f bump=major   # 0.6.0 -> 1.0.0
```

This runs CI, bumps the version, refreshes the lockfile and formatting, and
opens a `release/vX.Y.Z` PR. Nothing is tagged or published. Review and merge
that PR once required CI is green.

### Step 2 — finalize after the PR merges

```bash
gh workflow run release.yml --ref main -f finalize=v0.5.1
```

This verifies `main` really carries that version, tags `main`'s tip, and
publishes to PyPI. Running it before the PR merges fails with a clear error
rather than releasing anything.

### What the workflow does

**Prepare (`-f bump=...`)**

1. Verifies the dispatch runs from `refs/heads/main` and that exactly one mode input is set, then runs full CI (`.github/workflows/ci.yml`)
2. Fails fast if the run is pinned to a commit that is no longer `origin/main`'s tip
3. `bump-my-version bump <patch|minor|major>` — updates version in `pyproject.toml` and `token_throttle/__init__.py`
4. `devtools/bump_readme_version.py` — updates pip install version bounds in README.md
5. `uv lock` — syncs `uv.lock` with the new version
6. `ruff check --fix`, `ruff format`, then `ruff check` — applies safe lint autofixes, formats, and fails if issues remain
7. `uv build` as a gate, plus a porcelain check — fails if the tree is still dirty
8. Refuses to proceed if tag `vX.Y.Z` or branch `release/vX.Y.Z` already exists
9. Commits to `release/vX.Y.Z`, pushes **that branch only**, and opens a PR — `main` is never written
10. Explicitly dispatches `ci.yml` and `codeql.yml` onto the release branch (a branch pushed by `GITHUB_TOKEN` fires no `push`/`pull_request` events, so the PR would otherwise have no checks to satisfy)

**Finalize (`-f finalize=vX.Y.Z`)**

11. Verifies the run is on `main`'s live tip, that `pyproject.toml` is at that exact version, and that the tag does not already exist
12. Creates an annotated `vX.Y.Z` tag and pushes **the tag only** — the tag namespace is outside branch protection, so this needs no bypass
13. Internally re-dispatches `release.yml` on the immutable tag (a tag pushed by `GITHUB_TOKEN` does not fire `push: tags:`)
14. Re-runs full CI on the tag, then publishes to PyPI via OIDC trusted publishing — checking out the tested SHA (not the mutable tag name), verifying the tag matches the built version, and refusing any commit not reachable from `origin/main`

### Repository setting this depends on

Settings → Actions → General → Workflow permissions →
**"Allow GitHub Actions to create and approve pull requests"** must be enabled,
or step 1 cannot open its PR. Default token permissions stay read-only; this
toggle is separate from them. Auto-merge needs no PR approvals.

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
