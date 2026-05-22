# token-throttle

## Releasing a New Version

Releases are fully automated via GitHub Actions. Do NOT bump versions or publish locally.

### Trigger a release

```bash
gh workflow run release.yml -f bump=patch   # 0.5.0 -> 0.5.1
gh workflow run release.yml -f bump=minor   # 0.5.0 -> 0.6.0
gh workflow run release.yml -f bump=major   # 0.6.0 -> 1.0.0
```

### What the workflow does

1. Runs full CI test suite (`.github/workflows/ci.yml`)
2. `bump-my-version bump <patch|minor|major>` — updates version in `pyproject.toml` and `token_throttle/__init__.py`, creates a commit
3. `devtools/bump_readme_version.py` — updates pip install version bounds and badge in README.md
4. `uv lock` — syncs `uv.lock` with the new version
5. `ruff check --fix` + `ruff format` — applies any lint autofixes and formatting
6. Creates a second commit with lockfile/formatting changes (if any)
7. Porcelain check — fails the release if the working tree is still dirty after all fixes
8. Creates an annotated `vX.Y.Z` tag and atomically pushes `main` plus the tag
9. Internally re-dispatches `release.yml` on the immutable tag
10. Re-runs full CI test suite on the tag
11. Builds from the tag with `uv build` and publishes to PyPI via OIDC trusted publishing

### Version is tracked in two places

- `pyproject.toml` (`version` field + `[tool.bumpversion] current_version`)
- `token_throttle/__init__.py` (`__version__`)

Both are updated automatically by `bump-my-version`. Do not edit these manually.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```
