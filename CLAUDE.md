# token-throttle

## Releasing a New Version

Releases are fully automated via GitHub Actions. Do NOT bump versions or publish locally.

### Trigger a release

```bash
gh workflow run release.yml -f bump=minor   # 0.5.0 -> 0.6.0
gh workflow run release.yml -f bump=patch   # 0.5.0 -> 0.5.1
```

### What the workflow does

1. Runs full CI test suite (`.github/workflows/ci.yml`)
2. `bump-my-version bump <minor|patch>` — updates version in `pyproject.toml` and `token_throttle/__init__.py`, creates a commit
3. `devtools/bump_readme_version.py` — updates pip install version bounds and badge in README.md, amends the bump commit
4. Pushes to `main`
5. Builds with `uv build` and publishes to PyPI via OIDC trusted publishing

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
