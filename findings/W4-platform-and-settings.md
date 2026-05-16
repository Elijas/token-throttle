# W4 Platform and Release Settings Findings

Date: 2026-05-16
Branch: `fix/r8-platform`
Base: `ae9de48` (`v4.0.0`)

## Sub-task A: release settings

### GitHub `pypi` environment

Checked with:

```bash
gh api repos/Elijas/token-throttle/environments
gh api repos/Elijas/token-throttle/environments/pypi
```

Current state:

- Environment exists: `pypi`
- Created/updated: `2026-03-07T13:58:42Z`
- `can_admins_bypass`: `true`
- `protection_rules`: `[]`
- `deployment_branch_policy`: `null`

Assessment:

- The release workflow publish job already binds PyPI publishing to the
  `pypi` environment in `.github/workflows/release.yml`.
- GitHub environment protection is present only as an environment name. It has
  no reviewer, wait-timer, or deployment branch/tag policy configured.

KNOWN UNKNOWN: intended GitHub environment protection policy - the API shows no
protection rules, but this worker cannot infer which owners or teams should be
required reviewers. Repository owners should decide whether to require manual
approval, set a wait timer, disable admin bypass, and/or add deployment branch
policies for release refs.

### PyPI trusted publisher

PyPI trusted-publisher configuration is PyPI-side and is not exposed through
`gh`. No programmatic change was attempted.

Manual verification steps:

1. Sign in to PyPI with an owner/maintainer account for `token-throttle`.
2. Open the project management page for `token-throttle`.
3. Navigate to the project's publishing/trusted-publisher settings.
4. Confirm a GitHub publisher exists with:
   - Owner: `Elijas`
   - Repository: `token-throttle`
   - Workflow: `release.yml`
   - Environment: `pypi`
5. Confirm there are no stale publishers for old repositories, workflow names,
   or unprotected environments.

KNOWN UNKNOWN: PyPI trusted-publisher state - must be verified by a PyPI
project owner in the PyPI web UI.

### Pinned workflow actions

Skimmed `.github/workflows/release.yml` and `.github/workflows/ci.yml`.

All third-party or marketplace actions in those workflows are pinned by full
commit SHA:

- `actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd`
- `astral-sh/setup-uv@94527f2e458b27549849d47d273a16bec83a01e9`
- `actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405`
- `codecov/codecov-action@57e3a136b779b570ffcdbf80b3bdc90e7fab3de2`
- `pypa/gh-action-pypi-publish@cef221092ed1bacb1cc03d23a2d87d1d172e277b`

No pin-by-tag stragglers were found in the checked CI/release workflows.

## Sub-task B: CI Python/OS matrix

### Current support declaration

`pyproject.toml` declares:

- `requires-python = ">=3.12"`
- Classifiers for Python 3.12, 3.13, and 3.14

Python 3.10 and 3.11 are not declared as supported, so no CI jobs were added for
those versions.

### Previous matrix

Before this change, CI tested:

- Linux unit core: Python 3.12, 3.13, 3.14
- Linux unit full/all-extras: Python 3.12, 3.13, 3.14
- Linux Redis integration: Python 3.12, 3.13, 3.14
- Linux min-deps: Python 3.12
- Linux coverage: Python 3.13

### Expansion decision

Added a bounded cross-platform unit job:

- `test-unit-platform`
- OS matrix: `macos-latest`, `windows-latest`
- Python: 3.13
- Command: `uv run pytest tests/unit -v`
- Dependency set: all extras + dev

Reasoning:

- macOS is high-value and low-friction now that standard GitHub-hosted
  `macos-latest` runners are arm64.
- Windows unit coverage is worthwhile for path, locale, shell, and async event
  loop policy differences.
- The job uses a single stable Python version instead of multiplying every
  supported Python version across every OS, which limits CI-minute growth.

Deferred:

- Full Python x OS unit matrix: deferred to control CI cost.
- macOS/Windows Redis integration: deferred because GitHub Actions service
  containers and Redis integration coverage are Linux-oriented; the package's
  Redis behavior is already covered across supported Python versions on Linux.
- Python 3.10/3.11: deferred because they are outside the declared support
  contract.

## Changes applied

- Added `.github/workflows/ci.yml` job `test-unit-platform`.
- Updated `DEVELOPMENT.md` to reflect Python, OS, and Redis integration matrix
  scope.
