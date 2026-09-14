# Dependabot auto-merge deployment

Prepared locally; not active until published and the repository settings below
are applied.

Policy: automatically merge only patch/minor GitHub Actions updates and
explicitly listed direct development dependencies. Runtime, indirect, major,
unknown, mixed-ineligible, and maintainer-change updates remain manual.
Existing CodeQL grouping is unchanged.

The workflow runs on pull_request_target without checking out PR code. Metadata
commit verification stays enabled. The merge command is bound to the event head
SHA. Every listed CI check must be enforced by a ruleset and supplied by GitHub
Actions (integration 15368), or the workflow refuses to enable auto-merge.

## Activation prerequisites

1. Fix the Release workflow's direct push to main before enabling required CI.
   It currently creates a new, untested version commit and pushes it to main.
   A suitable migration is a release-preparation PR that runs CI before merging,
   followed by tagging and publishing the tested commit. Do not add a broad
   GitHub Actions bypass: the auto-merge workflow uses that identity too.
2. Publish this branch through the authorized owning orchestrator. The current
   session's push was denied by automatic approval review, which requires that
   orchestrator's SHA-bound approval and email audit.
3. Create the additional ruleset from .github/dependabot-ci-ruleset.json,
   preserving existing rulesets. Its strict checks require an up-to-date branch.
4. Enable the repository setting allow_auto_merge. Merge commits must remain
   enabled. No automatic approval permission or personal access token is needed.
5. Verify a real eligible Dependabot PR waits for all required checks, then
   merges. Verify major/runtime PRs do not get auto-merge enabled.

When CI matrix names or development dependencies change, update the explicit
lists in the workflow and the CI ruleset together. The workflow checks rulesets,
not legacy branch-protection settings.

