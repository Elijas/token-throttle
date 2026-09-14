# Dependabot auto-merge

Patch/minor GitHub Actions updates and explicitly listed direct development
dependencies are eligible for auto-merge. Runtime, indirect, major, unknown,
mixed-ineligible, and maintainer-change updates remain manual. CodeQL sub-actions
remain grouped so init and analyze update together.

The workflow runs on `pull_request_target` without checking out PR code.
Dependabot's commit verification stays enabled. Every PR update clears previous
auto-merge eligibility before checking the new metadata, and the final merge
command is bound to the event's head SHA.
Only a single verified Dependabot commit is eligible. PRs with additional
commits or manual merge resolutions stay manual; the metadata action checks
only the first commit, so the workflow verifies the complete PR separately.

## Required repository settings

- Enable **Allow auto-merge** and **Allow merge commits**.
- Install the additional ruleset in `.github/dependabot-ci-ruleset.json`.
  Preserve other rulesets. All listed CI checks must come from GitHub Actions,
  and the PR must be up to date with main. Do not configure bypass actors.
- Keep default workflow permissions **read-only**. Enable **Allow GitHub Actions
  to create and approve pull requests** so Release can create preparation PRs.
  Neither workflow automatically approves PRs; permissions are granted per job.

The auto-merge workflow fails closed if its required CI rules are absent. These
files alone do not change repository settings. They must be deployed together.

## Releases with protected main

Release preparation creates a branch and PR instead of pushing to main.
It explicitly dispatches CI and CodeQL because pushes and PRs created using
`GITHUB_TOKEN` do not trigger those workflows automatically.

1. Dispatch **Release** on `main` with `bump=patch|minor|major`.
2. Review and merge the generated release PR after required CI passes.
3. Dispatch **Release** on `main` with `finalize=vX.Y.Z` to tag the merged
   version and start the existing tag-test and publication sequence.

Do not use a broad GitHub Actions bypass to restore direct release pushes:
the auto-merge workflow runs under the same identity.

## Maintenance

When CI matrix names or allowed development dependencies change, update the
workflow lists and ruleset together. The protection check reads rulesets, not
legacy branch-protection settings. Unknown update types remain manual.

The next eligible Dependabot PR exercises the complete automatic path. Major
or runtime updates should remain open for review regardless of green CI.
