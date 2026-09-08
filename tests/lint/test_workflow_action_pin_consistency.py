"""
Every GitHub Action repo must be pinned to ONE commit across all workflows.

Motivating failure: Dependabot treats `github/codeql-action/init` and
`github/codeql-action/analyze` as two independent dependencies, so it raises a
separate PR for each. But they are two paths inside a single action repo, and
CodeQL aborts when the steps disagree:

    Loaded a configuration file for version '4.37.8', but running version '4.36.2'

Every such single-sub-action PR is therefore born failing, and the breakage is
invisible until CI runs. `.github/dependabot.yml` groups `github/codeql-action*`
so the bump is atomic; this test is the structural backstop for that config —
and it also catches hand-edits that bump one call site and miss the others.

Consistency is checked per action *repo* (`owner/name`), not per `uses:` string,
because the sub-path is exactly the distinction that caused the bug.

KNOWN UNKNOWN: the trailing `# vX.Y.Z` comment is checked only for agreement
between call sites, not against the upstream tag. A pin whose SHA and comment
are consistently wrong together still passes; verifying the SHA really is that
tag would need a network call, which CI lint must not make.
"""

from __future__ import annotations

import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass

_WORKFLOW_DIR = pathlib.Path(".github/workflows")

# `uses: owner/name[/sub/path]@ref [# comment]` — local (`./…`) and container
# (`docker://…`) references have no upstream repo to pin and are skipped by the
# leading owner/name shape.
_USES_RE = re.compile(
    r"^\s*-?\s*uses:\s*"
    r"(?P<owner>[\w.-]+)/(?P<name>[\w.-]+)"
    r"(?P<subpath>(?:/[\w.-]+)*)"
    r"@(?P<ref>\S+)"
    r"(?:\s*#\s*(?P<comment>.*?))?\s*$"
)


@dataclass(frozen=True)
class _Pin:
    location: str
    uses: str
    ref: str
    comment: str | None


def _collect_pins(repo_root: pathlib.Path) -> dict[str, list[_Pin]]:
    pins: dict[str, list[_Pin]] = defaultdict(list)
    workflow_dir = repo_root / _WORKFLOW_DIR

    for path in sorted(workflow_dir.glob("*.yml")) + sorted(
        workflow_dir.glob("*.yaml")
    ):
        relative_path = path.relative_to(repo_root).as_posix()
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = _USES_RE.match(line)
            if match is None:
                continue
            action_repo = f"{match['owner']}/{match['name']}"
            pins[action_repo].append(
                _Pin(
                    location=f"{relative_path}:{lineno}",
                    uses=f"{action_repo}{match['subpath']}",
                    ref=match["ref"],
                    comment=match["comment"],
                )
            )

    return pins


def test_workflow_dir_is_discoverable() -> None:
    """Guards against the collector silently finding nothing and passing."""
    repo_root = pathlib.Path(__file__).parent.parent.parent
    pins = _collect_pins(repo_root)

    assert pins, f"No pinned `uses:` steps found under {_WORKFLOW_DIR}"


def test_each_action_repo_is_pinned_to_one_commit() -> None:
    repo_root = pathlib.Path(__file__).parent.parent.parent
    pins = _collect_pins(repo_root)

    violations: list[str] = []
    for action_repo, entries in sorted(pins.items()):
        refs = {entry.ref for entry in entries}
        if len(refs) == 1:
            continue
        detail = "\n".join(
            f"    {entry.location}: {entry.uses}@{entry.ref}"
            f"{f'  # {entry.comment}' if entry.comment else ''}"
            for entry in entries
        )
        violations.append(
            f"  {action_repo} is pinned to {len(refs)} different refs:\n{detail}"
        )

    assert not violations, (
        "GitHub Action repos must be pinned to one commit across all workflows "
        "— sub-actions of the same repo (e.g. codeql-action/init and "
        "codeql-action/analyze) refuse to run at different versions. Bump every "
        "call site together:\n" + "\n".join(violations)
    )


def test_version_comments_agree_with_their_pinned_ref() -> None:
    """A stale `# vX.Y.Z` comment misreports what is actually running."""
    repo_root = pathlib.Path(__file__).parent.parent.parent
    pins = _collect_pins(repo_root)

    violations: list[str] = []
    for action_repo, entries in sorted(pins.items()):
        comments_by_ref: dict[str, set[str]] = defaultdict(set)
        for entry in entries:
            if entry.comment is not None:
                comments_by_ref[entry.ref].add(entry.comment)

        for ref, comments in sorted(comments_by_ref.items()):
            if len(comments) <= 1:
                continue
            locations = "\n".join(
                f"    {entry.location}: # {entry.comment}"
                for entry in entries
                if entry.ref == ref and entry.comment is not None
            )
            violations.append(
                f"  {action_repo}@{ref[:12]} carries "
                f"{len(comments)} different version comments:\n{locations}"
            )

    assert not violations, (
        "One pinned ref must carry one version comment — a disagreeing comment "
        "means at least one call site is mislabeled:\n" + "\n".join(violations)
    )
