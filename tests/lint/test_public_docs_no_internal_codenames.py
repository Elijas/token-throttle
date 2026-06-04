"""Guard: public-facing docs must not leak internal development codenames.

Public docs are read by users who cannot look up internal audit-round / lane /
finding / fix-tracker / test-case IDs. When documentation-accuracy work pulls a
fact in from an internal capsule or finding, it must be restated in user-facing
register (describe the *change*, not its tracker ID). This test fails CI on any
leak into the user-facing surface:

    README.md, MIGRATION.md, CHANGELOG.md, docs/*.md

Contributor/maintainer docs (DEVELOPMENT.md, CLAUDE.md) are intentionally out of
scope — they may reference internal IDs. See CLAUDE.md "Documentation
conventions" for the rule this enforces.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_PUBLIC_DOCS = [
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "MIGRATION.md",
    _REPO_ROOT / "CHANGELOG.md",
    *sorted((_REPO_ROOT / "docs").glob("*.md")),
]

# Internal development codenames that are meaningless to an external reader.
# Each entry is (human label, compiled pattern).
_FORBIDDEN: dict[str, re.Pattern[str]] = {
    "audit-round id (Rn)": re.compile(r"\bR\d{1,2}\b"),
    "audit-lane id (Lnn)": re.compile(r"\bL\d{2}\b"),
    "finding id (PFn)": re.compile(r"\bPF\d+\b"),
    "finding id (AD-n)": re.compile(r"\bAD-\d+\b"),
    "fix-tracker id (FIX-n)": re.compile(r"\bFIX-\d+\b"),
    "test-case id (TCn / TCn-nnn)": re.compile(r"\bTC\d+(?:-\d+)?\b"),
    "deferred-item id (PD-n)": re.compile(r"\bPD-\d+\b"),
    "capsule date-codename (YYMMDD-name)": re.compile(r"\b2\d{5}-[a-z]"),
}

# Narrow, reviewed exceptions: (doc filename, exact matched token) pairs that a
# pattern flags but are legitimate public content. Keep empty unless a real
# false positive is reviewed and justified here with a comment.
_ALLOWLIST: set[tuple[str, str]] = set()


def _violations() -> list[str]:
    found: list[str] = []
    for path in _PUBLIC_DOCS:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for label, pattern in _FORBIDDEN.items():
                for match in pattern.finditer(line):
                    token = match.group(0)
                    if (path.name, token) in _ALLOWLIST:
                        continue
                    found.append(
                        f"{path.name}:{lineno}: {label}: {token!r}  | {line.strip()[:90]}"
                    )
    return found


def test_public_docs_have_no_internal_codenames() -> None:
    violations = _violations()
    assert not violations, (
        "Internal development codenames leaked into public docs. Restate the fact "
        "in user-facing register (describe the change, not its tracker ID); see "
        "CLAUDE.md 'Documentation conventions'.\n  " + "\n  ".join(violations)
    )


def test_public_doc_set_is_non_empty() -> None:
    # Guards against the scan silently covering nothing (e.g. a path rename).
    existing = [p.name for p in _PUBLIC_DOCS if p.exists()]
    assert {"README.md", "MIGRATION.md", "CHANGELOG.md"}.issubset(set(existing)), (
        existing
    )
