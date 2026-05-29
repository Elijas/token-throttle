"""Execute README snippets that are documented as standalone examples."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_README = _REPO_ROOT / "README.md"


def _python_block_under_heading(markdown: str, heading: str) -> str:
    heading_marker = f"{heading}\n"
    try:
        section_start = markdown.index(heading_marker) + len(heading_marker)
    except ValueError as exc:
        raise AssertionError(f"README heading not found: {heading}") from exc

    next_heading = markdown.find("\n##", section_start)
    section = (
        markdown[section_start:]
        if next_heading == -1
        else markdown[section_start:next_heading]
    )
    match = re.search(r"```python\n(.*?)\n```", section, re.DOTALL)
    if match is None:
        raise AssertionError(f"README Python block not found under: {heading}")
    return match.group(1)


def _run_readme_example(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - executes trusted snippets from this repo's README.
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_memory_quickstart_readme_example_runs() -> None:
    markdown = _README.read_text(encoding="utf-8")
    result = _run_readme_example(
        _python_block_under_heading(markdown, "### Memory quickstart (zero-service)")
    )

    assert result.stdout.strip() == "reserved 1000 tokens, refunded 575 unused tokens"


def test_any_provider_readme_example_runs() -> None:
    markdown = _README.read_text(encoding="utf-8")
    result = _run_readme_example(
        _python_block_under_heading(markdown, "### Any provider (manual usage)")
    )

    assert (
        result.stdout.strip()
        == "unused 20 input tokens and 2800 output tokens returned to the pool"
    )


def test_sync_api_readme_example_runs() -> None:
    markdown = _README.read_text(encoding="utf-8")

    _run_readme_example(_python_block_under_heading(markdown, "## Sync API"))
