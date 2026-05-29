"""Execute README snippets that are documented as standalone examples."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
_README = _REPO_ROOT / "README.md"
_EXPECTED_STDOUT_BY_HEADING = {
    "### Memory quickstart (zero-service)": (
        "reserved 1000 tokens, refunded 575 unused tokens"
    ),
    "### Any provider (manual usage)": (
        "unused 20 input tokens and 2800 output tokens returned to the pool"
    ),
}


@dataclass(frozen=True)
class _ReadmePythonBlock:
    heading: str
    code: str
    start_line: int


def _heading_from_line(line: str) -> str | None:
    marker, separator, title = line.partition(" ")
    if (
        separator
        and marker
        and set(marker) == {"#"}
        and 1 <= len(marker) <= 6
        and title.strip()
    ):
        return f"{marker} {title.strip()}"
    return None


def _is_fragment_python_block(code: str) -> bool:
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.startswith(("# (fragment", "(fragment"))
    return False


def _standalone_python_blocks(markdown: str) -> list[_ReadmePythonBlock]:
    lines = markdown.splitlines()
    blocks: list[_ReadmePythonBlock] = []
    heading = "<README top>"
    index = 0
    while index < len(lines):
        line = lines[index]
        current_heading = _heading_from_line(line)
        if current_heading is not None:
            heading = current_heading
        if line.strip() != "```python":
            index += 1
            continue

        start_line = index + 2
        index += 1
        block_lines: list[str] = []
        while index < len(lines) and lines[index].strip() != "```":
            block_lines.append(lines[index])
            index += 1
        if index >= len(lines):
            raise AssertionError(
                f"Unterminated README Python block at line {start_line}"
            )

        code = "\n".join(block_lines)
        if not _is_fragment_python_block(code):
            blocks.append(
                _ReadmePythonBlock(
                    heading=heading,
                    code=code,
                    start_line=start_line,
                )
            )
        index += 1
    return blocks


def _run_readme_example(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - executes trusted snippets from this repo's README.
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


_STANDALONE_README_EXAMPLES = _standalone_python_blocks(
    _README.read_text(encoding="utf-8")
)


def test_readme_has_standalone_python_examples() -> None:
    assert _STANDALONE_README_EXAMPLES


@pytest.mark.parametrize(
    "example",
    _STANDALONE_README_EXAMPLES,
    ids=lambda example: f"{example.heading} line {example.start_line}",
)
def test_readme_standalone_python_examples_run(example: _ReadmePythonBlock) -> None:
    result = _run_readme_example(example.code)

    expected_stdout = _EXPECTED_STDOUT_BY_HEADING.get(example.heading)
    if expected_stdout is not None:
        assert result.stdout.strip() == expected_stdout
