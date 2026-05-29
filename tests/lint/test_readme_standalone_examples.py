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
_SUPPORTED_PYTHON_FENCE_TOKENS = {"python", "py"}


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


def _readme_fence_info(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("```") or stripped == "```":
        return None
    return stripped.removeprefix("```").strip()


def _is_unsupported_python_like_fence_token(token: str) -> bool:
    return token.startswith(("python", "py-", "py_")) or token in {"py3", "pycon"}


def _is_python_fence_info(info: str) -> bool:
    first_token = info.split(maxsplit=1)[0].lower()
    if first_token in _SUPPORTED_PYTHON_FENCE_TOKENS:
        return True
    if _is_unsupported_python_like_fence_token(first_token):
        raise AssertionError(
            "Unsupported README Python fence variant "
            f"`{info}`; use `python`, `python ...`, or `py` so examples "
            "are linted."
        )
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

        fence_info = _readme_fence_info(line)
        if fence_info is None:
            index += 1
            continue

        is_python_fence = _is_python_fence_info(fence_info)
        start_line = index + 2
        index += 1
        block_lines: list[str] = []
        while index < len(lines) and lines[index].strip() != "```":
            block_lines.append(lines[index])
            index += 1
        if index >= len(lines):
            block_kind = "Python" if is_python_fence else "code"
            raise AssertionError(
                f"Unterminated README {block_kind} block at line {start_line}"
            )

        if not is_python_fence:
            index += 1
            continue

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


def test_standalone_python_blocks_accept_supported_fence_variants() -> None:
    markdown = """
# Demo

```python
print("exact")
```

## Python attributes

```python title="example.py" linenums="1"
print("attrs")
```

## Py alias

```py
print("alias")
```
"""

    blocks = _standalone_python_blocks(markdown)

    assert [(block.heading, block.code) for block in blocks] == [
        ("# Demo", 'print("exact")'),
        ("## Python attributes", 'print("attrs")'),
        ("## Py alias", 'print("alias")'),
    ]


def test_standalone_python_blocks_preserve_fragment_skips() -> None:
    markdown = """
# Demo

```python

# (fragment - depends on earlier setup)
raise RuntimeError("fragment")
```

```python title="fragment.py"
(fragment - illustrative only)
raise RuntimeError("fragment")
```

```py
print("standalone")
```
"""

    blocks = _standalone_python_blocks(markdown)

    assert [(block.heading, block.code) for block in blocks] == [
        ("# Demo", 'print("standalone")')
    ]


def test_standalone_python_blocks_reject_unsupported_python_like_fences() -> None:
    markdown = """
# Demo

```python3
print("not linted")
```
"""

    with pytest.raises(AssertionError, match="Unsupported README Python fence"):
        _standalone_python_blocks(markdown)


def test_expected_stdout_attaches_by_exact_heading() -> None:
    exact_heading = "### Memory quickstart (zero-service)"
    near_heading = "### Memory quickstart (zero-service) extra"
    markdown = f"""
{exact_heading}

```python
print("exact heading")
```

{near_heading}

```python
print("near heading")
```
"""

    blocks = _standalone_python_blocks(markdown)

    assert [block.heading for block in blocks] == [exact_heading, near_heading]
    assert _EXPECTED_STDOUT_BY_HEADING.get(blocks[0].heading) is not None
    assert _EXPECTED_STDOUT_BY_HEADING.get(blocks[1].heading) is None


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
