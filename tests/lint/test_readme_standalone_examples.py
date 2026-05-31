"""Execute docs snippets that are documented as standalone examples."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
_README = _REPO_ROOT / "README.md"
_MIGRATION = _REPO_ROOT / "MIGRATION.md"
_DEVELOPMENT = _REPO_ROOT / "DEVELOPMENT.md"
_DOCS_WITH_STANDALONE_EXAMPLES = (_README, _MIGRATION, _DEVELOPMENT)


@dataclass(frozen=True)
class _ExpectedMigrationFragment:
    start_line: int
    reason: str
    heading: str
    first_non_empty_code_line: str


@dataclass(frozen=True)
class _StandaloneExampleIdentity:
    document_name: str
    start_line: int
    heading: str
    first_non_empty_code_line: str


@dataclass(frozen=True)
class _NonPythonFenceKey:
    document_name: str
    start_line: int
    language: str
    heading: str
    first_non_empty_content_line: str
    non_empty_content_lines: tuple[str, ...]


@dataclass(frozen=True)
class _NonPythonFenceIdentity:
    document_name: str
    start_line: int
    language: str
    classification: str
    heading: str
    first_non_empty_content_line: str
    non_empty_content_lines: tuple[str, ...]


@dataclass(frozen=True)
class _ExpectedNonPythonFence:
    document_name: str
    start_line: int
    language: str
    classification: str
    reason: str
    heading: str
    non_empty_content_lines: tuple[str, ...]


@dataclass(frozen=True)
class _ExpectedStdoutExample:
    document_name: str
    start_line: int
    heading: str
    first_non_empty_code_line: str
    expected_stdout: str


_EXPECTED_MIGRATION_FRAGMENTS = (
    _ExpectedMigrationFragment(
        start_line=37,
        reason="illustrates unsafe cross-limiter refund with undefined limiters",
        heading="### What changed",
        first_non_empty_code_line="reservation = await limiter_a.acquire_capacity(",
    ),
    _ExpectedMigrationFragment(
        start_line=47,
        reason="requires a live async limiter and provider call",
        heading="### What changed",
        first_non_empty_code_line="reservation = await limiter.acquire_capacity(",
    ),
    _ExpectedMigrationFragment(
        start_line=131,
        reason="uses illustrative recovery hook and limiter variables",
        heading="### What changed",
        first_non_empty_code_line="from token_throttle import AcquireRefundFailedError",
    ),
    _ExpectedMigrationFragment(
        start_line=143,
        reason="uses illustrative recovery hook and limiter variables",
        heading="### What changed",
        first_non_empty_code_line="from token_throttle import AcquireRefundFailedError",
    ),
    _ExpectedMigrationFragment(
        start_line=191,
        reason="illustrates callback behavior with undefined limiter and reservation",
        heading="### What changed",
        first_non_empty_code_line="async def on_capacity_refunded(**kwargs) -> None:",
    ),
    _ExpectedMigrationFragment(
        start_line=237,
        reason="requires a caller-provided backend builder",
        heading="### What changed",
        first_non_empty_code_line="from token_throttle import run_conformance_test_for",
    ),
    _ExpectedMigrationFragment(
        start_line=274,
        reason="uses illustrative model and recovery flow variables",
        heading="### AcquireRefundFailedError base class",
        first_non_empty_code_line="from token_throttle import AcquireRefundFailedError",
    ),
    _ExpectedMigrationFragment(
        start_line=409,
        reason="requires an operator-provided config mapping",
        heading="## 1. Preflight Config Dictionaries",
        first_non_empty_code_line=(
            "from token_throttle.migration import validate_config_for_v2_0"
        ),
    ),
    _ExpectedMigrationFragment(
        start_line=516,
        reason="requires an operator-supplied Redis client and target deployment prefix",
        heading="## 6a. Clean Up Pre-FIX-38 Redis Bucket Keys",
        first_non_empty_code_line=(
            "from token_throttle.migration import cleanup_legacy_buckets"
        ),
    ),
)
_EXPECTED_MIGRATION_FRAGMENT_START_LINES = frozenset(
    fragment.start_line for fragment in _EXPECTED_MIGRATION_FRAGMENTS
)
_EXPECTED_MIGRATION_REDIS_CLEANUP_SCAN_PATTERN = "`{key_prefix}:rate_limiting:bucket:*`"
_EXPECTED_MIGRATION_REDIS_BUCKET_KEY_SHAPE = (
    "{key_prefix}:rate_limiting:bucket:{model_family}:{metric}:{per_seconds}:{suffix}"
)
_EXPECTED_MIGRATION_REDIS_ACQUIRED_KEY_SHAPE = (
    "{key_prefix}:rate_limiting:acquired:{reservation_id}"
)
_NON_PYTHON_CLASSIFICATION_SHELL_SYNTAX = "syntax-checkable shell command block"
_NON_PYTHON_CLASSIFICATION_MANUAL = "intentionally non-executable/manual command block"
_NON_PYTHON_CLASSIFICATION_TEXT = "non-executable text/output/config/key-shape block"
_NON_PYTHON_FENCE_CLASSIFICATIONS = frozenset(
    {
        _NON_PYTHON_CLASSIFICATION_SHELL_SYNTAX,
        _NON_PYTHON_CLASSIFICATION_MANUAL,
        _NON_PYTHON_CLASSIFICATION_TEXT,
    }
)
_UNLABELED_FENCE_LANGUAGE = "<unlabeled>"
_SHELL_LIKE_FENCE_LANGUAGES = frozenset({"bash", "sh", "shell"})
_MANUAL_NON_PYTHON_FENCE_KEYS: frozenset[_NonPythonFenceKey] = frozenset()
_EXPECTED_NON_PYTHON_FENCES = (
    _ExpectedNonPythonFence(
        document_name="README.md",
        start_line=22,
        language="bash",
        classification=_NON_PYTHON_CLASSIFICATION_SHELL_SYNTAX,
        reason="package installation command; lint syntax-checks but does not execute it",
        heading="# token-throttle",
        non_empty_content_lines=(
            'pip install "token-throttle[redis,tiktoken]>=8.0.6,<8.1.0"   # OpenAI + Redis (recommended)',
            'pip install "token-throttle[redis]>=8.0.6,<8.1.0"            # Any provider + Redis',
            'pip install "token-throttle>=8.0.6,<8.1.0"                   # Any provider + in-memory',
        ),
    ),
    _ExpectedNonPythonFence(
        document_name="README.md",
        start_line=91,
        language="bash",
        classification=_NON_PYTHON_CLASSIFICATION_SHELL_SYNTAX,
        reason="package installation command; lint syntax-checks but does not execute it",
        heading="### OpenAI (built-in helpers)",
        non_empty_content_lines=(
            'pip install "token-throttle[redis,tiktoken]" openai',
        ),
    ),
    _ExpectedNonPythonFence(
        document_name="README.md",
        start_line=469,
        language="text",
        classification=_NON_PYTHON_CLASSIFICATION_TEXT,
        reason="Redis monitoring command inventory for operators, not a shell script",
        heading="#### Performance and capacity planning",
        non_empty_content_lines=(
            "INFO commandstats   # eval/evalsha, set, get, del latency and call volume",
            "INFO clients        # connected_clients, blocked_clients, maxclients pressure",
            "INFO memory         # used_memory, mem_fragmentation_ratio, evicted_keys",
            "INFO stats          # instantaneous_ops_per_sec, rejected_connections",
            "LATENCY LATEST      # server-side latency spikes",
            "SLOWLOG GET 128     # slow Lua scripts or lock commands",
        ),
    ),
    _ExpectedNonPythonFence(
        document_name="DEVELOPMENT.md",
        start_line=6,
        language="bash",
        classification=_NON_PYTHON_CLASSIFICATION_SHELL_SYNTAX,
        reason="environment setup command; lint syntax-checks but does not execute it",
        heading="## Setup",
        non_empty_content_lines=("uv sync --all-extras --group dev",),
    ),
    _ExpectedNonPythonFence(
        document_name="DEVELOPMENT.md",
        start_line=12,
        language="bash",
        classification=_NON_PYTHON_CLASSIFICATION_SHELL_SYNTAX,
        reason="test commands include the Redis full suite; lint syntax-checks but does not execute them",
        heading="## Running tests",
        non_empty_content_lines=(
            "# Unit tests only (no Redis required)",
            "uv run pytest tests/unit -v",
            "# Full suite (requires Redis on localhost:6379)",
            "uv run pytest tests/ -v --redis-url redis://localhost:6379",
        ),
    ),
    _ExpectedNonPythonFence(
        document_name="DEVELOPMENT.md",
        start_line=22,
        language="bash",
        classification=_NON_PYTHON_CLASSIFICATION_SHELL_SYNTAX,
        reason="setup and type-check commands; lint syntax-checks but does not execute them",
        heading="## Type checking",
        non_empty_content_lines=(
            "uv sync --all-extras --group dev",
            "uv run mypy",
        ),
    ),
    _ExpectedNonPythonFence(
        document_name="MIGRATION.md",
        start_line=439,
        language="text",
        classification=_NON_PYTHON_CLASSIFICATION_TEXT,
        reason="canonical migration error text, not an executable command",
        heading="## 2. Drain Reservations",
        non_empty_content_lines=(
            "legacy v1.4.x reservations no longer supported in v2.0.0; drain v1.4.x before upgrade",
        ),
    ),
    _ExpectedNonPythonFence(
        document_name="MIGRATION.md",
        start_line=564,
        language="text",
        classification=_NON_PYTHON_CLASSIFICATION_TEXT,
        reason="documented Redis bucket key shape",
        heading="### 7b. Redis key format and Lua compatibility",
        non_empty_content_lines=(_EXPECTED_MIGRATION_REDIS_BUCKET_KEY_SHAPE,),
    ),
    _ExpectedNonPythonFence(
        document_name="MIGRATION.md",
        start_line=570,
        language="text",
        classification=_NON_PYTHON_CLASSIFICATION_TEXT,
        reason="documented Redis acquire-marker key shape",
        heading="### 7b. Redis key format and Lua compatibility",
        non_empty_content_lines=(_EXPECTED_MIGRATION_REDIS_ACQUIRED_KEY_SHAPE,),
    ),
)
_EXPECTED_NON_README_STANDALONE_IDENTITIES = (
    _StandaloneExampleIdentity(
        document_name="MIGRATION.md",
        start_line=61,
        heading="### What changed",
        first_non_empty_code_line=(
            "from token_throttle import create_logging_callbacks"
        ),
    ),
    _StandaloneExampleIdentity(
        document_name="MIGRATION.md",
        start_line=70,
        heading="### What changed",
        first_non_empty_code_line="import logging",
    ),
    _StandaloneExampleIdentity(
        document_name="MIGRATION.md",
        start_line=182,
        heading="### What changed",
        first_non_empty_code_line="async def on_capacity_refunded(**kwargs) -> None:",
    ),
    _StandaloneExampleIdentity(
        document_name="MIGRATION.md",
        start_line=226,
        heading="### What changed",
        first_non_empty_code_line="from token_throttle import RateLimiterBackend",
    ),
    _StandaloneExampleIdentity(
        document_name="DEVELOPMENT.md",
        start_line=336,
        heading="### Redis connection pool sizing",
        first_non_empty_code_line="import redis.asyncio as aioredis",
    ),
)
_EXPLICIT_FRAGMENT_START_LINES_BY_DOCUMENT = {
    "MIGRATION.md": _EXPECTED_MIGRATION_FRAGMENT_START_LINES,
}
_EXPECTED_STDOUT_EXAMPLES = (
    _ExpectedStdoutExample(
        document_name="README.md",
        start_line=34,
        heading="### Memory quickstart (zero-service)",
        first_non_empty_code_line="import asyncio",
        expected_stdout="reserved 1000 tokens, refunded 575 unused tokens",
    ),
    _ExpectedStdoutExample(
        document_name="README.md",
        start_line=154,
        heading="### Any provider (manual usage)",
        first_non_empty_code_line="import asyncio",
        expected_stdout=(
            "unused 20 input tokens and 2800 output tokens returned to the pool"
        ),
    ),
)
_SUPPORTED_PYTHON_FENCE_TOKENS = {"python", "py"}
_UNSUPPORTED_PYTHON_LIKE_FENCE_TOKENS = frozenset(
    {"ipython", "ipython3", "pypy", "pycon", "pycon3"}
)


@dataclass(frozen=True)
class _MarkdownFenceBlock:
    document_name: str
    heading: str
    code: str
    start_line: int
    fence_info: str


@dataclass(frozen=True)
class _MarkdownPythonBlock:
    document_name: str
    heading: str
    code: str
    start_line: int
    classified_as_fragment: bool


def _document_label(document_name: str) -> str:
    return document_name.removesuffix(".md")


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
    return _first_non_empty_code_line(code).startswith(("# (fragment", "(fragment"))


def _non_empty_code_lines(code: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in code.splitlines() if line.strip())


def _first_non_empty_code_line(code: str) -> str:
    return next(iter(_non_empty_code_lines(code)), "")


def _unquoted_shell_comment_index(line: str) -> int | None:
    in_single_quote = False
    in_double_quote = False
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if character == "\\" and not in_single_quote:
            escaped = True
            continue
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if (
            character == "#"
            and not in_single_quote
            and not in_double_quote
            and (index == 0 or line[index - 1].isspace())
        ):
            return index
    return None


def _normalize_shell_like_non_python_content_line(line: str) -> str:
    stripped = line.strip()
    comment_index = _unquoted_shell_comment_index(stripped)
    if comment_index is None or comment_index == 0:
        return stripped
    return f"{stripped[:comment_index].rstrip()} {stripped[comment_index:]}"


def _non_python_content_lines(code: str, *, language: str) -> tuple[str, ...]:
    lines = _non_empty_code_lines(code)
    if language not in _SHELL_LIKE_FENCE_LANGUAGES:
        return lines
    return tuple(_normalize_shell_like_non_python_content_line(line) for line in lines)


def _fence_language(fence_info: str) -> str:
    if not fence_info.strip():
        return _UNLABELED_FENCE_LANGUAGE
    return fence_info.split(maxsplit=1)[0].lower()


def _standalone_example_identity(
    block: _MarkdownPythonBlock,
) -> _StandaloneExampleIdentity:
    return _StandaloneExampleIdentity(
        document_name=block.document_name,
        start_line=block.start_line,
        heading=block.heading,
        first_non_empty_code_line=_first_non_empty_code_line(block.code),
    )


def _expected_stdout_identity(
    expected: _ExpectedStdoutExample,
) -> _StandaloneExampleIdentity:
    return _StandaloneExampleIdentity(
        document_name=expected.document_name,
        start_line=expected.start_line,
        heading=expected.heading,
        first_non_empty_code_line=expected.first_non_empty_code_line,
    )


def _expected_stdout_by_identity() -> dict[_StandaloneExampleIdentity, str]:
    expected_stdout_by_identity: dict[_StandaloneExampleIdentity, str] = {}
    for expected in _EXPECTED_STDOUT_EXAMPLES:
        identity = _expected_stdout_identity(expected)
        assert identity not in expected_stdout_by_identity
        assert expected.expected_stdout.strip()
        expected_stdout_by_identity[identity] = expected.expected_stdout
    return expected_stdout_by_identity


def _is_markdown_fence_opener(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("```")


def _is_unsupported_python_like_fence_token(token: str) -> bool:
    py_version_alias = token.startswith("py") and token.removeprefix("py").isdigit()
    return (
        token.startswith(("python", "py-", "py_"))
        or py_version_alias
        or token in _UNSUPPORTED_PYTHON_LIKE_FENCE_TOKENS
    )


def _is_python_fence_info(info: str, *, document_name: str) -> bool:
    first_token = _fence_language(info)
    if first_token in _SUPPORTED_PYTHON_FENCE_TOKENS:
        return True
    if _is_unsupported_python_like_fence_token(first_token):
        raise AssertionError(
            f"Unsupported {_document_label(document_name)} Python fence variant "
            f"`{info}`; use `python`, `python ...`, or `py` so examples "
            "are linted."
        )
    return False


def _explicit_fragment_start_lines(document_name: str) -> frozenset[int]:
    return _EXPLICIT_FRAGMENT_START_LINES_BY_DOCUMENT.get(document_name, frozenset())


def _markdown_fence_blocks(
    markdown: str,
    *,
    document_name: str = _README.name,
) -> list[_MarkdownFenceBlock]:
    lines = markdown.splitlines()
    blocks: list[_MarkdownFenceBlock] = []
    heading = f"<{_document_label(document_name)} top>"
    index = 0
    while index < len(lines):
        line = lines[index]
        current_heading = _heading_from_line(line)
        if current_heading is not None:
            heading = current_heading

        if not _is_markdown_fence_opener(line):
            index += 1
            continue

        fence_info = line.strip().removeprefix("```").strip()
        start_line = index + 2
        index += 1
        block_lines: list[str] = []
        while index < len(lines) and lines[index].strip() != "```":
            block_lines.append(lines[index])
            index += 1
        if index >= len(lines):
            raise AssertionError(
                f"Unterminated {_document_label(document_name)} code block at line {start_line}"
            )

        blocks.append(
            _MarkdownFenceBlock(
                document_name=document_name,
                heading=heading,
                code="\n".join(block_lines),
                start_line=start_line,
                fence_info=fence_info,
            )
        )
        index += 1
    return blocks


def _python_blocks(
    markdown: str,
    *,
    document_name: str = _README.name,
) -> list[_MarkdownPythonBlock]:
    blocks: list[_MarkdownPythonBlock] = []
    explicit_fragment_start_lines = _explicit_fragment_start_lines(document_name)
    for fence in _markdown_fence_blocks(markdown, document_name=document_name):
        if not _is_python_fence_info(fence.fence_info, document_name=document_name):
            continue
        blocks.append(
            _MarkdownPythonBlock(
                document_name=fence.document_name,
                heading=fence.heading,
                code=fence.code,
                start_line=fence.start_line,
                classified_as_fragment=(
                    _is_fragment_python_block(fence.code)
                    or fence.start_line in explicit_fragment_start_lines
                ),
            )
        )
    return blocks


def _standalone_python_blocks(
    markdown: str,
    *,
    document_name: str = _README.name,
) -> list[_MarkdownPythonBlock]:
    return [
        block
        for block in _python_blocks(markdown, document_name=document_name)
        if not block.classified_as_fragment
    ]


def _standalone_python_blocks_from_document(path: Path) -> list[_MarkdownPythonBlock]:
    return _standalone_python_blocks(
        path.read_text(encoding="utf-8"),
        document_name=path.name,
    )


def _non_python_fence_blocks_from_document(path: Path) -> list[_MarkdownFenceBlock]:
    return [
        block
        for block in _markdown_fence_blocks(
            path.read_text(encoding="utf-8"),
            document_name=path.name,
        )
        if not _is_python_fence_info(block.fence_info, document_name=path.name)
    ]


def _skip_if_optional_dependency_is_missing(example: _MarkdownPythonBlock) -> None:
    if example.document_name == "DEVELOPMENT.md":
        pytest.importorskip("redis")


def _run_docs_example(
    example: _MarkdownPythonBlock,
) -> subprocess.CompletedProcess[str]:
    _skip_if_optional_dependency_is_missing(example)
    return subprocess.run(  # noqa: S603 - executes trusted snippets from this repo's docs.
        [sys.executable, "-c", example.code],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def _assert_migration_fragment_expectations(
    blocks: list[_MarkdownPythonBlock],
) -> None:
    fragments_by_start_line = {
        block.start_line: block for block in blocks if block.classified_as_fragment
    }

    assert (
        frozenset(fragments_by_start_line) == _EXPECTED_MIGRATION_FRAGMENT_START_LINES
    )
    for expected in _EXPECTED_MIGRATION_FRAGMENTS:
        block = fragments_by_start_line[expected.start_line]
        actual = _ExpectedMigrationFragment(
            start_line=block.start_line,
            reason=expected.reason,
            heading=block.heading,
            first_non_empty_code_line=_first_non_empty_code_line(block.code),
        )
        assert actual == expected
        assert expected.reason.strip()


def _non_python_fence_key(block: _MarkdownFenceBlock) -> _NonPythonFenceKey:
    language = _fence_language(block.fence_info)
    non_empty_content_lines = _non_python_content_lines(
        block.code,
        language=language,
    )
    return _NonPythonFenceKey(
        document_name=block.document_name,
        start_line=block.start_line,
        language=language,
        heading=block.heading,
        first_non_empty_content_line=next(iter(non_empty_content_lines), ""),
        non_empty_content_lines=non_empty_content_lines,
    )


def _expected_non_python_fence_key(
    expected: _ExpectedNonPythonFence,
) -> _NonPythonFenceKey:
    non_empty_content_lines = _non_python_content_lines(
        "\n".join(expected.non_empty_content_lines),
        language=expected.language,
    )
    first_non_empty_content_line = next(iter(non_empty_content_lines), "")
    return _NonPythonFenceKey(
        document_name=expected.document_name,
        start_line=expected.start_line,
        language=expected.language,
        heading=expected.heading,
        first_non_empty_content_line=first_non_empty_content_line,
        non_empty_content_lines=non_empty_content_lines,
    )


def _non_python_fence_classification_from_key(key: _NonPythonFenceKey) -> str:
    if key in _MANUAL_NON_PYTHON_FENCE_KEYS:
        return _NON_PYTHON_CLASSIFICATION_MANUAL
    if key.language in _SHELL_LIKE_FENCE_LANGUAGES:
        return _NON_PYTHON_CLASSIFICATION_SHELL_SYNTAX
    return _NON_PYTHON_CLASSIFICATION_TEXT


def _expected_non_python_fence_identity(
    expected: _ExpectedNonPythonFence,
) -> _NonPythonFenceIdentity:
    key = _expected_non_python_fence_key(expected)
    return _NonPythonFenceIdentity(
        document_name=key.document_name,
        start_line=key.start_line,
        language=key.language,
        classification=expected.classification,
        heading=key.heading,
        first_non_empty_content_line=key.first_non_empty_content_line,
        non_empty_content_lines=key.non_empty_content_lines,
    )


def _current_non_python_fence_identity(
    block: _MarkdownFenceBlock,
) -> _NonPythonFenceIdentity:
    key = _non_python_fence_key(block)
    return _NonPythonFenceIdentity(
        document_name=key.document_name,
        start_line=key.start_line,
        language=key.language,
        classification=_non_python_fence_classification_from_key(key),
        heading=key.heading,
        first_non_empty_content_line=key.first_non_empty_content_line,
        non_empty_content_lines=key.non_empty_content_lines,
    )


def _assert_manual_non_python_fence_keys_are_current(
    *,
    manual_keys: frozenset[_NonPythonFenceKey],
    current_keys: frozenset[_NonPythonFenceKey],
    expected_keys: frozenset[_NonPythonFenceKey],
) -> None:
    stale_current_keys = manual_keys - current_keys
    assert not stale_current_keys, (
        f"Manual non-Python fence keys are not current fences: {stale_current_keys!r}"
    )

    stale_expected_keys = manual_keys - expected_keys
    assert not stale_expected_keys, (
        f"Manual non-Python fence keys are not expected fences: {stale_expected_keys!r}"
    )


def _expected_markdown_fence_block(
    expected: _ExpectedNonPythonFence,
) -> _MarkdownFenceBlock:
    return _MarkdownFenceBlock(
        document_name=expected.document_name,
        heading=expected.heading,
        code="\n".join(expected.non_empty_content_lines),
        start_line=expected.start_line,
        fence_info=expected.language,
    )


_STANDALONE_README_EXAMPLES = _standalone_python_blocks(
    _README.read_text(encoding="utf-8"),
    document_name=_README.name,
)
_STANDALONE_DOC_EXAMPLES = [
    block
    for path in _DOCS_WITH_STANDALONE_EXAMPLES
    for block in _standalone_python_blocks_from_document(path)
]
_MIGRATION_PYTHON_BLOCKS = _python_blocks(
    _MIGRATION.read_text(encoding="utf-8"),
    document_name=_MIGRATION.name,
)
_NON_PYTHON_FENCE_BLOCKS = [
    block
    for path in _DOCS_WITH_STANDALONE_EXAMPLES
    for block in _non_python_fence_blocks_from_document(path)
]


def test_readme_has_standalone_python_examples() -> None:
    assert _STANDALONE_README_EXAMPLES


@pytest.mark.parametrize(
    "fence_info",
    [
        "python",
        'python title="example.py" linenums="1"',
        "py",
    ],
)
def test_standalone_python_blocks_accept_supported_fence_variants(
    fence_info: str,
) -> None:
    markdown = f"""
# Demo

```{fence_info}
print("exact")
```
"""

    blocks = _standalone_python_blocks(markdown)

    assert [(block.heading, block.code) for block in blocks] == [
        ("# Demo", 'print("exact")')
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


@pytest.mark.parametrize(
    "fence_info",
    [
        "python3",
        "python-console",
        "python_repl",
        "pythonish",
        "py2",
        "py3",
        "py10",
        "py-repl",
        "py_module",
        "pycon",
        "pycon3",
        "ipython",
        "ipython3",
        "pypy",
    ],
)
def test_standalone_python_blocks_reject_unsupported_python_like_fences(
    fence_info: str,
) -> None:
    markdown = f"""
# Demo

```{fence_info}
print("not linted")
```
"""

    with pytest.raises(AssertionError) as error:
        _standalone_python_blocks(markdown)
    assert "Unsupported README Python fence variant" in str(error.value)
    assert f"`{fence_info}`" in str(error.value)
    assert "use `python`, `python ...`, or `py` so examples are linted" in str(
        error.value
    )


def test_standalone_example_identity_catches_drift() -> None:
    expected = _EXPECTED_STDOUT_EXAMPLES[0]
    matching_block = _MarkdownPythonBlock(
        document_name=expected.document_name,
        heading=expected.heading,
        code=expected.first_non_empty_code_line,
        start_line=expected.start_line,
        classified_as_fragment=False,
    )
    expected_identity = _expected_stdout_identity(expected)

    assert _standalone_example_identity(matching_block) == expected_identity
    assert (
        _standalone_example_identity(
            _MarkdownPythonBlock(
                document_name=expected.document_name,
                heading=f"{expected.heading} extra",
                code=expected.first_non_empty_code_line,
                start_line=expected.start_line,
                classified_as_fragment=False,
            )
        )
        != expected_identity
    )
    assert (
        _standalone_example_identity(
            _MarkdownPythonBlock(
                document_name=expected.document_name,
                heading=expected.heading,
                code=expected.first_non_empty_code_line,
                start_line=expected.start_line + 1,
                classified_as_fragment=False,
            )
        )
        != expected_identity
    )
    assert (
        _standalone_example_identity(
            _MarkdownPythonBlock(
                document_name=expected.document_name,
                heading=expected.heading,
                code='print("changed")',
                start_line=expected.start_line,
                classified_as_fragment=False,
            )
        )
        != expected_identity
    )


def test_non_readme_standalone_python_examples_are_linted() -> None:
    current_identities = [
        _standalone_example_identity(block)
        for block in _STANDALONE_DOC_EXAMPLES
        if block.document_name != _README.name
    ]
    expected_identities = frozenset(_EXPECTED_NON_README_STANDALONE_IDENTITIES)

    assert len(expected_identities) == len(_EXPECTED_NON_README_STANDALONE_IDENTITIES)
    assert set(current_identities) == expected_identities
    for expected in _EXPECTED_NON_README_STANDALONE_IDENTITIES:
        assert current_identities.count(expected) == 1


def test_shell_like_non_python_classification_is_not_derived_from_expected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _EXPECTED_NON_PYTHON_FENCES[0]
    assert expected.language in _SHELL_LIKE_FENCE_LANGUAGES
    downgraded_expected = replace(
        expected,
        classification=_NON_PYTHON_CLASSIFICATION_TEXT,
        reason="incorrectly treats a shell command as text",
    )
    block = _expected_markdown_fence_block(expected)

    monkeypatch.setattr(
        sys.modules[__name__],
        "_EXPECTED_NON_PYTHON_FENCES",
        (downgraded_expected,),
    )

    current_identity = _current_non_python_fence_identity(block)
    assert current_identity.classification == _NON_PYTHON_CLASSIFICATION_SHELL_SYNTAX
    assert current_identity != _expected_non_python_fence_identity(downgraded_expected)


def test_non_python_fence_identity_includes_later_content_lines() -> None:
    expected = _EXPECTED_NON_PYTHON_FENCES[0]
    drifted_content_lines = list(expected.non_empty_content_lines)
    drifted_content_lines[1] = drifted_content_lines[1].replace(
        "[redis]",
        "[postgres]",
        1,
    )
    block = _MarkdownFenceBlock(
        document_name=expected.document_name,
        heading=expected.heading,
        code="\n".join(drifted_content_lines),
        start_line=expected.start_line,
        fence_info=expected.language,
    )

    current_identity = _current_non_python_fence_identity(block)
    expected_identity = _expected_non_python_fence_identity(expected)

    assert (
        current_identity.first_non_empty_content_line
        == expected_identity.first_non_empty_content_line
    )
    assert current_identity != expected_identity


def test_shell_like_non_python_identity_ignores_inline_comment_alignment() -> None:
    expected = _EXPECTED_NON_PYTHON_FENCES[0]
    realigned_content_lines = (
        'pip install "token-throttle[redis,tiktoken]>=8.0.6,<8.1.0" # OpenAI + Redis (recommended)',
        'pip install "token-throttle[redis]>=8.0.6,<8.1.0" # Any provider + Redis',
        'pip install "token-throttle>=8.0.6,<8.1.0" # Any provider + in-memory',
    )
    block = _MarkdownFenceBlock(
        document_name=expected.document_name,
        heading=expected.heading,
        code="\n".join(realigned_content_lines),
        start_line=expected.start_line,
        fence_info=expected.language,
    )

    assert _current_non_python_fence_identity(
        block
    ) == _expected_non_python_fence_identity(expected)


def test_shell_like_non_python_identity_keeps_inline_comment_text_guarded() -> None:
    expected = _EXPECTED_NON_PYTHON_FENCES[0]
    drifted_content_lines = list(expected.non_empty_content_lines)
    drifted_content_lines[1] = drifted_content_lines[1].replace(
        "Any provider + Redis",
        "Any provider + PostgreSQL",
        1,
    )
    block = _MarkdownFenceBlock(
        document_name=expected.document_name,
        heading=expected.heading,
        code="\n".join(drifted_content_lines),
        start_line=expected.start_line,
        fence_info=expected.language,
    )

    assert _current_non_python_fence_identity(
        block
    ) != _expected_non_python_fence_identity(expected)


def test_manual_non_python_fence_allowlist_rejects_stale_keys() -> None:
    expected_keys = frozenset(
        _expected_non_python_fence_key(expected)
        for expected in _EXPECTED_NON_PYTHON_FENCES
    )
    stale_key = replace(
        _expected_non_python_fence_key(_EXPECTED_NON_PYTHON_FENCES[0]),
        start_line=9999,
    )

    with pytest.raises(AssertionError, match="not current fences"):
        _assert_manual_non_python_fence_keys_are_current(
            manual_keys=frozenset({stale_key}),
            current_keys=expected_keys,
            expected_keys=expected_keys,
        )

    with pytest.raises(AssertionError, match="not expected fences"):
        _assert_manual_non_python_fence_keys_are_current(
            manual_keys=frozenset({stale_key}),
            current_keys=frozenset({*expected_keys, stale_key}),
            expected_keys=expected_keys,
        )


def test_non_python_fences_are_inventoried_and_classified() -> None:
    current_keys = frozenset(
        _non_python_fence_key(block) for block in _NON_PYTHON_FENCE_BLOCKS
    )
    expected_keys = frozenset(
        _expected_non_python_fence_key(expected)
        for expected in _EXPECTED_NON_PYTHON_FENCES
    )
    _assert_manual_non_python_fence_keys_are_current(
        manual_keys=_MANUAL_NON_PYTHON_FENCE_KEYS,
        current_keys=current_keys,
        expected_keys=expected_keys,
    )

    current_identities = [
        _current_non_python_fence_identity(block) for block in _NON_PYTHON_FENCE_BLOCKS
    ]
    expected_identities = frozenset(
        _expected_non_python_fence_identity(expected)
        for expected in _EXPECTED_NON_PYTHON_FENCES
    )

    assert len(expected_identities) == len(_EXPECTED_NON_PYTHON_FENCES)
    assert set(current_identities) == expected_identities
    for expected in _EXPECTED_NON_PYTHON_FENCES:
        assert expected.classification in _NON_PYTHON_FENCE_CLASSIFICATIONS
        assert expected.language != _UNLABELED_FENCE_LANGUAGE
        assert expected.non_empty_content_lines
        assert expected.classification == _non_python_fence_classification_from_key(
            _expected_non_python_fence_key(expected)
        )
        assert expected.reason.strip()
        assert (
            current_identities.count(_expected_non_python_fence_identity(expected)) == 1
        )


def test_shell_like_non_python_fences_are_bash_syntax_checked() -> None:
    bash_path = shutil.which("bash")
    if bash_path is None:
        pytest.skip("bash is unavailable")

    syntax_checked_keys = {
        _expected_non_python_fence_key(expected)
        for expected in _EXPECTED_NON_PYTHON_FENCES
        if expected.classification == _NON_PYTHON_CLASSIFICATION_SHELL_SYNTAX
    }
    assert syntax_checked_keys

    checked_keys: set[_NonPythonFenceKey] = set()
    for block in _NON_PYTHON_FENCE_BLOCKS:
        key = _non_python_fence_key(block)
        if key not in syntax_checked_keys:
            continue

        assert key.language in _SHELL_LIKE_FENCE_LANGUAGES
        result = subprocess.run(  # noqa: S603 - parses trusted docs snippets without executing them.
            [bash_path, "-n"],
            input=block.code,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"{block.document_name}:{block.start_line} failed bash -n:\n{result.stderr}"
        )
        checked_keys.add(key)

    assert checked_keys == syntax_checked_keys


def test_migration_fragments_are_classified_intentionally() -> None:
    _assert_migration_fragment_expectations(_MIGRATION_PYTHON_BLOCKS)


def test_migration_redis_key_shapes_are_guarded() -> None:
    migration = _MIGRATION.read_text(encoding="utf-8")

    assert _EXPECTED_MIGRATION_REDIS_CLEANUP_SCAN_PATTERN in migration
    assert _EXPECTED_MIGRATION_REDIS_BUCKET_KEY_SHAPE in migration
    assert _EXPECTED_MIGRATION_REDIS_ACQUIRED_KEY_SHAPE in migration


def test_expected_stdout_examples_match_current_docs_identity() -> None:
    current_identities = [
        _standalone_example_identity(block) for block in _STANDALONE_DOC_EXAMPLES
    ]
    expected_stdout_by_identity = _expected_stdout_by_identity()

    assert len(expected_stdout_by_identity) == len(_EXPECTED_STDOUT_EXAMPLES)
    for expected in _EXPECTED_STDOUT_EXAMPLES:
        identity = _expected_stdout_identity(expected)

        assert current_identities.count(identity) == 1
        assert expected_stdout_by_identity[identity] == expected.expected_stdout


@pytest.mark.parametrize(
    "example",
    _STANDALONE_DOC_EXAMPLES,
    ids=lambda example: (
        f"{example.document_name} {example.heading} line {example.start_line}"
    ),
)
def test_standalone_python_examples_run(example: _MarkdownPythonBlock) -> None:
    result = _run_docs_example(example)

    expected_stdout_by_identity = _expected_stdout_by_identity()
    identity = _standalone_example_identity(example)
    if identity in expected_stdout_by_identity:
        assert result.stdout.strip() == expected_stdout_by_identity[identity]
