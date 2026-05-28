"""Manual OpenAI tests must collect cleanly without the OpenAI SDK.

Manual tests are often run from local environments, and ``openai`` is a dev
dependency rather than a runtime or project extra. Importing OpenAI SDK modules
at collection time without a preceding ``pytest.importorskip("openai")`` turns
an optional manual test into a hard collection failure.
"""

from __future__ import annotations

import ast
import pathlib

import token_throttle

_REPO_ROOT = pathlib.Path(token_throttle.__file__).parent.parent
_MANUAL_TEST_DIR = _REPO_ROOT / "tests" / "manual"


def _node_imports_openai(node: ast.stmt) -> bool:
    """True if ``node`` is an ``import openai``/``from openai ...`` statement."""
    if isinstance(node, ast.Import):
        return any(
            alias.name == "openai" or alias.name.startswith("openai.")
            for alias in node.names
        )
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return module == "openai" or module.startswith("openai.")
    return False


def _is_openai_importorskip(node: ast.stmt) -> bool:
    """True for ``pytest.importorskip("openai"...)``, bare or assigned."""
    call: ast.Call | None = None
    if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)) or (
        isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
    ):
        call = node.value
    if call is None:
        return False
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == "importorskip"):
        return False
    if not call.args:
        return False
    first = call.args[0]
    return (
        isinstance(first, ast.Constant)
        and isinstance(first.value, str)
        and (first.value == "openai" or first.value.startswith("openai."))
    )


def _manual_test_files() -> list[pathlib.Path]:
    return sorted(_MANUAL_TEST_DIR.rglob("test_*.py"))


def test_manual_openai_tests_collect_without_openai_sdk() -> None:
    offenders: list[str] = []
    for path in _manual_test_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        seen_importorskip = False
        # Module level only -- function-local imports collect safely.
        for node in tree.body:
            if _is_openai_importorskip(node):
                seen_importorskip = True
            elif _node_imports_openai(node) and not seen_importorskip:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
    assert not offenders, (
        "Module-level import of openai without a preceding "
        '`pytest.importorskip("openai")` -- this breaks pytest collection in '
        "no-OpenAI environments. Move the import inside the test function, or "
        "importorskip at module top before the import. "
        f"Offenders: {offenders}"
    )
