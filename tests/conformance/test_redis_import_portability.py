"""Redis-dependent tests must collect cleanly without the ``redis`` extra.

CI runs ``pytest tests/unit`` in no-extras tiers (``test-unit-core``,
``test-min-deps``) to verify the optional-dependency boundary. A test that
imports a redis-gated backend module at module level fails *collection* there —
before any test-level skip can fire — because those modules eagerly
``import redis``.

The fix is to import the redis-gated symbol inside the test function (after a
``pytest.importorskip``) or to ``pytest.importorskip("redis")`` at module top
before the import. ``tests/integration`` and ``tests/property`` are included
because they are often run from developer environments that do not install the
Redis extra. This guard makes the requirement structural: instances #6 and #8
of the recurring "X lags Y" archetype were both this exact bug, each caught
only at release time with an opaque collection ImportError.

The set of redis-gated modules is auto-discovered (not hand-listed) so this
guard cannot itself drift behind the backend package.
"""

from __future__ import annotations

import ast
import pathlib

import token_throttle

_REPO_ROOT = pathlib.Path(token_throttle.__file__).parent.parent
_TEST_DIRS = (
    _REPO_ROOT / "tests" / "unit",
    _REPO_ROOT / "tests" / "integration",
    _REPO_ROOT / "tests" / "property",
)
_REDIS_PKG_DIR = _REPO_ROOT / "token_throttle" / "_limiter_backends" / "_redis"
_REDIS_PKG = "token_throttle._limiter_backends._redis"


def _node_imports_redis_pkg(node: ast.stmt) -> bool:
    """True if ``node`` is an ``import redis``/``from redis ...`` statement."""
    if isinstance(node, ast.Import):
        return any(
            alias.name == "redis" or alias.name.startswith("redis.")
            for alias in node.names
        )
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return module == "redis" or module.startswith("redis.")
    return False


def _module_is_redis_gated(path: pathlib.Path) -> bool:
    """A module is redis-gated if it imports the ``redis`` package at runtime —
    directly at module level or inside a module-level ``try`` (the descriptive-
    ImportError pattern). ``if TYPE_CHECKING:`` guards do not count.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if _node_imports_redis_pkg(node):
            return True
        if isinstance(node, ast.Try) and any(
            _node_imports_redis_pkg(child) for child in node.body
        ):
            return True
    return False


def _redis_gated_modules() -> set[str]:
    """Auto-discovered set of ``token_throttle..._redis.*`` module paths that
    cannot be imported without the ``redis`` package.
    """
    return {
        f"{_REDIS_PKG}.{path.stem}"
        for path in _REDIS_PKG_DIR.glob("*.py")
        if path.stem != "__init__" and _module_is_redis_gated(path)
    }


def _imports_redis_gated_backend(node: ast.stmt, gated: set[str]) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name in gated for alias in node.names)
    if not isinstance(node, ast.ImportFrom):
        return False

    module = node.module or ""
    if module in gated:
        return True
    if module != _REDIS_PKG:
        return False
    return any(f"{_REDIS_PKG}.{alias.name}" in gated for alias in node.names)


def _first_redis_optional_import_lineno(node: ast.stmt, gated: set[str]) -> int | None:
    if _node_imports_redis_pkg(node) or _imports_redis_gated_backend(node, gated):
        return node.lineno
    return None


def _is_redis_importorskip(node: ast.stmt) -> bool:
    """True for ``pytest.importorskip("redis"...)``, bare or assigned."""
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
        and first.value.startswith("redis")
    )


def _scanned_test_files() -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for test_dir in _TEST_DIRS:
        paths.extend(test_dir.rglob("test_*.py"))
        paths.extend(test_dir.rglob("conftest.py"))
    return sorted(set(paths))


def test_redis_gated_module_set_is_non_empty() -> None:
    """Sanity check: the auto-discovery found redis-gated modules. A future
    refactor that empties this set would silently disarm the guard below.
    """
    gated = _redis_gated_modules()
    assert gated, (
        "Auto-discovery found no redis-gated modules under "
        f"{_REDIS_PKG} — the discovery heuristic has broken"
    )


def test_tests_collect_without_redis_extra() -> None:
    gated = _redis_gated_modules()
    offenders: list[str] = []
    for path in _scanned_test_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        seen_importorskip = False
        for node in tree.body:  # module level only — function-local imports are safe
            if _is_redis_importorskip(node):
                seen_importorskip = True
            else:
                lineno = _first_redis_optional_import_lineno(node, gated)
                if lineno is not None and not seen_importorskip:
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}")
    assert not offenders, (
        "Module-level import of redis or a redis-gated backend module without a "
        'preceding `pytest.importorskip("redis")` — this breaks '
        "`pytest` collection in no-extras environments. Move the import inside "
        "the test function, or importorskip at module top before the import. "
        f"Offenders: {offenders}"
    )
