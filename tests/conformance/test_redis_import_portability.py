"""``tests/unit/`` must collect cleanly without the ``redis`` optional extra.

CI runs ``pytest tests/unit`` in no-extras tiers (``test-unit-core``,
``test-min-deps``) to verify the optional-dependency boundary. A test that
imports a redis-gated backend module at module level fails *collection* there —
before any test-level skip can fire — because those modules eagerly
``import redis``.

The fix is to import the redis-gated symbol inside the test function (after a
``pytest.importorskip``) or to ``pytest.importorskip("redis")`` at module top
before the import. This guard makes the requirement structural: instances #6
and #8 of the recurring "X lags Y" archetype were both this exact bug, each
caught only at release time with an opaque collection ImportError.

The set of redis-gated modules is auto-discovered (not hand-listed) so this
guard cannot itself drift behind the backend package.
"""

from __future__ import annotations

import ast
import pathlib

import token_throttle

_REPO_ROOT = pathlib.Path(token_throttle.__file__).parent.parent
_UNIT_TEST_DIR = _REPO_ROOT / "tests" / "unit"
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


def _imports_any(node: ast.stmt, modules: set[str]) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.module in modules
    if isinstance(node, ast.Import):
        return any(alias.name in modules for alias in node.names)
    return False


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


def test_redis_gated_module_set_is_non_empty() -> None:
    """Sanity check: the auto-discovery found redis-gated modules. A future
    refactor that empties this set would silently disarm the guard below.
    """
    gated = _redis_gated_modules()
    assert gated, (
        "Auto-discovery found no redis-gated modules under "
        f"{_REDIS_PKG} — the discovery heuristic has broken"
    )


def test_unit_tests_collect_without_redis_extra() -> None:
    gated = _redis_gated_modules()
    offenders: list[str] = []
    for path in sorted(_UNIT_TEST_DIR.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        seen_importorskip = False
        for node in tree.body:  # module level only — function-local imports are safe
            if _is_redis_importorskip(node):
                seen_importorskip = True
            elif _imports_any(node, gated) and not seen_importorskip:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
    assert not offenders, (
        "Module-level import of a redis-gated backend module without a "
        'preceding `pytest.importorskip("redis")` — this breaks '
        "`pytest tests/unit` collection in no-extras CI tiers. Move the import "
        "inside the test function, or importorskip at module top before the "
        f"import. Offenders: {offenders}"
    )
