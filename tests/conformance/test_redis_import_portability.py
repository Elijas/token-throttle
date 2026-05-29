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

The set of redis-gated modules is auto-discovered package-wide (not hand-listed,
not limited to the redis backend directory) so the guard covers redis-gated code
anywhere in ``token_throttle`` — including the OpenAI redis factories under
``_factories/_openai/`` — and cannot drift behind newly-added redis-gated
modules.
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
_PKG_NAME = token_throttle.__name__
_PKG_DIR = _REPO_ROOT / _PKG_NAME


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


def _module_dotted_path(path: pathlib.Path) -> str:
    """Dotted module path for a ``.py`` file inside the package, e.g.
    ``.../token_throttle/_factories/_openai/_x.py`` ->
    ``token_throttle._factories._openai._x``.
    """
    parts = path.relative_to(_REPO_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _redis_gated_modules() -> set[str]:
    """Auto-discovered set of ``token_throttle`` module paths that import the
    ``redis`` package at import time and so cannot be imported without the
    ``redis`` extra. Scans the whole package (not just the redis backend
    directory) so redis-gated code anywhere — e.g. the OpenAI redis factories
    under ``_factories/_openai/`` — is covered and the guard cannot drift behind
    newly-added redis-gated modules.
    """
    return {
        _module_dotted_path(path)
        for path in _PKG_DIR.rglob("*.py")
        if _module_is_redis_gated(path)
    }


def _imports_redis_gated_backend(node: ast.stmt, gated: set[str]) -> bool:
    """True if ``node`` imports a redis-gated module — covering: the module
    itself (``import pkg.mod`` / ``from pkg.mod import x``), the module as a
    submodule (``from pkg import mod``), and a redis-requiring public re-export
    off the top-level package (``from token_throttle import RedisBackend``),
    which resolves to a gated module lazily via ``__getattr__``.
    """
    if isinstance(node, ast.Import):
        return any(alias.name in gated for alias in node.names)
    if not isinstance(node, ast.ImportFrom):
        return False

    module = node.module or ""
    if module == _PKG_NAME:
        return any(alias.name in token_throttle._REDIS_ALL for alias in node.names)
    if module in gated:
        return True
    return any(f"{module}.{alias.name}" in gated for alias in node.names)


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


def _offending_redis_import_linenos(tree: ast.Module, gated: set[str]) -> list[int]:
    """Line numbers of module-level redis / redis-gated imports with no
    preceding ``pytest.importorskip("redis")`` — exactly the imports that break
    ``pytest`` collection without the ``redis`` extra. Only the module body is
    scanned; function-local imports are safe.
    """
    offenders: list[int] = []
    seen_importorskip = False
    for node in tree.body:
        if _is_redis_importorskip(node):
            seen_importorskip = True
            continue
        lineno = _first_redis_optional_import_lineno(node, gated)
        if lineno is not None and not seen_importorskip:
            offenders.append(lineno)
    return offenders


def test_redis_gated_module_set_is_non_empty() -> None:
    """Sanity check: the auto-discovery found redis-gated modules. A future
    refactor that empties this set would silently disarm the guard below.
    """
    gated = _redis_gated_modules()
    assert gated, (
        f"Auto-discovery found no redis-gated modules under {_PKG_NAME}/ — "
        "the discovery heuristic has broken"
    )


def test_redis_gated_discovery_covers_openai_factories() -> None:
    """Regression for the H2-F1 gap: discovery scoped to
    ``_limiter_backends/_redis/`` missed the redis-gated OpenAI factory modules
    under ``_factories/_openai/``. Package-wide discovery must cover both, so a
    future module-level import of them in a test cannot slip past the guard.
    """
    gated = _redis_gated_modules()
    expected = {
        "token_throttle._factories._openai._openai_rate_limiter",
        "token_throttle._factories._openai._openai_sync_rate_limiter",
    }
    missing = expected - gated
    assert not missing, (
        "Redis-gated discovery must cover the OpenAI factory modules; missing: "
        f"{sorted(missing)}. Discovered: {sorted(gated)}"
    )


def test_tests_collect_without_redis_extra() -> None:
    gated = _redis_gated_modules()
    offenders: list[str] = []
    for path in _scanned_test_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders.extend(
            f"{path.relative_to(_REPO_ROOT)}:{lineno}"
            for lineno in _offending_redis_import_linenos(tree, gated)
        )
    assert not offenders, (
        "Module-level import of redis or a redis-gated backend module without a "
        'preceding `pytest.importorskip("redis")` — this breaks '
        "`pytest` collection in no-extras environments. Move the import inside "
        "the test function, or importorskip at module top before the import. "
        f"Offenders: {offenders}"
    )


def test_guard_flags_unguarded_openai_factory_import() -> None:
    """Synthetic offender (in-memory AST, so no real file is left to break
    collection): a module-level import of a redis-gated OpenAI factory with no
    preceding ``pytest.importorskip("redis")`` must be flagged, while the same
    import guarded by ``importorskip`` must be accepted.
    """
    gated = _redis_gated_modules()
    factory = "token_throttle._factories._openai._openai_rate_limiter"
    unguarded = ast.parse(f"from {factory} import create_openai_redis_rate_limiter\n")
    guarded = ast.parse(
        f'import pytest\npytest.importorskip("redis")\nfrom {factory} import create_openai_redis_rate_limiter\n'
    )
    assert _offending_redis_import_linenos(unguarded, gated) == [1]
    assert _offending_redis_import_linenos(guarded, gated) == []


def test_guard_flags_unguarded_public_redis_symbol_import() -> None:
    """``from token_throttle import RedisBackend`` re-exports a redis-gated
    symbol resolved lazily via ``__getattr__``; at module level without
    ``importorskip`` it also breaks no-redis collection, so the guard must flag
    it. A redis-independent public symbol must NOT be flagged.
    """
    gated = _redis_gated_modules()
    redis_symbol = ast.parse("from token_throttle import RedisBackend\n")
    safe_symbol = ast.parse("from token_throttle import OpenAIUsageCounter\n")
    assert _offending_redis_import_linenos(redis_symbol, gated) == [1]
    assert _offending_redis_import_linenos(safe_symbol, gated) == []
