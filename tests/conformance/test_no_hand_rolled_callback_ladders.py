"""
Conformance check: production code must not hand-roll the critical-exception
ladder or the in-flight-reservation cleanup wrapper.

Archetype A (callback dispatch ladder) is centralised in
``token_throttle/_interfaces/_callbacks.py`` (constants + ``safe_invoke_*``
helpers + ``_exception_group_contains_critical``).

Archetype B (forget/refund on raise) is centralised in
``token_throttle/_rate_limiter.py`` and
``token_throttle/_sync_rate_limiter.py`` as
``_refund_or_forget_reservation_on_raise`` context managers.

This test scans the production tree for the syntactic signatures of the
former hand-rolled patterns and asserts they don't reappear. If a future
contributor copies an old ladder back, this test fails before the bug
recurs.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import textwrap
from dataclasses import dataclass

import pytest

import token_throttle

PRODUCTION_ROOT = pathlib.Path(token_throttle.__file__).parent
CALLBACKS_FILE = PRODUCTION_ROOT / "_interfaces" / "_callbacks.py"
RATE_LIMITER_FILE = PRODUCTION_ROOT / "_rate_limiter.py"
SYNC_RATE_LIMITER_FILE = PRODUCTION_ROOT / "_sync_rate_limiter.py"
MEMORY_BACKEND_FILE = PRODUCTION_ROOT / "_limiter_backends" / "_memory" / "_backend.py"
MEMORY_SYNC_BACKEND_FILE = (
    PRODUCTION_ROOT / "_limiter_backends" / "_memory" / "_sync_backend.py"
)
REDIS_BACKEND_FILE = PRODUCTION_ROOT / "_limiter_backends" / "_redis" / "_backend.py"
REDIS_SYNC_BACKEND_FILE = (
    PRODUCTION_ROOT / "_limiter_backends" / "_redis" / "_sync_backend.py"
)
AST_GUARD_SKIP_MARKER = "ast-guard: skip"
CANCELLED_ERROR_NAMES = {
    "asyncio.CancelledError",
    "concurrent.futures.CancelledError",
}
CRITICAL_EXCEPTION_LITERAL_NAMES = {
    "asyncio.CancelledError",
    "concurrent.futures.CancelledError",
    "KeyboardInterrupt",
    "SystemExit",
    "GeneratorExit",
    "MemoryError",
    "RecursionError",
}
CRITICAL_EXCEPTION_TUPLE_NAMES = {
    "BACKEND_CALLBACK_CRITICAL_EXCEPTIONS",
    "LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS",
}
KNOWN_CRITICAL_DISPATCH_HELPERS = {
    "_exception_group_contains_critical",
    "_raise_backend_external_error",
    "safe_invoke_async_callback",
    "safe_invoke_sync_callback",
}


@dataclass(frozen=True, order=True)
class _DiscoveredCancellationSite:
    relative_path: str
    function_name: str
    except_lineno: int

    def label(self) -> str:
        return f"{self.relative_path}:{self.function_name}:{self.except_lineno}"

    def key(self) -> tuple[str, str]:
        return (self.relative_path, self.function_name)


@dataclass(frozen=True)
class _CancellationSiteDetail:
    path: pathlib.Path
    site: _DiscoveredCancellationSite
    try_node: ast.Try
    cancelled_handler: ast.ExceptHandler


@dataclass(frozen=True)
class _CleanupSite:
    relative_path: str
    function_name: str
    cleanup_call: str

    @property
    def path(self) -> pathlib.Path:
        return PRODUCTION_ROOT / self.relative_path

    def key(self) -> tuple[str, str]:
        return (self.relative_path, self.function_name)


# Registered sites are an explicit acknowledgement that the discovered
# cancellation-composition function is load-bearing and needs BaseException
# cleanup. Discovery, not this list, decides which functions must be reviewed.
REGISTERED_CANCELLATION_CLEANUP_SITES = (
    _CleanupSite(
        "_rate_limiter.py",
        "_acquire_capacity",
        "_rollback_pending_acquire",
    ),
    _CleanupSite(
        "_rate_limiter.py",
        "_backend_task_succeeded_after_cancel",
        "exception",
    ),
    _CleanupSite(
        "_rate_limiter.py",
        "_wait_for_set_max_capacity_task_while_cancelled",
        "exception",
    ),
    _CleanupSite(
        "_rate_limiter.py",
        "_set_max_capacity_transactional",
        "_reconcile_runtime_max_capacity_after_failed_set",
    ),
    _CleanupSite(
        "_limiter_backends/_redis/_backend.py",
        "_check_and_consume_capacity",
        "_refund_cancelled_consumption",
    ),
    _CleanupSite(
        "_limiter_backends/_redis/_backend.py",
        "await_for_capacity",
        "_refund_cancelled_consumption",
    ),
)

ADDITIONAL_BASE_EXCEPTION_CLEANUP_SITES = (
    # Async memory backend coverage predates the CancelledError-specific AST
    # pass and remains load-bearing even though discovery does not find it.
    _CleanupSite(
        "_limiter_backends/_memory/_backend.py",
        "await_for_capacity",
        "_refund_cancelled_consumption",
    ),
    # Sync siblings use BaseException for interrupt-safe cleanup, not
    # CancelledError, so they are covered explicitly here.
    _CleanupSite(
        "_sync_rate_limiter.py",
        "_acquire_capacity",
        "_rollback_pending_acquire",
    ),
    _CleanupSite(
        "_sync_rate_limiter.py",
        "_set_max_capacity_transactional",
        "_reconcile_runtime_max_capacity_after_failed_set",
    ),
    _CleanupSite(
        "_limiter_backends/_memory/_sync_backend.py",
        "wait_for_capacity",
        "_refund_cancelled_consumption",
    ),
    _CleanupSite(
        "_limiter_backends/_redis/_sync_backend.py",
        "_check_and_consume_capacity",
        "_refund_cancelled_consumption",
    ),
    _CleanupSite(
        "_limiter_backends/_redis/_sync_backend.py",
        "wait_for_capacity",
        "_refund_cancelled_consumption",
    ),
)

REGISTERED_BASE_EXCEPTION_CLEANUP_SITES = (
    *REGISTERED_CANCELLATION_CLEANUP_SITES,
    *ADDITIONAL_BASE_EXCEPTION_CLEANUP_SITES,
)


def _attribute_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attribute_name(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _exception_type_names(node: ast.AST | None) -> tuple[str, ...]:
    if node is None:
        return ()
    if isinstance(node, ast.Tuple):
        names: list[str] = []
        for elt in node.elts:
            names.extend(_exception_type_names(elt))
        return tuple(names)
    name = _attribute_name(node)
    return () if name is None else (name,)


def _literal_names_in_assignment(path: pathlib.Path, assignment_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == assignment_name
            for target in targets
        ):
            continue
        value = node.value
        if not isinstance(value, ast.Tuple):
            return set()
        names: set[str] = set()
        for element in value.elts:
            if isinstance(element, ast.Starred):
                continue
            names.update(_exception_type_names(element))
        return names
    raise AssertionError(f"{assignment_name} not found in {path}")


def _handler_catches_base_exception(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    return "BaseException" in _exception_type_names(handler.type)


def _handler_catches_exception(handler: ast.ExceptHandler) -> bool:
    return "Exception" in _exception_type_names(handler.type)


def _handler_catches_cancelled_error(handler: ast.ExceptHandler) -> bool:
    return bool(CANCELLED_ERROR_NAMES.intersection(_exception_type_names(handler.type)))


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _handler_body_calls(handler: ast.ExceptHandler, function_name: str) -> bool:
    for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) == function_name:
            return True
    return False


def _function_has_base_exception_cleanup(
    path: pathlib.Path,
    function_name: str,
    cleanup_call: str,
) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
            continue
        if node.name != function_name:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.ExceptHandler):
                continue
            if not _handler_catches_base_exception(child):
                continue
            if _handler_body_calls(child, cleanup_call):
                return True
    return False


def _has_ast_guard_skip_marker(lines: list[str], lineno: int) -> bool:
    line_indexes = (lineno - 1, lineno - 2)
    return any(
        0 <= index < len(lines) and AST_GUARD_SKIP_MARKER in lines[index]
        for index in line_indexes
    )


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_function(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.AsyncFunctionDef | ast.FunctionDef | None:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, (ast.AsyncFunctionDef, ast.FunctionDef)):
            return current
        current = parents.get(current)
    return None


def _relative_production_path(path: pathlib.Path) -> str:
    return path.relative_to(PRODUCTION_ROOT).as_posix()


def _iter_production_python_files() -> list[pathlib.Path]:
    return sorted(
        (p for p in PRODUCTION_ROOT.rglob("*.py") if "__pycache__" not in p.parts),
        key=_relative_production_path,
    )


def _discover_cancellation_composition_details() -> list[_CancellationSiteDetail]:
    details: list[_CancellationSiteDetail] = []
    for path in _iter_production_python_files():
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        tree = ast.parse(text, filename=str(path))
        parents = _parent_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            function_node = _enclosing_function(node, parents)
            if function_node is None:
                continue
            for handler in node.handlers:
                if not _handler_catches_cancelled_error(handler):
                    continue
                if _has_ast_guard_skip_marker(lines, handler.lineno):
                    continue
                details.append(
                    _CancellationSiteDetail(
                        path=path,
                        site=_DiscoveredCancellationSite(
                            _relative_production_path(path),
                            function_node.name,
                            handler.lineno,
                        ),
                        try_node=node,
                        cancelled_handler=handler,
                    )
                )
    return sorted(
        details,
        key=lambda detail: (
            detail.site.relative_path,
            detail.site.except_lineno,
            detail.site.function_name,
        ),
    )


def _discover_cancellation_composition_sites() -> list[_DiscoveredCancellationSite]:
    return [detail.site for detail in _discover_cancellation_composition_details()]


def _handler_references_critical_tuple(handler: ast.ExceptHandler) -> bool:
    for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
        if isinstance(node, ast.Name) and node.id in CRITICAL_EXCEPTION_TUPLE_NAMES:
            return True
    return False


def _handler_reraises(handler: ast.ExceptHandler) -> bool:
    for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
        if isinstance(node, ast.Raise) and (
            node.exc is None
            or (
                handler.name is not None
                and isinstance(node.exc, ast.Name)
                and node.exc.id == handler.name
            )
        ):
            return True
    return False


def _handler_calls_known_critical_helper(handler: ast.ExceptHandler) -> bool:
    for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        if call_name in KNOWN_CRITICAL_DISPATCH_HELPERS:
            return True
    return False


def _exception_handler_preserves_critical_reachability(
    handler: ast.ExceptHandler,
) -> bool:
    if _handler_calls_known_critical_helper(handler):
        return True
    return _handler_references_critical_tuple(handler) and _handler_reraises(handler)


def _exception_handlers_before_base_exception(
    try_node: ast.Try,
) -> list[ast.ExceptHandler]:
    handlers: list[ast.ExceptHandler] = []
    for handler in try_node.handlers:
        if _handler_catches_base_exception(handler):
            return handlers
        if _handler_catches_exception(handler):
            handlers.append(handler)
    return []


# ---------------------------------------------------------------------------
# Archetype A guards
# ---------------------------------------------------------------------------


def test_no_module_level_critical_exception_tuple_outside_callbacks() -> None:
    pattern = re.compile(r"^_CRITICAL_[A-Z_]*EXCEPTION_TYPES\s*=", re.MULTILINE)
    offenders: list[pathlib.Path] = []
    for path in _iter_production_python_files():
        if path == CALLBACKS_FILE:
            continue
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(path)
    assert not offenders, (
        "Module-level _CRITICAL_*_EXCEPTION_TYPES tuples must live only in "
        "token_throttle/_interfaces/_callbacks.py. Use "
        "LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS or "
        "BACKEND_CALLBACK_CRITICAL_EXCEPTIONS instead. Offenders: "
        f"{[str(p) for p in offenders]}"
    )


def test_no_hand_rolled_group_contains_critical_helper() -> None:
    pattern = re.compile(
        r"^def\s+_\w*callback_exception_group_contains_critical\b", re.MULTILINE
    )
    offenders: list[pathlib.Path] = []
    for path in _iter_production_python_files():
        if path == CALLBACKS_FILE:
            continue
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(path)
    assert not offenders, (
        "Hand-rolled _*callback_exception_group_contains_critical helpers are "
        "forbidden outside _interfaces/_callbacks.py. Call "
        "_exception_group_contains_critical(exc, critical) from there. "
        f"Offenders: {[str(p) for p in offenders]}"
    )


def test_critical_exception_literal_names_match_lifecycle_tuple() -> None:
    source_names = _literal_names_in_assignment(
        CALLBACKS_FILE,
        "LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS",
    )
    assert source_names == CRITICAL_EXCEPTION_LITERAL_NAMES, (
        "Critical-exception literal ladder detection must stay aligned with "
        "LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS. Detector names: "
        f"{sorted(CRITICAL_EXCEPTION_LITERAL_NAMES)}; source tuple names: "
        f"{sorted(source_names)}"
    )


def test_no_hand_rolled_critical_exception_literal_ladder() -> None:
    """Detect literal critical-exception ladders in except-clause syntax.

    Four or more named lifecycle criticals in one ``try`` node is enough to
    flag a future hand-rolled callback dispatcher. The canonical dispatcher
    in ``_interfaces/_callbacks.py`` is the only place that should spell out
    these families.
    """
    offenders: list[tuple[pathlib.Path, int, tuple[str, ...]]] = []
    for path in _iter_production_python_files():
        if path == CALLBACKS_FILE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            names: set[str] = set()
            for handler in node.handlers:
                names.update(_exception_type_names(handler.type))
            matches = tuple(
                sorted(CRITICAL_EXCEPTION_LITERAL_NAMES.intersection(names))
            )
            if len(matches) >= 4:
                offenders.append((path, node.lineno, matches))
    assert not offenders, (
        "Hand-rolled critical-exception ladder detected. Replace with "
        "safe_invoke_async_callback / safe_invoke_sync_callback from "
        f"_interfaces/_callbacks.py. Offenders: {offenders}"
    )


# ---------------------------------------------------------------------------
# Archetype B guards
# ---------------------------------------------------------------------------


def test_no_hand_rolled_forget_in_flight_on_base_exception() -> None:
    """``except BaseException`` blocks that call ``_forget_in_flight_reservation``
    must go through ``_refund_or_forget_reservation_on_raise`` instead.

    The two exempt sites are the context-manager helpers themselves in
    ``_rate_limiter.py`` and ``_sync_rate_limiter.py``; both live in methods
    named ``_refund_or_forget_reservation_on_raise``. The scan skips lines
    inside the helper bodies by tracking helper start/end ranges.
    """
    pattern_open = re.compile(r"^\s*except\s+BaseException\b")
    forget_call = re.compile(r"_forget_in_flight_reservation\s*\(")
    helper_def = re.compile(
        r"^\s*(?:async\s+)?def\s+_refund_or_forget_reservation_on_raise\b"
    )
    method_start = re.compile(r"^\s*(?:async\s+)?def\s+|^\s*@")
    offenders: list[tuple[pathlib.Path, int]] = []
    for path in (RATE_LIMITER_FILE, SYNC_RATE_LIMITER_FILE):
        lines = path.read_text(encoding="utf-8").splitlines()
        helper_ranges: list[tuple[int, int]] = []
        for idx, line in enumerate(lines):
            if not helper_def.search(line):
                continue
            helper_indent = len(line) - len(line.lstrip())
            end = len(lines)
            # Skip to the end of the multi-line signature; the next method
            # at the same indent (possibly preceded by a decorator) marks
            # the helper's exit.
            for j in range(idx + 1, len(lines)):
                nxt = lines[j]
                if nxt.strip() == "":
                    continue
                nxt_indent = len(nxt) - len(nxt.lstrip())
                if nxt_indent <= helper_indent and method_start.search(nxt):
                    end = j
                    break
            helper_ranges.append((idx, end))

        for idx, line in enumerate(lines):
            if not pattern_open.search(line):
                continue
            if any(start <= idx < end for start, end in helper_ranges):
                continue
            window = "\n".join(lines[idx : idx + 12])
            if forget_call.search(window):
                offenders.append((path, idx + 1))
    assert not offenders, (
        "Hand-rolled `except BaseException` ... `_forget_in_flight_reservation` "
        "block detected. Use the `_refund_or_forget_reservation_on_raise` "
        f"context manager instead. Offenders: {offenders}"
    )


def test_registered_cleanup_sites_catch_base_exception() -> None:
    """Cleanup/reconciliation sites must not lag the critical taxonomy.

    Discovery finds every non-opted-out ``except asyncio.CancelledError`` /
    ``except concurrent.futures.CancelledError`` site in the production
    package. Registration only supplies the cleanup call each
    discovered load-bearing async site must preserve under ``BaseException``.
    Sync sibling sites are registered explicitly because they use
    interrupt-safe ``except BaseException`` cleanup directly.
    """
    discovered_sites = _discover_cancellation_composition_sites()
    registered_keys = {site.key() for site in REGISTERED_CANCELLATION_CLEANUP_SITES}
    unregistered = [
        site.label()
        for site in discovered_sites
        if (site.relative_path, site.function_name) not in registered_keys
    ]
    assert not unregistered, (
        "Discovered cancellation-composition sites must be registered with a "
        "cleanup_call or explicitly opted out with "
        f"`# {AST_GUARD_SKIP_MARKER} — <reason>` on/immediately above the "
        f"except line. Unregistered sites: {unregistered}. Full discovery: "
        f"{[site.label() for site in discovered_sites]}"
    )

    offenders = [
        (site.relative_path, site.function_name, site.cleanup_call)
        for site in REGISTERED_BASE_EXCEPTION_CLEANUP_SITES
        if not _function_has_base_exception_cleanup(
            site.path,
            site.function_name,
            site.cleanup_call,
        )
    ]
    assert not offenders, (
        "Cleanup/reconciliation sites must catch BaseException so "
        "KeyboardInterrupt/SystemExit/GeneratorExit/MemoryError/RecursionError "
        f"and callback critical exceptions cannot bypass cleanup. Offenders: "
        f"{offenders}"
    )


def test_async_cleanup_exception_handlers_preserve_critical_reachability() -> None:
    """``except Exception`` must not make later ``except BaseException`` dead.

    v6 added ``MemoryError`` and ``RecursionError`` to the critical callback
    taxonomy. They are ``Exception`` subclasses, so a cleanup-cancel try block
    with ``except Exception`` before ``except BaseException`` must visibly
    re-raise critical exceptions or delegate to the shared critical helper.
    """
    registered_keys = {site.key() for site in REGISTERED_CANCELLATION_CLEANUP_SITES}
    offenders: list[str] = []
    for detail in _discover_cancellation_composition_details():
        if (
            detail.site.relative_path,
            detail.site.function_name,
        ) not in registered_keys:
            continue
        for handler in _exception_handlers_before_base_exception(detail.try_node):
            if _exception_handler_preserves_critical_reachability(handler):
                continue
            offenders.append(
                f"{detail.site.label()} has except Exception at line "
                f"{handler.lineno} before except BaseException without a "
                "visible critical-exception re-raise"
            )

    assert not offenders, (
        "Cancellation cleanup try blocks with `except Exception` before "
        "`except BaseException` can intercept v6 critical Exception subclasses. "
        "Add an explicit isinstance(exc, LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS) "
        "check plus re-raise, or call a known helper that checks or preserves "
        "critical exceptions. "
        f"Offenders: {offenders}"
    )


def test_cancellation_composition_site_discovery_matches_snapshot() -> None:
    """Make the auto-discovered cancellation-composition surface reviewable."""
    expected = [
        ("_limiter_backends/_redis/_backend.py", "_check_and_consume_capacity"),
        ("_rate_limiter.py", "_acquire_capacity"),
        ("_rate_limiter.py", "_set_max_capacity_transactional"),
    ]
    actual_sites = _discover_cancellation_composition_sites()
    actual = [site.key() for site in actual_sites]
    assert actual == expected, (
        "Cancellation-composition discovery changed. If a new site is "
        "load-bearing, register its cleanup_call; if it is intentionally "
        f"narrower, add `# {AST_GUARD_SKIP_MARKER} — <reason>` on/immediately "
        "above the except line. Expected semantic keys: "
        f"{expected}. Actual semantic keys: {actual}. Actual discovery labels: "
        f"{[site.label() for site in actual_sites]}"
    )


def test_cancellation_assertions_scan_new_production_files(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    """A new production file must not be hidden by a fixed discovery tuple."""
    production_root = tmp_path / "token_throttle"
    production_root.mkdir()
    (production_root / "new_backend.py").write_text(
        textwrap.dedent(
            """\
            import asyncio

            async def cleanup_cell():
                try:
                    await do_work()
                except asyncio.CancelledError:
                    await rollback()
                    raise
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "PRODUCTION_ROOT", production_root)

    assert [site.label() for site in _discover_cancellation_composition_sites()] == [
        "new_backend.py:cleanup_cell:6"
    ]
    with pytest.raises(
        AssertionError,
        match=re.escape("new_backend.py:cleanup_cell:6"),
    ):
        test_registered_cleanup_sites_catch_base_exception()
