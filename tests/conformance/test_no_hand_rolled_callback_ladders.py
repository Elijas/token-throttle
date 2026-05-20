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

import token_throttle

PRODUCTION_ROOT = pathlib.Path(token_throttle.__file__).parent
CALLBACKS_FILE = PRODUCTION_ROOT / "_interfaces" / "_callbacks.py"
RATE_LIMITER_FILE = PRODUCTION_ROOT / "_rate_limiter.py"
SYNC_RATE_LIMITER_FILE = PRODUCTION_ROOT / "_sync_rate_limiter.py"


def _handler_catches_base_exception(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name):
        return handler.type.id == "BaseException"
    if isinstance(handler.type, ast.Tuple):
        return any(
            isinstance(elt, ast.Name) and elt.id == "BaseException"
            for elt in handler.type.elts
        )
    return False


def _handler_body_calls(handler: ast.ExceptHandler, function_name: str) -> bool:
    for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == function_name:
            return True
        if isinstance(func, ast.Name) and func.id == function_name:
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


def _iter_production_python_files() -> list[pathlib.Path]:
    return [p for p in PRODUCTION_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


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


def test_no_hand_rolled_critical_exception_literal_ladder() -> None:
    """Detect a ladder where all five lifecycle-critical literals appear in
    except-clause syntax within a ~10-line sliding window. Catches future
    contributors re-introducing the hand-rolled pattern.
    """
    sentinels = (
        "asyncio.CancelledError",
        "KeyboardInterrupt",
        "SystemExit",
        "GeneratorExit",
    )
    except_re = re.compile(r"^\s*except\s.*$")
    offenders: list[tuple[pathlib.Path, int]] = []
    for path in _iter_production_python_files():
        if path == CALLBACKS_FILE:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for start in range(len(lines)):
            window_lines = lines[start : start + 10]
            window_text = "\n".join(window_lines)
            if not all(sentinel in window_text for sentinel in sentinels):
                continue
            if not any(except_re.match(line) for line in window_lines):
                continue
            offenders.append((path, start + 1))
            break
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


def test_async_outer_cleanup_sites_catch_base_exception() -> None:
    """Async cleanup/reconciliation sites must not lag the critical taxonomy.

    These are heterogeneous state transitions, so they do not share a clean
    helper. The invariant is still common: if the cleanup call exists, at least
    one surrounding handler in that function must catch ``BaseException``.
    """
    # Maintenance contract: when adding an async cleanup/reconciliation site
    # whose cleanup must run for BaseException, register it here. Omitted sites
    # are invisible to this guard, so review cannot rely on auto-discovery.
    required_sites = [
        (
            PRODUCTION_ROOT / "_limiter_backends" / "_memory" / "_backend.py",
            "await_for_capacity",
            "_refund_cancelled_consumption",
        ),
        (
            PRODUCTION_ROOT / "_limiter_backends" / "_redis" / "_backend.py",
            "_check_and_consume_capacity",
            "_refund_cancelled_consumption",
        ),
        (
            RATE_LIMITER_FILE,
            "_acquire_capacity",
            "_rollback_pending_acquire",
        ),
        (
            RATE_LIMITER_FILE,
            "_set_max_capacity_transactional",
            "_reconcile_runtime_max_capacity_after_failed_set",
        ),
        (
            RATE_LIMITER_FILE,
            "_backend_task_succeeded_after_cancel",
            "exception",
        ),
        (
            RATE_LIMITER_FILE,
            "_wait_for_set_max_capacity_task_while_cancelled",
            "exception",
        ),
    ]
    offenders = [
        (path, function_name, cleanup_call)
        for path, function_name, cleanup_call in required_sites
        if not _function_has_base_exception_cleanup(
            path,
            function_name,
            cleanup_call,
        )
    ]
    assert not offenders, (
        "Async cleanup/reconciliation sites must catch BaseException so "
        "KeyboardInterrupt/SystemExit/GeneratorExit and callback critical "
        f"exceptions cannot bypass cleanup. Offenders: {offenders}"
    )
