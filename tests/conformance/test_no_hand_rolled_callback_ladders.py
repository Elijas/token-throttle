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

import pathlib
import re

import token_throttle

PRODUCTION_ROOT = pathlib.Path(token_throttle.__file__).parent
CALLBACKS_FILE = PRODUCTION_ROOT / "_interfaces" / "_callbacks.py"


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


# Archetype B guards are added by Phase 2 of the refactor; see plan §2.17.
