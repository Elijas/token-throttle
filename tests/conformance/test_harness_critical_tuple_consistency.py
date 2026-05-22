"""The conformance harness's treatment of canonical critical exceptions must
stay deliberate and locked — never silently drifted.

``token_throttle.conformance`` wraps backend failures in
``BackendConformanceError`` so custom-backend authors get a uniform taxonomy.
The canonical critical set ``LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS`` governs
which exceptions the harness must not treat as ordinary backend failures.

Before v7.0.1 the harness kept hand-copied exception tuples that silently
drifted behind the canonical set: ``RecursionError`` joined it in v6.0.0 but
was missing here, and the group-leaf classifier omitted
``concurrent.futures.CancelledError`` entirely. The tuples now derive from the
canonical set; these tests lock the resulting contract — including the one
deliberate asymmetry (a bare ``concurrent.futures.CancelledError`` is
normalized, while as a group leaf it propagates) — so neither can drift again.
This is the structural answer to instance #7's harness half of the recurring
"X lags Y" archetype.
"""

from __future__ import annotations

import asyncio
import concurrent.futures

import pytest

from token_throttle._exceptions import BackendConformanceError
from token_throttle._interfaces._callbacks import LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS
from token_throttle.conformance import (
    _NON_NORMALIZED_EXCEPTION_TYPES,
    _is_non_normalized_group_exception,
    _run_sync_step,
)

_CANCELLED_ERROR_TYPES = (asyncio.CancelledError, concurrent.futures.CancelledError)


def test_every_critical_type_propagates_as_group_leaf() -> None:
    """A critical exception inside a ``BaseExceptionGroup`` must be classified
    as control flow and propagated, never normalized.
    """
    for exc_type in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS:
        assert _is_non_normalized_group_exception(exc_type()), (
            f"{exc_type.__name__} is in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS "
            f"but the harness would normalize it as a group leaf"
        )


def test_every_non_cancellation_critical_is_non_normalized() -> None:
    """Non-group harness handlers re-raise ``_NON_NORMALIZED_EXCEPTION_TYPES``
    raw. Every critical type except the two CancelledError types must appear
    there. The CancelledError pair is excluded for distinct reasons (see the
    derivation comment in conformance.py): asyncio.CancelledError propagates
    via a dedicated clause / BaseException fall-through, while a bare
    concurrent.futures.CancelledError is deliberately normalized — that case
    is locked separately by the test below.
    """
    for exc_type in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS:
        if issubclass(exc_type, _CANCELLED_ERROR_TYPES):
            continue
        assert exc_type in _NON_NORMALIZED_EXCEPTION_TYPES, (
            f"{exc_type.__name__} is critical but absent from "
            f"_NON_NORMALIZED_EXCEPTION_TYPES — non-group harness handlers "
            f"would normalize it into BackendConformanceError"
        )


def test_bare_concurrent_futures_cancelled_error_is_normalized() -> None:
    """A bare concurrent.futures.CancelledError is an Exception subclass, so the
    non-group harness handlers catch it via ``except Exception`` and normalize
    it into BackendConformanceError rather than propagating it raw. This is the
    deliberate asymmetry with the group-leaf path (where it propagates): a
    backend raising it bare is a conformance failure; a group leaf is control
    flow. Locked here so the behavior cannot change silently.
    """

    def _raise_cf_cancelled() -> object:
        raise concurrent.futures.CancelledError("backend raised this")

    with pytest.raises(BackendConformanceError):
        _run_sync_step("teststep", _raise_cf_cancelled, deadline=5.0)
