"""The conformance harness must never normalize a canonical critical exception.

``token_throttle.conformance`` wraps backend failures in
``BackendConformanceError`` so custom-backend authors get a uniform taxonomy.
Critical exceptions — cancellation, process-health, interpreter-shutdown
signals — must escape that normalization and propagate raw.

Before v7.0.1 the harness kept hand-copied exception tuples that silently
drifted behind ``LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS``: ``RecursionError``
joined the canonical set in v6.0.0 but was missing here, and the group-leaf
classifier omitted ``concurrent.futures.CancelledError`` entirely. These tests
lock the contract so the harness cannot lag the canonical set again — they are
the structural answer to instance-of-"X lags Y" #7's harness half.
"""

from __future__ import annotations

import asyncio
import concurrent.futures

from token_throttle._interfaces._callbacks import LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS
from token_throttle.conformance import (
    _NON_NORMALIZED_EXCEPTION_TYPES,
    _is_non_normalized_group_exception,
)

_CANCELLED_ERROR_TYPES = (asyncio.CancelledError, concurrent.futures.CancelledError)


def test_every_critical_type_propagates_as_group_leaf() -> None:
    """A critical exception inside a ``BaseExceptionGroup`` must be classified
    as control-flow and propagated, never normalized.
    """
    for exc_type in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS:
        assert _is_non_normalized_group_exception(exc_type()), (
            f"{exc_type.__name__} is in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS "
            f"but the harness would normalize it as a group leaf"
        )


def test_every_non_cancellation_critical_is_non_normalized() -> None:
    """Non-group harness handlers re-raise ``_NON_NORMALIZED_EXCEPTION_TYPES``
    raw. Every critical type except the CancelledError pair (handled by a
    dedicated clause / BaseException fall-through) must appear there.
    """
    for exc_type in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS:
        if issubclass(exc_type, _CANCELLED_ERROR_TYPES):
            continue
        assert exc_type in _NON_NORMALIZED_EXCEPTION_TYPES, (
            f"{exc_type.__name__} is critical but absent from "
            f"_NON_NORMALIZED_EXCEPTION_TYPES — non-group harness handlers "
            f"would normalize it into BackendConformanceError"
        )
