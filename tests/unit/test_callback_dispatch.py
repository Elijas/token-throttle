"""
Unit tests for the unified critical-exception callback dispatch helpers.

Covers ``LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS``,
``BACKEND_CALLBACK_CRITICAL_EXCEPTIONS``, ``safe_invoke_async_callback``,
and ``safe_invoke_sync_callback`` in
``token_throttle._interfaces._callbacks``. These helpers are the single
source of truth for the "must propagate / suppress with warning" ladder
formerly hand-rolled at six dispatch sites.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import warnings

import pytest

from token_throttle._exceptions import AcquireRefundFailedError
from token_throttle._interfaces._callbacks import (
    BACKEND_CALLBACK_CRITICAL_EXCEPTIONS,
    LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
    _exception_group_contains_critical,
    safe_invoke_async_callback,
    safe_invoke_sync_callback,
)


def _make_refund_failed_error() -> AcquireRefundFailedError:
    refund_error = RuntimeError("refund failed")
    interrupted_by = RuntimeError("interruption")
    return AcquireRefundFailedError(
        reservation=None,
        refund_error=refund_error,
        interrupted_by=interrupted_by,
    )


def _instantiate_critical(exc_type: type[BaseException]) -> BaseException:
    if exc_type is AcquireRefundFailedError:
        return _make_refund_failed_error()
    if exc_type is SystemExit:
        return SystemExit(0)
    return exc_type("forced")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_lifecycle_critical_exceptions_contents() -> None:
    assert asyncio.CancelledError in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS
    assert concurrent.futures.CancelledError in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS
    assert KeyboardInterrupt in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS
    assert SystemExit in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS
    assert GeneratorExit in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS
    assert MemoryError in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS
    assert RecursionError in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS


def test_backend_critical_exceptions_extends_lifecycle() -> None:
    assert AcquireRefundFailedError in BACKEND_CALLBACK_CRITICAL_EXCEPTIONS
    for exc_type in LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS:
        assert exc_type in BACKEND_CALLBACK_CRITICAL_EXCEPTIONS


# ---------------------------------------------------------------------------
# _exception_group_contains_critical
# ---------------------------------------------------------------------------


def test_exception_group_contains_critical_returns_false_for_plain_exception() -> None:
    assert not _exception_group_contains_critical(
        RuntimeError("nope"), LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS
    )


def test_exception_group_contains_critical_returns_true_when_critical_present() -> None:
    group = BaseExceptionGroup("mix", [RuntimeError("x"), asyncio.CancelledError()])
    assert _exception_group_contains_critical(
        group, LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS
    )


def test_exception_group_contains_critical_returns_false_for_non_critical_group() -> (
    None
):
    group = BaseExceptionGroup("non-crit", [RuntimeError("x"), ValueError("y")])
    assert not _exception_group_contains_critical(
        group, LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS
    )


# ---------------------------------------------------------------------------
# safe_invoke_async_callback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc_type", LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS)
async def test_safe_invoke_async_callback_propagates_lifecycle_critical(
    exc_type: type[BaseException],
) -> None:
    async def callback() -> None:
        raise _instantiate_critical(exc_type)

    with pytest.raises(exc_type):
        await safe_invoke_async_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )


@pytest.mark.parametrize("exc_type", BACKEND_CALLBACK_CRITICAL_EXCEPTIONS)
async def test_safe_invoke_async_callback_propagates_backend_critical(
    exc_type: type[BaseException],
) -> None:
    async def callback() -> None:
        raise _instantiate_critical(exc_type)

    with pytest.raises(exc_type):
        await safe_invoke_async_callback(
            callback,
            critical=BACKEND_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter callback",
        )


async def test_safe_invoke_async_callback_propagates_group_containing_critical() -> (
    None
):
    async def callback() -> None:
        raise BaseExceptionGroup("mixed", [RuntimeError("a"), asyncio.CancelledError()])

    with pytest.raises(BaseExceptionGroup):
        await safe_invoke_async_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )


@pytest.mark.parametrize("exc_type", [ValueError, RuntimeError, Exception])
async def test_safe_invoke_async_callback_suppresses_non_critical(
    exc_type: type[Exception],
) -> None:
    async def callback() -> None:
        raise exc_type("nope")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await safe_invoke_async_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )

    assert any(
        issubclass(w.category, RuntimeWarning)
        and "Rate limiter lifecycle callback" in str(w.message)
        for w in caught
    )


async def test_safe_invoke_async_callback_suppresses_non_critical_group() -> None:
    async def callback() -> None:
        raise BaseExceptionGroup("non-crit", [RuntimeError("a"), ValueError("b")])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await safe_invoke_async_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )

    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


async def test_safe_invoke_async_callback_uses_log_label_in_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def callback() -> None:
        raise RuntimeError("boom")

    with (
        caplog.at_level(logging.WARNING, logger="token_throttle"),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        await safe_invoke_async_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="My custom label",
        )

    assert any("My custom label raised RuntimeError" in str(w.message) for w in caught)
    assert any(
        "My custom label raised RuntimeError" in r.getMessage() for r in caplog.records
    )


async def test_safe_invoke_async_callback_suppresses_warning_under_simplefilter_error() -> (
    None
):
    """MED-31 mitigation: ``simplefilter('error')`` must not reopen the leak."""

    async def callback() -> None:
        raise RuntimeError("boom")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await safe_invoke_async_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )


# ---------------------------------------------------------------------------
# safe_invoke_sync_callback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc_type", LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS)
def test_safe_invoke_sync_callback_propagates_lifecycle_critical(
    exc_type: type[BaseException],
) -> None:
    def callback() -> None:
        raise _instantiate_critical(exc_type)

    with pytest.raises(exc_type):
        safe_invoke_sync_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )


@pytest.mark.parametrize("exc_type", BACKEND_CALLBACK_CRITICAL_EXCEPTIONS)
def test_safe_invoke_sync_callback_propagates_backend_critical(
    exc_type: type[BaseException],
) -> None:
    def callback() -> None:
        raise _instantiate_critical(exc_type)

    with pytest.raises(exc_type):
        safe_invoke_sync_callback(
            callback,
            critical=BACKEND_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter callback",
        )


def test_safe_invoke_sync_callback_propagates_group_containing_critical() -> None:
    def callback() -> None:
        raise BaseExceptionGroup("mixed", [RuntimeError("a"), asyncio.CancelledError()])

    with pytest.raises(BaseExceptionGroup):
        safe_invoke_sync_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )


@pytest.mark.parametrize("exc_type", [ValueError, RuntimeError, Exception])
def test_safe_invoke_sync_callback_suppresses_non_critical(
    exc_type: type[Exception],
) -> None:
    def callback() -> None:
        raise exc_type("nope")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        safe_invoke_sync_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )

    assert any(
        issubclass(w.category, RuntimeWarning)
        and "Rate limiter lifecycle callback" in str(w.message)
        for w in caught
    )


def test_safe_invoke_sync_callback_suppresses_non_critical_group() -> None:
    def callback() -> None:
        raise BaseExceptionGroup("non-crit", [RuntimeError("a"), ValueError("b")])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        safe_invoke_sync_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )

    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_safe_invoke_sync_callback_uses_log_label_in_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def callback() -> None:
        raise RuntimeError("boom")

    with (
        caplog.at_level(logging.WARNING, logger="token_throttle"),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        safe_invoke_sync_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="My custom label",
        )

    assert any("My custom label raised RuntimeError" in str(w.message) for w in caught)
    assert any(
        "My custom label raised RuntimeError" in r.getMessage() for r in caplog.records
    )


def test_safe_invoke_sync_callback_suppresses_warning_under_simplefilter_error() -> (
    None
):
    """MED-31 mitigation: ``simplefilter('error')`` must not reopen the leak."""

    def callback() -> None:
        raise RuntimeError("boom")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        safe_invoke_sync_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )


def test_safe_invoke_sync_callback_detects_async_callable_via_sync_checked() -> None:
    """Sync helper goes through ``_invoke_sync_callback_checked``; awaitable-returning
    callables raise ``TypeError`` inside the helper. ``TypeError`` is non-critical
    so the error becomes a warning rather than propagating — matches the pre-refactor
    behavior of ``_emit_lifecycle_event`` and ``_invoke_callback_safe`` on sync
    backends, which never made the ``TypeError`` from ``_invoke_sync_callback_checked``
    critical.
    """

    async def callback() -> None:
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        safe_invoke_sync_callback(
            callback,
            critical=LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
            log_label="Rate limiter lifecycle callback",
        )

    assert any(
        issubclass(w.category, RuntimeWarning) and "awaitable" in str(w.message)
        for w in caught
    )
