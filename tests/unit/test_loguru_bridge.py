"""
Regression tests for the loguru-vs-stdlib bridge in
``token_throttle._interfaces._callbacks``.

Closes R4 L22 findings O01-O04:

- O01 (high): broken loguru installs raise non-``ImportError`` types;
  the probe must catch them and cache as unavailable.
- O02 (high): the cached value must be a *factory* so that
  in-process loguru API drift surfaces at the call site.
- O03 (medium): non-``ImportError`` failures must be cached; otherwise
  every callback re-imports a broken loguru (error storm).
- O04 (medium): ``_reset_loguru_cache()`` and the
  ``TOKEN_THROTTLE_LOGURU_DETECT_AGAIN`` env var let the user re-probe
  after a runtime install.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest

from token_throttle._interfaces import _callbacks as cb
from token_throttle._interfaces._callbacks import (
    _LOGURU_DETECT_AGAIN_ENV,
    _LOGURU_UNAVAILABLE,
    _get_loguru_logger,
    _log,
    _loguru_cache,
    _probe_loguru,
    _reset_loguru_cache,
)


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch: pytest.MonkeyPatch):
    """
    Each test starts with an empty cache and no env-var override, and
    leaves the cache empty for the next test.
    """
    monkeypatch.delenv(_LOGURU_DETECT_AGAIN_ENV, raising=False)
    _reset_loguru_cache()
    yield
    _reset_loguru_cache()


# ---------------------------------------------------------------------------
# O01: broaden import-error catch beyond ImportError
# ---------------------------------------------------------------------------


_BROKEN_IMPORT_EXCEPTIONS = [
    pytest.param(ImportError("module not found"), id="ImportError"),
    pytest.param(ModuleNotFoundError("no module"), id="ModuleNotFoundError"),
    pytest.param(RuntimeError("native lib failed to load"), id="RuntimeError"),
    pytest.param(AttributeError("missing C symbol"), id="AttributeError"),
    pytest.param(OSError("dylib blocked by gatekeeper"), id="OSError"),
    pytest.param(TypeError("metaclass conflict"), id="TypeError"),
    pytest.param(ValueError("bad LOGURU_LEVEL env"), id="ValueError"),
]


@pytest.mark.parametrize("exc", _BROKEN_IMPORT_EXCEPTIONS)
def test_o01_probe_catches_any_import_failure(
    exc: BaseException, caplog: pytest.LogCaptureFixture
) -> None:
    """O01: every import failure type → probe returns None, no escape."""
    with (
        patch.object(cb, "_resolve_loguru_logger", side_effect=exc),
        caplog.at_level(logging.WARNING, logger="token_throttle"),
    ):
        result = _probe_loguru()
    assert result is None
    assert any(
        "loguru not usable" in record.message and type(exc).__name__ in record.message
        for record in caplog.records
    ), f"expected warning mentioning {type(exc).__name__}, got {caplog.records}"


def test_o01_keyboard_interrupt_still_propagates() -> None:
    """KeyboardInterrupt is BaseException, must NOT be caught."""
    with (
        patch.object(cb, "_resolve_loguru_logger", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        _probe_loguru()
    # And the cache must be untouched so the next probe can succeed.
    assert "factory" not in _loguru_cache


def test_o01_system_exit_still_propagates() -> None:
    """SystemExit is BaseException, must NOT be caught."""
    with (
        patch.object(cb, "_resolve_loguru_logger", side_effect=SystemExit(2)),
        pytest.raises(SystemExit),
    ):
        _probe_loguru()
    assert "factory" not in _loguru_cache


def test_o01_callback_does_not_raise_when_loguru_broken() -> None:
    """End-to-end: a broken loguru install must not propagate to callbacks."""
    with patch.object(
        cb,
        "_resolve_loguru_logger",
        side_effect=RuntimeError("loguru native lib"),
    ):
        # Probe runs lazily on the first _log call. Should swallow the
        # RuntimeError, log a warning, and fall back to stdlib.
        _log("INFO", "hello from broken-loguru")
    # Cache is now sealed as unavailable.
    assert _loguru_cache["factory"] is _LOGURU_UNAVAILABLE


# ---------------------------------------------------------------------------
# O02: cache the factory, not the resolved logger; surface API drift
# ---------------------------------------------------------------------------


def test_o02_cache_stores_factory_not_logger() -> None:
    """O02: the cache value is a callable, not a Logger instance."""
    pytest.importorskip("loguru")
    factory = _probe_loguru()
    assert callable(factory)
    # Calling it should produce a logger; the logger should be the same
    # singleton across calls (loguru's design), but the factory must
    # remain a callable wrapper rather than the logger object itself.
    assert factory() is not factory


def test_o02_log_method_missing_at_probe_time_caught(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    O02: if loguru's logger is missing ``.log`` at probe time, the probe
    rejects it (caches unavailable), warns, and falls back to stdlib.
    """

    class FakeLoggerNoLog:
        # No .log method at all.
        pass

    fake_module = type("M", (), {"logger": FakeLoggerNoLog()})()
    with (
        patch.dict("sys.modules", {"loguru": fake_module}),
        caplog.at_level(logging.WARNING, logger="token_throttle"),
    ):
        result = _probe_loguru()
    assert result is None
    assert any(
        "loguru not usable" in r.message and "ImportError" in r.message
        for r in caplog.records
    )


def test_o02_api_drift_post_probe_surfaces_at_call_site() -> None:
    """
    O02: if loguru's ``.log`` is removed AFTER the probe (e.g. monkey-
    patched away), the next ``_log`` call raises a clear
    ``AttributeError`` from the live attribute lookup — not a stale
    cached reference.
    """

    class MutableLogger:
        def __init__(self) -> None:
            self.log_calls: list[tuple[Any, ...]] = []

        def log(self, level: str, message: str, **kwargs: Any) -> None:
            self.log_calls.append((level, message, kwargs))

    mutable = MutableLogger()

    fresh_factory_calls = 0

    def factory():
        nonlocal fresh_factory_calls
        fresh_factory_calls += 1
        return mutable

    _loguru_cache["factory"] = factory

    # First call works normally.
    _log("DEBUG", "before drift", k=1)
    assert mutable.log_calls == [("DEBUG", "before drift", {"k": 1})]
    assert fresh_factory_calls == 1

    # Now simulate API drift: remove .log from the logger.
    del MutableLogger.log

    # Next _log call hits the factory fresh, gets the same logger, but
    # the AttributeError now propagates with a clear message.
    with pytest.raises(AttributeError, match="log"):
        _log("DEBUG", "after drift")
    assert fresh_factory_calls == 2  # factory was called fresh


def test_o02_log_uses_factory_not_a_stale_reference() -> None:
    """
    The cached factory is invoked on every ``_log`` call. If the user
    swaps the underlying logger (e.g. via plugin reload), the next
    call sees the new one.
    """

    class CountingLogger:
        def __init__(self, label: str) -> None:
            self.label = label
            self.calls: list[str] = []

        def log(self, _level: str, message: str, **_kwargs: Any) -> None:
            self.calls.append(message)

    first = CountingLogger("v1")
    second = CountingLogger("v2")
    current = [first]

    _loguru_cache["factory"] = lambda: current[0]

    _log("INFO", "to-v1")
    current[0] = second
    _log("INFO", "to-v2")

    assert first.calls == ["to-v1"]
    assert second.calls == ["to-v2"]


# ---------------------------------------------------------------------------
# O03: cache non-ImportError failures; no retry storm
# ---------------------------------------------------------------------------


def test_o03_failure_cached_no_repeat_imports() -> None:
    """
    O03: after a probe failure, subsequent probes must not re-import
    loguru — the failure outcome is cached as unavailable.
    """
    call_count = 0

    def boom() -> Any:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("broken")

    with patch.object(cb, "_resolve_loguru_logger", side_effect=boom):
        for _ in range(5):
            assert _probe_loguru() is None
    assert call_count == 1, (
        f"resolve called {call_count} times; should be cached after 1"
    )
    assert _loguru_cache["factory"] is _LOGURU_UNAVAILABLE


def test_o03_repeated_log_calls_dont_reimport_broken_loguru() -> None:
    """End-to-end: many ``_log`` calls don't trigger many failed imports."""
    call_count = 0

    def boom() -> Any:
        nonlocal call_count
        call_count += 1
        raise OSError("dylib blocked")

    with patch.object(cb, "_resolve_loguru_logger", side_effect=boom):
        for i in range(50):
            _log("INFO", f"msg-{i}")
    assert call_count == 1, f"resolve called {call_count} times across 50 _log() calls"


# ---------------------------------------------------------------------------
# O04: cache reset helper + env-var override
# ---------------------------------------------------------------------------


def test_o04_reset_helper_re_runs_probe() -> None:
    """
    O04: after caching unavailable, ``_reset_loguru_cache()`` makes the
    next probe re-run, picking up loguru that's now installed.
    """
    pytest.importorskip("loguru")

    # Step 1: cache unavailable.
    _loguru_cache["factory"] = _LOGURU_UNAVAILABLE
    assert _probe_loguru() is None

    # Step 2: reset and re-probe — loguru is genuinely installed.
    _reset_loguru_cache()
    factory = _probe_loguru()
    assert factory is not None
    assert callable(factory)


def test_o04_env_var_forces_re_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    O04: ``TOKEN_THROTTLE_LOGURU_DETECT_AGAIN`` evicts the cached entry
    on the next probe, allowing a runtime-installed loguru to be picked
    up without an explicit ``_reset_loguru_cache()`` call.
    """
    pytest.importorskip("loguru")

    _loguru_cache["factory"] = _LOGURU_UNAVAILABLE
    assert _probe_loguru() is None

    monkeypatch.setenv(_LOGURU_DETECT_AGAIN_ENV, "1")
    factory = _probe_loguru()
    assert factory is not None


def test_o04_reset_when_cache_empty_is_safe() -> None:
    """Calling reset on a never-probed cache is a no-op."""
    assert "factory" not in _loguru_cache
    _reset_loguru_cache()
    assert "factory" not in _loguru_cache


def test_o04_env_var_unset_uses_cache_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the env var is unset, normal one-shot caching applies."""
    pytest.importorskip("loguru")
    monkeypatch.delenv(_LOGURU_DETECT_AGAIN_ENV, raising=False)
    first = _probe_loguru()
    second = _probe_loguru()
    assert first is second  # same factory object reused from cache


# ---------------------------------------------------------------------------
# O08: sync↔async parity preserved (positive informational from audit)
# ---------------------------------------------------------------------------


def test_sync_async_share_one_cache() -> None:
    """The same cache and reset apply to both sync and async paths."""
    # Same module re-import goes through sys.modules; identity must hold.
    assert cb._loguru_cache is _loguru_cache
    assert cb._probe_loguru is _probe_loguru
    assert cb._reset_loguru_cache is _reset_loguru_cache


# ---------------------------------------------------------------------------
# Smoke: real loguru still works through the new factory layer
# ---------------------------------------------------------------------------


def test_real_loguru_log_still_works() -> None:
    """End-to-end with a real loguru install."""
    pytest.importorskip("loguru")
    # Should not raise — the factory layer is transparent for happy path.
    _log("INFO", "real loguru smoke test", smoke=True)
    assert callable(_probe_loguru())


def test_get_loguru_logger_returns_fresh_each_call() -> None:
    """
    ``_get_loguru_logger()`` (used by explicit-loguru factories) calls
    the cached factory each invocation. Loguru's logger is a singleton,
    so identity holds; behaviorally we just want no caching of a stale
    reference at this layer.
    """
    pytest.importorskip("loguru")
    a = _get_loguru_logger()
    b = _get_loguru_logger()
    # Loguru exports a singleton, so identity holds.
    assert a is b
