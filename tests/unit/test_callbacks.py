"""Tests for callback infrastructure in token_throttle._interfaces._callbacks."""

import inspect
import logging
from unittest.mock import MagicMock, patch

import pytest
from frozendict import frozendict

from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    _get_loguru_logger,
    _log,
    _loguru_cache,
    _probe_loguru,
    create_logging_callbacks,
    create_loguru_callbacks,
    create_sync_logging_callbacks,
    create_sync_loguru_callbacks,
)

# Reusable kwargs for invoking callbacks
_WAIT_START_KWARGS = {
    "model_family": "test-fam",
    "usage": frozendict({"tokens": 100.0}),
    "preconsumption_capacities": frozendict({("tokens", 60): 900.0}),
}
_WAIT_END_KWARGS = {
    **_WAIT_START_KWARGS,
    "postconsumption_capacities": frozendict({("tokens", 60): 800.0}),
    "wait_time_s": 1.5,
}
_CONSUMED_KWARGS = {
    "model_family": "test-fam",
    "usage": frozendict({"tokens": 100.0}),
    "preconsumption_capacities": frozendict({("tokens", 60): 900.0}),
    "postconsumption_capacities": frozendict({("tokens", 60): 800.0}),
    "current_time": 1000.0,
}
_REFUNDED_KWARGS = {
    "model_family": "test-fam",
    "reserved_usage": frozendict({"tokens": 200.0}),
    "actual_usage": frozendict({"tokens": 100.0}),
    "refunded_usage": frozendict({"tokens": 100.0}),
    "prerefund_capacities": frozendict({("tokens", 60): 800.0}),
    "postrefund_capacities": frozendict({("tokens", 60): 900.0}),
}
_MISSING_DATA_KWARGS = {
    "model_family": "test-fam",
    "usage_metric": "tokens",
    "per_seconds": 60,
}


class TestRateLimiterCallbacks:
    def test_all_fields_default_to_none(self):
        callbacks = RateLimiterCallbacks()
        assert callbacks.on_wait_start is None
        assert callbacks.after_wait_end_consumption is None
        assert callbacks.on_capacity_consumed is None
        assert callbacks.on_capacity_refunded is None
        assert callbacks.on_missing_consumption_data is None


# ---------------------------------------------------------------------------
# _probe_loguru caching
# ---------------------------------------------------------------------------


class TestProbeLoguru:
    def test_returns_loguru_logger_when_installed(self):
        pytest.importorskip("loguru")
        _loguru_cache.clear()
        result = _probe_loguru()
        assert result is not None

    def test_cache_hit_on_second_call(self):
        _loguru_cache.clear()
        first = _probe_loguru()
        second = _probe_loguru()
        assert first is second

    def test_returns_none_when_loguru_unavailable(self):
        _loguru_cache.clear()
        with patch.dict("sys.modules", {"loguru": None}):
            # Force re-probe by clearing cache
            _loguru_cache.clear()
            with patch("builtins.__import__", side_effect=ImportError("no loguru")):
                result = _probe_loguru()
            assert result is None
        _loguru_cache.clear()  # Cleanup for other tests


# ---------------------------------------------------------------------------
# _log stdlib fallback
# ---------------------------------------------------------------------------


class TestLogStdlibFallback:
    def test_log_uses_stdlib_when_loguru_unavailable(self):
        _loguru_cache.clear()
        _loguru_cache["logger"] = None  # Force stdlib path
        try:
            with patch("token_throttle._interfaces._callbacks._stdlib_logger") as mock_logger:
                _log("DEBUG", "test message")
                mock_logger.log.assert_called_once_with(
                    logging.DEBUG, "test message"
                )
        finally:
            _loguru_cache.clear()

    def test_log_stdlib_with_kwargs(self):
        _loguru_cache.clear()
        _loguru_cache["logger"] = None
        try:
            with patch("token_throttle._interfaces._callbacks._stdlib_logger") as mock_logger:
                _log("INFO", "test message", model="gpt-4", count=5)
                mock_logger.log.assert_called_once()
                args = mock_logger.log.call_args[0]
                assert args[0] == logging.INFO
                # stdlib path uses format string: "%s | %s", message, extra
                assert args[1] == "%s | %s"
                assert args[2] == "test message"
                assert "model=" in args[3]
        finally:
            _loguru_cache.clear()

    def test_log_stdlib_level_mapping(self):
        _loguru_cache.clear()
        _loguru_cache["logger"] = None
        try:
            with patch("token_throttle._interfaces._callbacks._stdlib_logger") as mock_logger:
                _log("WARNING", "warn msg")
                assert mock_logger.log.call_args[0][0] == logging.WARNING
        finally:
            _loguru_cache.clear()

    def test_log_uses_loguru_when_available(self):
        pytest.importorskip("loguru")
        _loguru_cache.clear()
        mock_loguru = MagicMock()
        _loguru_cache["logger"] = mock_loguru
        try:
            _log("DEBUG", "test message", key="val")
            mock_loguru.log.assert_called_once_with("DEBUG", "test message", key="val")
        finally:
            _loguru_cache.clear()


# ---------------------------------------------------------------------------
# create_logging_callbacks (async)
# ---------------------------------------------------------------------------


class TestCreateLoggingCallbacks:
    def test_default_creates_all_callbacks(self):
        cbs = create_logging_callbacks()
        assert cbs.on_wait_start is not None
        assert cbs.after_wait_end_consumption is not None
        assert cbs.on_capacity_consumed is not None
        assert cbs.on_capacity_refunded is not None
        assert cbs.on_missing_consumption_data is not None

    def test_none_levels_suppress_callbacks(self):
        cbs = create_logging_callbacks(
            wait_start=None,
            wait_end_consumption=None,
            capacity_consumed=None,
            capacity_refunded=None,
            missing_consumption_data=None,
        )
        assert cbs.on_wait_start is None
        assert cbs.after_wait_end_consumption is None
        assert cbs.on_capacity_consumed is None
        assert cbs.on_capacity_refunded is None
        assert cbs.on_missing_consumption_data is None

    def test_all_callbacks_are_async(self):
        cbs = create_logging_callbacks()
        assert inspect.iscoroutinefunction(cbs.on_wait_start)
        assert inspect.iscoroutinefunction(cbs.after_wait_end_consumption)
        assert inspect.iscoroutinefunction(cbs.on_capacity_consumed)
        assert inspect.iscoroutinefunction(cbs.on_capacity_refunded)
        assert inspect.iscoroutinefunction(cbs.on_missing_consumption_data)

    async def test_on_wait_start_invokes_log(self):
        cbs = create_logging_callbacks(wait_start="DEBUG")
        with patch("token_throttle._interfaces._callbacks._log") as mock_log:
            await cbs.on_wait_start(**_WAIT_START_KWARGS)
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == "DEBUG"
            assert "wait starting" in mock_log.call_args[0][1].lower()

    async def test_after_wait_end_consumption_invokes_log(self):
        cbs = create_logging_callbacks(wait_end_consumption="INFO")
        with patch("token_throttle._interfaces._callbacks._log") as mock_log:
            await cbs.after_wait_end_consumption(**_WAIT_END_KWARGS)
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == "INFO"

    async def test_on_capacity_consumed_invokes_log(self):
        cbs = create_logging_callbacks(capacity_consumed="WARNING")
        with patch("token_throttle._interfaces._callbacks._log") as mock_log:
            await cbs.on_capacity_consumed(**_CONSUMED_KWARGS)
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == "WARNING"

    async def test_on_capacity_refunded_invokes_log(self):
        cbs = create_logging_callbacks(capacity_refunded="ERROR")
        with patch("token_throttle._interfaces._callbacks._log") as mock_log:
            await cbs.on_capacity_refunded(**_REFUNDED_KWARGS)
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == "ERROR"

    async def test_on_missing_consumption_data_invokes_log(self):
        cbs = create_logging_callbacks(missing_consumption_data="CRITICAL")
        with patch("token_throttle._interfaces._callbacks._log") as mock_log:
            await cbs.on_missing_consumption_data(**_MISSING_DATA_KWARGS)
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == "CRITICAL"


# ---------------------------------------------------------------------------
# create_sync_logging_callbacks
# ---------------------------------------------------------------------------


class TestCreateSyncLoggingCallbacks:
    def test_default_creates_all_callbacks(self):
        cbs = create_sync_logging_callbacks()
        assert cbs.on_wait_start is not None
        assert cbs.after_wait_end_consumption is not None
        assert cbs.on_capacity_consumed is not None
        assert cbs.on_capacity_refunded is not None
        assert cbs.on_missing_consumption_data is not None

    def test_none_levels_suppress_callbacks(self):
        cbs = create_sync_logging_callbacks(
            wait_start=None,
            wait_end_consumption=None,
            capacity_consumed=None,
            capacity_refunded=None,
            missing_consumption_data=None,
        )
        assert cbs.on_wait_start is None
        assert cbs.after_wait_end_consumption is None
        assert cbs.on_capacity_consumed is None
        assert cbs.on_capacity_refunded is None
        assert cbs.on_missing_consumption_data is None

    def test_all_callbacks_are_sync(self):
        cbs = create_sync_logging_callbacks()
        assert not inspect.iscoroutinefunction(cbs.on_wait_start)
        assert not inspect.iscoroutinefunction(cbs.after_wait_end_consumption)
        assert not inspect.iscoroutinefunction(cbs.on_capacity_consumed)
        assert not inspect.iscoroutinefunction(cbs.on_capacity_refunded)
        assert not inspect.iscoroutinefunction(cbs.on_missing_consumption_data)

    def test_on_wait_start_invokes_log(self):
        cbs = create_sync_logging_callbacks(wait_start="DEBUG")
        with patch("token_throttle._interfaces._callbacks._log") as mock_log:
            cbs.on_wait_start(**_WAIT_START_KWARGS)
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == "DEBUG"

    def test_after_wait_end_consumption_invokes_log(self):
        cbs = create_sync_logging_callbacks(wait_end_consumption="INFO")
        with patch("token_throttle._interfaces._callbacks._log") as mock_log:
            cbs.after_wait_end_consumption(**_WAIT_END_KWARGS)
            mock_log.assert_called_once()

    def test_on_capacity_consumed_invokes_log(self):
        cbs = create_sync_logging_callbacks(capacity_consumed="WARNING")
        with patch("token_throttle._interfaces._callbacks._log") as mock_log:
            cbs.on_capacity_consumed(**_CONSUMED_KWARGS)
            mock_log.assert_called_once()

    def test_on_capacity_refunded_invokes_log(self):
        cbs = create_sync_logging_callbacks(capacity_refunded="ERROR")
        with patch("token_throttle._interfaces._callbacks._log") as mock_log:
            cbs.on_capacity_refunded(**_REFUNDED_KWARGS)
            mock_log.assert_called_once()

    def test_on_missing_consumption_data_invokes_log(self):
        cbs = create_sync_logging_callbacks(missing_consumption_data="CRITICAL")
        with patch("token_throttle._interfaces._callbacks._log") as mock_log:
            cbs.on_missing_consumption_data(**_MISSING_DATA_KWARGS)
            mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# _get_loguru_logger error path
# ---------------------------------------------------------------------------


class TestGetLoguruLogger:
    def test_raises_import_error_when_loguru_unavailable(self):
        _loguru_cache.clear()
        _loguru_cache["logger"] = None
        try:
            with pytest.raises(ImportError, match="loguru"):
                _get_loguru_logger()
        finally:
            _loguru_cache.clear()

    def test_returns_logger_when_available(self):
        pytest.importorskip("loguru")
        _loguru_cache.clear()
        result = _get_loguru_logger()
        assert result is not None
        _loguru_cache.clear()


# ---------------------------------------------------------------------------
# create_loguru_callbacks (async) — invocation tests
# ---------------------------------------------------------------------------


class TestCreateLoguruCallbacks:
    def setup_method(self):
        pytest.importorskip("loguru")

    def test_no_levels_produces_all_none(self):
        callbacks = create_loguru_callbacks()
        assert callbacks.on_wait_start is None
        assert callbacks.after_wait_end_consumption is None
        assert callbacks.on_capacity_consumed is None
        assert callbacks.on_capacity_refunded is None
        assert callbacks.on_missing_consumption_data is None

    def test_wait_start_level_creates_callback(self):
        callbacks = create_loguru_callbacks(wait_start="INFO")
        assert callbacks.on_wait_start is not None
        assert callbacks.after_wait_end_consumption is None
        assert callbacks.on_capacity_consumed is None
        assert callbacks.on_capacity_refunded is None
        assert callbacks.on_missing_consumption_data is None

    def test_wait_end_consumption_level_creates_callback(self):
        callbacks = create_loguru_callbacks(wait_end_consumption="DEBUG")
        assert callbacks.after_wait_end_consumption is not None

    def test_capacity_consumed_level_creates_callback(self):
        callbacks = create_loguru_callbacks(capacity_consumed="WARNING")
        assert callbacks.on_capacity_consumed is not None

    def test_capacity_refunded_level_creates_callback(self):
        callbacks = create_loguru_callbacks(capacity_refunded="INFO")
        assert callbacks.on_capacity_refunded is not None

    def test_missing_consumption_data_level_creates_callback(self):
        callbacks = create_loguru_callbacks(missing_consumption_data="ERROR")
        assert callbacks.on_missing_consumption_data is not None

    def test_all_levels_set_creates_all_callbacks(self):
        callbacks = create_loguru_callbacks(
            wait_start="INFO",
            wait_end_consumption="DEBUG",
            capacity_consumed="WARNING",
            capacity_refunded="INFO",
            missing_consumption_data="ERROR",
        )
        assert callbacks.on_wait_start is not None
        assert callbacks.after_wait_end_consumption is not None
        assert callbacks.on_capacity_consumed is not None
        assert callbacks.on_capacity_refunded is not None
        assert callbacks.on_missing_consumption_data is not None

    def test_on_wait_start_is_async(self):
        callbacks = create_loguru_callbacks(wait_start="DEBUG")
        assert inspect.iscoroutinefunction(callbacks.on_wait_start)

    def test_after_wait_end_consumption_is_async(self):
        callbacks = create_loguru_callbacks(wait_end_consumption="DEBUG")
        assert inspect.iscoroutinefunction(callbacks.after_wait_end_consumption)

    def test_on_capacity_consumed_is_async(self):
        callbacks = create_loguru_callbacks(capacity_consumed="DEBUG")
        assert inspect.iscoroutinefunction(callbacks.on_capacity_consumed)

    def test_on_capacity_refunded_is_async(self):
        callbacks = create_loguru_callbacks(capacity_refunded="DEBUG")
        assert inspect.iscoroutinefunction(callbacks.on_capacity_refunded)

    def test_on_missing_consumption_data_is_async(self):
        callbacks = create_loguru_callbacks(missing_consumption_data="DEBUG")
        assert inspect.iscoroutinefunction(callbacks.on_missing_consumption_data)

    async def test_on_wait_start_invokes_loguru(self):
        cbs = create_loguru_callbacks(wait_start="DEBUG")
        with patch(
            "token_throttle._interfaces._callbacks._get_loguru_logger"
        ) as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            await cbs.on_wait_start(**_WAIT_START_KWARGS)
            mock_logger.log.assert_called_once()

    async def test_after_wait_end_invokes_loguru(self):
        cbs = create_loguru_callbacks(wait_end_consumption="INFO")
        with patch(
            "token_throttle._interfaces._callbacks._get_loguru_logger"
        ) as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            await cbs.after_wait_end_consumption(**_WAIT_END_KWARGS)
            mock_logger.log.assert_called_once()

    async def test_on_capacity_consumed_invokes_loguru(self):
        cbs = create_loguru_callbacks(capacity_consumed="WARNING")
        with patch(
            "token_throttle._interfaces._callbacks._get_loguru_logger"
        ) as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            await cbs.on_capacity_consumed(**_CONSUMED_KWARGS)
            mock_logger.log.assert_called_once()

    async def test_on_capacity_refunded_invokes_loguru(self):
        cbs = create_loguru_callbacks(capacity_refunded="ERROR")
        with patch(
            "token_throttle._interfaces._callbacks._get_loguru_logger"
        ) as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            await cbs.on_capacity_refunded(**_REFUNDED_KWARGS)
            mock_logger.log.assert_called_once()

    async def test_on_missing_consumption_data_invokes_loguru(self):
        cbs = create_loguru_callbacks(missing_consumption_data="CRITICAL")
        with patch(
            "token_throttle._interfaces._callbacks._get_loguru_logger"
        ) as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            await cbs.on_missing_consumption_data(**_MISSING_DATA_KWARGS)
            mock_logger.log.assert_called_once()


# ---------------------------------------------------------------------------
# create_sync_loguru_callbacks — invocation tests
# ---------------------------------------------------------------------------


class TestCreateSyncLoguruCallbacks:
    def setup_method(self):
        pytest.importorskip("loguru")

    def test_no_levels_produces_all_none(self):
        cbs = create_sync_loguru_callbacks()
        assert cbs.on_wait_start is None
        assert cbs.after_wait_end_consumption is None
        assert cbs.on_capacity_consumed is None
        assert cbs.on_capacity_refunded is None
        assert cbs.on_missing_consumption_data is None

    def test_all_levels_set_creates_all_callbacks(self):
        cbs = create_sync_loguru_callbacks(
            wait_start="INFO",
            wait_end_consumption="DEBUG",
            capacity_consumed="WARNING",
            capacity_refunded="INFO",
            missing_consumption_data="ERROR",
        )
        assert cbs.on_wait_start is not None
        assert cbs.after_wait_end_consumption is not None
        assert cbs.on_capacity_consumed is not None
        assert cbs.on_capacity_refunded is not None
        assert cbs.on_missing_consumption_data is not None

    def test_all_callbacks_are_sync(self):
        cbs = create_sync_loguru_callbacks(
            wait_start="DEBUG",
            wait_end_consumption="DEBUG",
            capacity_consumed="DEBUG",
            capacity_refunded="DEBUG",
            missing_consumption_data="DEBUG",
        )
        assert not inspect.iscoroutinefunction(cbs.on_wait_start)
        assert not inspect.iscoroutinefunction(cbs.after_wait_end_consumption)
        assert not inspect.iscoroutinefunction(cbs.on_capacity_consumed)
        assert not inspect.iscoroutinefunction(cbs.on_capacity_refunded)
        assert not inspect.iscoroutinefunction(cbs.on_missing_consumption_data)

    def test_on_wait_start_invokes_loguru(self):
        cbs = create_sync_loguru_callbacks(wait_start="DEBUG")
        with patch(
            "token_throttle._interfaces._callbacks._get_loguru_logger"
        ) as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            cbs.on_wait_start(**_WAIT_START_KWARGS)
            mock_logger.log.assert_called_once()

    def test_after_wait_end_invokes_loguru(self):
        cbs = create_sync_loguru_callbacks(wait_end_consumption="INFO")
        with patch(
            "token_throttle._interfaces._callbacks._get_loguru_logger"
        ) as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            cbs.after_wait_end_consumption(**_WAIT_END_KWARGS)
            mock_logger.log.assert_called_once()

    def test_on_capacity_consumed_invokes_loguru(self):
        cbs = create_sync_loguru_callbacks(capacity_consumed="WARNING")
        with patch(
            "token_throttle._interfaces._callbacks._get_loguru_logger"
        ) as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            cbs.on_capacity_consumed(**_CONSUMED_KWARGS)
            mock_logger.log.assert_called_once()

    def test_on_capacity_refunded_invokes_loguru(self):
        cbs = create_sync_loguru_callbacks(capacity_refunded="ERROR")
        with patch(
            "token_throttle._interfaces._callbacks._get_loguru_logger"
        ) as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            cbs.on_capacity_refunded(**_REFUNDED_KWARGS)
            mock_logger.log.assert_called_once()

    def test_on_missing_consumption_data_invokes_loguru(self):
        cbs = create_sync_loguru_callbacks(missing_consumption_data="CRITICAL")
        with patch(
            "token_throttle._interfaces._callbacks._get_loguru_logger"
        ) as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            cbs.on_missing_consumption_data(**_MISSING_DATA_KWARGS)
            mock_logger.log.assert_called_once()
