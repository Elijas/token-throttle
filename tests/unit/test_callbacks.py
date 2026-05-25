"""Tests for callback infrastructure in token_throttle._interfaces._callbacks."""

import inspect
import logging
from unittest.mock import patch

import pytest
from frozendict import frozendict
from pydantic import ValidationError

from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
    _log,
    create_logging_callbacks,
    create_sync_logging_callbacks,
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

    def test_rejects_sync_callbacks(self):
        def on_wait_start(**_kwargs) -> None:
            return None

        with pytest.raises(
            ValidationError,
            match="on_wait_start must be an async callable",
        ):
            RateLimiterCallbacks(on_wait_start=on_wait_start)


class TestSyncRateLimiterCallbacks:
    def test_rejects_async_callbacks(self):
        async def on_wait_start(**_kwargs) -> None:
            return None

        with pytest.raises(
            ValidationError,
            match="on_wait_start must be a synchronous callable",
        ):
            SyncRateLimiterCallbacks(on_wait_start=on_wait_start)


# ---------------------------------------------------------------------------
# _log stdlib
# ---------------------------------------------------------------------------


class TestLogStdlibFallback:
    def test_log_uses_stdlib(self):
        with patch("token_throttle._interfaces._callbacks._stdlib_logger") as mock:
            _log("DEBUG", "test message")
            mock.log.assert_called_once_with(logging.DEBUG, "test message")

    def test_log_stdlib_with_kwargs(self):
        with patch("token_throttle._interfaces._callbacks._stdlib_logger") as mock:
            _log("INFO", "test message", model="gpt-4", count=5)
            mock.log.assert_called_once()
            args = mock.log.call_args[0]
            kwargs = mock.log.call_args.kwargs
            assert args[0] == logging.INFO
            assert args[1] == "%s | %s"
            assert args[2] == "test message"
            assert "model=" in args[3]
            assert kwargs["extra"] == {"model": "gpt-4", "count": 5}

    def test_log_stdlib_level_mapping(self):
        with patch("token_throttle._interfaces._callbacks._stdlib_logger") as mock:
            _log("WARNING", "warn msg")
            assert mock.log.call_args[0][0] == logging.WARNING


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
