"""Tests for callback infrastructure in token_throttle._interfaces._callbacks."""

import inspect

import pytest

from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    create_loguru_callbacks,
)


class TestRateLimiterCallbacks:
    def test_all_fields_default_to_none(self):
        callbacks = RateLimiterCallbacks()
        assert callbacks.on_wait_start is None
        assert callbacks.after_wait_end_consumption is None
        assert callbacks.on_capacity_consumed is None
        assert callbacks.on_capacity_refunded is None
        assert callbacks.on_missing_consumption_data is None


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
