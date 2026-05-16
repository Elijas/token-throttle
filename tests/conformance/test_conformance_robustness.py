from __future__ import annotations

import asyncio
import traceback
from typing import cast

import pytest

from token_throttle import (
    BackendConformanceError,
    conformance,
    conformance_test_for,
    sync_conformance_test_for,
)
from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    RateLimiterBackend,
    SyncRateLimiterBackend,
    backend_uses_default_prepare_reconfigured_backend,
    sync_backend_uses_default_prepare_reconfigured_backend,
)
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle.conformance import ConformanceTiming


class _MinimalAsyncBackend(RateLimiterBackend):
    async def await_for_capacity(
        self,
        usage,
        *,
        timeout=None,
        reservation_id=None,
        reservation_lifetime_seconds=None,
    ):
        _ = timeout, reservation_id, reservation_lifetime_seconds
        if any(amount < 0 for amount in usage.values()):
            raise ValueError("negative usage")

    async def consume_capacity(
        self,
        usage,
        *,
        reservation_id=None,
        reservation_lifetime_seconds=None,
    ):
        _ = usage, reservation_id, reservation_lifetime_seconds

    async def refund_capacity(self, reserved_usage, actual_usage):
        _ = reserved_usage
        if any(amount < 0 for amount in actual_usage.values()):
            raise ValueError("negative actual usage")

    async def set_max_capacity(self, metric, per_seconds, value):
        _ = metric, per_seconds, value


class _MinimalAsyncBuilder:
    def __init__(self, backend):
        self._backend = backend

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks=None,
    ) -> RateLimiterBackend:
        _ = cfg, callbacks
        return self._backend

    async def aclose(self) -> None:
        pass

    def close(self) -> None:
        pass


class _RaisingBuildAsyncBuilder:
    def __init__(self) -> None:
        self.aclose_called = False
        self.close_called = False

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks=None,
    ) -> RateLimiterBackend:
        _ = cfg, callbacks
        raise RuntimeError("boom")

    async def aclose(self) -> None:
        self.aclose_called = True

    def close(self) -> None:
        self.close_called = True


class _RaisingBuildSyncBuilder:
    def __init__(self) -> None:
        self.close_called = False

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks=None,
    ) -> SyncRateLimiterBackend:
        _ = cfg, callbacks
        raise RuntimeError("boom")

    def close(self) -> None:
        self.close_called = True


class _SecondBuildJunkAsyncBuilder(_MinimalAsyncBuilder):
    def __init__(self):
        super().__init__(_MinimalAsyncBackend())
        self._build_count = 0

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks=None,
    ):
        _ = cfg, callbacks
        self._build_count += 1
        if self._build_count == 1:
            return self._backend
        return object()


class _HangingAsyncBackend(_MinimalAsyncBackend):
    async def await_for_capacity(
        self,
        usage,
        *,
        timeout=None,
        reservation_id=None,
        reservation_lifetime_seconds=None,
    ):
        _ = usage, timeout, reservation_id, reservation_lifetime_seconds
        await asyncio.sleep(60)


class _RaisingClaimAsyncBackend(_MinimalAsyncBackend):
    def supports_acquire_marker_authority(self) -> bool:
        raise KeyError("claim exploded")


class _CloseRaisesAwaitable:
    def __await__(self):
        if False:
            yield None
        return True

    def close(self) -> None:
        raise RuntimeError("close exploded")


class _AwaitableClaimAsyncBackend(_MinimalAsyncBackend):
    def supports_acquire_marker_authority(self):
        return _CloseRaisesAwaitable()


class _BadAsyncCallbackPayloadBuilder:
    def __init__(self) -> None:
        self._delegate = MemoryBackendBuilder(sleep_interval=0.01)

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks=None,
    ) -> RateLimiterBackend:
        if callbacks is None:
            return self._delegate.build(cfg, callbacks=callbacks)
        original_callbacks = callbacks

        async def on_capacity_consumed(**kwargs) -> None:
            original = original_callbacks.on_capacity_consumed
            if original is not None:
                kwargs["model_family"] = 123
                await original(**kwargs)

        callbacks = RateLimiterCallbacks(
            on_wait_start=original_callbacks.on_wait_start,
            after_wait_end_consumption=original_callbacks.after_wait_end_consumption,
            on_capacity_consumed=on_capacity_consumed,
            on_capacity_refunded=original_callbacks.on_capacity_refunded,
            on_missing_consumption_data=original_callbacks.on_missing_consumption_data,
            on_lifecycle_event=original_callbacks.on_lifecycle_event,
        )
        return self._delegate.build(cfg, callbacks=callbacks)

    async def aclose(self) -> None:
        return None

    def close(self) -> None:
        return None


class _BadSyncCallbackPayloadBuilder:
    def __init__(self) -> None:
        self._delegate = SyncMemoryBackendBuilder(sleep_interval=0.01)

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks=None,
    ) -> SyncRateLimiterBackend:
        if callbacks is None:
            return self._delegate.build(cfg, callbacks=callbacks)
        original_callbacks = callbacks

        def on_capacity_consumed(**kwargs) -> None:
            original = original_callbacks.on_capacity_consumed
            if original is not None:
                kwargs["model_family"] = 123
                original(**kwargs)

        callbacks = SyncRateLimiterCallbacks(
            on_wait_start=original_callbacks.on_wait_start,
            after_wait_end_consumption=original_callbacks.after_wait_end_consumption,
            on_capacity_consumed=on_capacity_consumed,
            on_capacity_refunded=original_callbacks.on_capacity_refunded,
            on_missing_consumption_data=original_callbacks.on_missing_consumption_data,
            on_lifecycle_event=original_callbacks.on_lifecycle_event,
        )
        return self._delegate.build(cfg, callbacks=callbacks)

    def close(self) -> None:
        return None


class _SlowPromptAsyncBuilder(MemoryBackendBuilder):
    def __init__(self) -> None:
        super().__init__(sleep_interval=0.01)
        self._delayed = False

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks=None,
    ) -> RateLimiterBackend:
        backend = super().build(cfg, callbacks=callbacks)
        if "async-basic" not in cfg.get_model_family():
            return backend

        original_await_for_capacity = backend.await_for_capacity

        async def await_for_capacity(
            usage,
            *,
            timeout=None,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        ):
            if (
                not self._delayed
                and dict(usage) == {"requests": 1}
                and timeout is None
                and reservation_id is None
            ):
                self._delayed = True
                await asyncio.sleep(1.2)
            return await original_await_for_capacity(
                usage,
                timeout=timeout,
                reservation_id=reservation_id,
                reservation_lifetime_seconds=reservation_lifetime_seconds,
            )

        backend.await_for_capacity = await_for_capacity
        return backend


async def test_builder_build_exception_is_normalized() -> None:
    builder = _RaisingBuildAsyncBuilder()
    with pytest.raises(
        BackendConformanceError,
        match=r"build\(async-protocol-probe\) raised RuntimeError: boom",
    ):
        await conformance_test_for(builder)
    assert builder.aclose_called
    assert builder.close_called


def test_sync_builder_build_exception_is_normalized_and_cleanup_runs() -> None:
    builder = _RaisingBuildSyncBuilder()
    with pytest.raises(
        BackendConformanceError,
        match=r"build\(sync-protocol-probe\) raised RuntimeError: boom",
    ):
        sync_conformance_test_for(builder)
    assert builder.close_called


async def test_every_build_result_is_runtime_checked() -> None:
    with pytest.raises(
        BackendConformanceError,
        match="async backend does not satisfy RateLimiterBackend",
    ):
        await conformance_test_for(_SecondBuildJunkAsyncBuilder())


async def test_async_backend_hang_is_bounded_by_operation_deadline() -> None:
    with pytest.raises(
        BackendConformanceError,
        match=r"await_for_capacity\(requests=1\) did not return within 0.05s",
    ):
        await conformance_test_for(
            _MinimalAsyncBuilder(_HangingAsyncBackend()),
            timing=ConformanceTiming(operation_deadline_seconds=0.05),
        )


async def test_async_callback_payload_shape_is_validated() -> None:
    with (
        pytest.warns(RuntimeWarning, match="Rate limiter callback raised"),
        pytest.raises(
            BackendConformanceError,
            match="on_capacity_consumed callback passed invalid model_family",
        ),
    ):
        await conformance_test_for(_BadAsyncCallbackPayloadBuilder())


def test_sync_callback_payload_shape_is_validated() -> None:
    with (
        pytest.warns(RuntimeWarning, match="Rate limiter callback raised"),
        pytest.raises(
            BackendConformanceError,
            match="on_capacity_consumed callback passed invalid model_family",
        ),
    ):
        sync_conformance_test_for(_BadSyncCallbackPayloadBuilder())


class TestConformanceTimingValidation:
    def test_scale_env(self, monkeypatch) -> None:
        monkeypatch.setenv("TOKEN_THROTTLE_CONFORMANCE_TIMING_SCALE", "0.5")

        assert conformance._resolve_timing(None) == ConformanceTiming(
            builder_deadline_seconds=2.5,
            operation_deadline_seconds=5.0,
            prompt_deadline_seconds=0.5,
            wait_budget_seconds=2.5,
        )

    @pytest.mark.parametrize(
        "value",
        ["", "0", "-1", "inf", "nan", "1e309", "not-a-number"],
    )
    def test_invalid_scale_env_is_rejected_without_value_leakage(
        self,
        monkeypatch,
        value,
    ) -> None:
        monkeypatch.setenv("TOKEN_THROTTLE_CONFORMANCE_TIMING_SCALE", value)

        with pytest.raises(
            ValueError,
            match="TOKEN_THROTTLE_CONFORMANCE_TIMING_SCALE "
            "must be a positive finite number",
        ) as exc_info:
            conformance._resolve_timing(None)

        assert exc_info.value.__cause__ is None
        if value in {"", "not-a-number"}:
            assert exc_info.value.__suppress_context__
        if value == "not-a-number":
            assert value not in "".join(traceback.format_exception(exc_info.value))

    def test_scale_env_overflow_is_reported_as_env_error(self, monkeypatch) -> None:
        monkeypatch.setenv("TOKEN_THROTTLE_CONFORMANCE_TIMING_SCALE", "1.0e308")

        with pytest.raises(
            ValueError,
            match="TOKEN_THROTTLE_CONFORMANCE_TIMING_SCALE "
            "must be a positive finite number",
        ):
            conformance._resolve_timing(None)

    def test_explicit_timing_ignores_scale_env(self, monkeypatch) -> None:
        monkeypatch.setenv("TOKEN_THROTTLE_CONFORMANCE_TIMING_SCALE", "3.0")

        assert conformance._resolve_timing(
            ConformanceTiming(operation_deadline_seconds=2.5)
        ) == ConformanceTiming(
            builder_deadline_seconds=5.0,
            operation_deadline_seconds=2.5,
            prompt_deadline_seconds=1.0,
            wait_budget_seconds=5.0,
        )

    @pytest.mark.parametrize("value", [True, "1.0", 1 + 0j])
    def test_explicit_timing_rejects_non_float_field_values(self, value) -> None:
        timing = ConformanceTiming(builder_deadline_seconds=cast("float", value))

        with pytest.raises(
            ValueError,
            match="builder_deadline_seconds must be a positive finite number",
        ) as exc_info:
            conformance._resolve_timing(timing)

        assert exc_info.value.__cause__ is None

    def test_explicit_timing_sanitizes_hostile_float_exception(self) -> None:
        class _HostileFloat:
            def __float__(self) -> float:
                raise RuntimeError("secret-token-123")

        timing = ConformanceTiming(
            builder_deadline_seconds=cast("float", _HostileFloat())
        )

        with pytest.raises(
            ValueError,
            match="builder_deadline_seconds must be a positive finite number",
        ) as exc_info:
            conformance._resolve_timing(timing)

        formatted = "".join(traceback.format_exception(exc_info.value))
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__
        assert "secret-token-123" not in formatted

    def test_timing_requires_conformance_timing_instance(self) -> None:
        class _DuckTiming:
            builder_deadline_seconds = 1.0
            operation_deadline_seconds = 2.0
            prompt_deadline_seconds = 3.0
            wait_budget_seconds = 4.0

        with pytest.raises(TypeError, match="timing must be a ConformanceTiming"):
            conformance._resolve_timing(cast("ConformanceTiming | None", _DuckTiming()))

        with pytest.raises(TypeError, match="timing must be a ConformanceTiming"):
            conformance._resolve_timing(cast("ConformanceTiming | None", {}))

    async def test_scale_env_extends_prompt_deadline(self, monkeypatch) -> None:
        monkeypatch.setenv("TOKEN_THROTTLE_CONFORMANCE_TIMING_SCALE", "1.0")

        with pytest.raises(
            BackendConformanceError,
            match="await_for_capacity\\(\\) did not return promptly",
        ):
            await conformance_test_for(_SlowPromptAsyncBuilder())

        monkeypatch.setenv("TOKEN_THROTTLE_CONFORMANCE_TIMING_SCALE", "2.0")

        await conformance_test_for(_SlowPromptAsyncBuilder())

    async def test_timing_context_resets_after_helper_failure(self) -> None:
        outer_timing = ConformanceTiming(operation_deadline_seconds=0.25)
        token = conformance._TIMING_CONTEXT.set(outer_timing)
        helper_timing = ConformanceTiming(operation_deadline_seconds=0.05)
        try:
            with pytest.raises(BackendConformanceError):
                await conformance_test_for(
                    _MinimalAsyncBuilder(_HangingAsyncBackend()),
                    timing=helper_timing,
                )
            assert conformance._TIMING_CONTEXT.get() == outer_timing
        finally:
            conformance._TIMING_CONTEXT.reset(token)

    async def test_nested_timing_scopes_restore_outer_context(
        self,
        monkeypatch,
    ) -> None:
        outer_timing = ConformanceTiming(operation_deadline_seconds=0.5)
        inner_timing = ConformanceTiming(operation_deadline_seconds=2.0)
        original_check = conformance._check_async_basic_capacity
        nested = False
        observed: list[ConformanceTiming] = []

        async def _nested_check(builder) -> None:
            nonlocal nested
            if nested:
                await original_check(builder)
                return
            nested = True
            observed.append(conformance._timing())
            await conformance_test_for(
                MemoryBackendBuilder(sleep_interval=0.01),
                timing=inner_timing,
            )
            observed.append(conformance._timing())
            raise BackendConformanceError("stop after nested timing probe")

        monkeypatch.setattr(conformance, "_check_async_basic_capacity", _nested_check)

        with pytest.raises(
            BackendConformanceError,
            match="stop after nested timing probe",
        ):
            await conformance_test_for(
                MemoryBackendBuilder(sleep_interval=0.01),
                timing=outer_timing,
            )

        assert observed == [outer_timing, outer_timing]
        assert conformance._TIMING_CONTEXT.get() is None


async def test_claim_method_exception_is_normalized() -> None:
    with pytest.raises(
        BackendConformanceError,
        match=r"supports_acquire_marker_authority\(\) raised KeyError: 'claim exploded'",
    ):
        await conformance_test_for(_MinimalAsyncBuilder(_RaisingClaimAsyncBackend()))


async def test_close_awaitable_exception_does_not_hide_claim_diagnostic() -> None:
    with pytest.raises(
        BackendConformanceError,
        match=r"supports_acquire_marker_authority\(\) must be synchronous and return bool",
    ):
        await conformance_test_for(_MinimalAsyncBuilder(_AwaitableClaimAsyncBackend()))


def test_instance_level_prepare_hook_is_not_treated_as_default() -> None:
    class _StructuralAsyncBackend:
        def __init__(self) -> None:
            async def prepare_reconfigured_backend(new_backend, cfg):
                _ = cfg
                return new_backend

            self.prepare_reconfigured_backend = prepare_reconfigured_backend

    class _StructuralSyncBackend:
        def __init__(self) -> None:
            def prepare_reconfigured_backend(new_backend, cfg):
                _ = cfg
                return new_backend

            self.prepare_reconfigured_backend = prepare_reconfigured_backend

    assert not backend_uses_default_prepare_reconfigured_backend(
        cast("RateLimiterBackend", _StructuralAsyncBackend())
    )
    assert not sync_backend_uses_default_prepare_reconfigured_backend(
        cast("SyncRateLimiterBackend", _StructuralSyncBackend())
    )
