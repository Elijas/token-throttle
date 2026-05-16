from __future__ import annotations

import asyncio

import pytest

from token_throttle import BackendConformanceError, conformance, conformance_test_for
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    RateLimiterBackend,
    backend_uses_default_prepare_reconfigured_backend,
    sync_backend_uses_default_prepare_reconfigured_backend,
)


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
    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks=None,
    ) -> RateLimiterBackend:
        _ = cfg, callbacks
        raise RuntimeError("boom")

    async def aclose(self) -> None:
        pass

    def close(self) -> None:
        pass


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


async def test_builder_build_exception_is_normalized() -> None:
    with pytest.raises(
        BackendConformanceError,
        match=r"build\(async-protocol-probe\) raised RuntimeError: boom",
    ):
        await conformance_test_for(_RaisingBuildAsyncBuilder())


async def test_every_build_result_is_runtime_checked() -> None:
    with pytest.raises(
        BackendConformanceError,
        match="async backend does not satisfy RateLimiterBackend",
    ):
        await conformance_test_for(_SecondBuildJunkAsyncBuilder())


async def test_async_backend_hang_is_bounded_by_operation_deadline(monkeypatch) -> None:
    monkeypatch.setattr(conformance, "_OPERATION_DEADLINE_SECONDS", 0.05)

    with pytest.raises(
        BackendConformanceError,
        match=r"await_for_capacity\(requests=1\) did not return within 0.05s",
    ):
        await conformance_test_for(_MinimalAsyncBuilder(_HangingAsyncBackend()))


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
        _StructuralAsyncBackend()
    )
    assert not sync_backend_uses_default_prepare_reconfigured_backend(
        _StructuralSyncBackend()
    )
