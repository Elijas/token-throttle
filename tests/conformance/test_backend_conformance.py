from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from token_throttle import (
    BackendConformanceError,
    conformance_test_for,
    sync_conformance_test_for,
)
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    RateLimiterBackend,
    SyncRateLimiterBackend,
)
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)

if TYPE_CHECKING:
    from token_throttle._interfaces._models import FrozenUsage


async def test_async_memory_backend_passes_public_conformance_suite() -> None:
    await conformance_test_for(MemoryBackendBuilder(sleep_interval=0.01))


def test_sync_memory_backend_passes_public_conformance_suite() -> None:
    sync_conformance_test_for(SyncMemoryBackendBuilder(sleep_interval=0.01))


class _MarkerLiarAsyncBackend(RateLimiterBackend):
    async def await_for_capacity(
        self,
        usage: FrozenUsage,
        *,
        timeout: float | None = None,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        _ = usage, timeout, reservation_id, reservation_lifetime_seconds
        return time.time()

    async def consume_capacity(
        self,
        usage: FrozenUsage,
        *,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        _ = usage, reservation_id, reservation_lifetime_seconds
        return time.time()

    async def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        _ = reserved_usage, actual_usage

    def supports_acquire_marker_authority(self) -> bool:
        return True

    async def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        _ = metric, per_seconds, value


class _MarkerLiarAsyncBuilder:
    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks=None,
    ) -> RateLimiterBackend:
        _ = cfg, callbacks
        return _MarkerLiarAsyncBackend()

    async def aclose(self) -> None:
        return None

    def close(self) -> None:
        return None


class _MarkerLiarSyncBackend(SyncRateLimiterBackend):
    def wait_for_capacity(
        self,
        usage: FrozenUsage,
        *,
        timeout: float | None = None,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        _ = usage, timeout, reservation_id, reservation_lifetime_seconds
        return time.time()

    def consume_capacity(
        self,
        usage: FrozenUsage,
        *,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> float | None:
        _ = usage, reservation_id, reservation_lifetime_seconds
        return time.time()

    def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        _ = reserved_usage, actual_usage

    def supports_acquire_marker_authority(self) -> bool:
        return True

    def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        _ = metric, per_seconds, value


class _MarkerLiarSyncBuilder:
    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks=None,
    ) -> SyncRateLimiterBackend:
        _ = cfg, callbacks
        return _MarkerLiarSyncBackend()

    def close(self) -> None:
        return None


async def test_async_conformance_rejects_marker_authority_lie() -> None:
    with pytest.raises(
        BackendConformanceError,
        match="supports_acquire_marker_authority=True",
    ):
        await conformance_test_for(_MarkerLiarAsyncBuilder())


def test_sync_conformance_rejects_marker_authority_lie() -> None:
    with pytest.raises(
        BackendConformanceError,
        match="supports_acquire_marker_authority=True",
    ):
        sync_conformance_test_for(_MarkerLiarSyncBuilder())
