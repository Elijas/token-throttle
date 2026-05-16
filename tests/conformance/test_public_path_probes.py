from __future__ import annotations

import pytest

from token_throttle import BackendConformanceError, conformance
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)


class _RejectReservationIdAsyncBackend:
    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    async def await_for_capacity(
        self,
        usage,
        *,
        timeout=None,
        reservation_id=None,
        reservation_lifetime_seconds=None,
    ):
        if reservation_id is not None:
            raise TypeError("reservation ids unsupported")
        return await self._inner.await_for_capacity(
            usage,
            timeout=timeout,
            reservation_id=reservation_id,
            reservation_lifetime_seconds=reservation_lifetime_seconds,
        )


class _RejectReservationIdAsyncBuilder:
    def __init__(self) -> None:
        self._builder = MemoryBackendBuilder()

    def build(self, cfg, *, callbacks=None):
        return _RejectReservationIdAsyncBackend(
            self._builder.build(cfg, callbacks=callbacks)
        )

    async def aclose(self) -> None:
        await self._builder.aclose()

    def close(self) -> None:
        self._builder.close()


class _RejectReservationIdSyncBackend:
    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def wait_for_capacity(
        self,
        usage,
        *,
        timeout=None,
        reservation_id=None,
        reservation_lifetime_seconds=None,
    ):
        if reservation_id is not None:
            raise TypeError("reservation ids unsupported")
        return self._inner.wait_for_capacity(
            usage,
            timeout=timeout,
            reservation_id=reservation_id,
            reservation_lifetime_seconds=reservation_lifetime_seconds,
        )


class _RejectReservationIdSyncBuilder:
    def __init__(self) -> None:
        self._builder = SyncMemoryBackendBuilder()

    def build(self, cfg, *, callbacks=None):
        return _RejectReservationIdSyncBackend(
            self._builder.build(cfg, callbacks=callbacks)
        )

    def close(self) -> None:
        self._builder.close()


async def test_async_public_round_trip_rejects_backend_without_reservation_id() -> None:
    with pytest.raises(BackendConformanceError, match="reservation ids unsupported"):
        await conformance._check_async_public_reservation_round_trip(
            _RejectReservationIdAsyncBuilder()
        )


def test_sync_public_round_trip_rejects_backend_without_reservation_id() -> None:
    with pytest.raises(BackendConformanceError, match="reservation ids unsupported"):
        conformance._check_sync_public_reservation_round_trip(
            _RejectReservationIdSyncBuilder()
        )


async def test_async_fix50_fault_probe_accepts_memory_backend() -> None:
    await conformance._check_async_acquire_refund_failed_error(MemoryBackendBuilder())


def test_sync_fix50_fault_probe_accepts_memory_backend() -> None:
    conformance._check_sync_acquire_refund_failed_error(SyncMemoryBackendBuilder())


async def test_async_fix50_fault_probe_requires_refund_failure(monkeypatch) -> None:
    async def refund_success(self, *args, **kwargs) -> bool:
        return True

    monkeypatch.setattr(
        conformance._AsyncRefundFailureBackend,
        "refund_capacity_for_buckets",
        refund_success,
    )

    with pytest.raises(BackendConformanceError, match="CancelledError"):
        await conformance._check_async_acquire_refund_failed_error(
            MemoryBackendBuilder()
        )


def test_sync_fix50_fault_probe_requires_refund_failure(monkeypatch) -> None:
    def refund_success(self, *args, **kwargs) -> bool:
        return True

    monkeypatch.setattr(
        conformance._SyncRefundFailureBackend,
        "refund_capacity_for_buckets",
        refund_success,
    )

    with pytest.raises(BackendConformanceError, match="_SyncAcquireInterrupted"):
        conformance._check_sync_acquire_refund_failed_error(SyncMemoryBackendBuilder())
