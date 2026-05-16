from __future__ import annotations

import time

import pytest

from token_throttle import (
    AcquireRefundFailedError,
    BackendConformanceError,
    CapacityReservation,
    conformance,
    frozen_usage,
)
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


class _RecordingRefundAsyncBackend:
    def __init__(self, inner, owner) -> None:
        self._inner = inner
        self._owner = owner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    async def refund_capacity_for_buckets(self, *args, **kwargs):
        self._owner.refund_calls += 1
        return await self._inner.refund_capacity_for_buckets(*args, **kwargs)


class _RecordingRefundAsyncBuilder:
    def __init__(self) -> None:
        self._builder = MemoryBackendBuilder()
        self.refund_calls = 0

    def build(self, cfg, *, callbacks=None):
        return _RecordingRefundAsyncBackend(
            self._builder.build(cfg, callbacks=callbacks),
            self,
        )

    async def aclose(self) -> None:
        await self._builder.aclose()

    def close(self) -> None:
        self._builder.close()


class _RecordingRefundSyncBackend:
    def __init__(self, inner, owner) -> None:
        self._inner = inner
        self._owner = owner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def refund_capacity_for_buckets(self, *args, **kwargs):
        self._owner.refund_calls += 1
        return self._inner.refund_capacity_for_buckets(*args, **kwargs)


class _RecordingRefundSyncBuilder:
    def __init__(self) -> None:
        self._builder = SyncMemoryBackendBuilder()
        self.refund_calls = 0

    def build(self, cfg, *, callbacks=None):
        return _RecordingRefundSyncBackend(
            self._builder.build(cfg, callbacks=callbacks),
            self,
        )

    def close(self) -> None:
        self._builder.close()


class _CloseFailAsyncBuilder(MemoryBackendBuilder):
    async def aclose(self) -> None:
        raise RuntimeError("async close boom")


class _CloseFailSyncBuilder(SyncMemoryBackendBuilder):
    def close(self) -> None:
        raise RuntimeError("sync close boom")


class _ConstructorFailAsyncBuilder(MemoryBackendBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.async_closed = False
        self.sync_closed = False

    def resolve_max_reservation_lifetime_seconds(self, max_lifetime):
        _ = max_lifetime
        raise RuntimeError("async constructor boom")

    async def aclose(self) -> None:
        self.async_closed = True

    def close(self) -> None:
        self.sync_closed = True


class _ConstructorFailSyncBuilder(SyncMemoryBackendBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def resolve_max_reservation_lifetime_seconds(self, max_lifetime):
        _ = max_lifetime
        raise RuntimeError("sync constructor boom")

    def close(self) -> None:
        self.closed = True


def _raise_acquire_refund_failed_error(
    reservation: CapacityReservation,
    *,
    interrupted_by: BaseException,
    refund_error: BaseException,
) -> None:
    raise AcquireRefundFailedError(
        reservation,
        interrupted_by=interrupted_by,
        refund_error=refund_error,
    ) from refund_error


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


async def test_async_public_round_trip_refunds_after_validation_failure(
    monkeypatch,
) -> None:
    builder = _RecordingRefundAsyncBuilder()

    def fail_validation_once(reservation, **kwargs):
        _ = reservation, kwargs
        raise BackendConformanceError("forced validation failure")

    monkeypatch.setattr(
        conformance,
        "_check_public_reservation_fields",
        fail_validation_once,
    )

    with pytest.raises(BackendConformanceError, match="forced validation failure"):
        await conformance._check_async_public_reservation_round_trip(builder)

    assert builder.refund_calls == 1


def test_sync_public_round_trip_refunds_after_validation_failure(monkeypatch) -> None:
    builder = _RecordingRefundSyncBuilder()

    def fail_validation_once(reservation, **kwargs):
        _ = reservation, kwargs
        raise BackendConformanceError("forced validation failure")

    monkeypatch.setattr(
        conformance,
        "_check_public_reservation_fields",
        fail_validation_once,
    )

    with pytest.raises(BackendConformanceError, match="forced validation failure"):
        conformance._check_sync_public_reservation_round_trip(builder)

    assert builder.refund_calls == 1


async def test_async_public_round_trip_fails_on_limiter_close_failure() -> None:
    with pytest.raises(BackendConformanceError, match="async close boom"):
        await conformance._check_async_public_reservation_round_trip(
            _CloseFailAsyncBuilder()
        )


def test_sync_public_round_trip_fails_on_limiter_close_failure() -> None:
    with pytest.raises(BackendConformanceError, match="sync close boom"):
        conformance._check_sync_public_reservation_round_trip(_CloseFailSyncBuilder())


async def test_async_public_constructor_failure_cleans_builder() -> None:
    builder = _ConstructorFailAsyncBuilder()

    with pytest.raises(BackendConformanceError, match="async constructor boom"):
        await conformance._check_async_public_reservation_round_trip(builder)

    assert builder.async_closed is True
    assert builder.sync_closed is True


def test_sync_public_constructor_failure_cleans_builder() -> None:
    builder = _ConstructorFailSyncBuilder()

    with pytest.raises(BackendConformanceError, match="sync constructor boom"):
        conformance._check_sync_public_reservation_round_trip(builder)

    assert builder.closed is True


def test_fix50_payload_validates_reservation_fields() -> None:
    refund_error = RuntimeError("refund")
    interrupted_by = RuntimeError("interrupted")
    now = time.time()
    reservation = CapacityReservation(
        reservation_id="rid",
        usage=frozen_usage({"requests": 2.0}),
        model_family="family",
        bucket_ids=frozenset({("requests", 1)}),
        model="conformance-model",
        limiter_instance_id="limiter",
        created_at_seconds=now,
    )

    try:
        _raise_acquire_refund_failed_error(
            reservation,
            interrupted_by=interrupted_by,
            refund_error=refund_error,
        )
    except AcquireRefundFailedError as exc:
        with pytest.raises(BackendConformanceError, match="usage"):
            conformance._check_acquire_refund_failed_payload(
                exc,
                refund_error=refund_error,
                interrupted_by=interrupted_by,
                expected_usage=frozen_usage({"requests": 1.0}),
                expected_model_family="family",
                expected_model="conformance-model",
                expected_bucket_ids=frozenset({("requests", 1)}),
                expected_limiter_instance_id="limiter",
                expected_reservation_id="rid",
                acquired_after_seconds=now - 1.0,
                acquired_before_seconds=now + 1.0,
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

    with pytest.raises(BackendConformanceError, match="_SyncAcquireInterruptedError"):
        conformance._check_sync_acquire_refund_failed_error(SyncMemoryBackendBuilder())
