from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from token_throttle import BackendConformanceError, conformance_test_for
from token_throttle._limiter_backends._memory._backend import (
    MemoryBackend,
    MemoryBackendBuilder,
)
from token_throttle._limiter_backends._memory._bucket import MemoryBucket
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackend,
    SyncMemoryBackendBuilder,
)
from token_throttle.conformance import sync_conformance_test_for

if TYPE_CHECKING:
    from token_throttle._interfaces._callbacks import RateLimiterCallbacks
    from token_throttle._interfaces._interfaces import (
        PerModelConfig,
        RateLimiterBackend,
        SyncRateLimiterBackend,
    )
    from token_throttle._interfaces._models import FrozenUsage


class _PatchingMemoryBuilder(MemoryBackendBuilder):
    def _patch_backend(
        self,
        backend: MemoryBackend,
        cfg: PerModelConfig,
    ) -> MemoryBackend:
        _ = cfg
        return backend

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> RateLimiterBackend:
        backend = super().build(cfg, callbacks=callbacks)
        assert isinstance(backend, MemoryBackend)
        return self._patch_backend(backend, cfg)


class _TruthfulMarkerAndDedupBuilder(_PatchingMemoryBuilder):
    def _patch_backend(
        self,
        backend: MemoryBackend,
        cfg: PerModelConfig,
    ) -> MemoryBackend:
        original_refund = backend.refund_capacity_for_buckets

        async def refund_capacity_for_buckets(  # noqa: PLR0913
            reserved_usage: FrozenUsage,
            actual_usage: FrozenUsage,
            *,
            bucket_ids=None,
            reservation_id=None,
            reservation_model_family=None,
            reservation_bucket_ids=None,
            reservation_reserved_usage=None,
        ):
            if reservation_id is not None:
                if reservation_model_family != cfg.get_model_family():
                    raise ValueError("reservation model family mismatch")
                expected_bucket_ids = frozenset(
                    (q.metric, q.per_seconds) for q in cfg.quotas
                )
                if reservation_bucket_ids != expected_bucket_ids:
                    raise ValueError("reservation bucket ids mismatch")
                if reservation_reserved_usage != reserved_usage:
                    raise ValueError("reservation usage mismatch")
            return await original_refund(
                reserved_usage,
                actual_usage,
                bucket_ids=bucket_ids,
                reservation_id=reservation_id,
                reservation_model_family=reservation_model_family,
                reservation_bucket_ids=reservation_bucket_ids,
                reservation_reserved_usage=reservation_reserved_usage,
            )

        backend.supports_acquire_marker_authority = lambda: True
        backend.supports_durable_refund_dedup = lambda: True
        backend.refund_capacity_for_buckets = refund_capacity_for_buckets
        return backend


class _MarkerMetadataLiarBuilder(_PatchingMemoryBuilder):
    def _patch_backend(
        self,
        backend: MemoryBackend,
        cfg: PerModelConfig,
    ) -> MemoryBackend:
        _ = cfg
        backend.supports_acquire_marker_authority = lambda: True
        return backend


class _MarkerBucketIdsLiarBuilder(_PatchingMemoryBuilder):
    def _patch_backend(
        self,
        backend: MemoryBackend,
        cfg: PerModelConfig,
    ) -> MemoryBackend:
        original_refund = backend.refund_capacity_for_buckets

        async def refund_capacity_for_buckets(  # noqa: PLR0913
            reserved_usage: FrozenUsage,
            actual_usage: FrozenUsage,
            *,
            bucket_ids=None,
            reservation_id=None,
            reservation_model_family=None,
            reservation_bucket_ids=None,
            reservation_reserved_usage=None,
        ):
            _ = reservation_bucket_ids
            if reservation_id is not None:
                if reservation_model_family != cfg.get_model_family():
                    raise ValueError("reservation model family mismatch")
                if reservation_reserved_usage != reserved_usage:
                    raise ValueError("reservation usage mismatch")
            return await original_refund(
                reserved_usage,
                actual_usage,
                bucket_ids=bucket_ids,
                reservation_id=reservation_id,
                reservation_model_family=reservation_model_family,
                reservation_bucket_ids=reservation_bucket_ids,
                reservation_reserved_usage=reservation_reserved_usage,
            )

        backend.supports_acquire_marker_authority = lambda: True
        backend.refund_capacity_for_buckets = refund_capacity_for_buckets
        return backend


class _MarkerReservedUsageLiarBuilder(_PatchingMemoryBuilder):
    def _patch_backend(
        self,
        backend: MemoryBackend,
        cfg: PerModelConfig,
    ) -> MemoryBackend:
        original_refund = backend.refund_capacity_for_buckets

        async def refund_capacity_for_buckets(  # noqa: PLR0913
            reserved_usage: FrozenUsage,
            actual_usage: FrozenUsage,
            *,
            bucket_ids=None,
            reservation_id=None,
            reservation_model_family=None,
            reservation_bucket_ids=None,
            reservation_reserved_usage=None,
        ):
            _ = reservation_reserved_usage
            if reservation_id is not None:
                if reservation_model_family != cfg.get_model_family():
                    raise ValueError("reservation model family mismatch")
                expected_bucket_ids = frozenset(
                    (q.metric, q.per_seconds) for q in cfg.quotas
                )
                if reservation_bucket_ids != expected_bucket_ids:
                    raise ValueError("reservation bucket ids mismatch")
            return await original_refund(
                reserved_usage,
                actual_usage,
                bucket_ids=bucket_ids,
                reservation_id=reservation_id,
                reservation_model_family=reservation_model_family,
                reservation_bucket_ids=reservation_bucket_ids,
                reservation_reserved_usage=reservation_reserved_usage,
            )

        backend.supports_acquire_marker_authority = lambda: True
        backend.refund_capacity_for_buckets = refund_capacity_for_buckets
        return backend


class _DurableDedupLiarBuilder(_PatchingMemoryBuilder):
    def _patch_backend(
        self,
        backend: MemoryBackend,
        cfg: PerModelConfig,
    ) -> MemoryBackend:
        _ = cfg
        original_refund = backend.refund_capacity_for_buckets

        async def refund_capacity_for_buckets(  # noqa: PLR0913
            reserved_usage: FrozenUsage,
            actual_usage: FrozenUsage,
            *,
            bucket_ids=None,
            reservation_id=None,
            reservation_model_family=None,
            reservation_bucket_ids=None,
            reservation_reserved_usage=None,
        ):
            _ = reservation_id
            return await original_refund(
                reserved_usage,
                actual_usage,
                bucket_ids=bucket_ids,
                reservation_id=None,
                reservation_model_family=reservation_model_family,
                reservation_bucket_ids=reservation_bucket_ids,
                reservation_reserved_usage=reservation_reserved_usage,
            )

        backend.supports_durable_refund_dedup = lambda: True
        backend.refund_capacity_for_buckets = refund_capacity_for_buckets
        return backend


class _BadMetricSetChangeBuilder(_PatchingMemoryBuilder):
    def _patch_backend(
        self,
        backend: MemoryBackend,
        cfg: PerModelConfig,
    ) -> MemoryBackend:
        _ = cfg

        async def prepare_reconfigured_backend(new_backend, new_cfg):
            _ = new_cfg
            return new_backend

        backend.prepare_reconfigured_backend = prepare_reconfigured_backend
        return backend


class _IsolationLiarBuilder(_PatchingMemoryBuilder):
    def __init__(self) -> None:
        super().__init__(sleep_interval=0.01)
        self._isolation_backend: MemoryBackend | None = None

    def _patch_backend(
        self,
        backend: MemoryBackend,
        cfg: PerModelConfig,
    ) -> MemoryBackend:
        backend.supports_metric_set_change = lambda: False
        family = cfg.get_model_family()
        if "async-isolation-a" in family:
            self._isolation_backend = backend
            return backend
        if "async-isolation-b" in family and self._isolation_backend is not None:
            replacement = [
                MemoryBucket(
                    metric=quota.metric,
                    per_seconds=quota.per_seconds,
                    limit=float(quota.limit),
                    model_family=cfg.get_model_family(),
                )
                for quota in cfg.quotas
            ]
            self._isolation_backend.install_reconfigured_state(
                condition=self._isolation_backend._condition,
                buckets=replacement,
                cfg=cfg,
            )
            return self._isolation_backend
        return backend


class _BadAsyncReturnBuilder(_PatchingMemoryBuilder):
    def _patch_backend(
        self,
        backend: MemoryBackend,
        cfg: PerModelConfig,
    ) -> MemoryBackend:
        _ = cfg

        async def await_for_capacity(
            usage: FrozenUsage,
            *,
            timeout=None,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        ):
            _ = usage, timeout, reservation_id, reservation_lifetime_seconds
            return "not-a-timestamp"

        backend.supports_metric_set_change = lambda: False
        backend.await_for_capacity = await_for_capacity
        return backend


class _AwaitableSyncReturnBuilder(SyncMemoryBackendBuilder):
    def build(self, cfg: PerModelConfig, *, callbacks=None):
        backend = super().build(cfg, callbacks=callbacks)

        async def wait_for_capacity(
            usage: FrozenUsage,
            *,
            timeout=None,
            reservation_id=None,
            reservation_lifetime_seconds=None,
        ):
            _ = usage, timeout, reservation_id, reservation_lifetime_seconds

        backend.supports_metric_set_change = lambda: False
        backend.wait_for_capacity = wait_for_capacity
        return backend


class _PatchingSyncMemoryBuilder(SyncMemoryBackendBuilder):
    def _patch_backend(
        self,
        backend: SyncMemoryBackend,
        cfg: PerModelConfig,
    ) -> SyncMemoryBackend:
        _ = cfg
        return backend

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks=None,
    ) -> SyncRateLimiterBackend:
        backend = super().build(cfg, callbacks=callbacks)
        assert isinstance(backend, SyncMemoryBackend)
        return self._patch_backend(backend, cfg)


class _SyncMarkerBucketIdsLiarBuilder(_PatchingSyncMemoryBuilder):
    def _patch_backend(
        self,
        backend: SyncMemoryBackend,
        cfg: PerModelConfig,
    ) -> SyncMemoryBackend:
        original_refund = backend.refund_capacity_for_buckets

        def refund_capacity_for_buckets(  # noqa: PLR0913
            reserved_usage: FrozenUsage,
            actual_usage: FrozenUsage,
            *,
            bucket_ids=None,
            reservation_id=None,
            reservation_model_family=None,
            reservation_bucket_ids=None,
            reservation_reserved_usage=None,
        ):
            _ = reservation_bucket_ids
            if reservation_id is not None:
                if reservation_model_family != cfg.get_model_family():
                    raise ValueError("reservation model family mismatch")
                if reservation_reserved_usage != reserved_usage:
                    raise ValueError("reservation usage mismatch")
            return original_refund(
                reserved_usage,
                actual_usage,
                bucket_ids=bucket_ids,
                reservation_id=reservation_id,
                reservation_model_family=reservation_model_family,
                reservation_bucket_ids=reservation_bucket_ids,
                reservation_reserved_usage=reservation_reserved_usage,
            )

        backend.supports_acquire_marker_authority = lambda: True
        backend.refund_capacity_for_buckets = refund_capacity_for_buckets
        return backend


class _SyncMarkerReservedUsageLiarBuilder(_PatchingSyncMemoryBuilder):
    def _patch_backend(
        self,
        backend: SyncMemoryBackend,
        cfg: PerModelConfig,
    ) -> SyncMemoryBackend:
        original_refund = backend.refund_capacity_for_buckets

        def refund_capacity_for_buckets(  # noqa: PLR0913
            reserved_usage: FrozenUsage,
            actual_usage: FrozenUsage,
            *,
            bucket_ids=None,
            reservation_id=None,
            reservation_model_family=None,
            reservation_bucket_ids=None,
            reservation_reserved_usage=None,
        ):
            _ = reservation_reserved_usage
            if reservation_id is not None:
                if reservation_model_family != cfg.get_model_family():
                    raise ValueError("reservation model family mismatch")
                expected_bucket_ids = frozenset(
                    (q.metric, q.per_seconds) for q in cfg.quotas
                )
                if reservation_bucket_ids != expected_bucket_ids:
                    raise ValueError("reservation bucket ids mismatch")
            return original_refund(
                reserved_usage,
                actual_usage,
                bucket_ids=bucket_ids,
                reservation_id=reservation_id,
                reservation_model_family=reservation_model_family,
                reservation_bucket_ids=reservation_bucket_ids,
                reservation_reserved_usage=reservation_reserved_usage,
            )

        backend.supports_acquire_marker_authority = lambda: True
        backend.refund_capacity_for_buckets = refund_capacity_for_buckets
        return backend


async def test_truthful_marker_authority_and_durable_dedup_claims_pass() -> None:
    await conformance_test_for(_TruthfulMarkerAndDedupBuilder(sleep_interval=0.01))


async def test_marker_authority_rejects_metadata_mismatch_lie() -> None:
    with pytest.raises(
        BackendConformanceError,
        match="reservation metadata mismatch",
    ):
        await conformance_test_for(_MarkerMetadataLiarBuilder(sleep_interval=0.01))


async def test_marker_authority_rejects_bucket_ids_metadata_lie() -> None:
    with pytest.raises(
        BackendConformanceError,
        match=r"reservation metadata mismatch \(bucket_ids\)",
    ):
        await conformance_test_for(_MarkerBucketIdsLiarBuilder(sleep_interval=0.01))


async def test_marker_authority_rejects_reserved_usage_metadata_lie() -> None:
    with pytest.raises(
        BackendConformanceError,
        match=r"reservation metadata mismatch \(reserved_usage\)",
    ):
        await conformance_test_for(_MarkerReservedUsageLiarBuilder(sleep_interval=0.01))


def test_sync_marker_authority_rejects_bucket_ids_metadata_lie() -> None:
    with pytest.raises(
        BackendConformanceError,
        match=r"reservation metadata mismatch \(bucket_ids\)",
    ):
        sync_conformance_test_for(_SyncMarkerBucketIdsLiarBuilder(sleep_interval=0.01))


def test_sync_marker_authority_rejects_reserved_usage_metadata_lie() -> None:
    with pytest.raises(
        BackendConformanceError,
        match=r"reservation metadata mismatch \(reserved_usage\)",
    ):
        sync_conformance_test_for(
            _SyncMarkerReservedUsageLiarBuilder(sleep_interval=0.01)
        )


async def test_durable_refund_dedup_rejects_duplicate_credit_lie() -> None:
    with pytest.raises(
        BackendConformanceError,
        match="supports_durable_refund_dedup=True must not credit duplicate refunds twice",
    ):
        await conformance_test_for(_DurableDedupLiarBuilder(sleep_interval=0.01))


async def test_metric_set_change_rejects_surviving_state_loss() -> None:
    with pytest.raises(
        BackendConformanceError,
        match=r"prepare_reconfigured_backend.*preserve surviving bucket consumption",
    ):
        await conformance_test_for(_BadMetricSetChangeBuilder(sleep_interval=0.01))


async def test_per_build_isolation_rejects_builder_state_reset() -> None:
    with pytest.raises(
        BackendConformanceError,
        match="building another backend must not change an existing backend's quota limits",
    ):
        await conformance_test_for(_IsolationLiarBuilder())


async def test_async_capacity_operation_rejects_bad_return_value() -> None:
    with pytest.raises(
        BackendConformanceError,
        match="must return None or a finite timestamp",
    ):
        await conformance_test_for(_BadAsyncReturnBuilder(sleep_interval=0.01))


def test_sync_operation_rejects_awaitable_return() -> None:
    with pytest.raises(
        BackendConformanceError,
        match="wait_for_capacity\\(\\) returned an awaitable",
    ):
        sync_conformance_test_for(_AwaitableSyncReturnBuilder(sleep_interval=0.01))
