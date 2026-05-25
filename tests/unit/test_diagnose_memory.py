import asyncio
import contextlib
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from token_throttle import RateLimiterDiagnostic
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter


class _NoIntrospectBackend:
    async def await_for_capacity(
        self,
        usage,
        *,
        timeout=None,
        reservation_id=None,
        reservation_lifetime_seconds=None,
    ):
        return time.time()

    async def consume_capacity(
        self,
        usage,
        *,
        reservation_id=None,
        reservation_lifetime_seconds=None,
    ):
        return time.time()

    async def refund_capacity(self, reserved_usage, actual_usage):
        return None

    async def set_max_capacity(self, metric, per_seconds, value):
        return None


class _NoIntrospectBuilder:
    def __init__(self) -> None:
        self.backend = _NoIntrospectBackend()

    def build(self, cfg, *, callbacks=None):
        return self.backend


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="diag-family",
        quotas=UsageQuotas(
            [
                Quota(metric="tokens", limit=100, per_seconds=60),
                Quota(metric="requests", limit=10, per_seconds=60),
            ]
        ),
    )


async def test_async_memory_diagnose_reports_schema_buckets_override_and_reservations():
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity(
        {"tokens": 20, "requests": 1},
        model="diag-model",
    )
    await limiter.set_max_capacity("diag-model", "tokens", 60, 50)

    diagnostic = await limiter.diagnose()

    assert isinstance(diagnostic, RateLimiterDiagnostic)
    assert diagnostic.schema_version == 1
    assert diagnostic.limiter_type == "async"
    assert diagnostic.backend_type == "memory"
    assert diagnostic.model_family_count == 1
    assert diagnostic.bucket_count == 2
    assert [bucket.metric for bucket in diagnostic.buckets] == [
        "requests",
        "tokens",
    ]

    tokens = next(bucket for bucket in diagnostic.buckets if bucket.metric == "tokens")
    assert tokens.configured_limit == pytest.approx(100.0)
    assert tokens.runtime_override == pytest.approx(50.0)
    assert tokens.override_source == "both"
    assert tokens.effective_max_capacity == pytest.approx(50.0)
    assert tokens.configured_to_effective_gap == pytest.approx(-50.0)
    assert tokens.refill_rate_per_second == pytest.approx(50.0 / 60.0)
    assert tokens.status == "ok"
    assert tokens.current_capacity is not None

    assert len(diagnostic.runtime_overrides) == 1
    assert diagnostic.runtime_overrides[0].metric == "tokens"
    assert diagnostic.reservations.total_count == 1
    assert diagnostic.reservations.in_flight_count == 1
    assert diagnostic.reservations.pending_acquire_count == 0
    assert diagnostic.reservations.delivery_cleanup_count == 0
    token_groups = [
        group
        for group in diagnostic.reservations.groups
        if group.metric == "tokens" and group.state == "all"
    ]
    assert token_groups
    assert token_groups[0].total_reserved_usage == pytest.approx(20.0)
    assert reservation.reservation_id in token_groups[0].reservation_ids
    assert diagnostic.backend_health.memory is not None
    assert diagnostic.backend_health.memory.bucket_count == 2
    assert diagnostic.backend_health.memory.acquired_reservation_id_count == 1
    assert diagnostic.waits.total_waiter_count == 0


def test_sync_memory_diagnose_reports_same_dto_shape():
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    reservation = limiter.acquire_capacity(
        {"tokens": 25, "requests": 2},
        model="diag-model",
    )
    limiter.set_max_capacity("diag-model", "requests", 60, 7)

    diagnostic = limiter.diagnose()

    assert isinstance(diagnostic, RateLimiterDiagnostic)
    assert diagnostic.limiter_type == "sync"
    assert diagnostic.backend_type == "memory"
    requests = next(
        bucket for bucket in diagnostic.buckets if bucket.metric == "requests"
    )
    assert requests.configured_limit == pytest.approx(10.0)
    assert requests.runtime_override == pytest.approx(7.0)
    assert requests.override_source == "both"
    assert diagnostic.reservations.total_count == 1
    request_groups = [
        group
        for group in diagnostic.reservations.groups
        if group.metric == "requests" and group.state == "all"
    ]
    assert request_groups[0].total_reserved_usage == pytest.approx(2.0)
    assert reservation.reservation_id in request_groups[0].reservation_ids


@settings(max_examples=10)
@given(
    tokens=st.floats(min_value=0.0, max_value=90.0, allow_nan=False),
    requests=st.floats(min_value=0.0, max_value=9.0, allow_nan=False),
)
def test_sync_memory_diagnose_is_valid_for_varied_limiter_state(
    tokens: float,
    requests: float,
):
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    limiter.acquire_capacity(
        {"tokens": tokens, "requests": requests},
        model="diag-model",
    )

    diagnostic = limiter.diagnose()

    assert RateLimiterDiagnostic.model_validate(diagnostic) == diagnostic
    assert diagnostic.reservations.total_count == 1


async def test_async_memory_diagnose_reports_wait_bottleneck():
    limiter = RateLimiter(
        PerModelConfig(
            model_family="wait-family",
            quotas=UsageQuotas([Quota(metric="tokens", limit=5, per_seconds=60)]),
        ),
        backend=MemoryBackendBuilder(sleep_interval=0.01),
    )
    await limiter.acquire_capacity({"tokens": 5}, model="wait-model")
    waiting = asyncio.create_task(
        limiter.acquire_capacity({"tokens": 4}, model="wait-model", timeout=5)
    )
    try:
        for _ in range(20):
            diagnostic = await limiter.diagnose()
            if diagnostic.waits.total_waiter_count:
                break
            await asyncio.sleep(0.01)
        else:  # pragma: no cover - failure path produces assertion below
            diagnostic = await limiter.diagnose()

        assert diagnostic.waits.total_waiter_count == 1
        waiter = diagnostic.waits.waiters[0]
        assert waiter.reservation_id is not None
        assert waiter.model_family == "wait-family"
        assert waiter.state == "waiting_for_capacity"
        assert waiter.usage["tokens"] == pytest.approx(4.0)
        assert waiter.primary_bottleneck is not None
        assert waiter.primary_bottleneck.metric == "tokens"
        assert waiter.primary_bottleneck.deficit > 0
        assert diagnostic.reservations.pending_acquire_count == 1
    finally:
        waiting.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waiting


async def test_async_diagnose_gracefully_degrades_without_backend_introspect():
    limiter = RateLimiter(_config(), backend=_NoIntrospectBuilder())
    await limiter.acquire_capacity(
        {"tokens": 1, "requests": 1},
        model="diag-model",
    )

    diagnostic = await limiter.diagnose()

    assert diagnostic.backend_type == "custom"
    assert diagnostic.bucket_count == 2
    assert {bucket.status for bucket in diagnostic.buckets} == {"missing"}
    assert all(bucket.current_capacity is None for bucket in diagnostic.buckets)
    assert diagnostic.backend_health.custom is not None
    assert diagnostic.backend_health.custom.introspection_supported is False
    assert any("introspect" in issue.message for issue in diagnostic.issues)
