from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from token_throttle import __version__
from token_throttle._exceptions import (
    AcquireRefundFailedError,
    CardinalityLimitExceededError,
    UnknownReservationError,
)
from token_throttle._interfaces._callbacks import (
    LifecycleEvent,
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="family/gpt",
        quotas=UsageQuotas([Quota(metric="requests", limit=10, per_seconds=60)]),
    )


def _request_config() -> PerModelConfig:
    class UsageCounter:
        def __call__(self, **_kwargs):
            return {"requests": 1}

        async def count_request_async(self, **_kwargs):
            return {"requests": 1}

    return PerModelConfig(
        model_family="family/gpt",
        quotas=UsageQuotas([Quota(metric="requests", limit=10, per_seconds=60)]),
        usage_counter=UsageCounter(),
    )


async def test_async_snapshot_state_and_lifecycle_events(caplog) -> None:
    events: list[LifecycleEvent] = []

    async def on_lifecycle_event(*, event: LifecycleEvent) -> None:
        events.append(event)

    with caplog.at_level(logging.INFO, logger="token_throttle"):
        limiter = RateLimiter(
            _request_config(),
            backend=MemoryBackendBuilder(),
            callbacks=RateLimiterCallbacks(on_lifecycle_event=on_lifecycle_event),
        )

    assert f"token_throttle version {__version__}" in caplog.text
    assert limiter.snapshot_state() == {
        "in_flight_reservations": 0,
        "model_families": 0,
        "backend_type": "memory",
    }

    reservation = await limiter.acquire_capacity_for_request(
        model="gpt-test",
        request_id="req-123",
    )

    assert limiter.snapshot_state() == {
        "in_flight_reservations": 1,
        "model_families": 1,
        "backend_type": "memory",
    }
    assert events[-1].event_type == "capacity_consumed"
    assert events[-1].reservation_id == reservation.reservation_id
    assert events[-1].request_id == "req-123"
    assert events[-1].model_family == "family/gpt"
    assert events[-1].model_alias == "gpt-test"
    assert events[-1].bucket_ids == frozenset({("requests", 60)})
    assert events[-1].usage == {"requests": 1.0}

    await limiter.refund_capacity({"requests": 1}, reservation)

    assert limiter.snapshot_state()["in_flight_reservations"] == 0
    assert events[-1].event_type == "capacity_refunded"
    assert events[-1].reservation_id == reservation.reservation_id


def test_sync_snapshot_state_and_lifecycle_events(caplog) -> None:
    events: list[LifecycleEvent] = []

    def on_lifecycle_event(*, event: LifecycleEvent) -> None:
        events.append(event)

    with caplog.at_level(logging.INFO, logger="token_throttle"):
        limiter = SyncRateLimiter(
            _request_config(),
            backend=SyncMemoryBackendBuilder(),
            callbacks=SyncRateLimiterCallbacks(on_lifecycle_event=on_lifecycle_event),
        )

    assert f"token_throttle version {__version__}" in caplog.text

    reservation = limiter.acquire_capacity_for_request(
        model="gpt-test",
        request_id="req-sync",
    )

    state = limiter.snapshot_state()
    assert state["backend_type"] == "memory"
    assert state["model_families"] == 1
    assert state["in_flight_reservations"] == 1
    assert events[-1].event_type == "capacity_consumed"
    assert events[-1].request_id == "req-sync"
    assert events[-1].model_alias == "gpt-test"

    limiter.refund_capacity({"requests": 1}, reservation)

    assert limiter.snapshot_state()["in_flight_reservations"] == 0
    assert events[-1].event_type == "capacity_refunded"


def test_lifecycle_callback_signature_validated() -> None:
    async def async_bad_callback() -> None:
        return None

    def sync_bad_callback() -> None:
        return None

    with pytest.raises(ValueError, match="on_lifecycle_event"):
        RateLimiterCallbacks(on_lifecycle_event=async_bad_callback)

    with pytest.raises(ValueError, match="on_lifecycle_event"):
        SyncRateLimiterCallbacks(on_lifecycle_event=sync_bad_callback)


def test_public_errors_have_structured_reason() -> None:
    assert AcquireRefundFailedError.reason == "acquire_refund_failed"
    assert CardinalityLimitExceededError("too many").reason == (
        "cardinality_limit_exceeded"
    )
    assert UnknownReservationError("missing").reason == "unknown_reservation"


def test_sync_redis_refund_dedup_debug_event(caplog) -> None:
    pytest.importorskip("redis")
    from token_throttle._limiter_backends._redis._sync_backend import (  # noqa: PLC0415
        SyncRedisBackend,
    )

    redis_client = MagicMock()
    redis_client.set.return_value = True
    backend = SyncRedisBackend([], redis_client, _config(), key_prefix="obs")

    with caplog.at_level(logging.DEBUG, logger="token_throttle.refund"):
        assert backend._commit_refund_dedup("resv-1") is True

    records = [
        record
        for record in caplog.records
        if getattr(record, "token_throttle_event", {}).get("event_type")
        == "redis_refund_dedup_write"
    ]
    assert records
    assert records[-1].token_throttle_event["reservation_id"] == "resv-1"
    assert records[-1].token_throttle_event["bucket_id"] is None
