"""The `Refund dropped` warning must not pre-empt the refund it describes.

When a callable config removes every bucket a reservation was issued against,
the refund is projected down to nothing and the limiter warns about it. The
warning has to be emitted *after* the refund is finalized: a caller running with
warnings promoted to errors turns it into a raise, and warning first would
abandon the reservation in the backend's acquired set and in
`in_flight_reservations`, where it would accumulate toward
`max_in_flight_reservations`.
"""

from __future__ import annotations

import warnings

import pytest

from token_throttle._exceptions import DuplicateRefundError
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

MODEL = "test-model"
MODEL_FAMILY = "test-family"


def _config(metric: str) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric=metric, limit=100.0, per_seconds=60)]),
        model_family=MODEL_FAMILY,
    )


async def test_async_refund_dropped_under_warnings_as_errors_still_releases() -> None:
    metric = "tokens"

    def config_getter(_model: str) -> PerModelConfig:
        return _config(metric)

    limiter = RateLimiter(config_getter, backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 30}, MODEL)
    rid = reservation.reservation_id

    # Rebuild the backend ahead of the refund so the only warning left for the
    # refund to raise is the "Refund dropped" one under test.
    metric = "requests"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        await limiter.record_usage({"requests": 0}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    assert rid in backend._acquired_reservation_ids

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning, match="Refund dropped"):
            await limiter.refund_capacity({"tokens": 0}, reservation)

    # The caller asked for the raise, but the reservation must be fully closed
    # out on both sides of the limiter/backend boundary before it happens.
    assert rid not in backend._acquired_reservation_ids
    assert rid in backend._refunded_reservation_ids
    assert rid not in limiter._in_flight_reservation_ids

    # Committed, not left pending or marked failed: refunding again is a
    # duplicate rather than a retry.
    metric = "tokens"
    await limiter.record_usage({"tokens": 0}, MODEL)
    with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
        await limiter.refund_capacity({"tokens": 0}, reservation)


def test_sync_refund_dropped_under_warnings_as_errors_still_releases() -> None:
    metric = "tokens"

    def config_getter(_model: str) -> PerModelConfig:
        return _config(metric)

    limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())
    reservation = limiter.acquire_capacity({"tokens": 30}, MODEL)
    rid = reservation.reservation_id

    metric = "requests"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        limiter.record_usage({"requests": 0}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    assert rid in backend._acquired_reservation_ids

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning, match="Refund dropped"):
            limiter.refund_capacity({"tokens": 0}, reservation)

    assert rid not in backend._acquired_reservation_ids
    assert rid in backend._refunded_reservation_ids
    assert rid not in limiter._in_flight_reservation_ids

    metric = "tokens"
    limiter.record_usage({"tokens": 0}, MODEL)
    with pytest.raises(DuplicateRefundError, match="reservation already refunded"):
        limiter.refund_capacity({"tokens": 0}, reservation)
