import pytest
from frozendict import frozendict

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)


class _FailingAddSet(set[str]):
    def add(self, value: str) -> None:
        raise MemoryError(f"refuse authority for {value}")


def _config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=60)]),
        model_family="fix-47",
    )


def _usage(amount: float = 10.0):
    return frozendict({"tokens": amount})


async def test_async_consume_records_authority_before_capacity_commit() -> None:
    backend = MemoryBackendBuilder().build(_config())
    backend._acquired_reservation_ids = _FailingAddSet()

    with pytest.raises(MemoryError, match="refuse authority"):
        await backend.consume_capacity(_usage(), reservation_id="reservation-1")

    assert backend._buckets[0].capacity is None

    backend._acquired_reservation_ids = set()
    await backend.consume_capacity(_usage(), reservation_id="reservation-1")

    assert backend._buckets[0].capacity == 90.0


async def test_async_wait_records_authority_before_capacity_commit() -> None:
    backend = MemoryBackendBuilder().build(_config())
    backend._acquired_reservation_ids = _FailingAddSet()

    with pytest.raises(MemoryError, match="refuse authority"):
        await backend.await_for_capacity(_usage(), reservation_id="reservation-1")

    assert backend._buckets[0].capacity is None

    backend._acquired_reservation_ids = set()
    await backend.await_for_capacity(_usage(), reservation_id="reservation-1")

    assert backend._buckets[0].capacity == 90.0


def test_sync_consume_records_authority_before_capacity_commit() -> None:
    backend = SyncMemoryBackendBuilder().build(_config())
    backend._acquired_reservation_ids = _FailingAddSet()

    with pytest.raises(MemoryError, match="refuse authority"):
        backend.consume_capacity(_usage(), reservation_id="reservation-1")

    assert backend._buckets[0].capacity is None

    backend._acquired_reservation_ids = set()
    backend.consume_capacity(_usage(), reservation_id="reservation-1")

    assert backend._buckets[0].capacity == 90.0


def test_sync_wait_records_authority_before_capacity_commit() -> None:
    backend = SyncMemoryBackendBuilder().build(_config())
    backend._acquired_reservation_ids = _FailingAddSet()

    with pytest.raises(MemoryError, match="refuse authority"):
        backend.wait_for_capacity(_usage(), reservation_id="reservation-1")

    assert backend._buckets[0].capacity is None

    backend._acquired_reservation_ids = set()
    backend.wait_for_capacity(_usage(), reservation_id="reservation-1")

    assert backend._buckets[0].capacity == 90.0


async def test_async_backend_refund_ids_are_fifo_bounded_under_200k_refunds() -> None:
    backend = MemoryBackendBuilder().build(_config())
    iterations = 200_000

    for index in range(iterations):
        reservation_id = f"reservation-{index}"
        await backend.consume_capacity(_usage(0), reservation_id=reservation_id)
        await backend.refund_capacity_for_buckets(
            _usage(0),
            _usage(0),
            reservation_id=reservation_id,
        )

    assert (
        len(backend._refunded_reservation_ids) == backend._refunded_reservation_ids_cap
    )
    assert "reservation-0" not in backend._refunded_reservation_ids
    assert f"reservation-{iterations - 1}" in backend._refunded_reservation_ids


def test_sync_backend_refund_ids_are_fifo_bounded_under_200k_refunds() -> None:
    backend = SyncMemoryBackendBuilder().build(_config())
    iterations = 200_000

    for index in range(iterations):
        reservation_id = f"reservation-{index}"
        backend.consume_capacity(_usage(0), reservation_id=reservation_id)
        backend.refund_capacity_for_buckets(
            _usage(0),
            _usage(0),
            reservation_id=reservation_id,
        )

    assert (
        len(backend._refunded_reservation_ids) == backend._refunded_reservation_ids_cap
    )
    assert "reservation-0" not in backend._refunded_reservation_ids
    assert f"reservation-{iterations - 1}" in backend._refunded_reservation_ids
