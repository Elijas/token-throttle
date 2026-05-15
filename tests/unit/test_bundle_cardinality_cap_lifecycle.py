import pytest

from token_throttle import (
    CardinalityLimitExceededError,
    PerModelConfig,
    Quota,
    RateLimiter,
    SyncRateLimiter,
    UsageQuotas,
)
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)


def _config(model_family: str) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="tokens", limit=1_000_000, per_seconds=60)]),
        model_family=model_family,
    )


class _CleanupBeforePendingSyncRateLimiter(SyncRateLimiter):
    def _begin_pending_acquire(self, reservation):
        if reservation.model_family != "family-0":
            self.clear_unused_model_families(0)
        super()._begin_pending_acquire(reservation)


def _assert_sync_baseline_only(limiter: SyncRateLimiter) -> None:
    assert limiter._model_name_to_model_family == {"base": "base"}
    assert set(limiter._model_family_to_validated_signature) == {"base"}
    assert set(limiter._model_family_signature_counts) == {"base"}
    assert limiter._model_family_alias_counts == {"base": 1}


def _assert_async_baseline_only(limiter: RateLimiter) -> None:
    assert limiter._model_name_to_model_family == {"base": "base"}
    assert set(limiter._model_family_to_validated_signature) == {"base"}
    assert set(limiter._model_family_signature_counts) == {"base"}
    assert limiter._model_family_alias_counts == {"base": 1}


def test_sync_cleanup_during_install_cannot_bypass_max_model_families():
    limiter = _CleanupBeforePendingSyncRateLimiter(
        _config,
        backend=SyncMemoryBackendBuilder(),
        max_model_families=2,
    )

    first = limiter.acquire_capacity({"tokens": 1}, model="family-0")
    second = limiter.acquire_capacity({"tokens": 1}, model="family-1")

    with pytest.raises(CardinalityLimitExceededError, match="max_model_families"):
        limiter.acquire_capacity({"tokens": 1}, model="family-2")

    assert set(limiter._model_family_to_backend) == {"family-0", "family-1"}
    assert len(limiter._model_family_to_backend) <= limiter._max_model_families

    limiter.refund_capacity({"tokens": 0}, second)
    limiter.refund_capacity({"tokens": 0}, first)


def test_sync_in_flight_cap_rejection_leaves_no_family_or_alias_residue():
    limiter = SyncRateLimiter(
        _config,
        backend=SyncMemoryBackendBuilder(),
        max_model_families=3,
        max_in_flight_reservations=1,
    )

    reservation = limiter.acquire_capacity({"tokens": 1}, model="base")
    baseline_backend_count = len(limiter._model_family_to_backend)
    _assert_sync_baseline_only(limiter)

    for index in range(10):
        with pytest.raises(CardinalityLimitExceededError, match="max_in_flight"):
            limiter.acquire_capacity({"tokens": 1}, model=f"rejected-{index}")

    _assert_sync_baseline_only(limiter)
    assert len(limiter._model_family_to_backend) == baseline_backend_count

    limiter.refund_capacity({"tokens": 0}, reservation)


async def test_async_in_flight_cap_rejection_leaves_no_family_or_alias_residue():
    limiter = RateLimiter(
        _config,
        backend=MemoryBackendBuilder(),
        max_model_families=3,
        max_in_flight_reservations=1,
    )

    reservation = await limiter.acquire_capacity({"tokens": 1}, model="base")
    baseline_backend_count = len(limiter._model_family_to_backend)
    _assert_async_baseline_only(limiter)

    for index in range(10):
        with pytest.raises(CardinalityLimitExceededError, match="max_in_flight"):
            await limiter.acquire_capacity({"tokens": 1}, model=f"rejected-{index}")

    _assert_async_baseline_only(limiter)
    assert len(limiter._model_family_to_backend) == baseline_backend_count

    await limiter.refund_capacity({"tokens": 0}, reservation)
