"""Regression tests for FIX-44a reservation authority hardening."""

from __future__ import annotations

from typing import Any

import pytest
from frozendict import frozendict
from pydantic import ValidationError

from token_throttle._capacity import CalculatedCapacity
from token_throttle._exceptions import UnknownReservationError
from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    RateLimiterBackend,
    RateLimiterBackendBuilderInterface,
)
from token_throttle._interfaces._models import (
    CapacityReservation,
    FrozenUsage,
    Quota,
    UsageQuotas,
)
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter
from token_throttle._validation import _revalidate_dto


def _quota() -> Quota:
    return Quota(metric="tokens", limit=100.0, per_seconds=60)


def _config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([_quota()]),
        model_family="family",
    )


def _reservation(limiter_instance_id: str = "limiter") -> CapacityReservation:
    return CapacityReservation(
        usage=frozendict({"tokens": 5.0}),
        model_family="family",
        bucket_ids=frozenset({("tokens", 60)}),
        model="model",
        limiter_instance_id=limiter_instance_id,
    )


async def _async_callback(**_kwargs: object) -> None:
    return None


def _sync_callback(**_kwargs: object) -> None:
    return None


def _portable_dump(dto: Any) -> dict[str, object]:
    dump = dto.model_dump()
    if isinstance(dto, PerModelConfig):
        dump["quotas"] = [quota.model_dump() for quota in dto.quotas]
    return dump


@pytest.mark.parametrize(
    ("dto", "mutator", "message"),
    [
        pytest.param(
            _quota(),
            lambda dto: object.__setattr__(dto, "limit", -1.0),
            "greater than",
            id="quota",
        ),
        pytest.param(
            _config(),
            lambda dto: object.__setattr__(dto, "model_family", "bad family"),
            "whitespace",
            id="per-model-config",
        ),
        pytest.param(
            _reservation(),
            lambda dto: object.__setattr__(dto, "usage", frozendict({"tokens": -1.0})),
            "non-negative",
            id="capacity-reservation",
        ),
        pytest.param(
            RateLimiterCallbacks(),
            lambda dto: object.__setattr__(dto, "on_wait_start", _sync_callback),
            "async callable",
            id="rate-limiter-callbacks",
        ),
        pytest.param(
            CalculatedCapacity(amount=1.0, is_fresh_start=False),
            lambda dto: object.__setattr__(dto, "amount", float("nan")),
            "finite",
            id="calculated-capacity",
        ),
    ],
)
def test_revalidate_rejects_malformed_exact_dtos(dto, mutator, message) -> None:
    mutator(dto)

    with pytest.raises(ValidationError, match=message):
        _revalidate_dto(dto)


def test_revalidate_rejects_uninitialized_quota_from_new() -> None:
    quota = Quota.__new__(Quota)

    with pytest.raises(ValidationError, match="Field required"):
        _revalidate_dto(quota)


def test_revalidate_rejects_malformed_quota_embedded_in_config() -> None:
    cfg = _config()
    embedded_quota = next(iter(cfg.quotas))
    object.__setattr__(embedded_quota, "limit", -1.0)

    with pytest.raises(ValidationError, match="greater than"):
        _revalidate_dto(cfg)


@pytest.mark.parametrize(
    "dto",
    [
        _quota(),
        _config(),
        _reservation(),
        RateLimiterCallbacks(on_wait_start=_async_callback),
        CalculatedCapacity(amount=-1.0, is_fresh_start=False),
    ],
)
def test_valid_dtos_pass_revalidation_unchanged(dto) -> None:
    revalidated = _revalidate_dto(dto)

    assert type(revalidated) is type(dto)
    assert _portable_dump(revalidated) == _portable_dump(dto)


async def test_acquire_revalidates_config_and_embedded_quotas() -> None:
    cfg = _config()
    embedded_quota = next(iter(cfg.quotas))
    object.__setattr__(embedded_quota, "limit", -1.0)
    limiter = RateLimiter(lambda _model: cfg, backend=MemoryBackendBuilder())

    with pytest.raises(ValidationError, match="greater than"):
        await limiter.acquire_capacity({"tokens": 1.0}, "model")


async def test_set_max_capacity_revalidates_resolved_config() -> None:
    cfg = _config()
    limiter = RateLimiter(lambda _model: cfg, backend=MemoryBackendBuilder())
    await limiter.acquire_capacity({"tokens": 1.0}, "model")
    object.__setattr__(cfg, "model_family", "bad family")

    with pytest.raises(ValidationError, match="whitespace"):
        await limiter.set_max_capacity("model", "tokens", 60, 200.0)


async def test_refund_revalidates_reservation() -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    reservation = await limiter.acquire_capacity({"tokens": 1.0}, "model")
    object.__setattr__(reservation, "usage", frozendict({"tokens": -1.0}))

    with pytest.raises(ValidationError, match="non-negative"):
        await limiter.refund_capacity({"tokens": 0.0}, reservation)


def test_callback_bundle_revalidated_at_async_limiter_boundary() -> None:
    callbacks = RateLimiterCallbacks()
    object.__setattr__(callbacks, "on_wait_start", _sync_callback)

    with pytest.raises(ValidationError, match="async callable"):
        RateLimiter(_config(), backend=MemoryBackendBuilder(), callbacks=callbacks)


def test_callback_bundle_revalidated_at_sync_limiter_boundary() -> None:
    callbacks = SyncRateLimiterCallbacks()
    object.__setattr__(callbacks, "on_wait_start", _async_callback)

    with pytest.raises(ValidationError, match="synchronous callable"):
        SyncRateLimiter(
            _config(),
            backend=SyncMemoryBackendBuilder(),
            callbacks=callbacks,
        )


class _DurableRefundBackend(RateLimiterBackend):
    def __init__(self) -> None:
        self.refunds: list[
            tuple[
                FrozenUsage,
                FrozenUsage,
                frozenset[tuple[str, int]] | None,
                str | None,
            ]
        ] = []

    async def await_for_capacity(
        self,
        usage: FrozenUsage,
        *,
        timeout: float | None = None,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> None:
        _ = usage, timeout, reservation_id, reservation_lifetime_seconds

    async def consume_capacity(
        self,
        usage: FrozenUsage,
        *,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> None:
        _ = usage, reservation_id, reservation_lifetime_seconds

    async def refund_capacity(
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
    ) -> None:
        await self.refund_capacity_for_buckets(reserved_usage, actual_usage)

    async def refund_capacity_for_buckets(  # noqa: PLR0913
        self,
        reserved_usage: FrozenUsage,
        actual_usage: FrozenUsage,
        *,
        bucket_ids: set[tuple[str, int]] | frozenset[tuple[str, int]] | None = None,
        reservation_id: str | None = None,
        reservation_model_family: str | None = None,
        reservation_bucket_ids: set[tuple[str, int]]
        | frozenset[tuple[str, int]]
        | None = None,
    ) -> bool:
        _ = reservation_model_family, reservation_bucket_ids
        self.refunds.append(
            (
                reserved_usage,
                actual_usage,
                None if bucket_ids is None else frozenset(bucket_ids),
                reservation_id,
            )
        )
        return True

    def supports_durable_refund_dedup(self) -> bool:
        return True

    async def set_max_capacity(
        self,
        metric: str,
        per_seconds: int,
        value: float,
    ) -> None:
        _ = metric, per_seconds, value


class _DurableRefundBackendBuilder(RateLimiterBackendBuilderInterface):
    def __init__(self) -> None:
        self.backend = _DurableRefundBackend()

    def build(
        self,
        cfg: PerModelConfig,
        *,
        callbacks: RateLimiterCallbacks | None = None,
    ) -> RateLimiterBackend:
        _ = cfg, callbacks
        return self.backend


async def test_manual_modern_reservation_without_acquire_marker_is_unknown() -> None:
    builder = _DurableRefundBackendBuilder()
    limiter = RateLimiter(_config(), backend=builder)
    await limiter.acquire_capacity({"tokens": 1.0}, "model")
    reservation = _reservation(limiter._limiter_instance_id)

    with pytest.raises(UnknownReservationError):
        await limiter.refund_capacity({"tokens": 3.0}, reservation)
    assert builder.backend.refunds == []
