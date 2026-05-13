"""Regression tests for FIX-03 BUNDLE-VALIDATOR.

Closes the ``is_unlimited`` reservation-bypass family identified in the
Round 4 audit: L02 V05/V10/V14, L05 I05/I06 (partial)/I10, L13 N06,
L01 F05.

Mechanism (see ``token_throttle/_interfaces/_models.py``):

1. ``CapacityReservation`` carries a ``@field_validator("is_unlimited",
   mode="after")`` that requires ``model_family == _UNLIMITED_FLAG``,
   empty ``usage``, and ``bucket_ids is None`` whenever
   ``is_unlimited=True``.
2. ``_unlimited_reservation`` on both async and sync limiters drops
   any caller-supplied ``usage`` and emits ``frozendict()`` so the
   factory is the only canonical producer of unlimited reservations
   (matches the documented "unlimited bypasses metering" semantics).
3. ``is_unlimited_reservation`` no longer treats the legacy
   sentinel-only path as unlimited; the flag is the single source of
   truth, paired with the validator above.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from frozendict import frozendict
from pydantic import ValidationError

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import (
    _UNLIMITED_FLAG as _MODELS_FLAG,
)
from token_throttle._interfaces._models import (
    CapacityReservation,
    Quota,
    UsageQuotas,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter
from token_throttle._validation import (
    _UNLIMITED_FLAG as _VALIDATION_FLAG,
)
from token_throttle._validation import (
    is_unlimited_reservation,
)


def _make_mock_async_builder():
    backend = AsyncMock()
    backend.await_for_capacity.return_value = None
    backend.refund_capacity.return_value = None
    builder = MagicMock()
    builder.build.return_value = backend
    return builder, backend


def _make_mock_sync_builder():
    backend = MagicMock()
    backend.wait_for_capacity.return_value = None
    backend.refund_capacity.return_value = None
    builder = MagicMock()
    builder.build.return_value = backend
    return builder, backend


def _unlimited_config() -> PerModelConfig:
    return PerModelConfig(quotas=UsageQuotas.unlimited(), model_family="unl-family")


def _limited_config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric="tokens", limit=1000.0, per_seconds=60),
                Quota(metric="requests", limit=100.0, per_seconds=60),
            ]
        ),
        model_family="real-family",
    )


class TestSentinelConstantReExport:
    """The ``_UNLIMITED_FLAG`` constant lives in ``_models`` after FIX-03.

    ``_validation`` re-exports it for back-compat; both must point at the
    same string. A regression here would silently misalign the validator
    and the factory.
    """

    def test_sentinel_constants_match(self):
        assert _MODELS_FLAG == _VALIDATION_FLAG == "__rate_limiting_disabled__"


class TestV14HandConstructionRejected:
    """V14 / L05 I-EVASION-1: hand-constructing an unlimited reservation
    with a mismatched ``model_family`` is rejected at construction time.
    """

    def test_nonsentinel_family_with_is_unlimited_true_rejected(self):
        with pytest.raises(
            ValidationError,
            match="requires model_family == '__rate_limiting_disabled__'",
        ):
            CapacityReservation(
                usage=frozendict(),
                model_family="real-family",
                is_unlimited=True,
                limiter_instance_id="limiter",
            )

    def test_empty_family_with_is_unlimited_true_rejected(self):
        with pytest.raises(ValidationError):
            CapacityReservation(
                usage=frozendict(),
                model_family="",
                is_unlimited=True,
                limiter_instance_id="limiter",
            )

    def test_canonical_unlimited_construction_still_works(self):
        reservation = CapacityReservation(
            usage=frozendict(),
            model_family=_VALIDATION_FLAG,
            is_unlimited=True,
            limiter_instance_id="limiter",
        )
        assert is_unlimited_reservation(reservation)
        assert reservation.is_unlimited is True
        assert reservation.bucket_ids is None
        assert dict(reservation.usage) == {}


class TestV05NonEmptyUsageRejected:
    """V05 / L05 I02: hand-construction with non-empty usage AND
    ``is_unlimited=True`` is rejected. Pre-FIX-03 this was the silent
    capacity-leak vector — refund short-circuits, metering bypassed.
    """

    def test_nonempty_usage_rejected(self):
        with pytest.raises(ValidationError, match="empty usage"):
            CapacityReservation(
                usage=frozendict({"tokens": 5.0}),
                model_family=_VALIDATION_FLAG,
                is_unlimited=True,
                limiter_instance_id="limiter",
            )

    def test_dump_roundtrip_with_forged_flag_rejected(self):
        # FIX-01 swapped internal ``model_copy(update={...})`` calls to
        # ``model_validate({**dump, **updates}, strict=True)``. This pins
        # the second-line-of-defense for that pattern: even if a future
        # call site uses the dump+update shape on an unlimited reservation
        # with non-empty usage, the validator rejects.
        legit = CapacityReservation(
            usage=frozendict({"tokens": 5.0}),
            model_family="real-family",
            limiter_instance_id="limiter",
        )
        with pytest.raises(ValidationError, match="empty usage"):
            CapacityReservation.model_validate(
                {
                    **legit.model_dump(),
                    "is_unlimited": True,
                    "model_family": _VALIDATION_FLAG,
                },
                strict=True,
            )


class TestV10JsonForgeRejected:
    """V10 / L13 N06 / L05 I-EVASION-6: JSON round-trip with a forged
    ``is_unlimited=true`` flag is rejected by ``model_validate_json``.

    Pre-FIX-03 a remote system serializing reservations could ship a
    payload that flipped the flag and the receiving limiter would
    silently no-op the refund.
    """

    def test_json_forge_with_realistic_family_rejected(self):
        forged = json.dumps(
            {
                "reservation_id": "fake",
                "usage": {"tokens": 10.0, "requests": 1.0},
                "model_family": "real-family",
                "bucket_ids": None,
                "model": "gpt-4",
                "is_unlimited": True,
                "limiter_instance_id": "limiter",
            }
        )
        with pytest.raises(ValidationError):
            CapacityReservation.model_validate_json(forged)

    def test_json_roundtrip_of_canonical_unlimited_succeeds(self):
        # Sanity: a valid serialized unlimited reservation still roundtrips.
        canonical = CapacityReservation(
            usage=frozendict(),
            model_family=_VALIDATION_FLAG,
            is_unlimited=True,
            model="gpt-4",
            limiter_instance_id="limiter",
        )
        recovered = CapacityReservation.model_validate_json(canonical.model_dump_json())
        assert is_unlimited_reservation(recovered)


class TestI05BucketIdsCoupling:
    """L05 I05: hand-constructing ``is_unlimited=True`` with non-None
    ``bucket_ids`` is rejected (semantically nonsense — unlimited
    reservations should have ``bucket_ids=None``).
    """

    def test_nonempty_bucket_ids_with_is_unlimited_rejected(self):
        with pytest.raises(ValidationError, match="bucket_ids=None"):
            CapacityReservation(
                usage=frozendict(),
                model_family=_VALIDATION_FLAG,
                bucket_ids=frozenset({("tokens", 60)}),
                is_unlimited=True,
                limiter_instance_id="limiter",
            )

    def test_empty_frozenset_bucket_ids_with_is_unlimited_rejected(self):
        # An empty frozenset is still "not None" — semantically
        # different from the canonical ``None`` and rejected.
        with pytest.raises(ValidationError, match="bucket_ids=None"):
            CapacityReservation(
                usage=frozendict(),
                model_family=_VALIDATION_FLAG,
                bucket_ids=frozenset(),
                is_unlimited=True,
                limiter_instance_id="limiter",
            )


class TestI10LegacyFallbackTightened:
    """L05 I10: a reservation with the sentinel ``model_family`` but
    ``is_unlimited=False`` is no longer treated as unlimited. Closes the
    second covert-bypass vector beyond V05 (anyone who learned the magic
    string could produce unlimited reservations without flipping the
    flag).
    """

    def test_sentinel_only_legacy_path_no_longer_unlimited(self):
        legacy = CapacityReservation(
            usage=frozendict(),
            model_family=_VALIDATION_FLAG,
            is_unlimited=False,
            limiter_instance_id="limiter",
        )
        assert is_unlimited_reservation(legacy) is False

    def test_flag_is_now_authoritative(self):
        canonical = CapacityReservation(
            usage=frozendict(),
            model_family=_VALIDATION_FLAG,
            is_unlimited=True,
            limiter_instance_id="limiter",
        )
        assert is_unlimited_reservation(canonical) is True


class TestUnlimitedFactoryDropsUsage:
    """End-to-end: the async + sync limiter factories drop user-supplied
    usage and emit ``frozendict()`` so the validator's invariant holds
    even when the caller passes non-empty usage to an unlimited config.
    Documented behavior change (option (b) from the L05 spec).
    """

    async def test_async_acquire_unlimited_drops_caller_usage(self):
        builder, _ = _make_mock_async_builder()
        limiter = RateLimiter(_unlimited_config(), backend=builder)

        reservation = await limiter.acquire_capacity({"tokens": 5}, model="gpt-4")

        assert reservation.is_unlimited is True
        assert reservation.model_family == _VALIDATION_FLAG
        assert dict(reservation.usage) == {}
        assert reservation.bucket_ids is None

    def test_sync_acquire_unlimited_drops_caller_usage(self):
        builder, _ = _make_mock_sync_builder()
        limiter = SyncRateLimiter(_unlimited_config(), backend=builder)

        reservation = limiter.acquire_capacity({"tokens": 5}, model="gpt-4")

        assert reservation.is_unlimited is True
        assert reservation.model_family == _VALIDATION_FLAG
        assert dict(reservation.usage) == {}
        assert reservation.bucket_ids is None


class TestN06EndToEndJsonForgeNoLeakage:
    """L13 N06: an external system that serialized a reservation, flipped
    the flag in transit, and called ``refund_capacity`` cannot leak
    capacity. The reservation cannot be deserialized in the first place
    so the refund call site is unreachable.
    """

    async def test_forged_json_cannot_reach_refund_capacity(self):
        builder, _ = _make_mock_async_builder()
        limiter = RateLimiter(_limited_config(), backend=builder)
        legit = await limiter.acquire_capacity(
            {"tokens": 100, "requests": 1}, model="gpt-4"
        )
        # Pre-FIX-03 attack vector: serialize a legit reservation,
        # flip the flag in transit, and the receiving end's
        # ``refund_capacity`` would silently no-op (capacity leak).
        forged_payload = legit.model_dump()
        forged_payload["is_unlimited"] = True
        # Receiving end deserializes via Pydantic — validator fires
        # so the forged reservation never reaches the refund call site.
        with pytest.raises(ValidationError):
            CapacityReservation.model_validate(forged_payload, strict=False)
