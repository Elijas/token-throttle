"""Regression tests for FIX-23 V05 model_copy closure."""

from __future__ import annotations

import pytest
from frozendict import frozendict
from pydantic import ValidationError

from token_throttle._interfaces._models import CapacityReservation
from token_throttle._validation import is_unlimited_reservation


def _normal_reservation() -> CapacityReservation:
    return CapacityReservation(
        usage=frozendict({"tokens": 30.0, "requests": 1.0}),
        model_family="real-family",
        bucket_ids=frozenset({("tokens", 60), ("requests", 60)}),
        model="gpt-4",
    )


def test_model_copy_update_revalidates_verifier_repro():
    normal = _normal_reservation()

    with pytest.raises(ValidationError, match="is_unlimited=True"):
        normal.model_copy(update={"is_unlimited": True})


def test_model_copy_update_still_accepts_valid_updates():
    normal = _normal_reservation()

    copied = normal.model_copy(update={"model": "gpt-4.1"})

    assert copied.model == "gpt-4.1"
    assert copied.model_family == normal.model_family
    assert copied.is_unlimited is False
    assert is_unlimited_reservation(copied) is False


def test_model_copy_without_update_uses_default_shallow_path():
    normal = _normal_reservation()

    copied = normal.model_copy()

    assert copied == normal
    assert copied is not normal


def test_model_copy_deep_without_update_still_uses_default_deep_path():
    normal = _normal_reservation()

    copied = normal.model_copy(deep=True)

    assert copied == normal
    assert copied is not normal
    assert copied.usage is not normal.usage
