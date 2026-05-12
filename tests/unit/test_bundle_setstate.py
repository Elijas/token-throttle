"""Regression tests for FIX-08 BUNDLE-SETSTATE."""

from __future__ import annotations

import copy
import pickle
from concurrent.futures import ProcessPoolExecutor

import pytest
from frozendict import frozendict
from pydantic import ValidationError

from token_throttle._interfaces._models import CapacityReservation


def _limited_reservation() -> CapacityReservation:
    return CapacityReservation(
        usage=frozendict({"tokens": 30.0, "requests": 1.0}),
        model_family="real-family",
        bucket_ids=frozenset({("tokens", 60), ("requests", 60)}),
        model="gpt-4",
    )


def _forged_unlimited_reservation() -> CapacityReservation:
    reservation = _limited_reservation()
    object.__setattr__(reservation, "is_unlimited", True)
    return reservation


def _forged_negative_usage_reservation() -> CapacityReservation:
    reservation = _limited_reservation()
    object.__setattr__(reservation, "usage", frozendict({"tokens": -1.0}))
    return reservation


def _loads_reservation_blob(blob: bytes) -> str:
    pickle.loads(blob)  # noqa: S301
    return "loaded"


def test_pickle_roundtrip_of_valid_reservation_still_succeeds():
    reservation = _limited_reservation()

    recovered = pickle.loads(pickle.dumps(reservation))  # noqa: S301

    assert recovered == reservation


@pytest.mark.parametrize("protocol", range(6), ids=lambda value: f"protocol-{value}")
def test_pickle_protocols_revalidate_forged_unlimited_reservation(protocol):
    blob = pickle.dumps(_forged_unlimited_reservation(), protocol=protocol)

    with pytest.raises(ValidationError, match="is_unlimited=True"):
        pickle.loads(blob)  # noqa: S301


@pytest.mark.parametrize(
    ("copy_fn", "message"),
    [
        pytest.param(copy.copy, "is_unlimited=True", id="copy"),
        pytest.param(copy.deepcopy, "must be non-negative", id="deepcopy"),
    ],
)
def test_copy_paths_revalidate_restored_state(copy_fn, message):
    reservation = (
        _forged_unlimited_reservation()
        if copy_fn is copy.copy
        else _forged_negative_usage_reservation()
    )

    with pytest.raises(ValidationError, match=message):
        copy_fn(reservation)


def test_cloudpickle_revalidates_forged_unlimited_reservation():
    cloudpickle = pytest.importorskip(
        "cloudpickle",
        reason="cloudpickle is optional; BUNDLE-SETSTATE uses the same __setstate__ path when installed",
    )
    blob = cloudpickle.dumps(_forged_unlimited_reservation())

    with pytest.raises(ValidationError, match="is_unlimited=True"):
        cloudpickle.loads(blob)


def test_process_pool_unpickle_revalidates_in_child_process():
    blob = pickle.dumps(_forged_unlimited_reservation())

    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_loads_reservation_blob, blob)
        with pytest.raises(ValidationError, match="is_unlimited=True"):
            future.result(timeout=10)
