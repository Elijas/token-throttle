# ruff: noqa: TC001, TC002, TC003, PLR0913
"""
Uniform op API over async and sync backends, returning comparable outcomes.

An outcome is ``("ok", value)`` or ``("exc", ExceptionTypeName, reason)`` where
``reason`` is the ``.reason`` attribute carried by ``DuplicateRefundError``
(``None`` for other exceptions). Messages are deliberately not compared: the
backends word the same decision differently.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from typing import Any

from frozendict import frozendict

from tests.differential._backends import Harnessed
from token_throttle._interfaces._models import BucketId, frozen_usage

Outcome = tuple[Any, ...]

REL_TOL = 1e-9
ABS_TOL = 1e-6


def _normalize_value(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    return value


class Driver:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def call(
        self, target: Harnessed, method: str, *args: Any, **kwargs: Any
    ) -> Outcome:
        fn = getattr(target.backend, method)
        try:
            if target.is_async:
                value = self.loop.run_until_complete(fn(*args, **kwargs))
            else:
                value = fn(*args, **kwargs)
        except Exception as exc:
            return ("exc", type(exc).__name__, getattr(exc, "reason", None))
        return ("ok", _normalize_value(value))

    # -- capacity operations -------------------------------------------------

    def acquire(
        self,
        target: Harnessed,
        usage: Mapping[str, float],
        *,
        timeout: float | None = 0.0,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> Outcome:
        method = "await_for_capacity" if target.is_async else "wait_for_capacity"
        return self.call(
            target,
            method,
            frozen_usage(usage),
            timeout=timeout,
            reservation_id=reservation_id,
            reservation_lifetime_seconds=reservation_lifetime_seconds,
        )

    def consume(
        self,
        target: Harnessed,
        usage: Mapping[str, float],
        *,
        reservation_id: str | None = None,
        reservation_lifetime_seconds: float | None = None,
    ) -> Outcome:
        return self.call(
            target,
            "consume_capacity",
            frozen_usage(usage),
            reservation_id=reservation_id,
            reservation_lifetime_seconds=reservation_lifetime_seconds,
        )

    def refund(
        self,
        target: Harnessed,
        reserved: Mapping[str, float],
        actual: Mapping[str, float],
    ) -> Outcome:
        return self.call(
            target, "refund_capacity", frozen_usage(reserved), frozen_usage(actual)
        )

    def refund_for_buckets(
        self,
        target: Harnessed,
        reserved: Mapping[str, float],
        actual: Mapping[str, float],
        *,
        bucket_ids: frozenset[BucketId] | None,
        reservation_id: str | None,
        reservation_model_family: str | None,
        reservation_bucket_ids: frozenset[BucketId] | None,
        reservation_reserved_usage: Mapping[str, float] | None,
    ) -> Outcome:
        return self.call(
            target,
            "refund_capacity_for_buckets",
            frozen_usage(reserved),
            frozen_usage(actual),
            bucket_ids=bucket_ids,
            reservation_id=reservation_id,
            reservation_model_family=reservation_model_family,
            reservation_bucket_ids=reservation_bucket_ids,
            reservation_reserved_usage=(
                None
                if reservation_reserved_usage is None
                else frozen_usage(reservation_reserved_usage)
            ),
        )

    def set_max_capacity(
        self, target: Harnessed, metric: str, per_seconds: int, value: float
    ) -> Outcome:
        return self.call(target, "set_max_capacity", metric, per_seconds, value)

    def apply_configured_max_capacity(
        self, target: Harnessed, metric: str, per_seconds: int, value: float
    ) -> Outcome:
        return self.call(
            target, "apply_configured_max_capacity", metric, per_seconds, value
        )

    # -- observation ---------------------------------------------------------

    def bucket_diagnostic(self, target: Harnessed, bucket_id: BucketId):
        outcome = self.call(target, "introspect")
        if outcome[0] != "ok":
            raise RuntimeError(f"{target.name}: introspect failed: {outcome!r}")
        for bucket in outcome[1].buckets:
            if (bucket.metric, int(bucket.per_seconds)) == bucket_id:
                return bucket
        raise KeyError(bucket_id)

    def capacities(self, target: Harnessed) -> dict[BucketId, tuple[float, float]]:
        outcome = self.call(target, "introspect")
        if outcome[0] != "ok":
            raise RuntimeError(f"{target.name}: introspect failed: {outcome!r}")
        diagnostic = outcome[1]
        result: dict[BucketId, tuple[float, float]] = {}
        for bucket in diagnostic.buckets:
            if bucket.current_capacity is None:
                raise RuntimeError(
                    f"{target.name}: introspect returned current_capacity=None for "
                    f"{bucket.metric}/{bucket.per_seconds}"
                )
            result[(bucket.metric, int(bucket.per_seconds))] = (
                float(bucket.current_capacity),
                float(bucket.effective_max_capacity),
            )
        return result


def capacities_close(
    left: Mapping[BucketId, tuple[float, float]],
    right: Mapping[BucketId, tuple[float, float]],
) -> bool:
    if set(left) != set(right):
        return False
    for bucket_id, (cap_l, max_l) in left.items():
        cap_r, max_r = right[bucket_id]
        if not math.isclose(cap_l, cap_r, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            return False
        if not math.isclose(max_l, max_r, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            return False
    return True


def describe(outcomes: Mapping[str, Outcome]) -> str:
    return "\n".join(f"  {name:<14} {outcome!r}" for name, outcome in outcomes.items())


def describe_capacities(
    snapshots: Mapping[str, Mapping[BucketId, tuple[float, float]]],
) -> str:
    lines = []
    for name, snap in snapshots.items():
        cells = ", ".join(
            f"{metric}/{window}s={cap:.6f}(max {maximum:g})"
            for (metric, window), (cap, maximum) in sorted(snap.items())
        )
        lines.append(f"  {name:<14} {cells}")
    return "\n".join(lines)


ALL_BUCKETS = frozenset


def usage_of(**amounts: float) -> frozendict[str, float]:
    return frozen_usage(amounts)
