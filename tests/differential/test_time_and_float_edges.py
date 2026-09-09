# ruff: noqa: TC003, FBT001
"""
Time and float edges, six backends side by side under the fake clock.

* backward clock steps (small: all clamp; > 1 s: SQLite repairs the stored
  timestamp, memory and Redis keep the future timestamp -- documented)
* the largest window with the smallest limit (rate ~5e-19/s)
* a limit near 2**53 (float precision plateau, documented)
* usage exactly equal to max, and one ulp above
* sub-normal-ish usage that is absorbed by the capacity value
"""

from __future__ import annotations

import asyncio
import math
import warnings
from pathlib import Path

import pytest

from tests.differential._backends import build_all, make_config, require_redis
from tests.differential._clock import FakeClock, patched_clock
from tests.differential._driver import Driver, capacities_close, describe_capacities
from token_throttle._capacity import MIN_MAX_CAPACITY
from token_throttle._interfaces._models import MAX_PER_SECONDS

FAMILY = "edges"


@pytest.fixture
def loop():
    loop_ = asyncio.new_event_loop()
    yield loop_
    loop_.close()


def _all(driver, targets, op):
    outcomes = {t.name: op(t) for t in targets}
    kinds = {(o[0], o[1] if o[0] == "exc" else None) for o in outcomes.values()}
    assert len(kinds) == 1, outcomes
    snaps = {t.name: driver.capacities(t) for t in targets}
    ref = next(iter(snaps.values()))
    assert all(capacities_close(ref, s) for s in snaps.values()), describe_capacities(
        snaps
    )
    return next(iter(outcomes.values())), ref


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "with_redis", [False, pytest.param(True, marks=pytest.mark.redis)]
)
def test_small_backward_clock_step_clamps_identically(
    with_redis: bool, loop, tmp_path: Path
) -> None:
    kinds = ("memory", "sqlite", "redis") if with_redis else ("memory", "sqlite")
    if with_redis:
        require_redis()
    clock = FakeClock()
    driver = Driver(loop)
    with patched_clock(clock), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        targets = build_all(
            make_config(FAMILY, (("requests", 60, 10.0),)),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            kinds=kinds,
        )
        try:
            _all(driver, targets, lambda t: driver.acquire(t, {"requests": 10.0}))
            clock.advance(-0.5)  # within the 1 s repair tolerance: everyone clamps
            # A zero-usage consume is a WRITE: every backend re-anchors last_checked
            # at the (earlier) current time, so refill resumes from T-0.5 on all.
            _, snap = _all(
                driver, targets, lambda t: driver.consume(t, {"requests": 0.0})
            )
            assert snap[("requests", 60)][0] == pytest.approx(0.0)
            clock.advance(0.5)
            _, snap = _all(
                driver, targets, lambda t: driver.consume(t, {"requests": 0.0})
            )
            assert snap[("requests", 60)][0] == pytest.approx(0.5 / 6.0)
            clock.advance(6.0)
            _, snap = _all(
                driver, targets, lambda t: driver.consume(t, {"requests": 0.0})
            )
            assert snap[("requests", 60)][0] == pytest.approx(6.5 / 6.0)
        finally:
            for t in targets:
                t.cleanup()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_large_backward_clock_step_sqlite_repairs_future_timestamp(
    loop, tmp_path: Path
) -> None:
    """Documented (docs/sqlite-backend.md 'Clock authority'): a stored timestamp
    more than one second in the future is repaired; memory keeps it.
    """
    clock = FakeClock()
    driver = Driver(loop)
    with patched_clock(clock), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        targets = build_all(
            make_config(FAMILY, (("requests", 60, 10.0),)),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            kinds=("memory", "sqlite"),
        )
        by_kind = {t.kind: [x for x in targets if x.kind == t.kind] for t in targets}
        try:
            for t in targets:
                assert driver.acquire(t, {"requests": 10.0})[0] == "ok"
            clock.advance(-3.0)
            # A FAILED try-acquire writes nothing on memory, but SQLite's
            # _load_states (writable) repairs the future timestamp in place.
            for t in targets:
                assert driver.acquire(t, {"requests": 5.0})[:2] == (
                    "exc",
                    "TimeoutError",
                )
            clock.advance(2.0)  # now 1 s before the original timestamp
            for t in by_kind["memory"]:
                assert driver.capacities(t)[("requests", 60)][0] == pytest.approx(
                    0.0
                )  # still clamped
            for t in by_kind["sqlite"]:
                # SQLite repaired last_checked to (T-3); 2 s elapsed -> 2/6 refill
                assert driver.capacities(t)[("requests", 60)][0] == pytest.approx(
                    2.0 / 6.0
                )
        finally:
            for t in targets:
                t.cleanup()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "with_redis", [False, pytest.param(True, marks=pytest.mark.redis)]
)
def test_extreme_rate_and_precision_edges(
    with_redis: bool, loop, tmp_path: Path
) -> None:
    kinds = ("memory", "sqlite", "redis") if with_redis else ("memory", "sqlite")
    if with_redis:
        require_redis()
    clock = FakeClock()
    driver = Driver(loop)
    quotas = (
        ("slow", MAX_PER_SECONDS, MIN_MAX_CAPACITY),  # rate ~ 4.7e-19 / s
        ("huge", 60, 2.0**53),  # float plateau
        ("exact", 60, 10.0),
    )
    with patched_clock(clock), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        targets = build_all(
            make_config(FAMILY, quotas),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            kinds=kinds,
            ttl_seconds=MAX_PER_SECONDS,
        )
        try:
            zero = {"slow": 0.0, "huge": 0.0, "exact": 0.0}
            # usage == max exactly is accepted everywhere
            out, snap = _all(
                driver, targets, lambda t: driver.acquire(t, {**zero, "exact": 10.0})
            )
            assert out[0] == "ok"
            assert snap[("exact", 60)][0] == pytest.approx(0.0)
            # one ulp above max is rejected everywhere with ValueError
            out, _ = _all(
                driver,
                targets,
                lambda t: driver.acquire(
                    t, {**zero, "exact": math.nextafter(10.0, math.inf)}
                ),
            )
            assert out[:2] == ("exc", "ValueError")
            # smallest limit, largest window: consume it all, then refill for a year
            out, snap = _all(
                driver,
                targets,
                lambda t: driver.acquire(t, {**zero, "slow": MIN_MAX_CAPACITY}),
            )
            assert out[0] == "ok"
            clock.advance(365 * 86400.0)
            _, snap = _all(driver, targets, lambda t: driver.consume(t, zero))
            assert snap[("slow", MAX_PER_SECONDS)][0] == pytest.approx(
                MIN_MAX_CAPACITY * 365 * 86400 / MAX_PER_SECONDS
            )
            # float plateau: 1 unit from 2**53 is absorbed identically everywhere
            out, snap = _all(
                driver, targets, lambda t: driver.acquire(t, {**zero, "huge": 1.0})
            )
            assert out[0] == "ok"
            assert (
                snap[("huge", 60)][0] == 2.0**53 - 1.0
                or snap[("huge", 60)][0] == 2.0**53
            )
            # tiny usage absorbed by the capacity value
            out, snap = _all(
                driver, targets, lambda t: driver.consume(t, {**zero, "exact": 1e-300})
            )
            assert out[0] == "ok"
        finally:
            for t in targets:
                t.cleanup()
