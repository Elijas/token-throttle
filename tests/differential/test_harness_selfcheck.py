# ruff: noqa: TC003
"""
Prove the differential comparator is not green by construction.

Build memory and SQLite on the fake clock, drive them identically, then apply
an operation to only one of them and assert the capacity comparator reports a
divergence. A comparator that cannot fail here is worth nothing.
"""

from __future__ import annotations

import asyncio
import warnings
from pathlib import Path

import pytest

from tests.differential._backends import build_all, make_config
from tests.differential._clock import FakeClock, patched_clock
from tests.differential._driver import Driver, capacities_close

QUOTAS = (("requests", 60, 10.0), ("tokens", 60, 1000.0))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_comparator_detects_injected_capacity_divergence(tmp_path: Path) -> None:
    loop = asyncio.new_event_loop()
    clock = FakeClock()
    driver = Driver(loop)
    with patched_clock(clock):
        targets = build_all(
            make_config("selfcheck", QUOTAS),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            kinds=("memory", "sqlite"),
        )
        try:
            usage = {"requests": 3.0, "tokens": 100.0}
            for target in targets:
                assert driver.acquire(target, usage)[0] == "ok"
            snapshots = [driver.capacities(t) for t in targets]
            assert all(capacities_close(snapshots[0], snap) for snap in snapshots[1:])

            # Consume on the SQLite pair only: the comparator must notice.
            sqlite_targets = [t for t in targets if t.kind == "sqlite"]
            for target in sqlite_targets:
                assert (
                    driver.consume(target, {"requests": 1.0, "tokens": 0.0})[0] == "ok"
                )
            memory_snapshot = driver.capacities(targets[0])
            for target in sqlite_targets:
                assert not capacities_close(memory_snapshot, driver.capacities(target))

            # Clock advance refills both identically again only after equal writes.
            clock.advance(60.0)
            for target in targets:
                assert (
                    driver.consume(target, {"requests": 0.0, "tokens": 0.0})[0] == "ok"
                )
            memory_snapshot = driver.capacities(targets[0])
            # Both are full after one window, so the injected gap has closed.
            for target in targets[1:]:
                assert capacities_close(memory_snapshot, driver.capacities(target))
        finally:
            for target in targets:
                target.cleanup()
    loop.close()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_fake_clock_drives_refill_on_every_backend(tmp_path: Path) -> None:
    """The injected clock, not the wall clock, must govern refill on each backend."""
    loop = asyncio.new_event_loop()
    clock = FakeClock()
    driver = Driver(loop)
    with warnings.catch_warnings(), patched_clock(clock):
        warnings.simplefilter("ignore", RuntimeWarning)
        targets = build_all(
            make_config("clockcheck", QUOTAS),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            kinds=("memory", "sqlite"),
        )
        try:
            for target in targets:
                assert (
                    driver.acquire(target, {"requests": 10.0, "tokens": 0.0})[0] == "ok"
                )
                assert driver.capacities(target)[("requests", 60)][0] == pytest.approx(
                    0.0
                )
            clock.advance(30.0)
            for target in targets:
                assert driver.capacities(target)[("requests", 60)][0] == pytest.approx(
                    5.0
                )
            clock.advance(30.0)
            for target in targets:
                assert driver.capacities(target)[("requests", 60)][0] == pytest.approx(
                    10.0
                )
        finally:
            for target in targets:
                target.cleanup()
    loop.close()
