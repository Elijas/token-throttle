"""Expiry must not manufacture capacity during runtime-limit transitions."""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from tests.differential._backends import build_one, make_config, require_redis
from tests.differential._clock import FakeClock, patched_clock
from tests.differential._driver import Driver


def _wait_for_server_duration(seconds: int) -> None:
    """Wait on an independent sentinel so accounting keys may retain history."""
    import redis  # noqa: PLC0415

    key = f"expiry-test-sentinel-{uuid.uuid4().hex}"
    with redis.Redis.from_url(require_redis()) as observer:
        observer.set(key, "1", ex=seconds)
        deadline = time.monotonic() + 5.0
        while observer.exists(key):
            assert time.monotonic() < deadline, "Owned test key did not expire"
            time.sleep(0.01)


@pytest.mark.parametrize("kind", ["sqlite", "redis"])
@pytest.mark.parametrize("mode", ["async", "sync"])
def test_override_expiry_preserves_refill_accrued_at_previous_rate(
    kind, mode, tmp_path
):
    loop = asyncio.new_event_loop()
    driver = Driver(loop)
    clock = FakeClock()
    with patched_clock(clock):
        target = build_one(
            kind,
            mode,
            make_config("expiry-rate", [("requests", 10, 10.0)]),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            ttl_seconds=100,
            override_ttl_seconds=1,
        )
        try:
            assert driver.consume(target, {"requests": 10.0})[0] == "ok"
            assert driver.set_max_capacity(target, "requests", 10, 1.0)[0] == "ok"
            if kind == "redis":
                _wait_for_server_duration(1)
            clock.advance(1.25)
            # 1 second at 0.1/s, followed by 0.25 seconds at 1/s.
            capacity, maximum = driver.capacities(target)[("requests", 10)]
            outcome = driver.acquire(target, {"requests": 1.0}, timeout=0)
            assert outcome[:2] == ("exc", "TimeoutError"), (
                f"admitted at capacity={capacity}, expected 0.35; {outcome=}"
            )
            assert maximum == 10.0
            assert capacity == pytest.approx(0.35)
        finally:
            target.cleanup()
            loop.close()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("kind", ["sqlite", "redis"])
@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("cold_observer", [False, True])
def test_expired_override_history_outlives_original_ttl_with_retained_debt(
    kind, mode, cold_observer, tmp_path
):
    loop = asyncio.new_event_loop()
    driver = Driver(loop)
    clock = FakeClock()
    prefix = f"expiry-retained-debt-{uuid.uuid4().hex}"
    config = make_config("expiry-retained-debt", [("requests", 1, 100.0)])
    targets = []
    with patched_clock(clock):

        def build():
            target = build_one(
                kind,
                mode,
                config,
                clock,
                loop=loop,
                tmp_path=tmp_path,
                key_prefix=prefix,
                db_path=tmp_path / "retained-debt.sqlite3",
                ttl_seconds=2,
                override_ttl_seconds=1,
                max_reservation_lifetime_seconds=0.5,
            )
            targets.append(target)
            return target

        try:
            writer = build()
            assert driver.consume(writer, {"requests": 200.0})[0] == "ok"
            assert driver.set_max_capacity(writer, "requests", 1, 1.0)[0] == "ok"
            if kind == "redis":
                _wait_for_server_duration(2)
            clock.advance(2.25)
            observer = build() if cold_observer else writer
            # Debt accrues 1 token before expiry, then 125 at the restored rate.
            # Expiring the old two-second state or its history would invent tokens.
            for _ in range(2):
                capacity, maximum = driver.capacities(observer)[("requests", 1)]
                assert maximum == 100.0
                assert capacity == pytest.approx(26.0)
            assert driver.acquire(observer, {"requests": 27.0}, timeout=0)[:2] == (
                "exc",
                "TimeoutError",
            )
        finally:
            for target in reversed(targets):
                target.cleanup()
            loop.close()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("kind", ["memory", "sqlite", "redis"])
@pytest.mark.parametrize("mode", ["async", "sync"])
def test_lowered_maximum_debt_outlives_original_two_window_bucket_ttl(
    kind, mode, tmp_path
):
    loop = asyncio.new_event_loop()
    driver = Driver(loop)
    clock = FakeClock()
    with patched_clock(clock):
        target = build_one(
            kind,
            mode,
            make_config("expiry-debt", [("requests", 1, 100.0)]),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            ttl_seconds=2,
            override_ttl_seconds=100,
            max_reservation_lifetime_seconds=0.5,
        )
        try:
            assert driver.consume(target, {"requests": 200.0})[0] == "ok"
            assert driver.set_max_capacity(target, "requests", 1, 1.0)[0] == "ok"
            assert driver.capacities(target)[("requests", 1)] == (-100.0, 1.0)
            if kind == "redis":
                _wait_for_server_duration(2)
            clock.advance(2.25)
            # The preserved -100 debt refills at the new 1/s rate.
            capacity, maximum = driver.capacities(target)[("requests", 1)]
            outcome = driver.acquire(target, {"requests": 1.0}, timeout=0)
            assert outcome[:2] == ("exc", "TimeoutError"), (
                f"admitted at capacity={capacity}, expected -97.75; {outcome=}"
            )
            assert maximum == 1.0
            assert capacity == pytest.approx(-97.75)
        finally:
            target.cleanup()
            loop.close()
