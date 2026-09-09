# ruff: noqa: TC003
"""
Pins for the divergences found by the differential harness (tt-audit, 2026-09).

Each test states the behaviour the specification supports and asserts it for
every backend. Where a backend is judged to diverge from the specification,
its parametrisation carries ``xfail(strict=True)`` with the finding id, so a
fix flips the pin to XPASS and forces the mark to be removed. Where the
divergence is documented on both sides (design), the test asserts each
backend's documented behaviour as-is.

Findings (see tt-audit.REPORT.md §4):
  D1 marker expiry: memory ignores reservation_lifetime_seconds; SQLite expires
     at exactly the lifetime (inclusive); Redis expires by server PX.
     Documented design divergence.
  D2 Redis introspect() reports the ORIGINAL quota limit after
     apply_configured_max_capacity while its decision path uses the new one.
  D3 recycled reservation_id after refund: memory re-acquires and re-refunds;
     SQLite raises DuplicateRefundError; Redis re-acquires then the refund
     raises RedisScriptResultError and the capacity is never credited back.
  D4 persisted runtime override without a configured-limit anchor: a new SQLite
     process with a changed configured limit applies the stale override; Redis
     ignores it (documented self-healing).
  D5 bucket_ttl_seconds == per_seconds with negative capacity: SQLite's bucket
     row expires under read-only polling and the debt is forgiven; Redis
     refreshes the TTL on every read and the debt refills linearly.
"""

from __future__ import annotations

import asyncio
import time
import uuid
import warnings
from pathlib import Path

import pytest

from tests.differential._backends import build_one, make_config, require_redis
from tests.differential._clock import FakeClock, RealClock, patched_clock
from tests.differential._driver import Driver
from token_throttle._exceptions import DuplicateRefundError, UnknownReservationError

FAMILY = "known"
QUOTAS = (("requests", 60, 10.0),)
BUCKETS = frozenset({("requests", 60)})
LIFETIME = 3600.0

KINDS = ("memory", "sqlite", "redis")
MODES = ("async", "sync")


def _params(*, xfail: dict[str, str] | None = None) -> list:
    params = []
    for kind in KINDS:
        for mode in MODES:
            marks = [pytest.mark.redis] if kind == "redis" else []
            if xfail and kind in xfail:
                marks.append(pytest.mark.xfail(strict=True, reason=xfail[kind]))
            params.append(pytest.param(kind, mode, id=f"{kind}-{mode}", marks=marks))
    return params


@pytest.fixture
def loop():
    loop_ = asyncio.new_event_loop()
    yield loop_
    loop_.close()


def _build(kind: str, mode: str, clock: FakeClock, loop, tmp_path: Path, **kwargs):
    if kind == "redis":
        require_redis()
    return build_one(
        kind,
        mode,
        make_config(FAMILY, QUOTAS),
        clock,
        loop=loop,
        tmp_path=tmp_path,
        **kwargs,
    )


def _marker_refund(driver: Driver, target, rid: str, reserved: dict, actual: dict):
    return driver.refund_for_buckets(
        target,
        reserved,
        actual,
        bucket_ids=BUCKETS,
        reservation_id=rid,
        reservation_model_family=FAMILY,
        reservation_bucket_ids=BUCKETS,
        reservation_reserved_usage=reserved,
    )


# --------------------------------------------------------------------------- D1


@pytest.mark.parametrize(("kind", "mode"), _params())
def test_d1_marker_expiry_boundary(kind: str, mode: str, loop, tmp_path: Path) -> None:
    """Documented: memory ignores the lifetime; SQLite expires at exactly lifetime."""
    if kind == "redis":
        pytest.skip(
            "Redis marker expiry is real-time PX; covered by repros/marker_expiry_boundary.py"
        )
    clock = FakeClock()
    driver = Driver(loop)
    with patched_clock(clock):
        target = _build(kind, mode, clock, loop, tmp_path)
        try:
            rid = f"d1-{uuid.uuid4().hex}"
            usage = {"requests": 2.0}
            assert (
                driver.acquire(
                    target,
                    usage,
                    reservation_id=rid,
                    reservation_lifetime_seconds=LIFETIME,
                )[0]
                == "ok"
            )
            clock.advance(LIFETIME)
            outcome = _marker_refund(driver, target, rid, usage, {"requests": 0.0})
            if kind == "memory":
                assert outcome == ("ok", True)
            else:
                assert outcome[:2] == ("exc", UnknownReservationError.__name__)
        finally:
            target.cleanup()


# --------------------------------------------------------------------------- D2


@pytest.mark.parametrize(
    ("kind", "mode"),
    _params(
        xfail={
            "redis": "D2: Redis introspect() reports the original quota limit after apply_configured_max_capacity"
        }
    ),
)
def test_d2_introspect_reflects_apply_configured_max_capacity(
    kind: str, mode: str, loop, tmp_path: Path
) -> None:
    clock = FakeClock()
    driver = Driver(loop)
    with patched_clock(clock):
        target = _build(kind, mode, clock, loop, tmp_path)
        try:
            assert driver.apply_configured_max_capacity(
                target, "requests", 60, 0.5
            ) == ("ok", None)
            # Decision path: every backend enforces the new limit.
            assert driver.acquire(target, {"requests": 1.0})[:2] == (
                "exc",
                "ValueError",
            )
            # Observability: introspect must agree with the decision path.
            current, effective = driver.capacities(target)[("requests", 60)]
            assert effective == pytest.approx(0.5)
            assert current == pytest.approx(0.5)
        finally:
            target.cleanup()


# --------------------------------------------------------------------------- D3


@pytest.mark.parametrize(
    ("kind", "mode"),
    _params(
        xfail={
            "memory": "D3: memory re-acquires a refunded reservation_id (its refunded-id set is not consulted at acquire)",
            "redis": "D3: Redis re-acquires a refunded reservation_id and the second refund raises RedisScriptResultError",
        }
    ),
)
def test_d3_reacquire_of_refunded_reservation_id_is_rejected(
    kind: str, mode: str, loop, tmp_path: Path
) -> None:
    """docs/custom-backends.md error taxonomy: DuplicateRefundError when the backend
    can prove a reservation was already refunded or already acquired.
    """
    clock = FakeClock()
    driver = Driver(loop)
    with patched_clock(clock):
        target = _build(kind, mode, clock, loop, tmp_path)
        try:
            rid = f"d3-{uuid.uuid4().hex}"
            usage = {"requests": 4.0}
            acquire = lambda: driver.acquire(  # noqa: E731
                target, usage, reservation_id=rid, reservation_lifetime_seconds=LIFETIME
            )
            assert acquire()[0] == "ok"
            assert _marker_refund(driver, target, rid, usage, {"requests": 0.0}) == (
                "ok",
                True,
            )
            second = acquire()
            assert second[:2] == ("exc", DuplicateRefundError.__name__), second
            # Nothing was consumed by the rejected re-acquire.
            assert driver.capacities(target)[("requests", 60)][0] == pytest.approx(10.0)
        finally:
            target.cleanup()


# --------------------------------------------------------------------------- D4


@pytest.mark.parametrize(
    ("kind", "mode"),
    _params(
        xfail={
            "sqlite": "D4: SQLite applies a persisted override written under a different configured limit"
        }
    ),
)
def test_d4_override_from_other_configured_limit_is_ignored_by_new_process(
    kind: str, mode: str, loop, tmp_path: Path
) -> None:
    """DEVELOPMENT.md 'Redis max_capacity_override self-heals on config mismatch':
    an override created under a previous quota configuration must not pin the
    new deployment to a stale limit.
    """
    if kind == "memory":
        pytest.skip("memory is process-local; no persisted override to inherit")
    clock = FakeClock()
    driver = Driver(loop)
    prefix = f"d4-{uuid.uuid4().hex}"
    db_path = tmp_path / "d4.sqlite3"
    with patched_clock(clock):
        first = (
            build_one(
                kind,
                mode,
                make_config(FAMILY, (("requests", 60, 100.0),)),
                clock,
                loop=loop,
                tmp_path=tmp_path,
                key_prefix=prefix,
                db_path=db_path,
            )
            if kind != "redis"
            else build_one(
                kind,
                mode,
                make_config(FAMILY, (("requests", 60, 100.0),)),
                clock,
                loop=loop,
                tmp_path=tmp_path,
                key_prefix=prefix,
            )
        )
        try:
            assert driver.set_max_capacity(first, "requests", 60, 50.0) == ("ok", None)
        finally:
            # Close the builder/backend but keep the store (Redis purge is the
            # last cleanup; run everything except it).
            purge = first.cleanups.pop() if kind == "redis" else None
            first.cleanup()
        second = build_one(
            kind,
            mode,
            make_config(FAMILY, (("requests", 60, 200.0),)),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            key_prefix=prefix,
            db_path=db_path,
        )
        try:
            _, effective = driver.capacities(second)[("requests", 60)]
            assert effective == pytest.approx(200.0)
            assert driver.acquire(second, {"requests": 150.0})[0] == "ok"
        finally:
            second.cleanup()
            if purge is not None:
                purge()


# --------------------------------------------------------------------------- D5


@pytest.mark.parametrize(
    ("kind", "mode"),
    _params(
        xfail={
            "sqlite": "D5: SQLite bucket row expires under read-only polling and the debt is forgiven"
        }
    ),
)
def test_d5_debt_is_not_forgiven_by_bucket_ttl_while_polled(
    kind: str, mode: str, loop, tmp_path: Path
) -> None:
    """Real clock, bucket_ttl_seconds == per_seconds == 1 (allowed by the validator).
    Capacity floored at -max needs two windows to refill; a poller must not see
    full capacity after one window.
    """
    if kind == "memory":
        pytest.skip("memory buckets never expire")
    clock = (
        RealClock()
    )  # real time drives TTLs here (build_one binds it into the engine)
    driver = Driver(loop)
    target = build_one(
        kind,
        mode,
        make_config(FAMILY, (("requests", 1, 10.0),)),
        clock,
        loop=loop,
        tmp_path=tmp_path,
        ttl_seconds=1,
        override_ttl_seconds=1,
    )
    try:
        assert driver.acquire(target, {"requests": 10.0})[0] == "ok"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert driver.consume(target, {"requests": 10.0})[0] == "ok"
        assert (
            driver.capacities(target)[("requests", 1)][0] < -8.0
        )  # real clock: a little refill already
        start = time.monotonic()
        granted_at = None
        while time.monotonic() - start < 1.4:
            # Honest curve: -10 + 10/s * t, so 5 units are never available before
            # t = 1.5 s. A success inside 1.4 windows means the debt was forgiven.
            if driver.acquire(target, {"requests": 5.0})[0] == "ok":
                granted_at = round(time.monotonic() - start, 2)
                break
            time.sleep(0.1)
        assert granted_at is None, (
            f"try_acquire(5) granted at t={granted_at}s from a -10 debt"
        )
    finally:
        target.cleanup()
