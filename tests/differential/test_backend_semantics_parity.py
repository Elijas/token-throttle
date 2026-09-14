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
            "Redis marker expiry is real-time PX; covered by devtools/repros/marker_expiry_boundary.py"
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
#
# introspect() is a diagnostic of the DECISION path: after
# apply_configured_max_capacity the configured limit IS the new value (no
# override in play); after set_max_capacity the override is reported as such.


@pytest.mark.parametrize(("kind", "mode"), _params())
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
            bucket = driver.bucket_diagnostic(target, ("requests", 60))
            assert bucket.effective_max_capacity == pytest.approx(0.5)
            assert bucket.current_capacity == pytest.approx(0.5)
            assert bucket.configured_limit == pytest.approx(0.5)
            assert bucket.override_source == "none"
            # A runtime override on top is reported as an override.
            assert driver.set_max_capacity(target, "requests", 60, 0.7) == ("ok", None)
            bucket = driver.bucket_diagnostic(target, ("requests", 60))
            assert bucket.effective_max_capacity == pytest.approx(0.7)
            assert bucket.configured_limit == pytest.approx(0.5)
            assert bucket.override_source == "backend"
        finally:
            target.cleanup()


# --------------------------------------------------------------------------- D3
#
# docs/operations.md ("duplicate_acquire ... only if a reservation_id is reused
# across two acquire attempts with different usage or buckets") and
# docs/custom-backends.md (DuplicateRefundError "when the backend can prove a
# reservation was already refunded or already acquired") fix ONE semantics:
#   * identical replay of a live reservation -> success, nothing consumed twice
#   * reuse with different usage while the reservation is live -> duplicate_acquire
#   * reuse after the reservation was refunded -> duplicate_acquire


@pytest.mark.parametrize(("kind", "mode"), _params())
def test_d3_identical_replay_of_live_reservation_is_idempotent(
    kind: str, mode: str, loop, tmp_path: Path
) -> None:
    clock = FakeClock()
    driver = Driver(loop)
    with patched_clock(clock):
        target = _build(kind, mode, clock, loop, tmp_path)
        try:
            rid = f"d3-{uuid.uuid4().hex}"
            usage = {"requests": 4.0}
            first = driver.acquire(
                target, usage, reservation_id=rid, reservation_lifetime_seconds=LIFETIME
            )
            assert first[0] == "ok", first
            replay = driver.acquire(
                target, usage, reservation_id=rid, reservation_lifetime_seconds=LIFETIME
            )
            assert replay[0] == "ok", replay
            assert driver.capacities(target)[("requests", 60)][0] == pytest.approx(6.0)
            # The reservation is still refundable exactly once.
            assert _marker_refund(driver, target, rid, usage, {"requests": 0.0}) == (
                "ok",
                True,
            )
            assert driver.capacities(target)[("requests", 60)][0] == pytest.approx(10.0)
        finally:
            target.cleanup()


@pytest.mark.parametrize(("kind", "mode"), _params())
def test_d3_reuse_of_live_reservation_with_different_usage_is_rejected(
    kind: str, mode: str, loop, tmp_path: Path
) -> None:
    clock = FakeClock()
    driver = Driver(loop)
    with patched_clock(clock):
        target = _build(kind, mode, clock, loop, tmp_path)
        try:
            rid = f"d3-{uuid.uuid4().hex}"
            assert (
                driver.acquire(
                    target,
                    {"requests": 4.0},
                    reservation_id=rid,
                    reservation_lifetime_seconds=LIFETIME,
                )[0]
                == "ok"
            )
            second = driver.acquire(
                target,
                {"requests": 1.0},
                reservation_id=rid,
                reservation_lifetime_seconds=LIFETIME,
            )
            assert second == (
                "exc",
                DuplicateRefundError.__name__,
                "duplicate_acquire",
            ), second
            assert driver.capacities(target)[("requests", 60)][0] == pytest.approx(6.0)
        finally:
            target.cleanup()


@pytest.mark.parametrize(("kind", "mode"), _params())
def test_d3_reacquire_of_refunded_reservation_id_is_rejected(
    kind: str, mode: str, loop, tmp_path: Path
) -> None:
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
            assert second == (
                "exc",
                DuplicateRefundError.__name__,
                "duplicate_acquire",
            ), second
            # Nothing was consumed by the rejected re-acquire.
            assert driver.capacities(target)[("requests", 60)][0] == pytest.approx(10.0)
        finally:
            target.cleanup()


# --------------------------------------------------------------------------- D4
#
# DEVELOPMENT.md "Redis max_capacity_override self-heals on config mismatch":
# an override created under a previous quota configuration must not pin a
# deployment with a different configured limit; a process whose configured
# limit still matches keeps honouring it.


def _persisted_pair(kind, mode, clock, loop, tmp_path, prefix, db_path, limit):  # noqa: PLR0913
    kwargs = {"key_prefix": prefix}
    if kind == "sqlite":
        kwargs["db_path"] = db_path
    return build_one(
        kind,
        mode,
        make_config(FAMILY, (("requests", 60, limit),)),
        clock,
        loop=loop,
        tmp_path=tmp_path,
        **kwargs,
    )


@pytest.mark.parametrize(("kind", "mode"), _params())
def test_d4_override_from_other_configured_limit_is_ignored_by_new_process(
    kind: str, mode: str, loop, tmp_path: Path
) -> None:
    if kind == "memory":
        pytest.skip("memory is process-local; no persisted override to inherit")
    clock = FakeClock()
    driver = Driver(loop)
    prefix = f"d4-{uuid.uuid4().hex}"
    db_path = tmp_path / "d4.sqlite3"
    with patched_clock(clock):
        first = _persisted_pair(
            kind, mode, clock, loop, tmp_path, prefix, db_path, 100.0
        )
        purge = first.cleanups.pop() if kind == "redis" else None
        try:
            assert driver.set_max_capacity(first, "requests", 60, 50.0) == ("ok", None)
            # The process whose configured limit matches keeps honouring it.
            same = _persisted_pair(
                kind, mode, clock, loop, tmp_path, prefix, db_path, 100.0
            )
            try:
                assert driver.bucket_diagnostic(
                    same, ("requests", 60)
                ).effective_max_capacity == pytest.approx(50.0)
                assert driver.acquire(same, {"requests": 60.0})[:2] == (
                    "exc",
                    "ValueError",
                )
            finally:
                if kind == "redis":
                    same.cleanups.pop()
                same.cleanup()
        finally:
            first.cleanup()
        second = _persisted_pair(
            kind, mode, clock, loop, tmp_path, prefix, db_path, 200.0
        )
        try:
            bucket = driver.bucket_diagnostic(second, ("requests", 60))
            assert bucket.effective_max_capacity == pytest.approx(200.0)
            assert bucket.override_source == "none"
            assert driver.acquire(second, {"requests": 150.0})[0] == "ok"
        finally:
            second.cleanup()
            if purge is not None:
                purge()


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_d4_sqlite_schema_version_1_database_is_migrated_in_place(
    mode: str, loop, tmp_path: Path
) -> None:
    """A database written by the previous schema (no override anchor column)
    opens, is upgraded, and keeps its bucket state.
    """
    import sqlite3  # noqa: PLC0415

    db_path = tmp_path / "v1.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta(key, value) VALUES ('schema_version', '1');
        CREATE TABLE buckets (
            key_prefix TEXT NOT NULL, model_family TEXT NOT NULL, metric TEXT NOT NULL,
            per_seconds INTEGER NOT NULL, capacity REAL, last_checked REAL,
            override_value REAL, override_expires_at REAL, updated_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            PRIMARY KEY (key_prefix, model_family, metric, per_seconds));
        CREATE TABLE acquire_markers (
            key_prefix TEXT NOT NULL, reservation_id TEXT NOT NULL, model_family TEXT NOT NULL,
            bucket_ids_json TEXT NOT NULL, reserved_usage_json TEXT NOT NULL,
            created_at REAL NOT NULL, expires_at REAL NOT NULL,
            PRIMARY KEY (key_prefix, reservation_id));
        CREATE TABLE refund_tombstones (
            key_prefix TEXT NOT NULL, reservation_id TEXT NOT NULL, refunded_at REAL NOT NULL,
            expires_at REAL NOT NULL, PRIMARY KEY (key_prefix, reservation_id));
        """
    )
    now = time.time()
    # A drained bucket (capacity 3 of 10) with a live override written by the old schema.
    connection.execute(
        "INSERT INTO buckets VALUES ('mig', ?, 'requests', 60, 3.0, ?, 8.0, ?, ?, ?)",
        (FAMILY, now, now + 3600, now, now + 3600),
    )
    connection.commit()
    connection.close()
    clock = FakeClock(now)
    driver = Driver(loop)
    with patched_clock(clock):
        target = build_one(
            "sqlite",
            mode,
            make_config(FAMILY, (("requests", 60, 10.0),)),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            key_prefix="mig",
            db_path=db_path,
        )
        try:
            bucket = driver.bucket_diagnostic(target, ("requests", 60))
            # Bucket state survived the upgrade; the un-anchored legacy override
            # cannot prove which configured limit it was set under, so it is
            # not applied.
            assert bucket.current_capacity == pytest.approx(3.0)
            assert bucket.effective_max_capacity == pytest.approx(10.0)
            assert bucket.override_source == "none"
            assert driver.set_max_capacity(target, "requests", 60, 8.0) == ("ok", None)
            assert driver.bucket_diagnostic(
                target, ("requests", 60)
            ).effective_max_capacity == pytest.approx(8.0)
        finally:
            target.cleanup()
    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone() == ("2",)
    connection.close()


# --------------------------------------------------------------------------- D5
#
# Negative capacity is preserved by design and floored at -max_capacity, so a
# bucket needs TWO quota windows to refill from its deepest debt. A bucket
# lifetime equal to one window lets expiry forgive that debt; the validator
# must therefore require bucket_ttl_seconds >= 2 * per_seconds.


@pytest.mark.parametrize(("kind", "mode"), _params())
def test_d5_bucket_ttl_below_two_windows_is_rejected(
    kind: str, mode: str, loop, tmp_path: Path
) -> None:
    if kind == "memory":
        pytest.skip("memory buckets have no TTL")
    for ttl in (1, 2, 3):  # window = 2 s: 1 and 3 are below two windows, 4 is the floor
        with pytest.raises(ValueError, match="bucket_ttl_seconds"):
            build_one(
                kind,
                mode,
                make_config(FAMILY, (("requests", 2, 10.0),)),
                RealClock(),
                loop=loop,
                tmp_path=tmp_path,
                ttl_seconds=ttl,
                override_ttl_seconds=ttl,
            )
    target = build_one(
        kind,
        mode,
        make_config(FAMILY, (("requests", 2, 10.0),)),
        RealClock(),
        loop=loop,
        tmp_path=tmp_path,
        ttl_seconds=4,
        override_ttl_seconds=4,
    )
    target.cleanup()


@pytest.mark.parametrize(("kind", "mode"), _params())
def test_d5_debt_is_not_forgiven_by_bucket_ttl_while_polled(
    kind: str, mode: str, loop, tmp_path: Path
) -> None:
    """Real clock, bucket_ttl_seconds == 2 * per_seconds (the floor). Window 4 s,
    limit 10 (2.5 units/s): from -10, 5 units are honestly available only from
    t = 6 s, so any success inside the first 5 s means expiry forgave debt.
    Slow refill keeps per-operation latency from masking the outcome.
    """
    if kind == "memory":
        pytest.skip("memory buckets never expire")
    driver = Driver(loop)
    target = build_one(
        kind,
        mode,
        make_config(FAMILY, (("requests", 4, 10.0),)),
        RealClock(),
        loop=loop,
        tmp_path=tmp_path,
        ttl_seconds=8,
        override_ttl_seconds=8,
    )
    try:
        assert driver.acquire(target, {"requests": 10.0})[0] == "ok"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert driver.consume(target, {"requests": 10.0})[0] == "ok"
        start = time.monotonic()
        assert driver.capacities(target)[("requests", 4)][0] < -5.0  # in debt
        granted_at = None
        while time.monotonic() - start < 5.0:
            if driver.acquire(target, {"requests": 5.0})[0] == "ok":
                granted_at = round(time.monotonic() - start, 2)
                break
            time.sleep(0.25)
        assert granted_at is None, (
            f"try_acquire(5) granted at t={granted_at}s from a -10 debt"
        )
    finally:
        target.cleanup()


# --------------------------------------------------------------------------- D8


@pytest.mark.parametrize(
    ("kind", "mode"),
    _params(),  # D8 fixed on this clone: the SQLite override now survives bucket-row expiry
)
def test_d8_override_lifetime_is_independent_of_bucket_row_expiry(
    kind: str, mode: str, loop, tmp_path: Path
) -> None:
    """docs/sqlite-backend.md TTL table: override_ttl_seconds is the "fixed
    lifetime of a shared set_max_capacity() override, measured from the call
    that writes it". Redis keeps the override on its own key, so bucket-key
    expiry does not touch it.
    """
    if kind == "memory":
        pytest.skip("memory buckets never expire")
    driver = Driver(loop)
    target = build_one(
        kind,
        mode,
        make_config(FAMILY, (("requests", 1, 10.0),)),
        RealClock(),
        loop=loop,
        tmp_path=tmp_path,
        ttl_seconds=2,
        override_ttl_seconds=6,
    )
    try:
        assert driver.set_max_capacity(target, "requests", 1, 4.0) == ("ok", None)
        assert driver.capacities(target)[("requests", 1)][1] == pytest.approx(4.0)
        time.sleep(2.4)  # idle past bucket_ttl, well inside override_ttl
        # The override must still govern both the diagnostic and the decision.
        assert driver.acquire(target, {"requests": 5.0})[:2] == ("exc", "ValueError")
        assert driver.capacities(target)[("requests", 1)][1] == pytest.approx(4.0)
    finally:
        target.cleanup()


# --------------------------------------------------------------------------- D9
#
# Operations on a closed SQLite backend raise RuntimeError on both twins
# (DEVELOPMENT.md reserves RuntimeError for broken invariants the caller cannot
# fix by changing arguments); the driver's own exception must not leak.


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("close_via", ["builder", "backend"])
def test_d9_closed_sqlite_backend_raises_runtime_error(
    mode: str, close_via: str, loop, tmp_path: Path
) -> None:
    clock = FakeClock()
    driver = Driver(loop)
    with patched_clock(clock):
        target = build_one(
            "sqlite",
            mode,
            make_config(FAMILY, QUOTAS),
            clock,
            loop=loop,
            tmp_path=tmp_path,
        )
        if close_via == "builder":
            target.cleanups.pop()()  # the builder close registered by build_one
        else:
            driver.call(target, "aclose" if mode == "async" else "close")
        try:
            for op in (
                lambda: driver.acquire(target, {"requests": 1.0}),
                lambda: driver.consume(target, {"requests": 1.0}),
                lambda: driver.refund(target, {"requests": 1.0}, {"requests": 0.0}),
                lambda: driver.set_max_capacity(target, "requests", 60, 5.0),
                lambda: driver.apply_configured_max_capacity(
                    target, "requests", 60, 5.0
                ),
                lambda: driver.call(target, "introspect"),
            ):
                outcome = op()
                assert outcome[:2] == ("exc", "RuntimeError"), outcome
        finally:
            target.cleanups.clear()
