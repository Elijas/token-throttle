# ruff: noqa: TC003, PLR0913
"""
Restart differential: SQLite and Redis are closed and reopened (new builder,
same store) in the middle of an operation sequence; the memory backend runs
continuously as the oracle for "what the state would be with no restart".

Checks: persisted capacity continues from where it was, refill across the
restart is identical, markers acquired before the restart are refundable after
it (backend-level, with marker metadata), tombstones survive (duplicate refund
still rejected), and a runtime override written before the restart is still
in force after it (same configured limit).
"""

from __future__ import annotations

import asyncio
import uuid
import warnings
from pathlib import Path

import pytest

from tests.differential._backends import (
    Harnessed,
    build_one,
    make_config,
    require_redis,
)
from tests.differential._clock import FakeClock, patched_clock
from tests.differential._driver import Driver, capacities_close

FAMILY = "restart"
QUOTAS = (("requests", 60, 10.0), ("tokens", 60, 1000.0))
BUCKETS = frozenset((m, w) for m, w, _ in QUOTAS)
LIFETIME = 7 * 86400.0


@pytest.fixture
def loop():
    loop_ = asyncio.new_event_loop()
    yield loop_
    loop_.close()


def _reopen(
    target: Harnessed,
    clock: FakeClock,
    loop,
    tmp_path: Path,
    prefix: str,
    db_path: Path,
) -> Harnessed:
    """Close builder + backend, keep the store, build a fresh backend on it."""
    purge = target.cleanups.pop() if target.kind == "redis" else None
    target.cleanup()
    mode = "async" if target.is_async else "sync"
    kwargs = {"key_prefix": prefix}
    if target.kind == "sqlite":
        kwargs["db_path"] = db_path
    reopened = build_one(
        target.kind,
        mode,
        make_config(FAMILY, QUOTAS),
        clock,
        loop=loop,
        tmp_path=tmp_path,
        **kwargs,
    )
    if purge is not None:
        reopened.cleanups.append(purge)
    return reopened


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize(
    "kind", ["sqlite", pytest.param("redis", marks=pytest.mark.redis)]
)
def test_state_survives_restart_and_matches_continuous_memory(
    kind: str, mode: str, loop, tmp_path: Path
) -> None:
    if kind == "redis":
        require_redis()
    clock = FakeClock()
    driver = Driver(loop)
    prefix = f"restart-{uuid.uuid4().hex}"
    db_path = tmp_path / "restart.sqlite3"
    with patched_clock(clock), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        oracle = build_one(
            "memory",
            mode,
            make_config(FAMILY, QUOTAS),
            clock,
            loop=loop,
            tmp_path=tmp_path,
        )
        durable = build_one(
            kind,
            mode,
            make_config(FAMILY, QUOTAS),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            key_prefix=prefix,
            db_path=db_path if kind == "sqlite" else None,
        )
        pair = [oracle, durable]
        try:

            def both(op):
                outcomes = [op(t) for t in pair]
                assert outcomes[0][:2] == outcomes[1][:2], outcomes
                assert capacities_close(
                    driver.capacities(oracle), driver.capacities(durable)
                ), (driver.capacities(oracle), driver.capacities(durable))
                return outcomes[0]

            rid_keep = f"keep-{uuid.uuid4().hex}"
            rid_done = f"done-{uuid.uuid4().hex}"
            u1 = {"requests": 3.0, "tokens": 300.0}
            u2 = {"requests": 2.0, "tokens": 100.0}
            assert (
                both(
                    lambda t: driver.acquire(
                        t,
                        u1,
                        reservation_id=rid_keep,
                        reservation_lifetime_seconds=LIFETIME,
                    )
                )[0]
                == "ok"
            )
            assert (
                both(
                    lambda t: driver.acquire(
                        t,
                        u2,
                        reservation_id=rid_done,
                        reservation_lifetime_seconds=LIFETIME,
                    )
                )[0]
                == "ok"
            )
            assert (
                both(
                    lambda t: driver.refund_for_buckets(
                        t,
                        u2,
                        {"requests": 1.0, "tokens": 50.0},
                        bucket_ids=BUCKETS,
                        reservation_id=rid_done,
                        reservation_model_family=FAMILY,
                        reservation_bucket_ids=BUCKETS,
                        reservation_reserved_usage=u2,
                    )
                )[0]
                == "ok"
            )
            assert (
                both(lambda t: driver.consume(t, {"requests": 4.0, "tokens": 0.0}))[0]
                == "ok"
            )
            assert (
                both(lambda t: driver.set_max_capacity(t, "tokens", 60, 800.0))[0]
                == "ok"
            )
            clock.advance(15.0)
            both(
                lambda t: driver.consume(t, {"requests": 0.0, "tokens": 0.0})
            )  # observe

            # ---- restart the durable backend; memory keeps running ----
            durable = _reopen(durable, clock, loop, tmp_path, prefix, db_path)
            pair[1] = durable
            assert capacities_close(
                driver.capacities(oracle), driver.capacities(durable)
            ), (driver.capacities(oracle), driver.capacities(durable))

            clock.advance(20.0)
            both(lambda t: driver.consume(t, {"requests": 0.0, "tokens": 0.0}))
            # Marker acquired before the restart is refundable after it.
            assert (
                both(
                    lambda t: driver.refund_for_buckets(
                        t,
                        u1,
                        {"requests": 0.0, "tokens": 0.0},
                        bucket_ids=BUCKETS,
                        reservation_id=rid_keep,
                        reservation_model_family=FAMILY,
                        reservation_bucket_ids=BUCKETS,
                        reservation_reserved_usage=u1,
                    )
                )[0]
                == "ok"
            )
            # Tombstone survived: duplicate refund of the pre-restart refund is rejected.
            outcome = driver.refund_for_buckets(
                durable,
                u2,
                {"requests": 1.0, "tokens": 50.0},
                bucket_ids=BUCKETS,
                reservation_id=rid_done,
                reservation_model_family=FAMILY,
                reservation_bucket_ids=BUCKETS,
                reservation_reserved_usage=u2,
            )
            assert outcome[:2] == ("exc", "DuplicateRefundError"), outcome
            # Override written before the restart still governs (same configured limit).
            assert driver.capacities(durable)[("tokens", 60)][1] == pytest.approx(800.0)
            clock.advance(3600.0)
            both(lambda t: driver.consume(t, {"requests": 0.0, "tokens": 0.0}))
        finally:
            for t in pair:
                t.cleanup()
