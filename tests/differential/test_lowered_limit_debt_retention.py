"""Lowered quotas must repay existing debt before idle expiry forgets it."""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from tests.differential._backends import build_one, make_config, require_redis
from tests.differential._clock import FakeClock, patched_clock
from tests.differential._driver import Driver


def _past_original_ttl(kind, clock):
    if kind == "redis":
        import redis  # noqa: PLC0415

        # This independent key must expire even when the debt keys are retained.
        with redis.Redis.from_url(require_redis()) as client:
            key = f"debt-retention-sentinel-{uuid.uuid4().hex}"
            client.set(key, "1", ex=2)
            deadline = time.monotonic() + 10
            while client.exists(key):
                assert time.monotonic() < deadline
                time.sleep(0.01)
    clock.advance(2.25)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("kind", ["memory", "sqlite", "redis"])
@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize(
    "change", ["runtime", "configured", "rebuild", "rebuild_runtime"]
)
def test_lowered_limit_retains_debt_through_idle_and_refresh(  # noqa: PLR0915
    kind, mode, change, tmp_path
):
    loop = asyncio.new_event_loop()
    driver = Driver(loop)
    clock = FakeClock()
    prefix = f"debt-{uuid.uuid4().hex}"
    cfg = make_config("debt", [("requests", 1, 100.0)])
    lowered = make_config("debt", [("requests", 1, 1.0)])
    targets = []

    def build(config):
        target = build_one(
            kind,
            mode,
            config,
            clock,
            loop=loop,
            tmp_path=tmp_path,
            key_prefix=prefix,
            db_path=tmp_path / "debt.sqlite3",
            ttl_seconds=2,
            override_ttl_seconds=100,
            max_reservation_lifetime_seconds=0.5,
        )
        targets.append(target)
        return target

    with patched_clock(clock):
        try:
            target = build(cfg)
            assert driver.consume(target, {"requests": 200})[0] == "ok"
            if change == "rebuild_runtime":
                assert driver.set_max_capacity(target, "requests", 1, 1)[0] == "ok"
                lowered = cfg
            if change == "runtime":
                assert driver.set_max_capacity(target, "requests", 1, 1)[0] == "ok"
            elif change == "configured":
                assert (
                    driver.apply_configured_max_capacity(target, "requests", 1, 1)[0]
                    == "ok"
                )
            else:
                if kind == "redis":
                    # Redis rebuilds retain the builder's client identity.
                    from tests.differential._backends import Harnessed  # noqa: PLC0415

                    replacement = Harnessed(
                        target.name,
                        kind,
                        target.is_async,
                        target.builder.build(lowered),
                        target.builder,
                    )
                else:
                    replacement = build(lowered)
                outcome = driver.call(
                    target, "prepare_reconfigured_backend", replacement.backend, lowered
                )
                assert outcome[0] == "ok"
                if kind == "sqlite":
                    target = replacement
                elif kind == "memory" and change == "rebuild_runtime":
                    # Memory overrides are owned/reapplied by the outer limiter.
                    assert driver.set_max_capacity(target, "requests", 1, 1)[0] == "ok"
            assert driver.capacities(target)[("requests", 1)] == (-100.0, 1.0)
            # A fresh observer must not depend on the writer's in-process cache.
            if kind != "memory":
                if kind == "sqlite":
                    target.cleanup()
                target = build(
                    cfg if change in {"runtime", "rebuild_runtime"} else lowered
                )
            for step in range(1, 4):
                _past_original_ttl(kind, clock)
                expected = -100 + step * 2.25
                assert driver.capacities(target)[("requests", 1)] == (expected, 1.0)
                assert driver.acquire(target, {"requests": 1}, timeout=0)[:2] == (
                    "exc",
                    "TimeoutError",
                )
                if step == 2:
                    # Also test a write, after an interval protected solely by
                    # blocked reads. Re-anchor without consume's debt floor.
                    assert driver.set_max_capacity(target, "requests", 1, 1)[0] == "ok"
            clock.advance(101)
            assert driver.acquire(target, {"requests": 1}, timeout=0)[0] == "ok"
            if kind == "redis":
                import redis  # noqa: PLC0415

                with redis.Redis.from_url(require_redis()) as client:
                    bucket = target.backend.sorted_buckets[0]
                    assert 0 < client.pttl(bucket._capacity_key) <= 2000
                    assert 0 < client.pttl(bucket._last_checked_key) <= 2000
            elif kind == "sqlite":
                expires_at = target.backend._engine._connection.execute(
                    "SELECT expires_at FROM buckets"
                ).fetchone()[0]
                assert expires_at == clock.time() + 2
        finally:
            for target in reversed(targets):
                target.cleanup()
            loop.close()


@pytest.mark.parametrize("kind", ["memory", "sqlite", "redis"])
@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("usage", [50, 100])
def test_nonnegative_state_keeps_ordinary_expiration(kind, mode, usage, tmp_path):
    loop = asyncio.new_event_loop()
    driver = Driver(loop)
    clock = FakeClock()
    with patched_clock(clock):
        target = build_one(
            kind,
            mode,
            make_config("ordinary", [("requests", 1, 100.0)]),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            ttl_seconds=2,
        )
        try:
            assert driver.consume(target, {"requests": usage})[0] == "ok"
            _past_original_ttl(kind, clock)
            assert driver.capacities(target)[("requests", 1)] == (100.0, 100.0)
            if kind == "redis":
                import redis  # noqa: PLC0415

                with redis.Redis.from_url(require_redis()) as client:
                    bucket = target.backend.sorted_buckets[0]
                    assert client.exists(bucket._capacity_key) == 0
                    assert client.exists(bucket._last_checked_key) == 0
            elif kind == "sqlite":
                row = target.backend._engine._connection.execute(
                    "SELECT expires_at FROM buckets"
                ).fetchone()
                assert row is None or row[0] <= clock.time()
        finally:
            target.cleanup()
            loop.close()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("mode", ["async", "sync"])
def test_sqlite_debt_retention_includes_expired_override_history(mode, tmp_path):
    loop = asyncio.new_event_loop()
    driver = Driver(loop)
    clock = FakeClock()
    with patched_clock(clock):
        target = build_one(
            "sqlite",
            mode,
            make_config("history", [("requests", 1, 100.0)]),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            ttl_seconds=2,
            override_ttl_seconds=1,
        )
        try:
            assert driver.consume(target, {"requests": 200})[0] == "ok"
            assert driver.set_max_capacity(target, "requests", 1, 1)[0] == "ok"
            clock.advance(2.25)
            # One second at 1/s, then 1.25 seconds at 100/s: -100 + 1 + 125.
            assert driver.capacities(target)[("requests", 1)] == (26.0, 100.0)
            assert driver.acquire(target, {"requests": 27}, timeout=0)[:2] == (
                "exc",
                "TimeoutError",
            )
            assert driver.acquire(target, {"requests": 26}, timeout=0)[0] == "ok"
        finally:
            target.cleanup()
            loop.close()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("kind", ["sqlite", "redis"])
@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("change", ["runtime", "configured", "rebuild"])
def test_unrepresentable_retention_rejects_before_changing_limit(
    kind, mode, change, tmp_path
):
    loop = asyncio.new_event_loop()
    driver = Driver(loop)
    clock = FakeClock()
    with patched_clock(clock):
        target = build_one(
            kind,
            mode,
            make_config("finite", [("requests", 1, 100.0)]),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            ttl_seconds=2,
        )
        try:
            assert driver.consume(target, {"requests": 200})[0] == "ok"
            if change == "runtime":
                outcome = driver.set_max_capacity(target, "requests", 1, 1e-9)
            elif change == "configured":
                outcome = driver.apply_configured_max_capacity(
                    target, "requests", 1, 1e-9
                )
            else:
                config = make_config("finite", [("requests", 1, 1e-9)])
                replacement = target.builder.build(config)
                outcome = driver.call(
                    target, "prepare_reconfigured_backend", replacement, config
                )
            assert outcome[:2] == ("exc", "ValueError")
            assert driver.capacities(target)[("requests", 1)] == (-100.0, 100.0)
            clock.advance(1)
            assert driver.capacities(target)[("requests", 1)] == (0.0, 100.0)
        finally:
            target.cleanup()
            loop.close()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("mode", ["async", "sync"])
def test_redis_failed_rate_increase_keeps_slow_rate_debt(mode, tmp_path, monkeypatch):
    loop = asyncio.new_event_loop()
    driver = Driver(loop)
    clock = FakeClock()
    with patched_clock(clock):
        target = build_one(
            "redis",
            mode,
            make_config("failed-increase", [("requests", 1, 100.0)]),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            ttl_seconds=2,
            override_ttl_seconds=100,
        )
        try:
            assert driver.consume(target, {"requests": 200})[0] == "ok"
            assert driver.set_max_capacity(target, "requests", 1, 1)[0] == "ok"

            def fail_write(_value):
                raise RuntimeError("Override write did not commit")

            monkeypatch.setattr(
                target.backend.sorted_buckets[0], "set_max_capacity", fail_write
            )
            assert driver.set_max_capacity(target, "requests", 1, 100)[:2] == (
                "exc",
                "RuntimeError",
            )
            _past_original_ttl("redis", clock)
            assert driver.capacities(target)[("requests", 1)] == (-97.75, 1.0)
        finally:
            target.cleanup()
            loop.close()
