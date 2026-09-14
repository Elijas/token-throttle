"""Redis expiry integrates each rate for only its own interval."""

import asyncio
import inspect
import json
import time
import uuid

import pytest

pytest.importorskip("redis", reason="redis package not installed")

import redis
import redis.asyncio

from tests.differential._backends import build_one, make_config, require_redis
from tests.differential._clock import FakeClock, patched_clock
from tests.differential._driver import Driver


def _expired(client, key):
    deadline = time.monotonic() + 4
    while client.exists(key):
        assert time.monotonic() < deadline, "Owned override did not expire"
        time.sleep(0.01)


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_failed_override_refresh_restores_history_value_and_ttl(mode, tmp_path):
    loop = asyncio.new_event_loop()
    clock = FakeClock(start=1000)
    target = None
    restricted_client = None
    username = f"expiry-{uuid.uuid4().hex}"
    password = uuid.uuid4().hex
    prefix = username

    def call(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        return (
            loop.run_until_complete(result) if inspect.isawaitable(result) else result
        )

    with patched_clock(clock), redis.Redis.from_url(require_redis()) as observer:
        try:
            cfg = make_config("rollback", [("requests", 10, 10)])
            target = build_one(
                "redis",
                mode,
                cfg,
                clock,
                loop=loop,
                tmp_path=tmp_path,
                key_prefix=prefix,
                ttl_seconds=100,
                override_ttl_seconds=1,
            )
            bucket = target.backend.sorted_buckets[0]
            call(bucket.set_max_capacity, 1.0)
            history = observer.get(bucket._override_expiry_key)
            ttl = observer.pttl(bucket._override_expiry_key)
            observer.execute_command(
                "ACL",
                "SETUSER",
                username,
                "on",
                f">{password}",
                f"~{prefix}:*",
                "+get",
                "+pttl",
                "+set",
                "+eval",
                "+time",
            )
            client_type = redis.asyncio.Redis if mode == "async" else redis.Redis
            restricted_client = client_type.from_url(
                require_redis(), username=username, password=password
            )
            restricted = type(bucket)(
                next(iter(cfg.quotas)),
                cfg,
                restricted_client,
                key_prefix=prefix,
                bucket_ttl_seconds=200,
                override_ttl_seconds=5,
            )
            before = vars(restricted).copy()
            # SET history succeeds, but EXPIRE override is denied by the owned ACL user.
            with pytest.raises(redis.ResponseError, match=r"permission|command|ACL"):
                call(restricted.refresh_max_capacity_from_redis)
            assert vars(restricted) == before
            assert observer.get(bucket._override_expiry_key) == history
            assert 0 < observer.pttl(bucket._override_expiry_key) <= ttl
        finally:
            if restricted_client is not None:
                call(
                    restricted_client.aclose
                    if mode == "async"
                    else restricted_client.close
                )
            observer.acl_deluser(username)
            if target:
                target.cleanup()
            loop.close()


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("configured,override", [(10.0, 1.0), (1.0, 10.0)])
@pytest.mark.parametrize(
    "operation",
    ["read", "direct", "maximum", "consume", "refund", "configure", "rebuild"],
)
def test_expiry_refill_for_new_observer(  # noqa: PLR0915
    mode, configured, override, operation, tmp_path
):
    loop = asyncio.new_event_loop()
    clock = FakeClock(start=1000)
    driver = Driver(loop)
    prefix = f"expiry-{uuid.uuid4().hex}"
    targets = []

    def call(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        return (
            loop.run_until_complete(result) if inspect.isawaitable(result) else result
        )

    with patched_clock(clock), redis.Redis.from_url(require_redis()) as observer:
        try:
            cfg = make_config("expiry", [("requests", 10, configured)])
            target = build_one(
                "redis",
                mode,
                cfg,
                clock,
                loop=loop,
                tmp_path=tmp_path,
                key_prefix=prefix,
                ttl_seconds=100,
                override_ttl_seconds=1,
            )
            targets.append(target)
            assert driver.consume(target, {"requests": configured})[0] == "ok"
            assert driver.set_max_capacity(target, "requests", 10, override)[0] == "ok"
            bucket = target.backend.sorted_buckets[0]
            _expired(observer, bucket._max_capacity_key)
            clock.advance(1.25)
            target = build_one(
                "redis",
                mode,
                cfg,
                clock,
                loop=loop,
                tmp_path=tmp_path,
                key_prefix=prefix,
                ttl_seconds=100,
                override_ttl_seconds=1,
            )
            targets.append(target)
            bucket = target.backend.sorted_buckets[0]
            # Independently integrate 1 second under the override and .25 under config.
            expected = min(configured, override / 10 + 0.25 * configured / 10)
            keys = list(observer.scan_iter(match=f"{prefix}:*"))
            before = {key: (observer.get(key), observer.pttl(key)) for key in keys}
            cache = vars(bucket).copy()
            diagnostic = call(target.backend.introspect)
            assert diagnostic.buckets[0].current_capacity == pytest.approx(expected)
            assert vars(bucket) == cache
            assert set(observer.scan_iter(match=f"{prefix}:*")) == set(keys)
            for key, (value, ttl) in before.items():
                assert observer.get(key) == value
                assert 0 <= observer.pttl(key) <= ttl

            if operation == "direct":
                assert call(bucket.get_capacity).amount == pytest.approx(expected)
            elif operation == "maximum":
                assert call(bucket.get_max_capacity) == configured
                assert bucket.calculate_capacity(
                    1000, 0, 1001.25
                ).amount == pytest.approx(expected)
            elif operation == "consume":
                assert driver.consume(target, {"requests": 0.1})[0] == "ok"
                assert driver.capacities(target)[("requests", 10)][0] == pytest.approx(
                    expected - 0.1
                )
            elif operation == "refund":
                assert (
                    driver.refund(target, {"requests": 0.1}, {"requests": 0})[0] == "ok"
                )
                assert driver.capacities(target)[("requests", 10)][0] == pytest.approx(
                    min(configured, expected + 0.1)
                )
            elif operation == "rebuild":
                rebuilt_cfg = make_config(
                    "expiry", [("requests", 10, configured), ("tokens", 10, 10)]
                )
                replacement = target.builder.build(rebuilt_cfg)
                target.backend = call(
                    target.backend.prepare_reconfigured_backend,
                    replacement,
                    rebuilt_cfg,
                )
                assert driver.capacities(target)[("requests", 10)][0] == pytest.approx(
                    expected
                )
            elif operation == "configure":
                call(target.backend.apply_configured_max_capacity, "requests", 10, 20.0)
                # Configuration snapshots preserve uncapped overflow at the transition.
                assert driver.capacities(target)[("requests", 10)][0] == pytest.approx(
                    override / 10 + 0.25 * configured / 10
                )
            else:
                assert driver.capacities(target)[("requests", 10)][0] == pytest.approx(
                    expected
                )
                assert call(target.backend._get_capacities_unsafe).capacities[
                    ("requests", 10)
                ] == pytest.approx(expected)
                assert bucket.calculate_capacity(
                    1000, 0, 1001.25
                ).amount == pytest.approx(expected)
                outcome = driver.acquire(target, {"requests": 1.0}, timeout=0)
                assert outcome[0] == ("exc" if expected < 1 else "ok")
                if expected < 1:
                    assert outcome[1] == "TimeoutError"
        finally:
            for target in reversed(targets):
                target.cleanup()
            loop.close()


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize(
    "scenario",
    ["legacy-expired", "legacy-live", "close-anchor", "manual-delete", "partial-loss"],
)
def test_history_compatibility_does_not_invent_missing_evidence(
    mode, scenario, tmp_path
):
    loop = asyncio.new_event_loop()
    driver = Driver(loop)
    clock = FakeClock(start=1000)
    target = None

    def call(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        return (
            loop.run_until_complete(result) if inspect.isawaitable(result) else result
        )

    with patched_clock(clock), redis.Redis.from_url(require_redis()) as observer:
        try:
            target = build_one(
                "redis",
                mode,
                make_config("compat", [("requests", 10, 10)]),
                clock,
                loop=loop,
                tmp_path=tmp_path,
                ttl_seconds=100,
                override_ttl_seconds=1,
            )
            assert driver.consume(target, {"requests": 10})[0] == "ok"
            assert driver.set_max_capacity(target, "requests", 10, 1)[0] == "ok"
            bucket = target.backend.sorted_buckets[0]
            if scenario.startswith("legacy"):
                # Established anchored JSON has no historical deadline by itself.
                observer.delete(bucket._override_expiry_key)
            if scenario == "close-anchor":
                observer.set(
                    bucket._max_capacity_key,
                    json.dumps(
                        {
                            "configured_max_capacity": 10 + 5e-12,
                            "override_max_capacity": 1,
                        }
                    ),
                    ex=1,
                )
            if scenario in ("legacy-live", "close-anchor"):
                clock.advance(0.25)
                call(target.backend._get_capacities_unsafe)
                _expired(observer, bucket._max_capacity_key)
                clock.advance(1.25)
                expected = 0.375  # 1.25*.1 + .25*1
            elif scenario == "manual-delete":
                observer.delete(bucket._max_capacity_key)
                clock.advance(0.25)
                expected = 0.25  # No claim about the unknown deletion instant.
            else:
                _expired(observer, bucket._max_capacity_key)
                clock.advance(1.25)
                expected = 1.25  # Expired legacy history is unrecoverable.
                if scenario == "partial-loss":
                    observer.delete(bucket._capacity_key)
                    expected = 0
            assert call(bucket.get_capacity).amount == pytest.approx(expected)
        finally:
            if target:
                target.cleanup()
            loop.close()


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("refresh", ["backend", "bucket", "maximum"])
def test_mutating_reads_slide_expiry_without_changing_prior_refill(
    mode, refresh, tmp_path
):
    loop = asyncio.new_event_loop()
    driver = Driver(loop)
    clock = FakeClock(start=1000)
    target = None

    def call(fn, *args, **kwargs):
        value = fn(*args, **kwargs)
        return loop.run_until_complete(value) if inspect.isawaitable(value) else value

    with patched_clock(clock), redis.Redis.from_url(require_redis()) as observer:
        try:
            cfg = make_config("sliding", [("requests", 10, 10.0)])
            target = build_one(
                "redis",
                mode,
                cfg,
                clock,
                loop=loop,
                tmp_path=tmp_path,
                ttl_seconds=100,
                override_ttl_seconds=1,
            )
            assert driver.consume(target, {"requests": 10.0})[0] == "ok"
            assert driver.set_max_capacity(target, "requests", 10, 1.0)[0] == "ok"
            bucket = target.backend.sorted_buckets[0]
            deadline = time.monotonic() + 3
            while observer.pttl(bucket._max_capacity_key) > 600:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            clock.advance(0.5)
            before = observer.pttl(bucket._max_capacity_key)
            if refresh == "backend":
                assert call(target.backend._get_capacities_unsafe).capacities[
                    ("requests", 10)
                ] == pytest.approx(0.05)
            elif refresh == "bucket":
                assert call(bucket.get_capacity).amount == pytest.approx(0.05)
            else:
                assert call(bucket.refresh_max_capacity_from_redis) == 1.0
            assert observer.pttl(bucket._max_capacity_key) > before + 200
            _expired(observer, bucket._max_capacity_key)
            clock.advance(1.25)
            # 1.5 seconds at .1/s followed by .25 seconds at 1/s.
            assert driver.capacities(target)[("requests", 10)][0] == pytest.approx(0.4)
        finally:
            if target:
                target.cleanup()
            loop.close()


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("ordering", ["before", "after"])
def test_history_retention_hook_follows_extended_accounting_keys(
    mode, ordering, tmp_path
):
    """Exercise the history hook independently of automatic debt retention."""
    loop = asyncio.new_event_loop()
    clock = FakeClock(start=1000)
    targets = []

    def call(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        return (
            loop.run_until_complete(result) if inspect.isawaitable(result) else result
        )

    with patched_clock(clock), redis.Redis.from_url(require_redis()) as observer:
        try:
            prefix = f"retention-{uuid.uuid4().hex}"
            cfg = make_config("retention", [("requests", 1, 100.0)])
            target = build_one(
                "redis",
                mode,
                cfg,
                clock,
                loop=loop,
                tmp_path=tmp_path,
                key_prefix=prefix,
                ttl_seconds=2,
                override_ttl_seconds=1,
                max_reservation_lifetime_seconds=0.5,
            )
            targets.append(target)
            bucket = target.backend.sorted_buckets[0]
            call(bucket.set_capacity, -100, allow_negative=True)
            call(target.backend.set_max_capacity, "requests", 1, 1.0)
            # The rate change already extends debt retention. Explicitly seed
            # the short-lived state this helper-boundary test is meant to model.
            for key in (
                bucket._capacity_key,
                bucket._last_checked_key,
                bucket._override_expiry_key,
            ):
                observer.expire(key, 2)
            if ordering == "before":
                call(bucket._refresh_override_expiry_retention, minimum_ttl_seconds=101)
                assert observer.pttl(bucket._override_expiry_key) > 100_000
                assert observer.pttl(bucket._capacity_key) <= 2000
                call(bucket.refresh_max_capacity_from_redis)
                assert observer.pttl(bucket._override_expiry_key) > 100_000
            for key in (bucket._capacity_key, bucket._last_checked_key):
                observer.expire(key, 101)
            if ordering == "after":
                call(bucket._refresh_override_expiry_retention)
            assert observer.pttl(bucket._override_expiry_key) > 100_000
            sentinel = f"{prefix}:sentinel"
            observer.set(sentinel, "1", ex=2)
            _expired(observer, sentinel)
            clock.advance(2.25)
            target = build_one(
                "redis",
                mode,
                cfg,
                clock,
                loop=loop,
                tmp_path=tmp_path,
                key_prefix=prefix,
                ttl_seconds=2,
                override_ttl_seconds=1,
                max_reservation_lifetime_seconds=0.5,
            )
            targets.append(target)
            assert call(target.backend.introspect).buckets[
                0
            ].current_capacity == pytest.approx(26)
        finally:
            for target in reversed(targets):
                target.cleanup()
            loop.close()


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("stored", [-20.0, 20.0])
def test_expiry_snapshot_preserves_uncapped_overflow_and_debt(mode, stored, tmp_path):
    loop = asyncio.new_event_loop()
    clock = FakeClock(start=1000)
    driver = Driver(loop)
    target = None

    def call(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        return (
            loop.run_until_complete(result) if inspect.isawaitable(result) else result
        )

    with patched_clock(clock), redis.Redis.from_url(require_redis()) as observer:
        try:
            target = build_one(
                "redis",
                mode,
                make_config("raw", [("requests", 10, 10.0)]),
                clock,
                loop=loop,
                tmp_path=tmp_path,
                ttl_seconds=100,
                override_ttl_seconds=1,
            )
            bucket = target.backend.sorted_buckets[0]
            call(bucket.set_capacity, stored, allow_negative=True)
            call(target.backend.set_max_capacity, "requests", 10, 1.0)
            _expired(observer, bucket._max_capacity_key)
            clock.advance(1.25)
            call(target.backend.set_max_capacity, "requests", 10, 100.0)
            assert driver.capacities(target)[("requests", 10)][0] == pytest.approx(
                stored + 0.35
            )
        finally:
            if target:
                target.cleanup()
            loop.close()
