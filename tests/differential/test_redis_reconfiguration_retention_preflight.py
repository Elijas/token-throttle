"""Predictable retention rejection must precede every rebuild mutation."""

from __future__ import annotations

import asyncio

import pytest

from tests.differential._backends import build_one, make_config, require_redis
from tests.differential._clock import FakeClock, patched_clock
from tests.differential._driver import Driver


def _persisted(client, prefix):
    # Absolute expiry from Redis' own atomic clock, without refreshing any key.
    script = """
local now = redis.call('TIME')
local ttl = redis.call('PTTL', KEYS[1])
local deadline = ttl
if ttl >= 0 then
    deadline = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000) + ttl
end
return {redis.call('GET', KEYS[1]), deadline}
"""
    return {
        key: client.eval(script, 1, key)
        for key in sorted(client.scan_iter(match=f"{prefix}:*"))
        if not key.endswith(b":lock")
    }


def _local(backend):
    return [
        (
            bucket.configured_max_capacity,
            bucket._max_capacity_cached,
            bucket._max_capacity_cache_time,
            bucket._rate_per_sec,
            bucket._state_confirmed_at_server_time,
            bucket._state_confirmed_ttl_seconds,
        )
        for bucket in backend.sorted_buckets
    ]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("shape", ["surviving", "removed", "new_identity"])
def test_rebuild_rejects_late_debt_before_any_bucket_changes(mode, shape, tmp_path):
    import redis  # noqa: PLC0415

    loop = asyncio.new_event_loop()
    driver = Driver(loop)
    clock = FakeClock()
    cfg = make_config("preflight", [("alpha", 1, 100), ("beta", 1, 100)])
    with patched_clock(clock):
        target = build_one(
            "redis",
            mode,
            cfg,
            clock,
            loop=loop,
            tmp_path=tmp_path,
            ttl_seconds=2,
            override_ttl_seconds=100,
        )
        try:
            assert driver.consume(target, {"alpha": 200, "beta": 200})[0] == "ok"
            assert driver.set_max_capacity(target, "alpha", 1, 7)[0] == "ok"
            if shape == "new_identity":
                # beta's durable state survives an earlier worker, but is new
                # to the active backend's bucket set.
                target.backend = target.builder.build(
                    make_config("preflight", [("alpha", 1, 100)])
                )
            quotas = [("beta", 1, 1e-9)]
            if shape != "removed":
                quotas.insert(0, ("alpha", 1, 50))
            new_cfg = make_config("preflight", quotas)
            replacement = target.builder.build(new_cfg)
            clock.advance(0.25)
            prefix = target.backend._key_prefix
            with redis.Redis.from_url(require_redis()) as observer:
                before = _persisted(observer, prefix)
                old_local, new_local = _local(target.backend), _local(replacement)
                old_config = target.backend._limit_config
                outcome = driver.call(
                    target, "prepare_reconfigured_backend", replacement, new_cfg
                )
                assert outcome[:2] == ("exc", "ValueError")
                assert _persisted(observer, prefix) == before
                assert _local(target.backend) == old_local
                assert _local(replacement) == new_local
                assert target.backend._limit_config is old_config
        finally:
            target.cleanup()
            loop.close()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("mode", ["async", "sync"])
def test_valid_rebuild_preserves_mixed_surviving_overrides(mode, tmp_path):
    loop = asyncio.new_event_loop()
    driver = Driver(loop)
    clock = FakeClock()
    cfg = make_config("valid-preflight", [("alpha", 1, 100), ("beta", 1, 100)])
    with patched_clock(clock):
        target = build_one(
            "redis",
            mode,
            cfg,
            clock,
            loop=loop,
            tmp_path=tmp_path,
            ttl_seconds=2,
            override_ttl_seconds=100,
        )
        try:
            assert driver.consume(target, {"alpha": 200, "beta": 200})[0] == "ok"
            assert driver.set_max_capacity(target, "alpha", 1, 7)[0] == "ok"
            assert driver.set_max_capacity(target, "beta", 1, 3)[0] == "ok"
            clock.advance(0.25)
            new_cfg = make_config(
                "valid-preflight",
                [("alpha", 1, 50), ("beta", 1, 100), ("gamma", 1, 10)],
            )
            replacement = target.builder.build(new_cfg)
            assert (
                driver.call(
                    target, "prepare_reconfigured_backend", replacement, new_cfg
                )[0]
                == "ok"
            )
            assert driver.capacities(target) == {
                ("alpha", 1): (-98.25, 50),
                ("beta", 1): (-99.25, 3),
                ("gamma", 1): (10, 10),
            }
        finally:
            target.cleanup()
            loop.close()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Callable config.*:UserWarning")
@pytest.mark.parametrize("mode", ["async", "sync"])
def test_public_failed_callable_rebuild_keeps_metadata_and_storage(mode, tmp_path):
    import redis  # noqa: PLC0415

    from token_throttle import RateLimiter, SyncRateLimiter  # noqa: PLC0415

    loop = asyncio.new_event_loop()
    clock = FakeClock()
    configs = [make_config("public-preflight", [("alpha", 1, 100), ("beta", 1, 100)])]
    with patched_clock(clock):
        target = build_one(
            "redis",
            mode,
            configs[0],
            clock,
            loop=loop,
            tmp_path=tmp_path,
            ttl_seconds=2,
            override_ttl_seconds=100,
        )
        limiter_class = RateLimiter if mode == "async" else SyncRateLimiter
        limiter = limiter_class(lambda _model: configs[0], backend=target.builder)

        def call(method, *args):
            result = getattr(limiter, method)(*args)
            return loop.run_until_complete(result) if mode == "async" else result

        try:
            call("record_usage", {"alpha": 200, "beta": 200}, "model")
            call("set_max_capacity", "model", "alpha", 1, 7)
            backend = limiter._model_family_to_backend["public-preflight"]
            before_local = _local(backend)
            before_quotas = dict(limiter._model_family_to_quotas)
            before_overrides = {
                family: dict(values)
                for family, values in limiter._model_family_to_runtime_max_capacity.items()
            }
            configs[0] = make_config(
                "public-preflight",
                [("alpha", 1, 50), ("beta", 1, 1e-9), ("gamma", 1, 10)],
            )
            clock.advance(0.25)
            with redis.Redis.from_url(require_redis()) as observer:
                before = _persisted(observer, backend._key_prefix)
                with pytest.raises(ValueError, match="retention longer"):
                    call("record_usage", {"alpha": 0, "beta": 0, "gamma": 0}, "model")
                assert _persisted(observer, backend._key_prefix) == before
            assert limiter._model_family_to_backend["public-preflight"] is backend
            assert limiter._model_family_to_quotas == before_quotas
            assert limiter._model_family_to_runtime_max_capacity == before_overrides
            assert _local(backend) == before_local
        finally:
            call("aclose" if mode == "async" else "close")
            target.cleanup()
            loop.close()
