"""Accounting expiry must never overtake the history needed to interpret it."""

from __future__ import annotations

import asyncio

import pytest

from tests.differential._backends import build_one, make_config, require_redis
from tests.differential._clock import FakeClock, patched_clock
from tests.differential._driver import Driver


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("operation", ["write", "failed_write", "read"])
def test_history_is_retained_at_state_extension_boundary(
    mode, operation, tmp_path, monkeypatch
):
    import redis  # noqa: PLC0415

    from token_throttle._limiter_backends._redis._override_expiry import (  # noqa: PLC0415
        expiry_payload,
    )

    loop = asyncio.new_event_loop()
    clock = FakeClock()
    with patched_clock(clock):
        target = build_one(
            "redis",
            mode,
            make_config("history-boundary", [("requests", 1, 1.0)]),
            clock,
            loop=loop,
            tmp_path=tmp_path,
            ttl_seconds=2,
            override_ttl_seconds=1,
        )
        bucket = target.backend.sorted_buckets[0]
        try:
            with redis.Redis.from_url(require_redis()) as observer:
                # Seed a valid expired-override interval and outstanding debt.
                # Short retention models an existing bucket needing extension.
                now = clock.time()
                observer.set(bucket._last_checked_key, now, ex=2)
                observer.set(bucket._capacity_key, -100, ex=2)
                observer.set(
                    bucket._override_expiry_key,
                    expiry_payload(1.0, 100.0, now),
                    ex=2,
                )
                before = observer.mget(bucket._last_checked_key, bucket._capacity_key)
                if operation == "read":
                    result = bucket.get_capacity(current_time=now)
                    if mode == "async":
                        loop.run_until_complete(result)
                    assert Driver(loop).capacities(target)[("requests", 1)] == (-100, 1)
                else:
                    pipeline = bucket._redis.pipeline()
                    original_execute = pipeline.execute

                    def check_boundary():
                        # -100 at 1/s plus the full-bucket window needs 101s.
                        # This check runs BEFORE the accounting transaction.
                        assert observer.pttl(bucket._override_expiry_key) >= 100_000
                        if operation == "failed_write":
                            raise OSError("injected before state transaction")

                    if mode == "async":

                        async def execute(*args, **kwargs):
                            check_boundary()
                            return await original_execute(*args, **kwargs)

                    else:

                        def execute(*args, **kwargs):
                            check_boundary()
                            return original_execute(*args, **kwargs)

                    monkeypatch.setattr(pipeline, "execute", execute)

                    def write():
                        result = bucket.set_capacity(
                            -100,
                            pipeline=pipeline,
                            current_time=now,
                            allow_negative=True,
                        )
                        if mode == "async":
                            loop.run_until_complete(result)

                    if operation == "failed_write":
                        with pytest.raises(OSError, match="before state transaction"):
                            write()
                        assert (
                            observer.mget(
                                bucket._last_checked_key, bucket._capacity_key
                            )
                            == before
                        )
                    else:
                        write()
                history_ttl = observer.pttl(bucket._override_expiry_key)
                assert history_ttl >= 100_000
                assert history_ttl >= observer.pttl(bucket._capacity_key) - 100
        finally:
            target.cleanup()
            loop.close()
