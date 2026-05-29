"""FIX-45 fake-client coverage for lost Redis EVAL replies.

The tests below simulate a committed Lua transaction whose reply is lost by
raising after the fake Redis client's ``eval`` has mutated in-memory state. R7
deferred a real TCP proxy that drops replies/ACKs against Redis itself; the
library relies on deterministic fake-client checks plus Redis Lua atomicity,
with end-to-end proxy/server failure validation left to operators.
"""

from __future__ import annotations

import warnings

import pytest

from token_throttle._exceptions import DuplicateRefundError, UnknownReservationError
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

from .test_bundle_acquire_marker_authority import (
    _HAS_REDIS,
    FAMILY,
    MODEL,
    PREFIX,
    _AsyncRedis,
    _AsyncRedisBuilder,
    _config,
    _SyncRedis,
    _SyncRedisBuilder,
    redis_acquired_marker_key,
    redis_refund_dedup_key,
)

pytestmark = pytest.mark.skipif(not _HAS_REDIS, reason="redis package not installed")
redis = pytest.importorskip("redis", reason="redis package not installed")

_CAPACITY_KEY = f"{PREFIX}:rate_limiting:bucket:{FAMILY}:tokens:60:capacity"


class _AsyncLostReplyRedis(_AsyncRedis):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_eval_after_commit = False

    async def eval(self, *args: object, **kwargs: object) -> str:
        result = await super().eval(*args, **kwargs)
        if self.fail_next_eval_after_commit:
            self.fail_next_eval_after_commit = False
            raise redis.exceptions.ConnectionError(
                "simulated lost EVAL reply after commit"
            )
        return result


class _SyncLostReplyRedis(_SyncRedis):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_eval_after_commit = False

    def eval(self, *args: object, **kwargs: object) -> str:
        result = super().eval(*args, **kwargs)
        if self.fail_next_eval_after_commit:
            self.fail_next_eval_after_commit = False
            raise redis.exceptions.ConnectionError(
                "simulated lost EVAL reply after commit"
            )
        return result


async def test_async_lost_acquire_eval_reply_is_reconciled_to_success() -> None:
    redis_client = _AsyncLostReplyRedis()
    redis_client.fail_next_eval_after_commit = True
    limiter = RateLimiter(_config(), backend=_AsyncRedisBuilder(redis_client))

    reservation = await limiter.acquire_capacity({"tokens": 30}, MODEL)

    assert reservation.reservation_id in limiter._in_flight_reservation_ids
    assert reservation.reservation_id not in limiter._pending_acquire_reservations
    assert redis_client.store[_CAPACITY_KEY] == 70.0
    assert redis_acquired_marker_key(PREFIX, reservation.reservation_id) in (
        redis_client.store
    )


def test_sync_lost_acquire_eval_reply_is_reconciled_to_success() -> None:
    redis_client = _SyncLostReplyRedis()
    redis_client.fail_next_eval_after_commit = True
    limiter = SyncRateLimiter(_config(), backend=_SyncRedisBuilder(redis_client))

    reservation = limiter.acquire_capacity({"tokens": 30}, MODEL)

    assert reservation.reservation_id in limiter._in_flight_reservation_ids
    assert reservation.reservation_id not in limiter._pending_acquire_reservations
    assert redis_client.store[_CAPACITY_KEY] == 70.0
    assert redis_acquired_marker_key(PREFIX, reservation.reservation_id) in (
        redis_client.store
    )


async def test_async_acquire_retry_same_reservation_id_does_not_double_consume() -> (
    None
):
    backend = _AsyncRedisBuilder(_AsyncRedis()).build(_config())

    await backend.await_for_capacity(
        {"tokens": 30},
        reservation_id="retry-rid",
        reservation_lifetime_seconds=5,
    )
    await backend.await_for_capacity(
        {"tokens": 30},
        reservation_id="retry-rid",
        reservation_lifetime_seconds=5,
    )

    assert backend._redis.store[_CAPACITY_KEY] == 70.0


def test_sync_acquire_retry_same_reservation_id_does_not_double_consume() -> None:
    backend = _SyncRedisBuilder(_SyncRedis()).build(_config())

    backend.wait_for_capacity(
        {"tokens": 30},
        reservation_id="retry-rid",
        reservation_lifetime_seconds=5,
    )
    backend.wait_for_capacity(
        {"tokens": 30},
        reservation_id="retry-rid",
        reservation_lifetime_seconds=5,
    )

    assert backend._redis.store[_CAPACITY_KEY] == 70.0


async def test_async_refund_tombstone_replay_skips_capacity_write() -> None:
    from token_throttle._limiter_backends._redis._backend import (  # noqa: PLC0415
        RedisScriptResultError,
    )

    redis_client = _AsyncRedis()
    backend = _AsyncRedisBuilder(redis_client).build(_config())
    await backend.await_for_capacity(
        {"tokens": 30},
        reservation_id="replay-rid",
        reservation_lifetime_seconds=5,
    )
    tombstone_key = redis_refund_dedup_key(PREFIX, "replay-rid")
    redis_client.store[tombstone_key] = "1"

    with pytest.raises(RedisScriptResultError, match="incoherent"):
        await backend.refund_capacity_for_buckets(
            {"tokens": 30},
            {"tokens": 0},
            bucket_ids=frozenset({("tokens", 60)}),
            reservation_id="replay-rid",
            reservation_model_family=FAMILY,
            reservation_bucket_ids=frozenset({("tokens", 60)}),
            reservation_reserved_usage={"tokens": 30},
        )

    assert redis_client.store[_CAPACITY_KEY] == 70.0


def test_sync_refund_tombstone_replay_skips_capacity_write() -> None:
    from token_throttle._limiter_backends._redis._sync_backend import (  # noqa: PLC0415
        RedisScriptResultError,
    )

    redis_client = _SyncRedis()
    backend = _SyncRedisBuilder(redis_client).build(_config())
    backend.wait_for_capacity(
        {"tokens": 30},
        reservation_id="replay-rid",
        reservation_lifetime_seconds=5,
    )
    tombstone_key = redis_refund_dedup_key(PREFIX, "replay-rid")
    redis_client.store[tombstone_key] = "1"

    with pytest.raises(RedisScriptResultError, match="incoherent"):
        backend.refund_capacity_for_buckets(
            {"tokens": 30},
            {"tokens": 0},
            bucket_ids=frozenset({("tokens", 60)}),
            reservation_id="replay-rid",
            reservation_model_family=FAMILY,
            reservation_bucket_ids=frozenset({("tokens", 60)}),
            reservation_reserved_usage={"tokens": 30},
        )

    assert redis_client.store[_CAPACITY_KEY] == 70.0


async def test_async_marker_mismatch_does_not_delete_refund_authority() -> None:
    redis_client = _AsyncRedis()
    builder = _AsyncRedisBuilder(redis_client)
    limiter_a = RateLimiter(_config(), backend=builder)
    limiter_b = RateLimiter(_config(), backend=builder)
    reservation = await limiter_a.acquire_capacity({"tokens": 30}, MODEL)
    marker_key = redis_acquired_marker_key(PREFIX, reservation.reservation_id)
    drifted = reservation.model_copy(
        update={"bucket_ids": frozenset({("tokens", 120)})}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(UnknownReservationError):
            await limiter_b.refund_capacity({"tokens": 0}, drifted)

    assert marker_key in redis_client.store
    assert reservation.reservation_id in limiter_a._in_flight_reservation_ids

    await limiter_a.refund_capacity({"tokens": 0}, reservation)

    assert marker_key not in redis_client.store
    assert redis_client.store[_CAPACITY_KEY] == 100.0


def test_sync_marker_mismatch_does_not_delete_refund_authority() -> None:
    redis_client = _SyncRedis()
    builder = _SyncRedisBuilder(redis_client)
    limiter_a = SyncRateLimiter(_config(), backend=builder)
    limiter_b = SyncRateLimiter(_config(), backend=builder)
    reservation = limiter_a.acquire_capacity({"tokens": 30}, MODEL)
    marker_key = redis_acquired_marker_key(PREFIX, reservation.reservation_id)
    drifted = reservation.model_copy(
        update={"bucket_ids": frozenset({("tokens", 120)})}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(UnknownReservationError):
            limiter_b.refund_capacity({"tokens": 0}, drifted)

    assert marker_key in redis_client.store
    assert reservation.reservation_id in limiter_a._in_flight_reservation_ids

    limiter_a.refund_capacity({"tokens": 0}, reservation)

    assert marker_key not in redis_client.store
    assert redis_client.store[_CAPACITY_KEY] == 100.0


async def test_async_lost_refund_eval_reply_commits_local_state() -> None:
    redis_client = _AsyncLostReplyRedis()
    limiter = RateLimiter(_config(), backend=_AsyncRedisBuilder(redis_client))
    reservation = await limiter.acquire_capacity({"tokens": 30}, MODEL)
    redis_client.fail_next_eval_after_commit = True

    await limiter.refund_capacity({"tokens": 10}, reservation)

    marker_key = redis_acquired_marker_key(PREFIX, reservation.reservation_id)
    tombstone_key = redis_refund_dedup_key(PREFIX, reservation.reservation_id)
    assert marker_key not in redis_client.store
    assert redis_client.store[tombstone_key] == "1"
    assert redis_client.store[_CAPACITY_KEY] == 90.0
    assert limiter._refunded_reservation_ids[reservation.reservation_id] == "committed"
    assert reservation.reservation_id not in limiter._in_flight_reservation_ids
    with pytest.raises(DuplicateRefundError):
        await limiter.refund_capacity({"tokens": 10}, reservation)


def test_sync_lost_refund_eval_reply_commits_local_state() -> None:
    redis_client = _SyncLostReplyRedis()
    limiter = SyncRateLimiter(_config(), backend=_SyncRedisBuilder(redis_client))
    reservation = limiter.acquire_capacity({"tokens": 30}, MODEL)
    redis_client.fail_next_eval_after_commit = True

    limiter.refund_capacity({"tokens": 10}, reservation)

    marker_key = redis_acquired_marker_key(PREFIX, reservation.reservation_id)
    tombstone_key = redis_refund_dedup_key(PREFIX, reservation.reservation_id)
    assert marker_key not in redis_client.store
    assert redis_client.store[tombstone_key] == "1"
    assert redis_client.store[_CAPACITY_KEY] == 90.0
    assert limiter._refunded_reservation_ids[reservation.reservation_id] == "committed"
    assert reservation.reservation_id not in limiter._in_flight_reservation_ids
    with pytest.raises(DuplicateRefundError):
        limiter.refund_capacity({"tokens": 10}, reservation)


async def test_async_marker_absence_unknown_clears_local_in_flight() -> None:
    redis_client = _AsyncRedis()
    limiter = RateLimiter(_config(), backend=_AsyncRedisBuilder(redis_client))
    reservation = await limiter.acquire_capacity({"tokens": 30}, MODEL)
    await redis_client.delete(
        redis_acquired_marker_key(PREFIX, reservation.reservation_id)
    )

    with pytest.raises(UnknownReservationError):
        await limiter.refund_capacity({"tokens": 0}, reservation)

    assert reservation.reservation_id not in limiter._in_flight_reservation_ids
    assert reservation.reservation_id not in limiter._in_flight_reservation_family
    assert reservation.reservation_id not in limiter._refunded_reservation_ids


def test_sync_marker_absence_unknown_clears_local_in_flight() -> None:
    redis_client = _SyncRedis()
    limiter = SyncRateLimiter(_config(), backend=_SyncRedisBuilder(redis_client))
    reservation = limiter.acquire_capacity({"tokens": 30}, MODEL)
    redis_client.delete(redis_acquired_marker_key(PREFIX, reservation.reservation_id))

    with pytest.raises(UnknownReservationError):
        limiter.refund_capacity({"tokens": 0}, reservation)

    assert reservation.reservation_id not in limiter._in_flight_reservation_ids
    assert reservation.reservation_id not in limiter._in_flight_reservation_family
    assert reservation.reservation_id not in limiter._refunded_reservation_ids
