"""Expiry history must not relax Redis parsing or mutate partially validated reads."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("redis", reason="redis package not installed")

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackend
from token_throttle._limiter_backends._redis._bucket import RedisBucket
from token_throttle._limiter_backends._redis._override_expiry import (
    OverrideExpiry,
    accrue_with_expiry,
    calculate_with_expiry,
    expiry_payload,
    parse_expiry,
)
from token_throttle._limiter_backends._redis._sync_backend import SyncRedisBackend
from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket


@pytest.mark.parametrize("stored", [-100.0, 0.0, 20.0])
@pytest.mark.parametrize("configured,maximum", [(10.0, 1.0), (1.0, 10.0)])
def test_expiry_math_keeps_signed_balance_and_caps_only_final_read(
    stored, configured, maximum
):
    result = calculate_with_expiry(
        history=OverrideExpiry(configured, maximum, 1001),
        override=None,
        configured=configured,
        per_seconds=10,
        last_checked=1000,
        outdated_capacity=stored,
        current_time=1001.25,
        bucket_id="test",
    )
    assert result.amount == pytest.approx(
        min(configured, stored + maximum / 10 + configured * 0.25 / 10)
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b"[]",
        b"null",
        b"bad",
        b'"bad"',
        b'{"version":1,"configured":10,"maximum":1,"expires_at":true}',
    ],
)
def test_malformed_history_is_rejected(raw):
    with pytest.raises(ValueError, match=r"Redis|Expecting value"):
        parse_expiry(raw)


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("bad", [b"nan", b"not-a-number"])
def test_late_invalid_capacity_does_not_update_any_cache_or_history(mode, bad):
    cfg = PerModelConfig(
        model_family="expiry",
        quotas=UsageQuotas(
            [
                Quota(metric="a", per_seconds=10, limit=10),
                Quota(metric="b", per_seconds=10, limit=10),
            ]
        ),
    )
    client = MagicMock()
    raw_history = expiry_payload(10, 1, 1001).encode()
    client.get = (
        AsyncMock(return_value=raw_history)
        if mode == "async"
        else MagicMock(return_value=raw_history)
    )
    client.eval = (
        AsyncMock(return_value=1) if mode == "async" else MagicMock(return_value=1)
    )
    bucket_type = RedisBucket if mode == "async" else SyncRedisBucket
    backend_type = RedisBackend if mode == "async" else SyncRedisBackend
    buckets = [bucket_type(q, cfg, client, key_prefix="test") for q in cfg.quotas]
    backend = backend_type(buckets, client, cfg, key_prefix="test")
    pipeline = MagicMock()
    results = [b"1000", b"0", True, True, b"1000", bad, True, True, None, None]
    pipeline.execute = (
        AsyncMock(return_value=results)
        if mode == "async"
        else MagicMock(return_value=results)
    )
    before = [vars(bucket).copy() for bucket in buckets]

    def read():
        if mode == "async":
            asyncio.run(
                backend._get_capacities_unsafe(pipeline=pipeline, current_time=1001.25)
            )
        else:
            backend._get_capacities_unsafe(pipeline=pipeline, current_time=1001.25)

    with pytest.raises(ValueError, match="Invalid last_checked or capacity"):
        read()
    assert [vars(bucket) for bucket in buckets] == before
    client.eval.assert_not_called()


def test_absent_history_and_early_manual_deletion_use_legacy_fallback():
    for history in (None, OverrideExpiry(10, 1, 1001)):
        result = calculate_with_expiry(
            history=history,
            override=None,
            configured=10,
            per_seconds=10,
            last_checked=1000,
            outdated_capacity=0,
            current_time=1000.25,
            bucket_id="test",
        )
        assert result.amount == 0.25


@pytest.mark.parametrize(
    "configured,maximum,now,expected",
    [
        (1e308, 1.0, 2.0, 2.0),
        (1e16, 1.0, 2.0, 2.0),
        (1e16, 1.0, 2.5, 5e15 + 2),
        (1.0, 1e308, 2.0, 1.0),
    ],
)
def test_expiry_rates_do_not_overflow_or_cancel_before_final_clamp(
    configured, maximum, now, expected
):
    result = calculate_with_expiry(
        history=OverrideExpiry(configured, maximum, 2.0),
        override=None,
        configured=configured,
        per_seconds=1,
        last_checked=0.0,
        outdated_capacity=0.0,
        current_time=now,
        bucket_id="test",
    )
    assert result.amount == expected


def test_uncapped_expiry_accrual_does_not_overflow_before_debt_cancellation():
    assert (
        accrue_with_expiry(
            OverrideExpiry(1, 1e308, 2),
            override=None,
            configured=1,
            per_seconds=1,
            last_checked=0,
            current_time=2,
            stored=-1e308,
            rate_per_sec=1,
        )
        == 1e308
    )
