"""Regression tests for FIX-53 OpenAI Redis factory TTL/lifetime knobs."""

import math
from unittest.mock import MagicMock

import pytest

pytest.importorskip("redis", reason="redis package not installed")

import redis as _sync_redis
import redis.asyncio as _async_redis

from token_throttle._factories._openai._openai_rate_limiter import (
    create_openai_redis_rate_limiter,
)
from token_throttle._factories._openai._openai_sync_rate_limiter import (
    create_openai_redis_sync_rate_limiter,
)
from token_throttle._limiter_backends._redis._keys import (
    DEFAULT_REFUND_DEDUP_TTL_SECONDS,
)
from token_throttle._limiter_backends._redis._ttl import DEFAULT_BUCKET_TTL_SECONDS


def _async_redis_mock() -> MagicMock:
    return MagicMock(spec=_async_redis.Redis)


def _sync_redis_mock() -> MagicMock:
    return MagicMock(spec=_sync_redis.Redis)


def test_async_openai_factory_preserves_redis_lifetime_defaults() -> None:
    limiter = create_openai_redis_rate_limiter(
        _async_redis_mock(),
        key_prefix="test",
        rpm=100,
        tpm=10_000,
    )

    assert limiter._backend._bucket_ttl_seconds == DEFAULT_BUCKET_TTL_SECONDS
    assert (
        limiter._backend._refund_dedup_ttl_seconds == DEFAULT_REFUND_DEDUP_TTL_SECONDS
    )
    expected_default_lifetime = math.nextafter(
        min(DEFAULT_BUCKET_TTL_SECONDS, DEFAULT_REFUND_DEDUP_TTL_SECONDS) / 2,
        0.0,
    )
    assert limiter._max_reservation_lifetime_seconds == expected_default_lifetime


def test_async_openai_factory_passes_redis_lifetime_and_dedup_ttl_knobs() -> None:
    limiter = create_openai_redis_rate_limiter(
        _async_redis_mock(),
        key_prefix="test",
        rpm=100,
        tpm=10_000,
        bucket_ttl_seconds=61,
        max_reservation_lifetime_seconds=30,
        refund_dedup_ttl_seconds=62,
    )

    assert limiter._backend._bucket_ttl_seconds == 61
    assert limiter._backend._refund_dedup_ttl_seconds == 62
    assert limiter._max_reservation_lifetime_seconds == 30.0


def test_async_openai_factory_none_dedup_ttl_preserves_default() -> None:
    limiter = create_openai_redis_rate_limiter(
        _async_redis_mock(),
        key_prefix="test",
        rpm=100,
        tpm=10_000,
        refund_dedup_ttl_seconds=None,
    )

    assert (
        limiter._backend._refund_dedup_ttl_seconds == DEFAULT_REFUND_DEDUP_TTL_SECONDS
    )


def test_async_openai_factory_validates_lifetime_against_redis_ttls() -> None:
    with pytest.raises(ValueError, match="Redis TTLs must exceed"):
        create_openai_redis_rate_limiter(
            _async_redis_mock(),
            key_prefix="test",
            rpm=100,
            tpm=10_000,
            bucket_ttl_seconds=60,
            refund_dedup_ttl_seconds=60,
            max_reservation_lifetime_seconds=30,
        )


def test_sync_openai_factory_preserves_redis_lifetime_defaults() -> None:
    limiter = create_openai_redis_sync_rate_limiter(
        _sync_redis_mock(),
        key_prefix="test",
        rpm=100,
        tpm=10_000,
    )

    assert limiter._backend._bucket_ttl_seconds == DEFAULT_BUCKET_TTL_SECONDS
    assert (
        limiter._backend._refund_dedup_ttl_seconds == DEFAULT_REFUND_DEDUP_TTL_SECONDS
    )
    expected_default_lifetime = math.nextafter(
        min(DEFAULT_BUCKET_TTL_SECONDS, DEFAULT_REFUND_DEDUP_TTL_SECONDS) / 2,
        0.0,
    )
    assert limiter._max_reservation_lifetime_seconds == expected_default_lifetime


def test_sync_openai_factory_passes_redis_lifetime_and_dedup_ttl_knobs() -> None:
    limiter = create_openai_redis_sync_rate_limiter(
        _sync_redis_mock(),
        key_prefix="test",
        rpm=100,
        tpm=10_000,
        bucket_ttl_seconds=61,
        max_reservation_lifetime_seconds=30,
        refund_dedup_ttl_seconds=62,
    )

    assert limiter._backend._bucket_ttl_seconds == 61
    assert limiter._backend._refund_dedup_ttl_seconds == 62
    assert limiter._max_reservation_lifetime_seconds == 30.0


def test_sync_openai_factory_none_dedup_ttl_preserves_default() -> None:
    limiter = create_openai_redis_sync_rate_limiter(
        _sync_redis_mock(),
        key_prefix="test",
        rpm=100,
        tpm=10_000,
        refund_dedup_ttl_seconds=None,
    )

    assert (
        limiter._backend._refund_dedup_ttl_seconds == DEFAULT_REFUND_DEDUP_TTL_SECONDS
    )


def test_sync_openai_factory_validates_lifetime_against_redis_ttls() -> None:
    with pytest.raises(ValueError, match="Redis TTLs must exceed"):
        create_openai_redis_sync_rate_limiter(
            _sync_redis_mock(),
            key_prefix="test",
            rpm=100,
            tpm=10_000,
            bucket_ttl_seconds=60,
            refund_dedup_ttl_seconds=60,
            max_reservation_lifetime_seconds=30,
        )
