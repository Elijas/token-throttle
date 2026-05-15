"""Regression coverage for bounded Redis key segments."""

import pytest
from pydantic import ValidationError

from token_throttle._exceptions import CardinalityLimitExceededError
from token_throttle._interfaces._models import (
    MAX_KEY_PREFIX_LENGTH,
    MAX_RESERVATION_ID_LENGTH,
    CapacityReservation,
)
from token_throttle._limiter_backends._redis._keys import (
    redis_namespace_key,
    redis_refund_dedup_key,
    validate_redis_key_prefix,
)
from token_throttle._validation import (
    MAX_TOTAL_KEY_LENGTH,
    _validate_key_prefix,
)
from token_throttle.migration import validate_config_for_v2_0


def test_key_prefix_rejects_megabyte_scale_value() -> None:
    with pytest.raises(
        CardinalityLimitExceededError,
        match=f"key_prefix must be at most {MAX_KEY_PREFIX_LENGTH} characters",
    ):
        validate_redis_key_prefix("a" * 1_000_000)


def test_key_prefix_accepts_length_boundary() -> None:
    assert validate_redis_key_prefix("a" * MAX_KEY_PREFIX_LENGTH) == (
        "a" * MAX_KEY_PREFIX_LENGTH
    )


def test_key_prefix_rejects_one_character_over_boundary() -> None:
    with pytest.raises(
        CardinalityLimitExceededError,
        match=f"key_prefix must be at most {MAX_KEY_PREFIX_LENGTH} characters",
    ):
        validate_redis_key_prefix("a" * (MAX_KEY_PREFIX_LENGTH + 1))


def test_capacity_reservation_rejects_too_long_reservation_id() -> None:
    with pytest.raises(ValidationError, match="reservation_id"):
        CapacityReservation(
            reservation_id="a" * (MAX_RESERVATION_ID_LENGTH + 1),
            usage={"tokens": 1},
            model_family="gpt-4o",
            limiter_instance_id="limiter",
        )


def test_refund_dedup_key_revalidates_reservation_id_boundary() -> None:
    reservation_id = "a" * (MAX_RESERVATION_ID_LENGTH + 1)

    with pytest.raises(
        CardinalityLimitExceededError,
        match=f"reservation_id must be at most {MAX_RESERVATION_ID_LENGTH} characters",
    ):
        redis_refund_dedup_key("tenant", reservation_id)


@pytest.mark.parametrize(
    "key_prefix",
    [
        "",
        " ",
        "bad prefix",
        "bad:prefix",
        "{bad}",
        "control\x00char",
        "a" * (MAX_KEY_PREFIX_LENGTH + 1),
    ],
)
def test_migration_rejects_same_key_prefix_patterns_as_runtime(
    key_prefix: object,
) -> None:
    with pytest.raises(ValueError, match="key_prefix") as canonical_error:
        _validate_key_prefix(key_prefix)

    issues = validate_config_for_v2_0({"backend": "redis", "key_prefix": key_prefix})

    assert [issue.field_path for issue in issues] == ["redis.key_prefix"]
    assert issues[0].reason == str(canonical_error.value)


@pytest.mark.parametrize(
    "key_prefix",
    [
        "prod-tenant-abc-123-deployment-v2",
        "a" * MAX_KEY_PREFIX_LENGTH,
        "tenant_europe.1",
    ],
)
def test_migration_accepts_same_key_prefix_patterns_as_runtime(
    key_prefix: str,
) -> None:
    assert _validate_key_prefix(key_prefix) == key_prefix

    assert (
        validate_config_for_v2_0({"backend": "redis", "key_prefix": key_prefix}) == []
    )


def test_total_key_length_rejects_valid_segments_that_exceed_global_cap() -> None:
    segments = tuple("s" * MAX_KEY_PREFIX_LENGTH for _ in range(64))

    with pytest.raises(
        CardinalityLimitExceededError,
        match=f"Redis key must be at most {MAX_TOTAL_KEY_LENGTH} characters",
    ):
        redis_namespace_key("tenant", *segments)
