"""Debt retention must stay finite without truncating its repayment horizon."""

import pytest

from token_throttle._limiter_backends._debt_ttl import (
    MAX_STATE_TTL_SECONDS,
    debt_ttl_seconds,
)


@pytest.mark.parametrize(
    ("capacity", "maximum", "configured", "expected"),
    [
        (-100, 1, 100, 101),
        (-100, 100, 1, 101),
        (-100, 100, 100, 2),
        (-100.25, 1, 100, 102),
        (0, 1, 100, 2),
        (1, 1, 100, 2),
        (-(MAX_STATE_TTL_SECONDS - 1), 1, 1, MAX_STATE_TTL_SECONDS),
    ],
)
def test_retention_covers_repayment(capacity, maximum, configured, expected):
    assert (
        debt_ttl_seconds(
            capacity,
            bucket_ttl_seconds=2,
            per_seconds=1,
            max_capacity=maximum,
            configured_max_capacity=configured,
        )
        == expected
    )


@pytest.mark.parametrize("capacity", [-MAX_STATE_TTL_SECONDS, -1e308])
def test_retention_does_not_saturate_at_finite_ceiling(capacity):
    with pytest.raises(ValueError, match="retention longer"):
        debt_ttl_seconds(
            capacity,
            bucket_ttl_seconds=2,
            per_seconds=1,
            max_capacity=1,
            configured_max_capacity=1,
        )
