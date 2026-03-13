"""Tests for shared token-bucket capacity math."""

import pytest

from token_throttle._capacity import calculate_capacity


@pytest.mark.parametrize("last_checked", [float("nan"), float("inf"), float("-inf")])
def test_calculate_capacity_rejects_non_finite_last_checked(last_checked):
    with pytest.raises(ValueError, match="Invalid last_checked or capacity values"):
        calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=40.0,
            current_time=1030.0,
            max_capacity=100.0,
            rate_per_sec=1.0,
            bucket_id="memory:test-model:tokens:60",
        )


@pytest.mark.parametrize(
    "outdated_capacity",
    [float("nan"), float("inf"), float("-inf")],
)
def test_calculate_capacity_rejects_non_finite_outdated_capacity(outdated_capacity):
    with pytest.raises(ValueError, match="Invalid last_checked or capacity values"):
        calculate_capacity(
            last_checked=1000.0,
            outdated_capacity=outdated_capacity,
            current_time=1030.0,
            max_capacity=100.0,
            rate_per_sec=1.0,
            bucket_id="memory:test-model:tokens:60",
        )
