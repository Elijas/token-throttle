"""Tests for shared token-bucket capacity math."""

import warnings

import pytest

import token_throttle._capacity as _cap
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


class TestClockSkewBehavior:
    """Verify calculate_capacity behavior under clock-skew conditions.

    calculate_capacity is a pure function — it doesn't know or care where
    current_time came from. These tests document the math so that the
    caller-level fix (using Redis server time) can be validated separately.
    """

    def test_negative_time_passed_clamps_to_zero(self, caplog):
        """When current_time < last_checked, time_passed is clamped to 0."""
        _cap._backward_clock_warned = False
        _cap._backward_clock_last_warning_at = None
        with (
            caplog.at_level("WARNING", logger="token_throttle"),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            result = calculate_capacity(
                last_checked=100.0,
                outdated_capacity=50.0,
                current_time=90.0,  # 10s "behind" last_checked
                max_capacity=100.0,
                rate_per_sec=1.0,
                bucket_id="memory:family:tokens:60",
            )
        # Capacity preserved (no negative refill), but also not reduced
        assert result.amount == 50.0
        assert result.is_fresh_start is False
        assert len(w) == 1
        assert "Negative time_passed" in str(w[0].message)
        assert "metric=tokens" in caplog.text
        assert "model_family=family" in caplog.text
        assert "value=-10.0000" in caplog.text
        assert "bucket_id=memory:family:tokens:60" in caplog.text

    def test_negative_time_warning_is_throttled_not_global_once(self, monkeypatch):
        """Backward-clock warnings re-emit after the throttle window."""
        _cap._backward_clock_warned = False
        _cap._backward_clock_last_warning_at = None
        ticks = iter([0.0, 1.0, 301.0])
        monkeypatch.setattr(_cap.time, "monotonic", lambda: next(ticks))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for _ in range(3):
                calculate_capacity(
                    last_checked=100.0,
                    outdated_capacity=50.0,
                    current_time=90.0,
                    max_capacity=100.0,
                    rate_per_sec=1.0,
                    bucket_id="test:clock-skew",
                )

        assert len(w) == 2
        assert all("Negative time_passed" in str(warning.message) for warning in w)

    def test_clock_ahead_host_causes_premature_refill(self):
        """A clock-ahead host sees inflated time_passed, fully refilling a drained bucket.

        This is the core of the clock-skew bug: a host with clock +60s computes
        time_passed=60 on a bucket drained at t=0, causing max refill. The fix
        is at the caller level (use Redis server time), not in this function.
        """
        result = calculate_capacity(
            last_checked=0.0,
            outdated_capacity=0.0,
            current_time=60.0,  # Host A thinks 60s passed
            max_capacity=60.0,
            rate_per_sec=1.0,
            bucket_id="test:clock-skew",
        )
        assert result.amount == 60.0

    def test_subsequent_reader_preserves_inflated_capacity(self):
        """After a clock-ahead host writes, a correct-clock host sees full capacity.

        Simulates the second half of the bug: Host A wrote last_checked=60 and
        capacity=60 from its skewed clock. Host B (correct clock at t=0) reads
        this and gets negative time_passed, which clamps to 0, preserving the
        inflated capacity=60 unchanged.
        """
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = calculate_capacity(
                last_checked=60.0,  # Future timestamp from Host A
                outdated_capacity=60.0,  # Inflated capacity from Host A
                current_time=0.0,  # Host B's correct clock
                max_capacity=60.0,
                rate_per_sec=1.0,
                bucket_id="test:clock-skew",
            )
        # The inflated capacity is preserved due to clamp-to-zero
        assert result.amount == 60.0
