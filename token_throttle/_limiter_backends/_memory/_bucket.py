import math
import warnings

from token_throttle._capacity import CalculatedCapacity, calculate_capacity


class MemoryBucket:
    """
    In-memory token bucket state holder.

    Plain Python object — no locks, no async. Shared by both async and sync backends.
    Concurrency is handled by the backend that owns this bucket.
    """

    def __init__(
        self,
        metric: str,
        per_seconds: int,
        limit: float,
        model_family: str,
    ) -> None:
        self.capacity: float | None = None
        self.last_checked: float | None = None
        self.max_capacity = float(limit)
        self._rate_per_sec = float(limit) / float(per_seconds)
        self.usage_metric = metric
        self.per_seconds = per_seconds
        self._bucket_id = f"memory:{model_family}:{metric}:{int(per_seconds)}"

    def get_capacity(self, current_time: float) -> CalculatedCapacity:
        """Calculate current capacity using shared calculate_capacity()."""
        return calculate_capacity(
            last_checked=self.last_checked,
            outdated_capacity=self.capacity,
            current_time=current_time,
            max_capacity=self.max_capacity,
            rate_per_sec=self._rate_per_sec,
            bucket_id=self._bucket_id,
        )

    def set_capacity(
        self, value: float, current_time: float, *, allow_negative: bool = False
    ) -> None:
        """
        Set bucket capacity and update the timestamp.

        allow_negative controls whether capacity can go below zero:
        - False (default): used by acquire_capacity — the blocking path
          guarantees capacity >= usage before consuming, so negatives
          indicate a logic error.
        - True: used by consume_capacity (speedometer / record_usage) and
          refund_capacity. Speedometer intentionally overshoots; refund
          must preserve negative debt so the token-bucket refill handles
          recovery naturally.
        """
        self.capacity = value if allow_negative else max(0.0, value)
        self.last_checked = current_time

    def set_max_capacity(self, value: float, current_time: float) -> None:
        """
        Update max_capacity and recalculate refill rate.

        Anchors the stored capacity at the OLD rate before swapping so the
        new rate is not applied retroactively to time that elapsed under
        the old rate. ``calculate_capacity`` integrates a single
        ``rate_per_sec`` across ``[last_checked, current_time]``; any rate
        change must therefore reset ``last_checked`` to ``now``.

        The anchor is the *uncapped* old-rate integration — we preserve any
        raw value above ``max_capacity`` (``calculate_capacity`` applies
        ``min(max_capacity, …)`` at read time). This keeps lower-then-raise
        cap sequences recoverable: overflow hidden under the low cap is
        re-exposed when the cap rises.
        """
        if isinstance(value, bool):
            raise ValueError("max_capacity must not be a boolean")  # noqa: TRY004
        if not (math.isfinite(value) and value > 0):
            raise ValueError("max_capacity must be finite and greater than 0")
        if self.last_checked is not None and self.capacity is not None:
            time_passed = current_time - self.last_checked
            if time_passed < 0:
                warnings.warn(
                    f"Negative time_passed ({time_passed:.4f}s) detected in bucket "
                    f"'{self._bucket_id}' — likely NTP clock correction. "
                    f"Clamping to 0.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                time_passed = 0.0
            self.capacity = self.capacity + time_passed * self._rate_per_sec
            self.last_checked = current_time
        self.max_capacity = value
        self._rate_per_sec = value / float(self.per_seconds)
