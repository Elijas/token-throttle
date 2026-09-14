"""Finite inactivity retention for debt preserved across quota changes."""

import math

MAX_STATE_TTL_SECONDS = 2**31 - 1


def debt_ttl_seconds(
    capacity: float,
    *,
    bucket_ttl_seconds: int,
    per_seconds: int,
    max_capacity: float,
    configured_max_capacity: float,
) -> int:
    """
    Keep debt until refill can reach full at either side of override expiry.

    The configured two-window minimum covers ordinary debt. A lowered maximum
    can leave deeper debt, so use the slower possible rate to bound repayment,
    followed by one full quota window. Do not silently saturate an unsafe TTL.
    """
    if capacity >= 0:
        return bucket_ttl_seconds
    required = (
        1.0 + (-capacity / min(max_capacity, configured_max_capacity))
    ) * per_seconds
    if not math.isfinite(required) or required > MAX_STATE_TTL_SECONDS:
        raise ValueError(
            "Unpaid capacity debt requires retention longer than "
            f"{MAX_STATE_TTL_SECONDS} seconds; repay debt or choose a higher maximum"
        )
    return max(bucket_ttl_seconds, math.ceil(required))
