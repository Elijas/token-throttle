import math
from collections.abc import Iterable

from token_throttle._interfaces._models import Quota

DEFAULT_BUCKET_TTL_SECONDS = 7 * 24 * 60 * 60
RESERVATION_LIFETIME_TTL_SAFETY_MARGIN = 2.0
MAX_REDIS_TTL_SECONDS = 2**31 - 1


def validate_redis_ttl_seconds(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(
            f"{name} must be an exact int number of seconds "
            f"(got {type(value).__name__}); use a plain int such as 604800"
        )
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0 (got {value!r})")
    if value > MAX_REDIS_TTL_SECONDS:
        raise ValueError(
            f"{name} must be <= {MAX_REDIS_TTL_SECONDS} seconds "
            f"(got {value!r}); choose a smaller Redis TTL"
        )
    return value


def validate_max_reservation_lifetime_seconds(value: object) -> float | None:
    if value is None:
        return None
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValueError(
            "max_reservation_lifetime_seconds must be finite and greater than 0"
        )
    value_float = float(value)
    if not math.isfinite(value_float) or value_float <= 0:
        raise ValueError(
            "max_reservation_lifetime_seconds must be finite and greater than 0"
        )
    if value_float > MAX_REDIS_TTL_SECONDS:
        raise ValueError(
            "max_reservation_lifetime_seconds must be <= "
            f"{MAX_REDIS_TTL_SECONDS} seconds (got {value!r})"
        )
    return value_float


def _validate_reservation_lifetime_ttl_safety_margin(value: object) -> float:
    if (
        type(value) is bool
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 1.0
    ):
        raise ValueError(
            "reservation lifetime TTL safety_margin must be greater than 1"
        )
    return float(value)


def _finite_ttls_by_name(
    *,
    bucket_ttl_seconds: int | None,
    refund_dedup_ttl_seconds: int | None,
) -> dict[str, int]:
    ttl_by_name: dict[str, int] = {}
    if bucket_ttl_seconds is not None:
        ttl_by_name["bucket_ttl_seconds"] = validate_redis_ttl_seconds(
            bucket_ttl_seconds,
            name="bucket_ttl_seconds",
        )
    if refund_dedup_ttl_seconds is not None:
        ttl_by_name["refund_dedup_ttl_seconds"] = validate_redis_ttl_seconds(
            refund_dedup_ttl_seconds,
            name="refund_dedup_ttl_seconds",
        )
    return ttl_by_name


def derive_default_max_reservation_lifetime_seconds_from_ttls(
    *,
    bucket_ttl_seconds: int | None,
    refund_dedup_ttl_seconds: int | None,
    safety_margin: float = RESERVATION_LIFETIME_TTL_SAFETY_MARGIN,
) -> float | None:
    """
    Derive Redis' default reservation lifetime from configured TTLs.

    Redis reservations need their acquire marker, bucket state, and refund
    dedup tombstone to outlive the period in which a reservation may be
    refunded. When at least one Redis TTL is finite, the derived lifetime is
    just below ``min(bucket_ttl_seconds, refund_dedup_ttl_seconds) /
    safety_margin``. The default safety margin is 2, so the public rule of
    thumb is ``min(bucket_ttl_seconds, refund_dedup_ttl_seconds) / 2``.

    Returning just below the quotient preserves the strict invariant enforced
    for explicit lifetimes: every finite Redis TTL must be greater than
    ``max_reservation_lifetime_seconds * safety_margin``. If both TTL inputs
    are None, no Redis-derived lifetime can be calculated and None is returned.
    """
    margin = _validate_reservation_lifetime_ttl_safety_margin(safety_margin)
    ttl_by_name = _finite_ttls_by_name(
        bucket_ttl_seconds=bucket_ttl_seconds,
        refund_dedup_ttl_seconds=refund_dedup_ttl_seconds,
    )
    if not ttl_by_name:
        return None

    # The invariant is strict (ttl > lifetime * margin), so derive just below
    # the exact quotient instead of returning a boundary value that rejects.
    return math.nextafter(float(min(ttl_by_name.values())) / margin, 0.0)


def resolve_max_reservation_lifetime_seconds_from_ttls(
    *,
    max_reservation_lifetime_seconds: float | None,
    bucket_ttl_seconds: int | None,
    refund_dedup_ttl_seconds: int | None,
    safety_margin: float = RESERVATION_LIFETIME_TTL_SAFETY_MARGIN,
) -> float | None:
    """
    Resolve caller-provided or Redis-derived reservation lifetime.

    A caller-provided ``max_reservation_lifetime_seconds`` is validated against
    the Redis TTL invariant: each finite Redis TTL must be strictly greater
    than ``max_reservation_lifetime_seconds * safety_margin``. This keeps Redis
    bucket state, acquire markers, and refund dedup tombstones available for
    the full refund window.

    When the caller leaves the lifetime as None, Redis derives the default from
    the smaller finite TTL using the same safety margin. With the default
    margin of 2, that means the effective lifetime is just below
    ``min(bucket_ttl_seconds, refund_dedup_ttl_seconds) / 2``.
    """
    max_lifetime = validate_max_reservation_lifetime_seconds(
        max_reservation_lifetime_seconds
    )
    if max_lifetime is None:
        return derive_default_max_reservation_lifetime_seconds_from_ttls(
            bucket_ttl_seconds=bucket_ttl_seconds,
            refund_dedup_ttl_seconds=refund_dedup_ttl_seconds,
            safety_margin=safety_margin,
        )

    validate_reservation_lifetime_ttl_invariant(
        max_reservation_lifetime_seconds=max_lifetime,
        bucket_ttl_seconds=bucket_ttl_seconds,
        refund_dedup_ttl_seconds=refund_dedup_ttl_seconds,
        safety_margin=safety_margin,
    )
    return max_lifetime


def validate_reservation_lifetime_ttl_invariant(
    *,
    max_reservation_lifetime_seconds: float | None,
    bucket_ttl_seconds: int | None,
    refund_dedup_ttl_seconds: int | None,
    safety_margin: float = RESERVATION_LIFETIME_TTL_SAFETY_MARGIN,
) -> None:
    max_lifetime = validate_max_reservation_lifetime_seconds(
        max_reservation_lifetime_seconds
    )
    if max_lifetime is None:
        return
    margin = _validate_reservation_lifetime_ttl_safety_margin(safety_margin)

    required_ttl = max_lifetime * margin
    ttl_by_name = _finite_ttls_by_name(
        bucket_ttl_seconds=bucket_ttl_seconds,
        refund_dedup_ttl_seconds=refund_dedup_ttl_seconds,
    )
    too_short = [
        f"{name}={ttl}"
        for name, ttl in ttl_by_name.items()
        if float(ttl) <= required_ttl
    ]
    if too_short:
        raise ValueError(
            "Redis TTLs must exceed max_reservation_lifetime_seconds * "
            f"{margin:g}; required > {required_ttl:g}s, got "
            f"{', '.join(too_short)}"
        )


def validate_bucket_ttl_covers_quota_windows(
    *,
    bucket_ttl_seconds: int,
    quotas: Iterable[Quota],
) -> None:
    """
    Fail fast when a quota window outlives the bucket state TTL.

    Redis expires idle ``last_checked``/capacity keys after
    ``bucket_ttl_seconds``. If a quota's ``per_seconds`` window is longer than
    that, an idle gap between ``bucket_ttl_seconds`` and ``per_seconds``
    silently expires the bucket state, and the next read re-grants the full
    ``max_capacity`` instead of the drained state the long window should have
    preserved. Equality (``per_seconds == bucket_ttl_seconds``) is allowed:
    the bucket only needs to survive gaps *shorter* than the window itself.
    """
    too_long = [
        f"{quota.metric}: per_seconds={quota.per_seconds}"
        for quota in quotas
        if quota.per_seconds > bucket_ttl_seconds
    ]
    if too_long:
        raise ValueError(
            "bucket_ttl_seconds must be >= every configured quota's "
            f"per_seconds (got bucket_ttl_seconds={bucket_ttl_seconds}); "
            f"offending quotas: {', '.join(too_long)}. Raise "
            "bucket_ttl_seconds to at least the longest quota window, or "
            "shorten that quota's per_seconds."
        )
