import math
from collections.abc import Iterable

from token_throttle._interfaces._models import Quota

DEFAULT_BUCKET_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_REFUND_DEDUP_TTL_SECONDS = 7 * 24 * 60 * 60
RESERVATION_LIFETIME_TTL_SAFETY_MARGIN = 2.0
MAX_SQLITE_TTL_SECONDS = 2**31 - 1


def validate_sqlite_ttl_seconds(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(
            f"{name} must be an exact int number of seconds "
            f"(got {type(value).__name__}); use a plain int such as 604800"
        )
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0 (got {value!r})")
    if value > MAX_SQLITE_TTL_SECONDS:
        raise ValueError(
            f"{name} must be <= {MAX_SQLITE_TTL_SECONDS} seconds "
            f"(got {value!r}); choose a smaller SQLite TTL"
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
    if value_float > MAX_SQLITE_TTL_SECONDS:
        raise ValueError(
            "max_reservation_lifetime_seconds must be <= "
            f"{MAX_SQLITE_TTL_SECONDS} seconds (got {value!r})"
        )
    return value_float


def _validate_safety_margin(value: object) -> float:
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


def derive_default_max_reservation_lifetime_seconds_from_ttls(
    *,
    bucket_ttl_seconds: int,
    refund_dedup_ttl_seconds: int,
    safety_margin: float = RESERVATION_LIFETIME_TTL_SAFETY_MARGIN,
) -> float:
    margin = _validate_safety_margin(safety_margin)
    bucket_ttl = validate_sqlite_ttl_seconds(
        bucket_ttl_seconds,
        name="bucket_ttl_seconds",
    )
    refund_ttl = validate_sqlite_ttl_seconds(
        refund_dedup_ttl_seconds,
        name="refund_dedup_ttl_seconds",
    )
    return math.nextafter(float(min(bucket_ttl, refund_ttl)) / margin, 0.0)


def resolve_max_reservation_lifetime_seconds_from_ttls(
    *,
    max_reservation_lifetime_seconds: float | None,
    bucket_ttl_seconds: int,
    refund_dedup_ttl_seconds: int,
    safety_margin: float = RESERVATION_LIFETIME_TTL_SAFETY_MARGIN,
) -> float:
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
    bucket_ttl_seconds: int,
    refund_dedup_ttl_seconds: int,
    safety_margin: float = RESERVATION_LIFETIME_TTL_SAFETY_MARGIN,
) -> None:
    max_lifetime = validate_max_reservation_lifetime_seconds(
        max_reservation_lifetime_seconds
    )
    if max_lifetime is None:
        return
    margin = _validate_safety_margin(safety_margin)
    ttl_by_name = {
        "bucket_ttl_seconds": validate_sqlite_ttl_seconds(
            bucket_ttl_seconds,
            name="bucket_ttl_seconds",
        ),
        "refund_dedup_ttl_seconds": validate_sqlite_ttl_seconds(
            refund_dedup_ttl_seconds,
            name="refund_dedup_ttl_seconds",
        ),
    }
    required_ttl = max_lifetime * margin
    too_short = [
        f"{name}={ttl}"
        for name, ttl in ttl_by_name.items()
        if float(ttl) <= required_ttl
    ]
    if too_short:
        raise ValueError(
            "SQLite TTLs must exceed max_reservation_lifetime_seconds * "
            f"{margin:g}; required > {required_ttl:g}s, got "
            f"{', '.join(too_short)}"
        )


def validate_bucket_ttl_covers_quota_windows(
    *,
    bucket_ttl_seconds: int,
    quotas: Iterable[Quota],
) -> None:
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
