import math

DEFAULT_BUCKET_TTL_SECONDS = 7 * 24 * 60 * 60
RESERVATION_LIFETIME_TTL_SAFETY_MARGIN = 2.0


def validate_redis_ttl_seconds(value: object, *, name: str) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise TypeError(f"{name} must be an int number of seconds")
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
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
    return value_float


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
    if (
        type(safety_margin) is bool
        or not isinstance(safety_margin, (int, float))
        or not math.isfinite(float(safety_margin))
        or float(safety_margin) <= 1.0
    ):
        raise ValueError(
            "reservation lifetime TTL safety_margin must be greater than 1"
        )

    required_ttl = max_lifetime * float(safety_margin)
    ttl_by_name = {
        "bucket_ttl_seconds": validate_redis_ttl_seconds(
            bucket_ttl_seconds,
            name="bucket_ttl_seconds",
        ),
        "refund_dedup_ttl_seconds": validate_redis_ttl_seconds(
            refund_dedup_ttl_seconds,
            name="refund_dedup_ttl_seconds",
        ),
    }
    too_short = [
        f"{name}={ttl}"
        for name, ttl in ttl_by_name.items()
        if float(ttl) <= required_ttl
    ]
    if too_short:
        raise ValueError(
            "Redis TTLs must exceed max_reservation_lifetime_seconds * "
            f"{float(safety_margin):g}; required > {required_ttl:g}s, got "
            f"{', '.join(too_short)}"
        )
