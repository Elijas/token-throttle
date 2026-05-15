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
