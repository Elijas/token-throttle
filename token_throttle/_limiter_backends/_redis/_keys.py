from token_throttle._validation import (
    _validate_key_prefix,
    _validate_reservation_id,
    _validate_total_key_length,
)

_REDIS_NAMESPACE = "rate_limiting"
DEFAULT_REFUND_DEDUP_TTL_SECONDS = 7 * 24 * 60 * 60


def validate_redis_key_prefix(value: object) -> str:
    """Validate the deployment-scoped Redis key prefix."""
    return _validate_key_prefix(value)


def redis_namespace_key(key_prefix: str, *segments: object) -> str:
    key = ":".join(
        (
            validate_redis_key_prefix(key_prefix),
            _REDIS_NAMESPACE,
            *(str(segment) for segment in segments),
        )
    )
    return _validate_total_key_length(key)


def redis_key_with_suffix(key: str, *suffixes: object) -> str:
    return _validate_total_key_length(
        ":".join((key, *(str(suffix) for suffix in suffixes)))
    )


def redis_refund_dedup_key(key_prefix: str, reservation_id: str) -> str:
    return redis_namespace_key(
        key_prefix,
        "refund_dedup",
        _validate_reservation_id(reservation_id),
    )


def validate_refund_dedup_ttl_seconds(value: object) -> int:
    if type(value) is not int:
        raise TypeError("refund_dedup_ttl_seconds must be an int number of seconds")
    if value <= 0:
        raise ValueError("refund_dedup_ttl_seconds must be greater than 0")
    return value
