import json
import math
from collections.abc import Mapping

from token_throttle._interfaces._models import BucketId
from token_throttle._validation import (
    _validate_key_prefix,
    _validate_reservation_id,
    _validate_total_key_length,
)

_REDIS_NAMESPACE = "rate_limiting"
DEFAULT_REFUND_DEDUP_TTL_SECONDS = 7 * 24 * 60 * 60
_ACQUIRED_MARKER_VERSION = 3


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


def redis_acquired_marker_key(key_prefix: str, reservation_id: str) -> str:
    return redis_namespace_key(
        key_prefix,
        "acquired",
        _validate_reservation_id(reservation_id),
    )


def redis_acquired_marker_value(
    *,
    reservation_id: str,
    model_family: str,
    bucket_ids: set[BucketId] | frozenset[BucketId],
    usage: Mapping[str, float] | None = None,
) -> str:
    # KNOWN UNKNOWN: this internal JSON marker schema is intentionally stricter
    # than the public CapacityReservation shape; no external compatibility is
    # promised until a future release documents marker payloads as operator API.
    payload: dict[str, object] = {
        "v": _ACQUIRED_MARKER_VERSION,
        "reservation_id": _validate_reservation_id(reservation_id),
        "model_family": model_family,
        "buckets": [
            [metric, int(per_seconds)] for metric, per_seconds in sorted(bucket_ids)
        ],
    }
    if usage is not None:
        normalized_usage = []
        for metric, amount in sorted(usage.items()):
            amount_float = float(amount)
            if not math.isfinite(amount_float) or amount_float < 0:
                raise ValueError(
                    f"Reserved usage value for {metric} must be finite and non-negative"
                )
            normalized_usage.append([metric, amount_float])
        payload["usage"] = normalized_usage
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def validate_refund_dedup_ttl_seconds(value: object) -> int:
    if type(value) is not int:
        raise TypeError("refund_dedup_ttl_seconds must be an int number of seconds")
    if value <= 0:
        raise ValueError("refund_dedup_ttl_seconds must be greater than 0")
    return value
