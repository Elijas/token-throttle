DEFAULT_BUCKET_TTL_SECONDS = 7 * 24 * 60 * 60


def validate_redis_ttl_seconds(value: object, *, name: str) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise TypeError(f"{name} must be an int number of seconds")
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value
