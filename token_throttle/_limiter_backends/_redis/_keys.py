import unicodedata

_REDIS_NAMESPACE = "rate_limiting"


def validate_redis_key_prefix(value: object) -> str:
    """Validate the deployment-scoped Redis key prefix."""
    if type(value) is not str:
        raise ValueError(f"key_prefix must be a str (got {type(value).__name__})")
    if not value:
        raise ValueError("key_prefix must not be empty")

    normalized = unicodedata.normalize("NFC", value)
    if not normalized.strip():
        raise ValueError("key_prefix must not be whitespace-only")
    if normalized != normalized.strip():
        raise ValueError("key_prefix must not contain leading/trailing whitespace")
    if any(char.isspace() for char in normalized):
        raise ValueError("key_prefix must not contain whitespace")
    if any(not char.isprintable() for char in normalized):
        raise ValueError("key_prefix must not contain non-printable characters")
    if any(unicodedata.category(char).startswith("C") for char in normalized):
        raise ValueError("key_prefix must not contain Unicode control characters")
    if ":" in normalized:
        raise ValueError(
            "key_prefix must not contain ':' (used as Redis key separator)"
        )
    if "{" in normalized or "}" in normalized:
        raise ValueError(
            "key_prefix must not contain '{' or '}' (Redis Cluster hash tag delimiters)"
        )
    return normalized


def redis_namespace_key(key_prefix: str, *segments: object) -> str:
    return ":".join(
        (key_prefix, _REDIS_NAMESPACE, *(str(segment) for segment in segments))
    )
