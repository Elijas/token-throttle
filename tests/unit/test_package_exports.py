"""Tests for top-level package exports."""

import importlib.util

import token_throttle

_REDIS_EXPORTS = {
    "LOCK_TIMEOUT_SECONDS",
    "CapacitiesGetterResult",
    "RedisBackend",
    "RedisBackendBuilder",
    "RedisBucket",
    "SyncRedisBackend",
    "SyncRedisBackendBuilder",
    "SyncRedisBucket",
    "create_openai_redis_rate_limiter",
    "openai_model_family_getter",
}


def test_star_import_works_without_optional_dependency_failures():
    namespace: dict[str, object] = {}

    exec("from token_throttle import *", namespace)

    assert namespace["RateLimiter"] is token_throttle.RateLimiter
    assert namespace["SyncRateLimiter"] is token_throttle.SyncRateLimiter


def test_redis_exports_match_installed_dependency():
    has_redis = importlib.util.find_spec("redis") is not None

    for export_name in _REDIS_EXPORTS:
        assert (export_name in token_throttle.__all__) is has_redis


def test_dir_contains_public_names():
    """Cover __dir__() in __init__.py."""
    d = dir(token_throttle)
    assert "RateLimiter" in d
    assert "SyncRateLimiter" in d
    assert "MemoryBackendBuilder" in d
    assert "Quota" in d


def test_getattr_raises_for_unknown_name():
    """Cover the AttributeError branch in __getattr__."""
    import pytest

    with pytest.raises(AttributeError, match="has no attribute"):
        token_throttle.this_does_not_exist_at_all
