"""Test that _openai_rate_limiter.py raises ImportError when redis is missing.

Covers lines 5-6 of token_throttle/_factories/_openai/_openai_rate_limiter.py.
"""

import importlib
import sys

import pytest


def test_openai_rate_limiter_raises_without_redis():
    module_key = "token_throttle._factories._openai._openai_rate_limiter"

    # Save modules that will be manipulated
    saved = {}
    keys_to_save = [module_key, "redis", "redis.asyncio"]
    for key in keys_to_save:
        if key in sys.modules:
            saved[key] = sys.modules.pop(key)

    try:
        # Make redis unavailable
        sys.modules["redis"] = None
        sys.modules["redis.asyncio"] = None

        with pytest.raises(ImportError, match="redis"):
            importlib.import_module(module_key)
    finally:
        # Clean up: remove our None entries
        sys.modules.pop("redis", None)
        sys.modules.pop("redis.asyncio", None)
        sys.modules.pop(module_key, None)

        # Restore originals
        for key, mod in saved.items():
            sys.modules[key] = mod


def test_top_level_openai_model_family_getter_works_without_redis():
    package_key = "token_throttle"
    helper_key = "token_throttle._factories._openai._model_family"
    rate_limiter_key = "token_throttle._factories._openai._openai_rate_limiter"

    saved = {}
    keys_to_save = [package_key, helper_key, rate_limiter_key, "redis", "redis.asyncio"]
    for key in keys_to_save:
        if key in sys.modules:
            saved[key] = sys.modules.pop(key)

    try:
        sys.modules["redis"] = None
        sys.modules["redis.asyncio"] = None

        package = importlib.import_module(package_key)

        assert package.openai_model_family_getter("gpt-4-0613") == "gpt-4"
    finally:
        sys.modules.pop("redis", None)
        sys.modules.pop("redis.asyncio", None)
        sys.modules.pop(package_key, None)
        sys.modules.pop(helper_key, None)
        sys.modules.pop(rate_limiter_key, None)

        for key, mod in saved.items():
            sys.modules[key] = mod
