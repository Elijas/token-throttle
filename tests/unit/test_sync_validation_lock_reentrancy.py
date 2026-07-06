"""Regression test: config_getter calling back into the limiter while
_validate_shared_model_family_config holds _validation_lock must not deadlock.

SyncRateLimiter resolves a conflicting shared model_family by re-invoking the
user's config_getter for the family's representative model, while still
holding _validation_lock. If that callback calls back into the limiter (e.g.
clear_unused_model_families, which takes the same lock), a plain
threading.Lock self-deadlocks. _validation_lock is a threading.RLock so
same-thread re-entry completes instead, mirroring the async RateLimiter
(which has no lock at all, since it has no await points to interleave on).
"""

import threading

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._sync_rate_limiter import SyncRateLimiter

MODEL_FAMILY = "shared-family"


def _config(limit: int) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas([Quota(metric="requests", limit=limit, per_seconds=60)]),
        model_family=MODEL_FAMILY,
    )


def test_config_getter_callback_into_clear_unused_model_families_does_not_deadlock():
    limiter_holder: list[SyncRateLimiter] = []

    def config_getter(model_name: str) -> PerModelConfig:
        if model_name == "model_a":
            # Simulate a config_getter that calls back into the limiter while
            # _validate_shared_model_family_config holds _validation_lock.
            limiter_holder[0].clear_unused_model_families(0)
            return _config(100)
        if model_name == "model_b":
            return _config(1)
        raise AssertionError(f"unexpected model {model_name!r}")

    limiter = SyncRateLimiter(config_getter, backend=SyncMemoryBackendBuilder())
    limiter_holder.append(limiter)

    # Register model_a as the model_family's representative alias.
    limiter.acquire_capacity({"requests": 1}, model="model_a")

    errors: list[BaseException] = []

    def acquire_conflicting_model() -> None:
        try:
            limiter.acquire_capacity({"requests": 1}, model="model_b")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=acquire_conflicting_model, daemon=True)
    thread.start()
    thread.join(timeout=10)

    assert not thread.is_alive(), (
        "acquire_capacity deadlocked; _validation_lock is not reentrant"
    )
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "inconsistent across models" in str(errors[0])
