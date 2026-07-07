"""Conformance guard: ``RateLimiter`` / ``SyncRateLimiter`` public-surface parity.

The two limiter classes are hand-maintained siblings. Their public surface is
the highest-blast-radius parity boundary in the library (it is the user-facing
API), yet it was pinned only by *parallel* per-surface test files — exactly the
maintenance pattern that lets siblings drift. This module is the committed
regression guard for finding H2-F1: it pins, head-to-head,

* the public method set (only documented flavor differences are allowed),
* the signatures of the shared public callables and the constructor,
* the ``RateLimiterCallbacks`` / ``SyncRateLimiterCallbacks`` field set,
* the paired public exports / factories, and
* the ``(type(exc), str(exc))`` matrix over the validation bad-input paths.

Runtime parity was audited clean when this was written; the guard exists so a
future one-sided edit (a new method, a reworded message on one surface only)
cannot silently reintroduce drift. The error matrix is lifted and simplified
from ``findings/_repros/h2_error_parity.py`` in the audit capsule.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

import token_throttle
from token_throttle import (
    CapacityReservation,
    MemoryBackendBuilder,
    PerModelConfig,
    Quota,
    RateLimiter,
    RateLimiterCallbacks,
    SyncMemoryBackendBuilder,
    SyncRateLimiter,
    SyncRateLimiterCallbacks,
    UsageQuotas,
    frozen_usage,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 1. Public method-set parity
# ---------------------------------------------------------------------------

# By-design flavor differences (finding §Non-findings): the async limiter
# exposes the native coroutine ``aclose()``; its ``close()`` is a convenience
# that refuses to run inside a live loop and otherwise delegates to ``aclose``.
# The sync limiter has only ``close()``. No other public name may differ.
_ASYNC_ONLY_PUBLIC_METHODS = frozenset({"aclose"})
_SYNC_ONLY_PUBLIC_METHODS: frozenset[str] = frozenset()


def _public_names(cls: type) -> set[str]:
    return {name for name in dir(cls) if not name.startswith("_")}


def test_public_method_set_parity() -> None:
    async_only = _public_names(RateLimiter) - _public_names(SyncRateLimiter)
    sync_only = _public_names(SyncRateLimiter) - _public_names(RateLimiter)
    assert async_only == _ASYNC_ONLY_PUBLIC_METHODS, (
        "async-only public methods drifted from the documented set; "
        f"got {sorted(async_only)}, expected {sorted(_ASYNC_ONLY_PUBLIC_METHODS)}"
    )
    assert sync_only == _SYNC_ONLY_PUBLIC_METHODS, (
        "SyncRateLimiter grew a public method with no async counterpart: "
        f"{sorted(sync_only)}"
    )


# ---------------------------------------------------------------------------
# 2. Signature parity for shared public callables + the constructor
# ---------------------------------------------------------------------------

_DOTTED_PATH = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")


def _canonical_annotation(annotation: object) -> str:
    """Render an annotation for flavor-insensitive comparison.

    The limiter modules do not use ``from __future__ import annotations``, so
    ``inspect.signature`` yields live objects. Module qualifiers are dropped
    (``pkg.mod.Name`` -> ``Name``) and the ``Sync`` flavor prefix is removed so
    ``SyncRateLimiterCallbacks`` and ``RateLimiterCallbacks`` compare equal.

    Return annotations need no unwrapping: ``inspect.signature`` reports the
    declared ``-> X`` for ``async def`` (not ``Coroutine[..., X]``), so the
    async and sync flavors already agree.
    """
    if annotation is inspect.Parameter.empty:
        return "<empty>"
    text = annotation.__name__ if isinstance(annotation, type) else str(annotation)
    text = _DOTTED_PATH.sub(lambda m: m.group(0).rsplit(".", 1)[-1], text)
    return text.replace("Sync", "")


def _normalized_signature(func: Callable[..., object]) -> tuple[Any, str]:
    sig = inspect.signature(func)
    params = tuple(
        (
            param.name,
            param.kind,
            repr(param.default),
            _canonical_annotation(param.annotation),
        )
        for param in sig.parameters.values()
    )
    return params, _canonical_annotation(sig.return_annotation)


def _shared_public_methods() -> list[str]:
    return sorted(_public_names(RateLimiter) & _public_names(SyncRateLimiter))


@pytest.mark.parametrize("method_name", _shared_public_methods())
def test_shared_public_method_signature_parity(method_name: str) -> None:
    async_sig = _normalized_signature(getattr(RateLimiter, method_name))
    sync_sig = _normalized_signature(getattr(SyncRateLimiter, method_name))
    assert async_sig == sync_sig, (
        f"{method_name}: signature drift between flavors\n"
        f"  async = {async_sig}\n"
        f"  sync  = {sync_sig}"
    )


def test_constructor_signature_parity() -> None:
    async_sig = _normalized_signature(RateLimiter.__init__)
    sync_sig = _normalized_signature(SyncRateLimiter.__init__)
    assert async_sig == sync_sig, (
        "constructor signature drift between flavors (after normalizing the "
        "expected backend-builder / callbacks flavor types)\n"
        f"  async = {async_sig}\n"
        f"  sync  = {sync_sig}"
    )


# ---------------------------------------------------------------------------
# 3. Callback DTO field parity
# ---------------------------------------------------------------------------


def _field_shape(model: type[BaseModel]) -> dict[str, tuple[bool, Any]]:
    return {
        name: (info.is_required(), info.default)
        for name, info in model.model_fields.items()
    }


def test_callback_dto_field_parity() -> None:
    async_fields = _field_shape(RateLimiterCallbacks)
    sync_fields = _field_shape(SyncRateLimiterCallbacks)
    assert async_fields == sync_fields, (
        "callback DTO field set / requiredness / defaults drifted\n"
        f"  RateLimiterCallbacks     = {async_fields}\n"
        f"  SyncRateLimiterCallbacks = {sync_fields}"
    )


# ---------------------------------------------------------------------------
# 4. Paired public exports / factories
# ---------------------------------------------------------------------------

# (async_name, sync_name): each pair must be present in ``__all__`` together or
# absent together. Redis-flavored pairs are absent without the redis extra; the
# present/absent parity assertion handles that automatically.
_EXPORT_PAIRS: tuple[tuple[str, str], ...] = (
    ("RateLimiter", "SyncRateLimiter"),
    ("RateLimiterCallbacks", "SyncRateLimiterCallbacks"),
    ("RateLimiterBackend", "SyncRateLimiterBackend"),
    ("RateLimiterBackendBuilderInterface", "SyncRateLimiterBackendBuilderInterface"),
    ("BackendIntrospectable", "SyncBackendIntrospectable"),
    ("OnWaitStartCallback", "SyncOnWaitStartCallback"),
    ("OnWaitEndCallback", "SyncOnWaitEndCallback"),
    ("OnCapacityConsumedCallback", "SyncOnCapacityConsumedCallback"),
    ("OnCapacityRefundedCallback", "SyncOnCapacityRefundedCallback"),
    ("OnMissingConsumptionDataCallback", "SyncOnMissingConsumptionDataCallback"),
    ("OnLifecycleEventCallback", "SyncOnLifecycleEventCallback"),
    ("create_logging_callbacks", "create_sync_logging_callbacks"),
    ("MemoryBackend", "SyncMemoryBackend"),
    ("MemoryBackendBuilder", "SyncMemoryBackendBuilder"),
    ("RedisBackend", "SyncRedisBackend"),
    ("RedisBackendBuilder", "SyncRedisBackendBuilder"),
    ("RedisBucket", "SyncRedisBucket"),
    ("create_openai_redis_rate_limiter", "create_openai_redis_sync_rate_limiter"),
)


@pytest.mark.parametrize(("async_name", "sync_name"), _EXPORT_PAIRS)
def test_paired_public_exports(async_name: str, sync_name: str) -> None:
    exported = set(token_throttle.__all__)
    assert (async_name in exported) == (sync_name in exported), (
        f"export pairing drift: {async_name!r} and {sync_name!r} must appear in "
        "token_throttle.__all__ together (or both be absent without the extra)"
    )


# ---------------------------------------------------------------------------
# 5. Validation error-contract parity matrix
#
# Identical bad inputs are fed to both surfaces; each must raise the same
# exception type with the same message. These are validation paths that fail
# before any backend I/O, so the (intentionally flavor-correct) builders below
# are never exercised — but using the matching builder per flavor keeps the
# scenarios honest. Lifted from findings/_repros/h2_error_parity.py and
# extended with direct refund_capacity value-paths (finding KNOWN UNKNOWN #1).
# ---------------------------------------------------------------------------


def good_counter(**_kwargs: Any) -> dict[str, float]:
    return {"tokens": 100.0, "requests": 1.0}


async def good_counter_async(**_kwargs: Any) -> dict[str, float]:
    return {"tokens": 100.0, "requests": 1.0}


def counter_returning_awaitable(**_kwargs: Any) -> Any:
    # A synchronous counter (so it passes config validation) whose *result* is
    # an awaitable. Both surfaces must reject the awaitable result identically.
    return good_counter_async()


def cfg(
    *,
    usage_counter: Any = None,
    unlimited: bool = False,
    model_family: str | None = None,
) -> PerModelConfig:
    if unlimited:
        return PerModelConfig(quotas=UsageQuotas.unlimited(), model_family=model_family)
    return PerModelConfig(
        quotas=UsageQuotas(
            [Quota(metric="tokens", limit=1000), Quota(metric="requests", limit=10)]
        ),
        model_family=model_family,
        usage_counter=usage_counter,
    )


def reservation_for(limiter: Any) -> CapacityReservation:
    return CapacityReservation(
        usage=frozen_usage({"tokens": 100.0, "requests": 1.0}),
        model_family="gpt-4",
        limiter_instance_id=limiter._limiter_instance_id,
    )


@dataclass(frozen=True)
class _Scenario:
    name: str
    cfg_kwargs: dict[str, Any]
    # A single thunk is run against both flavors: on the async limiter it
    # returns a coroutine (awaited below); on the sync limiter it returns the
    # value directly. Sharing one thunk guarantees byte-identical inputs.
    thunk: Callable[[Any], Any]


_SCENARIOS: tuple[_Scenario, ...] = (
    # --- acquire_capacity ---
    _Scenario(
        "acq: usage not mapping",
        {},
        lambda lim: lim.acquire_capacity([], model="gpt-4"),
    ),
    _Scenario(
        "acq: empty model",
        {},
        lambda lim: lim.acquire_capacity({"tokens": 1, "requests": 1}, model=""),
    ),
    _Scenario(
        "acq: model bool",
        {},
        lambda lim: lim.acquire_capacity({"tokens": 1, "requests": 1}, model=True),
    ),
    _Scenario(
        "acq: model int",
        {},
        lambda lim: lim.acquire_capacity({"tokens": 1, "requests": 1}, model=42),
    ),
    _Scenario(
        "acq: model None",
        {},
        lambda lim: lim.acquire_capacity({"tokens": 1, "requests": 1}, model=None),
    ),
    _Scenario(
        "acq: mismatched keys",
        {},
        lambda lim: lim.acquire_capacity({"tokens": 1}, model="gpt-4"),
    ),
    _Scenario(
        "acq: negative usage",
        {},
        lambda lim: lim.acquire_capacity({"tokens": -1, "requests": 1}, model="gpt-4"),
    ),
    _Scenario(
        "acq: boolean usage",
        {},
        lambda lim: lim.acquire_capacity(
            {"tokens": True, "requests": 1}, model="gpt-4"
        ),
    ),
    _Scenario(
        "acq: non-numeric usage",
        {},
        lambda lim: lim.acquire_capacity(
            {"tokens": object(), "requests": 1}, model="gpt-4"
        ),
    ),
    # --- timeout validation (unlimited cfg so timeout is the only failure) ---
    _Scenario(
        "acq: timeout bool",
        {"unlimited": True},
        lambda lim: lim.acquire_capacity({}, model="gpt-4", timeout=True),
    ),
    _Scenario(
        "acq: timeout nan",
        {"unlimited": True},
        lambda lim: lim.acquire_capacity({}, model="gpt-4", timeout=float("nan")),
    ),
    _Scenario(
        "acq: timeout inf",
        {"unlimited": True},
        lambda lim: lim.acquire_capacity({}, model="gpt-4", timeout=float("inf")),
    ),
    _Scenario(
        "acq: timeout negative",
        {"unlimited": True},
        lambda lim: lim.acquire_capacity({}, model="gpt-4", timeout=-1.0),
    ),
    _Scenario(
        "acq: timeout string",
        {"unlimited": True},
        lambda lim: lim.acquire_capacity({}, model="gpt-4", timeout="fast"),
    ),
    # --- acquire_capacity_for_request ---
    _Scenario(
        "acqreq: extra not mapping",
        {"usage_counter": good_counter},
        lambda lim: lim.acquire_capacity_for_request(model="gpt-4", extra_usage=[]),
    ),
    _Scenario(
        "acqreq: missing model",
        {},
        lambda lim: lim.acquire_capacity_for_request(extra_usage=None),
    ),
    _Scenario(
        "acqreq: model False",
        {},
        lambda lim: lim.acquire_capacity_for_request(model=False),
    ),
    _Scenario(
        "acqreq: model empty",
        {},
        lambda lim: lim.acquire_capacity_for_request(model=""),
    ),
    _Scenario(
        "acqreq: model int", {}, lambda lim: lim.acquire_capacity_for_request(model=0)
    ),
    _Scenario(
        "acqreq: none usage_counter",
        {},
        lambda lim: lim.acquire_capacity_for_request(model="gpt-4", extra_usage=None),
    ),
    _Scenario(
        "acqreq: unknown extra key",
        {"usage_counter": good_counter},
        lambda lim: lim.acquire_capacity_for_request(
            model="gpt-4", extra_usage={"unknown_metric": 5}
        ),
    ),
    _Scenario(
        "acqreq: extra boolean",
        {"usage_counter": good_counter},
        lambda lim: lim.acquire_capacity_for_request(
            model="gpt-4", extra_usage={"tokens": True}
        ),
    ),
    _Scenario(
        "acqreq: extra non-numeric",
        {"usage_counter": good_counter},
        lambda lim: lim.acquire_capacity_for_request(
            model="gpt-4", extra_usage={"tokens": object()}
        ),
    ),
    _Scenario(
        "acqreq: extra negative",
        {"usage_counter": good_counter},
        lambda lim: lim.acquire_capacity_for_request(
            model="gpt-4", extra_usage={"tokens": -1}
        ),
    ),
    _Scenario(
        "acqreq: timeout negative",
        {"unlimited": True},
        lambda lim: lim.acquire_capacity_for_request(model="gpt-4", timeout=-1.0),
    ),
    _Scenario(
        "acqreq: awaitable counter",
        {"usage_counter": counter_returning_awaitable},
        lambda lim: lim.acquire_capacity_for_request(model="gpt-4"),
    ),
    # --- refund_capacity_from_response (total_tokens validation, pre-backend) ---
    _Scenario(
        "refresp: no response no usage",
        {},
        lambda lim: lim.refund_capacity_from_response(reservation_for(lim)),
    ),
    _Scenario(
        "refresp: total_tokens None",
        {},
        lambda lim: lim.refund_capacity_from_response(
            reservation_for(lim), usage={"total_tokens": None}
        ),
    ),
    _Scenario(
        "refresp: total_tokens bool",
        {},
        lambda lim: lim.refund_capacity_from_response(
            reservation_for(lim), usage={"total_tokens": True}
        ),
    ),
    _Scenario(
        "refresp: total_tokens nan",
        {},
        lambda lim: lim.refund_capacity_from_response(
            reservation_for(lim), usage={"total_tokens": float("nan")}
        ),
    ),
    _Scenario(
        "refresp: total_tokens inf",
        {},
        lambda lim: lim.refund_capacity_from_response(
            reservation_for(lim), usage={"total_tokens": float("inf")}
        ),
    ),
    _Scenario(
        "refresp: total_tokens negative",
        {},
        lambda lim: lim.refund_capacity_from_response(
            reservation_for(lim), usage={"total_tokens": -5}
        ),
    ),
    _Scenario(
        "refresp: total_tokens 'abc'",
        {},
        lambda lim: lim.refund_capacity_from_response(
            reservation_for(lim), usage={"total_tokens": "abc"}
        ),
    ),
    _Scenario(
        "refresp: total_tokens '80'",
        {},
        lambda lim: lim.refund_capacity_from_response(
            reservation_for(lim), usage={"total_tokens": "80"}
        ),
    ),
    _Scenario(
        "refresp: usage missing total_tokens",
        {},
        lambda lim: lim.refund_capacity_from_response(
            reservation_for(lim), usage={"prompt_tokens": 50}
        ),
    ),
    # --- refund_capacity direct value-paths (finding KNOWN UNKNOWN #1) ---
    _Scenario(
        "refund: actual_usage is reservation",
        {},
        lambda lim: lim.refund_capacity(reservation_for(lim), reservation_for(lim)),
    ),
    _Scenario(
        "refund: mismatched keys",
        {},
        lambda lim: lim.refund_capacity({"nope": 1}, reservation_for(lim)),
    ),
    _Scenario(
        "refund: boolean usage",
        {},
        lambda lim: lim.refund_capacity(
            {"tokens": True, "requests": 1}, reservation_for(lim)
        ),
    ),
    _Scenario(
        "refund: non-numeric usage",
        {},
        lambda lim: lim.refund_capacity(
            {"tokens": object(), "requests": 1}, reservation_for(lim)
        ),
    ),
    _Scenario(
        "refund: negative usage",
        {},
        lambda lim: lim.refund_capacity(
            {"tokens": -1, "requests": 1}, reservation_for(lim)
        ),
    ),
    # --- set_max_capacity ---
    _Scenario(
        "setmax: no backend",
        {},
        lambda lim: lim.set_max_capacity("gpt-4", "tokens", 60, 500.0),
    ),
    _Scenario(
        "setmax: unlimited",
        {"unlimited": True},
        lambda lim: lim.set_max_capacity("gpt-4", "tokens", 60, 500.0),
    ),
)


def _async_outcome(scenario: _Scenario) -> tuple[type[BaseException], str] | None:
    async def _run() -> None:
        limiter = RateLimiter(
            cfg(**scenario.cfg_kwargs), backend=MemoryBackendBuilder()
        )
        try:
            await scenario.thunk(limiter)
        finally:
            await limiter.aclose()

    try:
        asyncio.run(_run())
    except BaseException as exc:
        return type(exc), str(exc)
    return None


def _sync_outcome(scenario: _Scenario) -> tuple[type[BaseException], str] | None:
    limiter = SyncRateLimiter(
        cfg(**scenario.cfg_kwargs), backend=SyncMemoryBackendBuilder()
    )
    try:
        scenario.thunk(limiter)
    except BaseException as exc:
        return type(exc), str(exc)
    finally:
        limiter.close()
    return None


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda scenario: scenario.name)
def test_validation_error_parity(scenario: _Scenario) -> None:
    async_outcome = _async_outcome(scenario)
    sync_outcome = _sync_outcome(scenario)
    assert async_outcome is not None, f"{scenario.name}: async surface did not raise"
    assert sync_outcome is not None, f"{scenario.name}: sync surface did not raise"
    assert async_outcome == sync_outcome, (
        f"{scenario.name}: validation error parity drift\n"
        f"  async = {async_outcome}\n"
        f"  sync  = {sync_outcome}"
    )
