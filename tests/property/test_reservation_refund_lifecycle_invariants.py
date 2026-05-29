"""Property / interleaving / taxonomy tests for the reservation refund-on-raise
state machine.

These exercise the ``_refund_or_forget_reservation_on_raise`` context managers
on ``RateLimiter`` and ``SyncRateLimiter`` and the refund state machine they
delegate to (``_refund_capacity``: a per-reservation lock plus a tri-state
``pending`` / ``failed`` / ``committed`` dedup map).

They pin invariants that existing tests only *bracket*:

* ``tests/unit/test_reservation_cleanup_on_raise.py`` exercises the context
  manager's branch *dispatch* but mocks ``_forget_in_flight_reservation`` and
  the refund helpers, so real capacity accounting, idempotency and the full
  ``BaseException`` taxonomy are never run; only ``CancelledError`` /
  ``RuntimeError`` are injected.
* ``tests/unit/test_r3_audit_fixes.py`` (F02.R3.01) and
  ``tests/unit/test_bundle_perf_refund_guard_narrow.py`` cover concurrent
  same-reservation refunds, but only at a fixed N=2 and assert a *mock*
  ``call_count`` / ``await_count`` rather than the final credited capacity.

Net-new coverage here:

1. Refund idempotency under fuzzed sequential repetition (real accounting).
2. No double-credit under fuzzed N-actor concurrency (real accounting) — the
   F02.R3.01 concurrent-duplicate-refund TOCTOU guard, generalized past N=2 and
   asserting the credited capacity rather than a call count.
3. Exception taxonomy during cleanup: every member of
   ``LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS`` injected at the real
   context-manager body via a lifecycle callback, on both the refund and forget
   branches.
4. Refund-vs-forget branch mutual exclusivity (real cleanup, not mocked).

All tests use the in-memory backend and require no Redis. Each invariant has
sync and async siblings asserting identical outcomes.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from token_throttle._exceptions import DuplicateRefundError
from token_throttle._interfaces._callbacks import (
    LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS,
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)
from token_throttle._rate_limiter import RateLimiter
from token_throttle._sync_rate_limiter import SyncRateLimiter

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

MODEL = "model"
MODEL_FAMILY = "fam"
METRIC = "tokens"
# Slow window so natural refill is negligible during a test (1000/3600 ~=
# 0.28 tok/s; a multi-second test refills well under the tolerance).
SLOW_WINDOW = 3600
LIMIT = 1000.0

# Drain most capacity then refund a *small* credit, so that even N erroneous
# extra credits stay well below ``max_capacity`` and cap-clipping can never
# mask a double-credit regression.
ACQUIRE = 900.0
REFUND_ACTUAL = 850.0
CREDIT = ACQUIRE - REFUND_ACTUAL  # 50.0
AFTER_ACQUIRE_CAP = LIMIT - ACQUIRE  # 100.0
AFTER_SINGLE_REFUND_CAP = AFTER_ACQUIRE_CAP + CREDIT  # 150.0

# Absorbs refill during the test window plus float noise (mirrors the
# REFILL_TOLERANCE idiom in tests/property/test_concurrent_accounting.py).
REFILL_TOLERANCE = 2.0


def _config() -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [Quota(metric=METRIC, limit=LIMIT, per_seconds=SLOW_WINDOW)]
        ),
        model_family=MODEL_FAMILY,
    )


def _get_capacity(backend) -> float:
    """Read current capacity from the first memory bucket."""
    return backend._buckets[0].get_capacity(time.time()).amount


# Distinct critical exception classes (the tuple may list CancelledError twice
# via the asyncio / concurrent.futures aliases). Parametrizing off the library
# tuple keeps the test honest if a member is ever added or removed.
_CRITICAL_CLASSES = list(dict.fromkeys(LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS))
_CRITICAL_IDS = [cls.__name__ for cls in _CRITICAL_CLASSES]


# ---------------------------------------------------------------------------
# Group A: refund idempotency under fuzzed sequential repetition
# ---------------------------------------------------------------------------


@hypothesis_settings(max_examples=40, deadline=None)
@given(repeats=st.integers(min_value=2, max_value=8))
def test_async_refund_idempotent_under_sequential_repetition(repeats: int) -> None:
    """Refunding the same reservation K times credits the backend exactly once.

    Exactly one call succeeds; every later call raises ``DuplicateRefundError``
    (never a different exception); the final capacity equals the single-refund
    outcome; refund state is committed; per-reservation locks are released.
    """

    async def run() -> None:
        limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
        reservation = await limiter.acquire_capacity({METRIC: ACQUIRE}, MODEL)
        backend = limiter._model_family_to_backend[MODEL_FAMILY]

        outcomes: list[BaseException | None] = []
        for _ in range(repeats):
            try:
                await limiter.refund_capacity({METRIC: REFUND_ACTUAL}, reservation)
                outcomes.append(None)
            except BaseException as exc:  # classified below
                outcomes.append(exc)

        successes = [o for o in outcomes if o is None]
        dups = [o for o in outcomes if isinstance(o, DuplicateRefundError)]
        others = [
            o
            for o in outcomes
            if o is not None and not isinstance(o, DuplicateRefundError)
        ]
        assert others == [], f"unexpected exceptions: {others}"
        assert len(successes) == 1
        assert len(dups) == repeats - 1
        assert _get_capacity(backend) == pytest.approx(
            AFTER_SINGLE_REFUND_CAP, abs=REFILL_TOLERANCE
        )
        assert limiter._refunded_reservation_ids[reservation.reservation_id] == (
            "committed"
        )
        assert reservation.reservation_id not in limiter._in_flight_reservation_ids
        assert limiter._refund_locks == {}

    asyncio.run(run())


@hypothesis_settings(max_examples=40, deadline=None)
@given(repeats=st.integers(min_value=2, max_value=8))
def test_sync_refund_idempotent_under_sequential_repetition(repeats: int) -> None:
    """Sync sibling of the async sequential-idempotency property."""
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    reservation = limiter.acquire_capacity({METRIC: ACQUIRE}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]

    outcomes: list[BaseException | None] = []
    for _ in range(repeats):
        try:
            limiter.refund_capacity({METRIC: REFUND_ACTUAL}, reservation)
            outcomes.append(None)
        except BaseException as exc:  # classified below
            outcomes.append(exc)

    successes = [o for o in outcomes if o is None]
    dups = [o for o in outcomes if isinstance(o, DuplicateRefundError)]
    others = [
        o for o in outcomes if o is not None and not isinstance(o, DuplicateRefundError)
    ]
    assert others == [], f"unexpected exceptions: {others}"
    assert len(successes) == 1
    assert len(dups) == repeats - 1
    assert _get_capacity(backend) == pytest.approx(
        AFTER_SINGLE_REFUND_CAP, abs=REFILL_TOLERANCE
    )
    assert limiter._refunded_reservation_ids[reservation.reservation_id] == "committed"
    assert reservation.reservation_id not in limiter._in_flight_reservation_ids
    assert limiter._refund_locks == {}


# ---------------------------------------------------------------------------
# Group B: no double-credit under fuzzed N-actor concurrency.
#
# Generalizes the N=2 concurrent-duplicate-refund TOCTOU guard (F02.R3.01 in
# tests/unit/test_r3_audit_fixes.py) to fuzzed N, asserting the final credited
# capacity rather than a mock call count.
# ---------------------------------------------------------------------------


@hypothesis_settings(max_examples=25, deadline=None)
@given(n_actors=st.integers(min_value=2, max_value=8), widen=st.booleans())
def test_async_concurrent_same_reservation_credits_once(
    n_actors: int, *, widen: bool
) -> None:
    """N concurrent ``refund_capacity`` calls for one reservation credit once.

    A TOCTOU double-credit regression (the F02.R3.01 shape) would push the
    final capacity above the single-refund value by ``CREDIT`` per extra
    crediting actor. ``widen`` injects an await inside the backend write to
    enlarge the interleaving window.
    """

    async def run() -> None:
        limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
        reservation = await limiter.acquire_capacity({METRIC: ACQUIRE}, MODEL)
        backend = limiter._model_family_to_backend[MODEL_FAMILY]
        if widen:
            original = backend.refund_capacity_for_buckets

            async def slow_write(*args, **kwargs):
                await asyncio.sleep(0.01)
                return await original(*args, **kwargs)

            backend.refund_capacity_for_buckets = slow_write

        results = await asyncio.gather(
            *[
                limiter.refund_capacity({METRIC: REFUND_ACTUAL}, reservation)
                for _ in range(n_actors)
            ],
            return_exceptions=True,
        )

        successes = [r for r in results if r is None]
        dups = [r for r in results if isinstance(r, DuplicateRefundError)]
        others = [
            r
            for r in results
            if r is not None and not isinstance(r, DuplicateRefundError)
        ]
        assert others == [], f"unexpected exceptions: {others}"
        assert len(successes) == 1
        assert len(dups) == n_actors - 1
        assert _get_capacity(backend) == pytest.approx(
            AFTER_SINGLE_REFUND_CAP, abs=REFILL_TOLERANCE
        ), "double-credit: capacity exceeds single-refund outcome"
        assert limiter._refund_locks == {}

    asyncio.run(run())


@hypothesis_settings(max_examples=25, deadline=None)
@given(n_actors=st.integers(min_value=2, max_value=8), widen=st.booleans())
def test_sync_concurrent_same_reservation_credits_once(
    n_actors: int, *, widen: bool
) -> None:
    """Sync sibling: N threads racing one reservation credit it exactly once."""
    limiter = SyncRateLimiter(_config(), backend=SyncMemoryBackendBuilder())
    reservation = limiter.acquire_capacity({METRIC: ACQUIRE}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    if widen:
        original = backend.refund_capacity_for_buckets

        def slow_write(*args, **kwargs):
            time.sleep(0.01)
            return original(*args, **kwargs)

        backend.refund_capacity_for_buckets = slow_write

    barrier = threading.Barrier(n_actors)
    results: list[BaseException | None] = []
    results_lock = threading.Lock()

    def refund_once(_: int) -> None:
        barrier.wait()
        try:
            limiter.refund_capacity({METRIC: REFUND_ACTUAL}, reservation)
            with results_lock:
                results.append(None)
        except BaseException as exc:  # classified below
            with results_lock:
                results.append(exc)

    with ThreadPoolExecutor(max_workers=n_actors) as pool:
        list(pool.map(refund_once, range(n_actors)))

    successes = [r for r in results if r is None]
    dups = [r for r in results if isinstance(r, DuplicateRefundError)]
    others = [
        r for r in results if r is not None and not isinstance(r, DuplicateRefundError)
    ]
    assert others == [], f"unexpected exceptions: {others}"
    assert len(successes) == 1
    assert len(dups) == n_actors - 1
    assert _get_capacity(backend) == pytest.approx(
        AFTER_SINGLE_REFUND_CAP, abs=REFILL_TOLERANCE
    ), "double-credit: capacity exceeds single-refund outcome"
    assert limiter._refund_locks == {}


# ---------------------------------------------------------------------------
# Group C: exception taxonomy during context-manager cleanup.
#
# A lifecycle callback that raises on ``capacity_consumed`` is the real
# (non-mocked) trigger for ``_refund_or_forget_reservation_on_raise``. Only
# critical exceptions propagate out of the callback (``safe_invoke_*`` swallows
# ordinary ones), so the taxonomy under test is exactly
# ``LIFECYCLE_CALLBACK_CRITICAL_EXCEPTIONS``.
# ---------------------------------------------------------------------------


class _AsyncRaisingCallbacks:
    """Build async callbacks that raise ``exc_type`` once on capacity_consumed."""

    def __init__(self, exc_type: type[BaseException]) -> None:
        self.exc_type = exc_type
        self.armed = False
        self.refund_events = 0

    def build(self) -> RateLimiterCallbacks:
        async def on_event(event) -> None:
            if event.event_type == "capacity_refunded":
                self.refund_events += 1
            if event.event_type == "capacity_consumed" and self.armed:
                raise self.exc_type("injected")

        return RateLimiterCallbacks(on_lifecycle_event=on_event)


class _SyncRaisingCallbacks:
    def __init__(self, exc_type: type[BaseException]) -> None:
        self.exc_type = exc_type
        self.armed = False
        self.refund_events = 0

    def build(self) -> SyncRateLimiterCallbacks:
        def on_event(event) -> None:
            if event.event_type == "capacity_refunded":
                self.refund_events += 1
            if event.event_type == "capacity_consumed" and self.armed:
                raise self.exc_type("injected")

        return SyncRateLimiterCallbacks(on_lifecycle_event=on_event)


@pytest.mark.parametrize("exc_type", _CRITICAL_CLASSES, ids=_CRITICAL_IDS)
async def test_async_refund_branch_propagates_critical_and_refunds(exc_type) -> None:
    """Refund branch: a critical exception in the ctx body propagates unwrapped
    and capacity is refunded (process signals are not swallowed).
    """
    cb = _AsyncRaisingCallbacks(exc_type)
    limiter = RateLimiter(
        _config(), backend=MemoryBackendBuilder(), callbacks=cb.build()
    )
    primer = await limiter.acquire_capacity({METRIC: 0}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    before = _get_capacity(backend)

    cb.armed = True
    with pytest.raises(exc_type):
        await limiter.acquire_capacity({METRIC: ACQUIRE}, MODEL)

    assert _get_capacity(backend) == pytest.approx(before, abs=REFILL_TOLERANCE)
    extra_in_flight = limiter._in_flight_reservation_ids - {primer.reservation_id}
    assert extra_in_flight == set()
    assert cb.refund_events == 1
    assert limiter._refund_locks == {}


@pytest.mark.parametrize("exc_type", _CRITICAL_CLASSES, ids=_CRITICAL_IDS)
async def test_async_forget_branch_propagates_critical_and_keeps_consumed(
    exc_type,
) -> None:
    """Forget branch (record_usage): a critical exception propagates and the
    consumed capacity is left consumed (not refunded); the reservation is
    forgotten.
    """
    cb = _AsyncRaisingCallbacks(exc_type)
    limiter = RateLimiter(
        _config(), backend=MemoryBackendBuilder(), callbacks=cb.build()
    )
    primer = await limiter.acquire_capacity({METRIC: 0}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    before = _get_capacity(backend)

    cb.armed = True
    with pytest.raises(exc_type):
        await limiter.record_usage({METRIC: ACQUIRE}, MODEL)

    assert _get_capacity(backend) == pytest.approx(
        before - ACQUIRE, abs=REFILL_TOLERANCE
    )
    extra_in_flight = limiter._in_flight_reservation_ids - {primer.reservation_id}
    assert extra_in_flight == set()
    assert cb.refund_events == 0
    assert limiter._refund_locks == {}


@pytest.mark.parametrize("exc_type", _CRITICAL_CLASSES, ids=_CRITICAL_IDS)
def test_sync_refund_branch_propagates_critical_and_refunds(exc_type) -> None:
    """Sync sibling of the refund-branch taxonomy test."""
    cb = _SyncRaisingCallbacks(exc_type)
    limiter = SyncRateLimiter(
        _config(), backend=SyncMemoryBackendBuilder(), callbacks=cb.build()
    )
    primer = limiter.acquire_capacity({METRIC: 0}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    before = _get_capacity(backend)

    cb.armed = True
    with pytest.raises(exc_type):
        limiter.acquire_capacity({METRIC: ACQUIRE}, MODEL)

    assert _get_capacity(backend) == pytest.approx(before, abs=REFILL_TOLERANCE)
    extra_in_flight = limiter._in_flight_reservation_ids - {primer.reservation_id}
    assert extra_in_flight == set()
    assert cb.refund_events == 1
    assert limiter._refund_locks == {}


@pytest.mark.parametrize("exc_type", _CRITICAL_CLASSES, ids=_CRITICAL_IDS)
def test_sync_forget_branch_propagates_critical_and_keeps_consumed(exc_type) -> None:
    """Sync sibling of the forget-branch taxonomy test."""
    cb = _SyncRaisingCallbacks(exc_type)
    limiter = SyncRateLimiter(
        _config(), backend=SyncMemoryBackendBuilder(), callbacks=cb.build()
    )
    primer = limiter.acquire_capacity({METRIC: 0}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]
    before = _get_capacity(backend)

    cb.armed = True
    with pytest.raises(exc_type):
        limiter.record_usage({METRIC: ACQUIRE}, MODEL)

    assert _get_capacity(backend) == pytest.approx(
        before - ACQUIRE, abs=REFILL_TOLERANCE
    )
    extra_in_flight = limiter._in_flight_reservation_ids - {primer.reservation_id}
    assert extra_in_flight == set()
    assert cb.refund_events == 0
    assert limiter._refund_locks == {}


# ---------------------------------------------------------------------------
# Group D: refund-vs-forget branch mutual exclusivity (real cleanup)
# ---------------------------------------------------------------------------


async def test_async_refund_and_forget_branches_are_mutually_exclusive() -> None:
    """For the same injected critical exception, the blocking-acquire path
    refunds (capacity restored, refund event fired) while the record_usage
    path forgets (capacity left consumed, no refund event). The two outcomes
    are mutually exclusive and both correct.
    """
    # Refund branch.
    refund_cb = _AsyncRaisingCallbacks(asyncio.CancelledError)
    refund_limiter = RateLimiter(
        _config(), backend=MemoryBackendBuilder(), callbacks=refund_cb.build()
    )
    await refund_limiter.acquire_capacity({METRIC: 0}, MODEL)
    refund_backend = refund_limiter._model_family_to_backend[MODEL_FAMILY]
    refund_before = _get_capacity(refund_backend)
    refund_cb.armed = True
    with pytest.raises(asyncio.CancelledError):
        await refund_limiter.acquire_capacity({METRIC: ACQUIRE}, MODEL)
    refund_after = _get_capacity(refund_backend)

    # Forget branch.
    forget_cb = _AsyncRaisingCallbacks(asyncio.CancelledError)
    forget_limiter = RateLimiter(
        _config(), backend=MemoryBackendBuilder(), callbacks=forget_cb.build()
    )
    await forget_limiter.acquire_capacity({METRIC: 0}, MODEL)
    forget_backend = forget_limiter._model_family_to_backend[MODEL_FAMILY]
    forget_before = _get_capacity(forget_backend)
    forget_cb.armed = True
    with pytest.raises(asyncio.CancelledError):
        await forget_limiter.record_usage({METRIC: ACQUIRE}, MODEL)
    forget_after = _get_capacity(forget_backend)

    # Mutually exclusive: refund restored capacity AND fired a refund event;
    # forget left capacity consumed AND fired no refund event.
    assert refund_after == pytest.approx(refund_before, abs=REFILL_TOLERANCE)
    assert refund_cb.refund_events == 1
    assert forget_after == pytest.approx(forget_before - ACQUIRE, abs=REFILL_TOLERANCE)
    assert forget_cb.refund_events == 0


async def test_async_unlimited_reservation_forgets_without_crediting() -> None:
    """Unlimited reservations take the forget branch: a critical exception in
    the ctx body propagates and the reservation is forgotten with no backend
    credit (unlimited never consumed capacity).
    """
    raised = {"armed": False}

    async def on_event(event) -> None:
        if event.event_type == "capacity_consumed" and raised["armed"]:
            raise asyncio.CancelledError("injected")

    unlimited_cfg = PerModelConfig(
        quotas=UsageQuotas.unlimited(), model_family=MODEL_FAMILY
    )
    limiter = RateLimiter(
        unlimited_cfg,
        backend=MemoryBackendBuilder(),
        callbacks=RateLimiterCallbacks(on_lifecycle_event=on_event),
    )
    raised["armed"] = True
    with pytest.raises(asyncio.CancelledError):
        await limiter.acquire_capacity({}, MODEL)

    assert limiter._in_flight_reservation_ids == set()
    assert limiter._refund_locks == {}


def test_sync_unlimited_reservation_forgets_without_crediting() -> None:
    """Sync sibling: an unlimited reservation interrupted in the ctx body is
    forgotten with no backend credit (parity with the async unlimited case).
    """
    raised = {"armed": False}

    def on_event(event) -> None:
        if event.event_type == "capacity_consumed" and raised["armed"]:
            raise asyncio.CancelledError("injected")

    unlimited_cfg = PerModelConfig(
        quotas=UsageQuotas.unlimited(), model_family=MODEL_FAMILY
    )
    limiter = SyncRateLimiter(
        unlimited_cfg,
        backend=SyncMemoryBackendBuilder(),
        callbacks=SyncRateLimiterCallbacks(on_lifecycle_event=on_event),
    )
    raised["armed"] = True
    with pytest.raises(asyncio.CancelledError):
        limiter.acquire_capacity({}, MODEL)

    assert limiter._in_flight_reservation_ids == set()
    assert limiter._refund_locks == {}
