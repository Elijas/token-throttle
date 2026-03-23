"""
Concurrent & blocking-acquire property tests for quota accounting.

Unlike the sequential property tests (shadow-model machines with frozen time),
these tests use **real time** and exercise the `threading.Condition` /
`asyncio.Condition` notification paths under randomized concurrent workloads.

Tests verify post-condition invariants rather than step-by-step shadow tracking,
since concurrent interleavings are nondeterministic.

Coverage targets:
1. Thread-safe concurrent ops (no corruption)
2. Async concurrent ops (no corruption)
3. No double-spend under contention
4. Blocked acquire wakes on refund (notify_all path)
5. Multiple waiters wake on refund
6. Blocked acquire wakes on set_max_capacity
7. set_max_capacity below acquire amount raises ValueError
8. Concurrent refund cap-clipping atomicity
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._limiter_backends._memory._backend import (
    MemoryBackend,
    MemoryBackendBuilder,
)
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackend,
    SyncMemoryBackendBuilder,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRIC = "requests"
# Slow refill so natural refill is negligible during tests.
# 100 tokens / 3600s = 0.028 tokens/sec; a 5s test refills ~0.14 tokens.
SLOW_WINDOW = 3600
LIMIT = 100.0

# Tolerance for conservation checks. Generous enough to absorb refill during
# the test window (max ~5s at 100/3600 rate = 0.14 tokens) plus float noise.
REFILL_TOLERANCE = 2.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_sync_backend(
    *, limit: float = LIMIT, per_seconds: int = SLOW_WINDOW, metric: str = METRIC
) -> SyncMemoryBackend:
    builder = SyncMemoryBackendBuilder()
    config = PerModelConfig(
        model_family="test",
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
    )
    return builder.build(config)


def _build_async_backend(
    *, limit: float = LIMIT, per_seconds: int = SLOW_WINDOW, metric: str = METRIC
) -> MemoryBackend:
    builder = MemoryBackendBuilder()
    config = PerModelConfig(
        model_family="test",
        quotas=UsageQuotas(
            [Quota(metric=metric, limit=limit, per_seconds=per_seconds)]
        ),
    )
    return builder.build(config)


def _get_capacity(backend) -> float:
    """Read current capacity from the first memory bucket."""
    return backend._buckets[0].get_capacity(time.time()).amount


def _get_max_capacity(backend) -> float:
    return backend._buckets[0].max_capacity


def verify_post_conditions(
    backend,
    *,
    initial_cap: float,
    total_consumed: float,
    total_refunded: float,
    label: str = "",
) -> None:
    """Shared invariant checks for concurrent tests 1-4."""
    final_cap = _get_capacity(backend)
    max_cap = _get_max_capacity(backend)

    # Invariant 1: capacity must not exceed max_capacity
    assert final_cap <= max_cap + 0.01, (
        f"{label}capacity {final_cap} exceeded max {max_cap}"
    )

    # Invariant 2: conservation (with refill tolerance)
    # final_cap + total_consumed - total_refunded <= initial_cap + tolerance
    conservation = final_cap + total_consumed - total_refunded
    assert conservation <= initial_cap + REFILL_TOLERANCE, (
        f"{label}conservation violated: {final_cap} + {total_consumed} - "
        f"{total_refunded} = {conservation} > {initial_cap} + {REFILL_TOLERANCE}"
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Operation types for the concurrent fuzz tests
OP_CONSUME = "consume"
OP_REFUND = "refund"
OP_SET_MAX = "set_max"
OP_ACQUIRE = "acquire"

op_strategy = st.sampled_from([OP_CONSUME, OP_REFUND, OP_SET_MAX, OP_ACQUIRE])
amount_strategy = st.floats(
    min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False
)
max_cap_strategy = st.floats(
    min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False
)
op_with_amount = st.tuples(op_strategy, amount_strategy, max_cap_strategy)
ops_list_strategy = st.lists(op_with_amount, min_size=10, max_size=30)
n_threads_strategy = st.integers(min_value=4, max_value=8)

# For the double-spend test
n_requesters_strategy = st.integers(min_value=5, max_value=10)

# For the blocked-acquire tests
refund_amount_strategy = st.floats(
    min_value=10.0, max_value=50.0, allow_nan=False, allow_infinity=False
)
acquire_amount_strategy = st.floats(
    min_value=5.0, max_value=40.0, allow_nan=False, allow_infinity=False
)

# For the multiple-waiters test
n_waiters_strategy = st.integers(min_value=2, max_value=5)
small_amount_strategy = st.floats(
    min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False
)

# For concurrent refund cap-clipping
refund_clip_amount_strategy = st.floats(
    min_value=5.0, max_value=30.0, allow_nan=False, allow_infinity=False
)
n_refunders_strategy = st.integers(min_value=4, max_value=8)


# ---------------------------------------------------------------------------
# Test 1: Sync concurrent ops — no corruption
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@hypothesis_settings(max_examples=50, deadline=None)
@given(ops=ops_list_strategy, n_threads=n_threads_strategy)
def test_sync_concurrent_ops_no_corruption(ops, n_threads):
    """Hypothesis-generated ops dispatched to multiple threads on SyncMemoryBackend.

    Verifies no unexpected exceptions and post-condition invariants hold.
    """
    backend = _build_sync_backend()

    # Thread-safe accumulators
    consumed_lock = threading.Lock()
    consumed_total = [0.0]
    refunded_total = [0.0]
    errors: list[BaseException] = []

    def execute_op(op_type, amount, max_cap_val):
        try:
            if op_type == OP_CONSUME:
                backend.consume_capacity(frozen_usage({METRIC: amount}))
                with consumed_lock:
                    consumed_total[0] += amount
            elif op_type == OP_REFUND:
                # Refund as if we reserved `amount` but used 0
                backend.refund_capacity(
                    reserved_usage=frozen_usage({METRIC: amount}),
                    actual_usage=frozen_usage({METRIC: 0}),
                )
                with consumed_lock:
                    refunded_total[0] += amount
            elif op_type == OP_SET_MAX:
                backend.set_max_capacity(METRIC, SLOW_WINDOW, max_cap_val)
            elif op_type == OP_ACQUIRE:
                backend.wait_for_capacity(frozen_usage({METRIC: amount}), timeout=0.0)
                with consumed_lock:
                    consumed_total[0] += amount
        except (TimeoutError, ValueError):
            pass  # Expected under contention
        except BaseException as exc:
            errors.append(exc)

    # Partition ops across threads
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [
            pool.submit(execute_op, op_type, amount, max_cap_val)
            for op_type, amount, max_cap_val in ops
        ]
        for f in as_completed(futures, timeout=30):
            f.result()

    assert errors == [], f"Unexpected errors: {errors}"

    final_cap = _get_capacity(backend)
    max_cap = _get_max_capacity(backend)
    assert final_cap <= max_cap + 0.01, f"capacity {final_cap} exceeded max {max_cap}"


# ---------------------------------------------------------------------------
# Test 2: Async concurrent ops — no corruption
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@hypothesis_settings(max_examples=50, deadline=None)
@given(ops=ops_list_strategy, n_tasks=n_threads_strategy)
def test_async_concurrent_ops_no_corruption(ops, n_tasks):
    """Async mirror of test 1 using asyncio.gather with MemoryBackend."""

    async def run():
        backend = _build_async_backend()
        errors: list[BaseException] = []

        async def execute_op(op_type, amount, max_cap_val):
            try:
                if op_type == OP_CONSUME:
                    await backend.consume_capacity(frozen_usage({METRIC: amount}))
                elif op_type == OP_REFUND:
                    await backend.refund_capacity(
                        reserved_usage=frozen_usage({METRIC: amount}),
                        actual_usage=frozen_usage({METRIC: 0}),
                    )
                elif op_type == OP_SET_MAX:
                    await backend.set_max_capacity(METRIC, SLOW_WINDOW, max_cap_val)
                elif op_type == OP_ACQUIRE:
                    await backend.await_for_capacity(
                        frozen_usage({METRIC: amount}), timeout=0.0
                    )
            except (TimeoutError, ValueError):
                pass
            except BaseException as exc:
                errors.append(exc)

        tasks = [
            execute_op(op_type, amount, max_cap_val)
            for op_type, amount, max_cap_val in ops
        ]
        await asyncio.gather(*tasks)

        assert errors == [], f"Unexpected errors: {errors}"

        final_cap = _get_capacity(backend)
        max_cap = _get_max_capacity(backend)
        assert final_cap <= max_cap + 0.01, (
            f"capacity {final_cap} exceeded max {max_cap}"
        )

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Test 3: Sync concurrent acquires — no double-spend
# ---------------------------------------------------------------------------


@hypothesis_settings(max_examples=50, deadline=None)
@given(n_requesters=n_requesters_strategy)
def test_sync_concurrent_all_acquires_no_double_spend(n_requesters):
    """N threads each requesting LIMIT/N tokens from a fresh bucket.

    All must succeed (bucket starts full). Final capacity must be near 0.
    """
    per_thread = LIMIT / n_requesters
    backend = _build_sync_backend(limit=LIMIT)

    with ThreadPoolExecutor(max_workers=n_requesters) as pool:
        futures = [
            pool.submit(backend.wait_for_capacity, frozen_usage({METRIC: per_thread}))
            for _ in range(n_requesters)
        ]
        results = [f.result() for f in as_completed(futures, timeout=10)]

    assert len(results) == n_requesters
    final_cap = _get_capacity(backend)
    assert final_cap == pytest.approx(0.0, abs=REFILL_TOLERANCE), (
        f"Expected ~0 after exact drain by {n_requesters} threads, got {final_cap}"
    )


# ---------------------------------------------------------------------------
# Test 4: Async concurrent acquires — no double-spend
# ---------------------------------------------------------------------------


@hypothesis_settings(max_examples=50, deadline=None)
@given(n_requesters=n_requesters_strategy)
def test_async_concurrent_all_acquires_no_double_spend(n_requesters):
    """Async mirror of test 3."""

    async def run():
        per_task = LIMIT / n_requesters
        backend = _build_async_backend(limit=LIMIT)

        results = await asyncio.gather(
            *[
                backend.await_for_capacity(frozen_usage({METRIC: per_task}))
                for _ in range(n_requesters)
            ],
            return_exceptions=True,
        )

        failures = [r for r in results if isinstance(r, BaseException)]
        assert failures == [], f"Unexpected failures: {failures}"

        final_cap = _get_capacity(backend)
        assert final_cap == pytest.approx(0.0, abs=REFILL_TOLERANCE), (
            f"Expected ~0 after exact drain by {n_requesters} tasks, got {final_cap}"
        )

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Test 5: Sync blocked acquire wakes on refund
# ---------------------------------------------------------------------------


@hypothesis_settings(max_examples=30, deadline=None)
@given(acquire_amount=acquire_amount_strategy)
def test_sync_blocked_acquire_wakes_on_refund(acquire_amount):
    """Refund -> notify_all() -> blocked waiter wakes path.

    1. Drain all capacity
    2. Launch background thread with blocking acquire
    3. Short sleep, then refund enough capacity
    4. Verify background thread succeeded
    """
    backend = _build_sync_backend(limit=LIMIT)

    # Drain all capacity
    backend.wait_for_capacity(frozen_usage({METRIC: LIMIT}))

    acquired = threading.Event()
    waiter_errors: list[BaseException] = []

    def blocking_acquire():
        try:
            backend.wait_for_capacity(
                frozen_usage({METRIC: acquire_amount}), timeout=5.0
            )
            acquired.set()
        except BaseException as exc:
            waiter_errors.append(exc)

    t = threading.Thread(target=blocking_acquire)
    t.start()

    # Give the thread time to enter the wait
    time.sleep(0.15)

    # Refund enough capacity for the waiter
    backend.refund_capacity(
        reserved_usage=frozen_usage({METRIC: acquire_amount}),
        actual_usage=frozen_usage({METRIC: 0}),
    )

    t.join(timeout=5.0)
    assert not t.is_alive(), "Waiter thread did not exit"
    assert waiter_errors == [], f"Waiter errors: {waiter_errors}"
    assert acquired.is_set(), "Waiter should have acquired after refund"

    # Conservation: started with 0 (drained), refunded acquire_amount,
    # waiter consumed acquire_amount => final should be ~0
    final_cap = _get_capacity(backend)
    assert final_cap == pytest.approx(0.0, abs=REFILL_TOLERANCE), (
        f"Expected ~0, got {final_cap}"
    )


# ---------------------------------------------------------------------------
# Test 6: Async blocked acquire wakes on refund
# ---------------------------------------------------------------------------


@hypothesis_settings(max_examples=30, deadline=None)
@given(acquire_amount=acquire_amount_strategy)
def test_async_blocked_acquire_wakes_on_refund(acquire_amount):
    """Async mirror of test 5."""

    async def run():
        backend = _build_async_backend(limit=LIMIT)

        # Drain all capacity
        await backend.await_for_capacity(frozen_usage({METRIC: LIMIT}))

        waiter_task = asyncio.create_task(
            backend.await_for_capacity(
                frozen_usage({METRIC: acquire_amount}), timeout=5.0
            )
        )

        # Give the task time to enter the wait
        await asyncio.sleep(0.15)

        # Refund enough
        await backend.refund_capacity(
            reserved_usage=frozen_usage({METRIC: acquire_amount}),
            actual_usage=frozen_usage({METRIC: 0}),
        )

        # Waiter should complete
        await asyncio.wait_for(waiter_task, timeout=5.0)

        final_cap = _get_capacity(backend)
        assert final_cap == pytest.approx(0.0, abs=REFILL_TOLERANCE), (
            f"Expected ~0, got {final_cap}"
        )

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Test 7: Sync multiple waiters wake on refund
# ---------------------------------------------------------------------------


@hypothesis_settings(max_examples=30, deadline=None)
@given(
    n_waiters=n_waiters_strategy,
    per_waiter=small_amount_strategy,
)
def test_sync_multiple_waiters_wake_on_refund(n_waiters, per_waiter):
    """Multiple blocked threads each wanting a small amount. Refund enough for all.

    Verify all wake up and succeed. Conservation: refund - sum(acquired) ~ final_cap.
    """
    backend = _build_sync_backend(limit=LIMIT)

    # Drain all capacity
    backend.wait_for_capacity(frozen_usage({METRIC: LIMIT}))

    acquired_count = [0]
    count_lock = threading.Lock()
    waiter_errors: list[BaseException] = []

    def blocking_acquire():
        try:
            backend.wait_for_capacity(frozen_usage({METRIC: per_waiter}), timeout=5.0)
            with count_lock:
                acquired_count[0] += 1
        except BaseException as exc:
            waiter_errors.append(exc)

    threads = [threading.Thread(target=blocking_acquire) for _ in range(n_waiters)]
    for t in threads:
        t.start()

    # Give threads time to enter the wait
    time.sleep(0.2)

    # Refund enough for all waiters
    total_needed = per_waiter * n_waiters
    backend.refund_capacity(
        reserved_usage=frozen_usage({METRIC: total_needed}),
        actual_usage=frozen_usage({METRIC: 0}),
    )

    for t in threads:
        t.join(timeout=5.0)

    alive = [t for t in threads if t.is_alive()]
    assert alive == [], f"{len(alive)} waiter thread(s) still alive"
    assert waiter_errors == [], f"Waiter errors: {waiter_errors}"
    assert acquired_count[0] == n_waiters, (
        f"Expected {n_waiters} acquires, got {acquired_count[0]}"
    )

    # Conservation: refunded total_needed, each waiter consumed per_waiter
    final_cap = _get_capacity(backend)
    assert final_cap == pytest.approx(0.0, abs=REFILL_TOLERANCE), (
        f"Expected ~0 after {n_waiters} waiters consumed {per_waiter} each, "
        f"got {final_cap}"
    )


# ---------------------------------------------------------------------------
# Test 8: Sync blocked acquire wakes on set_max_capacity
# ---------------------------------------------------------------------------


@hypothesis_settings(max_examples=30, deadline=None)
@given(
    acquire_amount=st.floats(
        min_value=2.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
)
def test_sync_blocked_acquire_wakes_on_set_max_capacity(acquire_amount):
    """Blocked acquire wakes when set_max_capacity increases refill rate.

    1. Create backend with per_seconds=1, limit=20 (rate 20/s), drain capacity
    2. Launch blocking acquire for acquire_amount (2-10, within max)
    3. set_max_capacity(200) => fires notify_all + rate jumps to 200/s
    4. Waiter wakes, get_capacity uses new rate on elapsed time => succeeds

    The key mechanic: after draining, the waiter's computed sleep is
    acquire_amount / 20 ≈ 0.1-0.5s. We call set_max_capacity after 0.15s,
    which fires notify_all (instant wake) AND boosts the rate so that
    elapsed_time * new_rate provides enough capacity.
    """
    # Rate = 20/s. acquire_amount (2-10) is within max_capacity (20).
    initial_limit = 20.0
    backend = _build_sync_backend(limit=initial_limit, per_seconds=1)

    # Drain all capacity
    backend.wait_for_capacity(frozen_usage({METRIC: initial_limit}))

    acquired = threading.Event()
    waiter_errors: list[BaseException] = []

    def blocking_acquire():
        try:
            backend.wait_for_capacity(
                frozen_usage({METRIC: acquire_amount}), timeout=5.0
            )
            acquired.set()
        except BaseException as exc:
            waiter_errors.append(exc)

    t = threading.Thread(target=blocking_acquire)
    t.start()

    # Give thread time to enter the wait
    time.sleep(0.15)

    # Increase max_capacity — fires notify_all and increases refill rate.
    # New rate = 200/s. After 0.15s elapsed since drain, capacity ≈ 0.15*200 = 30.
    backend.set_max_capacity(METRIC, 1, 200.0)

    t.join(timeout=5.0)
    assert not t.is_alive(), "Waiter thread did not exit"
    assert waiter_errors == [], f"Waiter errors: {waiter_errors}"
    assert acquired.is_set(), "Waiter should have acquired after set_max_capacity"


# ---------------------------------------------------------------------------
# Test 9: set_max_capacity below acquire amount raises ValueError
# ---------------------------------------------------------------------------


@hypothesis_settings(max_examples=30, deadline=None)
@given(
    acquire_amount=st.floats(
        min_value=10.0, max_value=50.0, allow_nan=False, allow_infinity=False
    ),
    new_max=st.floats(
        min_value=1.0, max_value=9.0, allow_nan=False, allow_infinity=False
    ),
)
def test_sync_set_max_below_acquire_raises_valueerror(acquire_amount, new_max):
    """Blocked acquire for amount X. set_max_capacity(Y) where Y < X.

    The waiter should raise ValueError (exceeds bucket max capacity), not TimeoutError.
    """
    # Ensure acquire_amount > new_max
    if acquire_amount <= new_max:
        return  # Skip degenerate cases

    backend = _build_sync_backend(limit=LIMIT)

    # Drain capacity so acquire will block
    backend.wait_for_capacity(frozen_usage({METRIC: LIMIT}))

    waiter_result: list[BaseException | None] = [None]

    def blocking_acquire():
        try:
            backend.wait_for_capacity(
                frozen_usage({METRIC: acquire_amount}), timeout=5.0
            )
        except BaseException as exc:
            waiter_result[0] = exc

    t = threading.Thread(target=blocking_acquire)
    t.start()

    # Give thread time to enter the wait
    time.sleep(0.15)

    # Lower max below the acquire amount
    backend.set_max_capacity(METRIC, SLOW_WINDOW, new_max)

    t.join(timeout=5.0)
    assert not t.is_alive(), "Waiter thread did not exit"

    exc = waiter_result[0]
    assert isinstance(exc, ValueError), (
        f"Expected ValueError, got {type(exc).__name__}: {exc}"
    )
    assert "exceeds bucket max capacity" in str(exc)


# ---------------------------------------------------------------------------
# Test 10: Sync concurrent refund cap-clipping
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@hypothesis_settings(max_examples=50, deadline=None)
@given(
    n_refunders=n_refunders_strategy,
    refund_amount=refund_clip_amount_strategy,
)
def test_sync_concurrent_refund_cap_clipping(n_refunders, refund_amount):
    """Multiple threads refunding simultaneously when capacity is near max.

    Verify capacity <= max_capacity even under contention (cap-clipping atomicity).
    """
    backend = _build_sync_backend(limit=LIMIT)
    # Consume a small amount so there's room to refund into
    small_consume = 10.0
    backend.consume_capacity(frozen_usage({METRIC: small_consume}))

    errors: list[BaseException] = []

    def do_refund():
        try:
            backend.refund_capacity(
                reserved_usage=frozen_usage({METRIC: refund_amount}),
                actual_usage=frozen_usage({METRIC: 0}),
            )
        except BaseException as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=n_refunders) as pool:
        futures = [pool.submit(do_refund) for _ in range(n_refunders)]
        for f in as_completed(futures, timeout=10):
            f.result()

    assert errors == [], f"Unexpected errors: {errors}"

    final_cap = _get_capacity(backend)
    max_cap = _get_max_capacity(backend)
    assert final_cap <= max_cap + 0.01, (
        f"Cap-clipping failed: capacity {final_cap} > max {max_cap}"
    )
