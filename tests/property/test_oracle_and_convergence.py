"""
Property tests closing three informational findings from the 2026-04 audit:

INF-07  Independent oracle for token-bucket capacity math.
INF-09  Cross-worker convergence (multiple coroutines sharing one backend).
INF-10  Redis backend Hypothesis property tests (pure-math layer).
"""

import asyncio
import warnings
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import assume, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)

from token_throttle._capacity import calculate_capacity
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage
from token_throttle._limiter_backends._memory._backend import MemoryBackend
from token_throttle._limiter_backends._memory._bucket import MemoryBucket
from token_throttle._limiter_backends._memory._sync_backend import SyncMemoryBackend
from token_throttle._limiter_backends._redis._bucket import RedisBucket

# ---------------------------------------------------------------------------
# Oracle — INF-07
# ---------------------------------------------------------------------------


def oracle_calculate_capacity(
    last_checked: float | None,
    outdated_capacity: float | None,
    current_time: float,
    max_capacity: float,
    rate_per_sec: float,
) -> tuple[float, bool]:
    """Obviously-correct reference implementation of token-bucket capacity math.

    Returns (amount, is_fresh_start).

    Deliberately minimal: no validation, no warnings, no edge-case handling
    beyond the two core rules (fresh-start on None, refill + clamp otherwise).
    A reviewer should be able to verify this in under a minute.
    """
    if last_checked is None or outdated_capacity is None:
        return max_capacity, True
    elapsed = max(0.0, current_time - last_checked)
    amount = min(max_capacity, outdated_capacity + elapsed * rate_per_sec)
    return amount, False


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

limits = st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False)
per_seconds = st.integers(min_value=1, max_value=86400)
capacities = st.floats(
    min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
)
negative_capacities = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
times = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)


# ===========================================================================
# INF-07: Oracle vs. production calculate_capacity
# ===========================================================================


class TestOracleAgreesWithProduction:
    """Compare the independent oracle against the shared calculate_capacity().

    The oracle is a 10-line pure-Python function with no imports. If both
    agree on random inputs, confidence that the production code is correct
    is much higher than a shadow model that mirrors the same logic.
    """

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        outdated_capacity=capacities,
        last_checked=times,
        time_delta=times,
    )
    def test_agreement_on_normal_inputs(
        self, limit, per_seconds_val, outdated_capacity, last_checked, time_delta
    ):
        rate = limit / per_seconds_val
        current_time = last_checked + time_delta

        result = calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=current_time,
            max_capacity=limit,
            rate_per_sec=rate,
            bucket_id="oracle-test",
        )
        oracle_amount, oracle_fresh = oracle_calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=current_time,
            max_capacity=limit,
            rate_per_sec=rate,
        )

        assert result.amount == pytest.approx(oracle_amount, abs=1e-9), (
            f"Amount mismatch: production={result.amount}, oracle={oracle_amount}"
        )
        assert result.is_fresh_start == oracle_fresh

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        outdated_capacity=negative_capacities,
        last_checked=times,
        time_delta=times,
    )
    def test_agreement_on_negative_capacity(
        self, limit, per_seconds_val, outdated_capacity, last_checked, time_delta
    ):
        rate = limit / per_seconds_val
        current_time = last_checked + time_delta

        result = calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=current_time,
            max_capacity=limit,
            rate_per_sec=rate,
            bucket_id="oracle-test",
        )
        oracle_amount, oracle_fresh = oracle_calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=current_time,
            max_capacity=limit,
            rate_per_sec=rate,
        )

        assert result.amount == pytest.approx(oracle_amount, abs=1e-9), (
            f"Amount mismatch (negative cap): production={result.amount}, oracle={oracle_amount}"
        )
        assert result.is_fresh_start == oracle_fresh

    @given(limit=limits, per_seconds_val=per_seconds, current_time=times)
    def test_fresh_start_none_last_checked(self, limit, per_seconds_val, current_time):
        rate = limit / per_seconds_val

        result = calculate_capacity(
            last_checked=None,
            outdated_capacity=50.0,
            current_time=current_time,
            max_capacity=limit,
            rate_per_sec=rate,
            bucket_id="oracle-test",
        )
        oracle_amount, oracle_fresh = oracle_calculate_capacity(
            last_checked=None,
            outdated_capacity=50.0,
            current_time=current_time,
            max_capacity=limit,
            rate_per_sec=rate,
        )

        assert result.amount == oracle_amount
        assert result.is_fresh_start is True
        assert oracle_fresh is True

    @given(limit=limits, per_seconds_val=per_seconds, current_time=times)
    def test_fresh_start_none_outdated_capacity(
        self, limit, per_seconds_val, current_time
    ):
        rate = limit / per_seconds_val

        result = calculate_capacity(
            last_checked=100.0,
            outdated_capacity=None,
            current_time=current_time,
            max_capacity=limit,
            rate_per_sec=rate,
            bucket_id="oracle-test",
        )
        oracle_amount, oracle_fresh = oracle_calculate_capacity(
            last_checked=100.0,
            outdated_capacity=None,
            current_time=current_time,
            max_capacity=limit,
            rate_per_sec=rate,
        )

        assert result.amount == oracle_amount
        assert result.is_fresh_start is True
        assert oracle_fresh is True

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        outdated_capacity=capacities,
        last_checked=times,
        backward_delta=st.floats(
            min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_backward_clock_clamped_to_zero(
        self, limit, per_seconds_val, outdated_capacity, last_checked, backward_delta
    ):
        """When current_time < last_checked (clock went backward),
        production clamps elapsed to 0. Oracle does the same via max(0, ...).
        """
        rate = limit / per_seconds_val
        current_time = last_checked - backward_delta
        assume(current_time >= 0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = calculate_capacity(
                last_checked=last_checked,
                outdated_capacity=outdated_capacity,
                current_time=current_time,
                max_capacity=limit,
                rate_per_sec=rate,
                bucket_id="oracle-test",
            )
        oracle_amount, oracle_fresh = oracle_calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=current_time,
            max_capacity=limit,
            rate_per_sec=rate,
        )

        assert result.amount == pytest.approx(oracle_amount, abs=1e-9), (
            f"Backward-clock mismatch: production={result.amount}, oracle={oracle_amount}"
        )
        assert result.is_fresh_start == oracle_fresh


class TestOracleVsMemoryBucket:
    """Cross-check the oracle against MemoryBucket.get_capacity() end-to-end."""

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        outdated_capacity=negative_capacities,
        last_checked=times,
        time_delta=times,
    )
    def test_memory_bucket_matches_oracle(
        self, limit, per_seconds_val, outdated_capacity, last_checked, time_delta
    ):
        bucket = MemoryBucket(
            metric="tokens",
            per_seconds=per_seconds_val,
            limit=limit,
            model_family="oracle-test",
        )
        bucket.capacity = outdated_capacity
        bucket.last_checked = last_checked
        current_time = last_checked + time_delta

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = bucket.get_capacity(current_time)

        oracle_amount, oracle_fresh = oracle_calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=current_time,
            max_capacity=limit,
            rate_per_sec=limit / per_seconds_val,
        )

        assert result.amount == pytest.approx(oracle_amount, abs=1e-9)
        assert result.is_fresh_start == oracle_fresh


# ===========================================================================
# INF-09: Cross-worker convergence
# ===========================================================================

CONVERGENCE_INITIAL_TIME = 1_000_000.0
CONVERGENCE_METRIC = "tokens"
CONVERGENCE_WINDOW = 60
CONVERGENCE_LIMIT = 1000.0

worker_consume_amounts = st.floats(
    min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False
)
worker_refund_fractions = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
worker_time_deltas = st.floats(
    min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False
)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class CrossWorkerConvergenceMachine(RuleBasedStateMachine):
    """Multiple workers sharing a single SyncMemoryBackend.

    Each worker does consume/refund cycles concurrently (sequentially simulated
    via Hypothesis state machine rules). After all operations, we verify that
    total capacity is consistent: no tokens leaked or created from nothing.

    Conservation invariant:
        capacity + total_consumed - total_refunded <= initial_capacity + cumulative_refill
    """

    def __init__(self):
        super().__init__()
        self.backend: SyncMemoryBackend | None = None
        self.bucket: MemoryBucket | None = None
        self.current_time: float = CONVERGENCE_INITIAL_TIME

        self.total_consumed: float = 0.0
        self.total_refunded: float = 0.0
        self.cumulative_max_refill: float = 0.0
        self.rate: float = CONVERGENCE_LIMIT / CONVERGENCE_WINDOW

        self.outstanding_per_worker: dict[int, list[float]] = {}

        self._time_patcher = patch(
            "token_throttle._limiter_backends._memory._sync_backend.time"
        )
        self._mock_time = self._time_patcher.start()
        self._mock_time.time.side_effect = lambda: self.current_time
        self._mock_time.monotonic.side_effect = lambda: self.current_time

    @initialize()
    def init_backend(self):
        config = PerModelConfig(
            model_family="convergence-test",
            quotas=UsageQuotas(
                [
                    Quota(
                        metric=CONVERGENCE_METRIC,
                        limit=CONVERGENCE_LIMIT,
                        per_seconds=CONVERGENCE_WINDOW,
                    )
                ]
            ),
        )
        bucket = MemoryBucket(
            metric=CONVERGENCE_METRIC,
            per_seconds=CONVERGENCE_WINDOW,
            limit=CONVERGENCE_LIMIT,
            model_family="convergence-test",
        )
        self.backend = SyncMemoryBackend(buckets=[bucket], limit_config=config)
        self.bucket = bucket
        self.current_time = CONVERGENCE_INITIAL_TIME
        self.total_consumed = 0.0
        self.total_refunded = 0.0
        self.cumulative_max_refill = 0.0
        self.rate = CONVERGENCE_LIMIT / CONVERGENCE_WINDOW
        self.outstanding_per_worker = {i: [] for i in range(4)}

    @rule(delta=worker_time_deltas)
    def advance_time(self, delta):
        self.cumulative_max_refill += delta * self.rate
        self.current_time += delta

    @rule(
        worker_id=st.sampled_from([0, 1, 2, 3]),
        amount=worker_consume_amounts,
    )
    def worker_consume(self, worker_id, amount):
        before = self.bucket.get_capacity(self.current_time).amount
        self.backend.consume_capacity(frozen_usage({CONVERGENCE_METRIC: amount}))
        after = self.bucket.get_capacity(self.current_time).amount
        self.total_consumed += before - after

    @rule(
        worker_id=st.sampled_from([0, 1, 2, 3]),
        amount=worker_consume_amounts,
    )
    def worker_acquire(self, worker_id, amount):
        if amount > CONVERGENCE_LIMIT:
            return
        cap = self.bucket.get_capacity(self.current_time).amount
        if amount <= cap:
            self.backend.wait_for_capacity(
                frozen_usage({CONVERGENCE_METRIC: amount}), timeout=0.0
            )
            self.total_consumed += amount
            self.outstanding_per_worker[worker_id].append(amount)
        else:
            with pytest.raises(TimeoutError):
                self.backend.wait_for_capacity(
                    frozen_usage({CONVERGENCE_METRIC: amount}), timeout=0.0
                )

    @rule(
        worker_id=st.sampled_from([0, 1, 2, 3]),
        actual_fraction=worker_refund_fractions,
    )
    def worker_refund(self, worker_id, actual_fraction):
        reservations = self.outstanding_per_worker[worker_id]
        if not reservations:
            return
        reserved = reservations.pop()
        actual = reserved * actual_fraction
        before = self.bucket.get_capacity(self.current_time).amount
        self.backend.refund_capacity(
            frozen_usage({CONVERGENCE_METRIC: reserved}),
            frozen_usage({CONVERGENCE_METRIC: actual}),
        )
        after = self.bucket.get_capacity(self.current_time).amount
        self.total_refunded += after - before

    @invariant()
    def capacity_never_exceeds_max(self):
        if self.bucket is None:
            return
        cap = self.bucket.get_capacity(self.current_time).amount
        assert cap <= CONVERGENCE_LIMIT + 1e-9, (
            f"Capacity {cap} exceeded max {CONVERGENCE_LIMIT}"
        )

    @invariant()
    def no_tokens_created_from_nothing(self):
        """Conservation: consumed minus refunded cannot exceed what was available."""
        if self.bucket is None:
            return
        cap = self.bucket.get_capacity(self.current_time).amount
        balance = cap + self.total_consumed - self.total_refunded
        budget = CONVERGENCE_LIMIT + self.cumulative_max_refill
        assert balance <= budget + 1e-6, (
            f"Conservation violation: capacity={cap}, consumed={self.total_consumed}, "
            f"refunded={self.total_refunded}, balance={balance}, budget={budget}"
        )

    @invariant()
    def oracle_cross_check(self):
        """Cross-check current capacity against the independent oracle."""
        if self.bucket is None:
            return
        result = self.bucket.get_capacity(self.current_time)
        oracle_amount, _ = oracle_calculate_capacity(
            last_checked=self.bucket.last_checked,
            outdated_capacity=self.bucket.capacity,
            current_time=self.current_time,
            max_capacity=self.bucket.max_capacity,
            rate_per_sec=self.rate,
        )
        assert result.amount == pytest.approx(oracle_amount, abs=1e-9), (
            f"Oracle mismatch during convergence: actual={result.amount}, oracle={oracle_amount}"
        )

    def teardown(self):
        self._time_patcher.stop()


StatefulCrossWorkerConvergence = CrossWorkerConvergenceMachine.TestCase
StatefulCrossWorkerConvergence.settings = hypothesis_settings(
    max_examples=300, stateful_step_count=80, deadline=None
)


# ===========================================================================
# INF-09 (parametric): Async multi-worker drain-and-refund accounting
# ===========================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@hypothesis_settings(max_examples=100, deadline=None)
@given(
    n_workers=st.integers(min_value=2, max_value=6),
    per_worker_amount=st.floats(
        min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False
    ),
    refund_fraction=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
def test_async_multi_worker_accounting(n_workers, per_worker_amount, refund_fraction):
    """N async workers each consume then partially refund on a shared backend.

    After all workers finish, verify conservation holds.
    """
    clock = [CONVERGENCE_INITIAL_TIME]

    with (
        patch("token_throttle._limiter_backends._memory._backend.time") as mock_time,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        mock_time.time.side_effect = lambda: clock[0]
        mock_time.monotonic.side_effect = lambda: clock[0]

        config = PerModelConfig(
            model_family="convergence-test",
            quotas=UsageQuotas(
                [
                    Quota(
                        metric=CONVERGENCE_METRIC,
                        limit=CONVERGENCE_LIMIT,
                        per_seconds=CONVERGENCE_WINDOW,
                    )
                ]
            ),
        )
        bucket = MemoryBucket(
            metric=CONVERGENCE_METRIC,
            per_seconds=CONVERGENCE_WINDOW,
            limit=CONVERGENCE_LIMIT,
            model_family="convergence-test",
        )
        backend = MemoryBackend(buckets=[bucket], limit_config=config)

        total_consumed = per_worker_amount * n_workers
        if total_consumed > CONVERGENCE_LIMIT:
            return

        async def worker(worker_id: int):
            await backend.consume_capacity(
                frozen_usage({CONVERGENCE_METRIC: per_worker_amount})
            )
            refund_amount = per_worker_amount * (1.0 - refund_fraction)
            if refund_amount > 0:
                await backend.refund_capacity(
                    frozen_usage({CONVERGENCE_METRIC: per_worker_amount}),
                    frozen_usage(
                        {CONVERGENCE_METRIC: per_worker_amount * refund_fraction}
                    ),
                )
            return refund_amount

        async def run():
            tasks = [worker(i) for i in range(n_workers)]
            refunds = await asyncio.gather(*tasks)
            return sum(refunds)

        loop = asyncio.new_event_loop()
        try:
            total_refunded = loop.run_until_complete(run())
        finally:
            loop.close()

        final_cap = bucket.get_capacity(clock[0]).amount
        net_consumed = total_consumed - total_refunded
        expected_final = min(CONVERGENCE_LIMIT, CONVERGENCE_LIMIT - net_consumed)

        assert final_cap == pytest.approx(expected_final, abs=1e-6), (
            f"Accounting error: final={final_cap}, expected={expected_final}, "
            f"consumed={total_consumed}, refunded={total_refunded}"
        )
        assert final_cap <= CONVERGENCE_LIMIT + 1e-9


# ===========================================================================
# INF-10: Redis bucket pure-math property tests
# ===========================================================================


def make_redis_bucket(limit: float = 100.0, per_seconds_val: int = 60) -> RedisBucket:
    quota = Quota(metric="tokens", limit=limit, per_seconds=per_seconds_val)
    config = PerModelConfig(
        model_family="redis-oracle-test", quotas=UsageQuotas([quota])
    )
    return RedisBucket(quota=quota, limit_config=config, redis_client=AsyncMock())


class TestRedisBucketOracleAgreement:
    """Property tests for RedisBucket.calculate_capacity against the oracle.

    RedisBucket.calculate_capacity delegates to the shared calculate_capacity()
    function, but wires in its own max_capacity and rate_per_sec. These tests
    verify that wiring is correct for arbitrary quota configurations.
    """

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        outdated_capacity=capacities,
        last_checked=times,
        time_delta=times,
    )
    def test_redis_bucket_matches_oracle(
        self, limit, per_seconds_val, outdated_capacity, last_checked, time_delta
    ):
        bucket = make_redis_bucket(limit=limit, per_seconds_val=per_seconds_val)
        current_time = last_checked + time_delta

        result = bucket.calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=current_time,
        )
        oracle_amount, oracle_fresh = oracle_calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=current_time,
            max_capacity=bucket.max_capacity,
            rate_per_sec=bucket._rate_per_sec,
        )

        assert result.amount == pytest.approx(oracle_amount, abs=1e-9), (
            f"Redis bucket oracle mismatch: production={result.amount}, oracle={oracle_amount}"
        )
        assert result.is_fresh_start == oracle_fresh

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        outdated_capacity=negative_capacities,
        last_checked=times,
        time_delta=times,
    )
    def test_redis_bucket_negative_capacity_matches_oracle(
        self, limit, per_seconds_val, outdated_capacity, last_checked, time_delta
    ):
        bucket = make_redis_bucket(limit=limit, per_seconds_val=per_seconds_val)
        current_time = last_checked + time_delta

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = bucket.calculate_capacity(
                last_checked=last_checked,
                outdated_capacity=outdated_capacity,
                current_time=current_time,
            )
        oracle_amount, oracle_fresh = oracle_calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=current_time,
            max_capacity=bucket.max_capacity,
            rate_per_sec=bucket._rate_per_sec,
        )

        assert result.amount == pytest.approx(oracle_amount, abs=1e-9)
        assert result.is_fresh_start == oracle_fresh

    @given(limit=limits, per_seconds_val=per_seconds, current_time=times)
    def test_redis_bucket_fresh_start_on_none(
        self, limit, per_seconds_val, current_time
    ):
        bucket = make_redis_bucket(limit=limit, per_seconds_val=per_seconds_val)

        result = bucket.calculate_capacity(
            last_checked=None, outdated_capacity=None, current_time=current_time
        )
        oracle_amount, oracle_fresh = oracle_calculate_capacity(
            last_checked=None,
            outdated_capacity=None,
            current_time=current_time,
            max_capacity=bucket.max_capacity,
            rate_per_sec=bucket._rate_per_sec,
        )

        assert result.amount == oracle_amount
        assert result.is_fresh_start is True
        assert oracle_fresh is True

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        outdated_capacity=capacities,
        last_checked=times,
        backward_delta=st.floats(
            min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_redis_bucket_backward_clock(
        self, limit, per_seconds_val, outdated_capacity, last_checked, backward_delta
    ):
        bucket = make_redis_bucket(limit=limit, per_seconds_val=per_seconds_val)
        current_time = last_checked - backward_delta
        assume(current_time >= 0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = bucket.calculate_capacity(
                last_checked=last_checked,
                outdated_capacity=outdated_capacity,
                current_time=current_time,
            )
        oracle_amount, oracle_fresh = oracle_calculate_capacity(
            last_checked=last_checked,
            outdated_capacity=outdated_capacity,
            current_time=current_time,
            max_capacity=bucket.max_capacity,
            rate_per_sec=bucket._rate_per_sec,
        )

        assert result.amount == pytest.approx(oracle_amount, abs=1e-9)
        assert result.is_fresh_start == oracle_fresh


class TestRedisBucketInvariants:
    """Additional property invariants for RedisBucket math not covered by
    test_bucket_invariants.py — these target the oracle comparison rather
    than self-referential identity checks.
    """

    @given(data=st.data())
    def test_refill_monotonicity_via_oracle(self, data):
        """More elapsed time => equal or higher capacity (oracle-verified)."""
        limit = data.draw(limits, label="limit")
        per_seconds_val = data.draw(per_seconds, label="per_seconds")
        outdated_capacity = data.draw(negative_capacities, label="outdated_capacity")
        last_checked = data.draw(times, label="last_checked")
        delta1 = data.draw(times, label="delta1")
        extra = data.draw(times, label="extra_delta")

        bucket = make_redis_bucket(limit=limit, per_seconds_val=per_seconds_val)
        t1 = last_checked + delta1
        t2 = t1 + extra

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r1 = bucket.calculate_capacity(last_checked, outdated_capacity, t1)
            r2 = bucket.calculate_capacity(last_checked, outdated_capacity, t2)

        o1, _ = oracle_calculate_capacity(
            last_checked,
            outdated_capacity,
            t1,
            bucket.max_capacity,
            bucket._rate_per_sec,
        )
        o2, _ = oracle_calculate_capacity(
            last_checked,
            outdated_capacity,
            t2,
            bucket.max_capacity,
            bucket._rate_per_sec,
        )

        assert r1.amount == pytest.approx(o1, abs=1e-9)
        assert r2.amount == pytest.approx(o2, abs=1e-9)
        assert r2.amount >= r1.amount - 1e-9

    @given(
        limit=limits,
        per_seconds_val=per_seconds,
        outdated_capacity=capacities,
        last_checked=times,
        time_delta=times,
    )
    def test_redis_wiring_rate_matches_quota(
        self, limit, per_seconds_val, outdated_capacity, last_checked, time_delta
    ):
        """Verify RedisBucket._rate_per_sec == limit / per_seconds (quota wiring)."""
        bucket = make_redis_bucket(limit=limit, per_seconds_val=per_seconds_val)
        expected_rate = limit / per_seconds_val
        assert bucket._rate_per_sec == pytest.approx(expected_rate, rel=1e-12)
        assert bucket.max_capacity == pytest.approx(limit, rel=1e-12)
