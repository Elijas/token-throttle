# ruff: noqa: FBT001
"""
Hypothesis stateful differential test: one operation sequence, six backends.

Every rule runs the same operation on memory/SQLite/Redis x async/sync under a
shared fake clock and asserts (1) identical outcome kind, exception type and
``DuplicateRefundError.reason``, (2) identical returned value, (3) identical
per-bucket ``current_capacity`` / ``effective_max_capacity`` from
``introspect()``. A failing sequence is a divergence; Hypothesis shrinks it.

Known, documented divergences are excluded by default so the machine keeps
searching for undocumented ones; ``test_known_divergence_*`` in
``test_known_divergences.py`` pin those explicitly.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
import warnings
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    consumes,
    initialize,
    precondition,
    rule,
)

from tests.differential._backends import (
    ALL_KINDS,
    ALL_MODES,
    Harnessed,
    build_all,
    make_config,
    require_redis,
)
from tests.differential._clock import FakeClock, patched_clock
from tests.differential._driver import (
    Driver,
    Outcome,
    capacities_close,
    describe,
    describe_capacities,
)

FAMILY = "diff-family"
REQUESTS_LIMIT = 10.0
TOKENS_MINUTE_LIMIT = 1000.0
TOKENS_HOUR_LIMIT = 5000.0
QUOTAS = (
    ("requests", 60, REQUESTS_LIMIT),
    ("tokens", 60, TOKENS_MINUTE_LIMIT),
    ("tokens", 3600, TOKENS_HOUR_LIMIT),
)
ALL_BUCKET_IDS = frozenset((metric, window) for metric, window, _ in QUOTAS)
# 400 days: far beyond any clock advance the machine can accumulate, so acquire
# markers never expire inside the general machine (expiry is pinned separately).
RESERVATION_LIFETIME = 400 * 86400.0

advance_seconds = st.sampled_from(
    [0.0, 0.001, 0.25, 1.0, 7.5, 30.0, 59.999, 60.0, 61.0, 3600.0, 86400.0]
)
request_amounts = st.sampled_from([0.0, 1.0, 2.0, 5.0, 9.0, 10.0, 10.5, 11.0])
token_amounts = st.sampled_from(
    [0.0, 1.0, 0.1, 333.3, 500.0, 999.0, 1000.0, 1000.1, 1500.0, 5000.0, 5001.0]
)
actual_ratio = st.sampled_from([0.0, 0.5, 1.0, 1.25, 2.0])
max_values = st.sampled_from([0.5, 1.0, 5.0, 10.0, 20.0, 500.0, 1000.0, 2000.0, 9999.0])
bucket_choice = st.sampled_from(sorted(ALL_BUCKET_IDS))


_GLOBAL_RULE_COUNTS: dict[str, int] = {}


class Reservation:
    def __init__(self, reservation_id: str, usage: dict[str, float]) -> None:
        self.reservation_id = reservation_id
        self.usage = usage

    def __repr__(self) -> str:
        return f"Reservation({self.reservation_id[:8]}, {self.usage})"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class DifferentialMachine(RuleBasedStateMachine):
    kinds: tuple[str, ...] = ALL_KINDS
    modes: tuple[str, ...] = ALL_MODES
    # Redis introspect() reports the original quota limit after
    # apply_configured_max_capacity (finding D2); exclude the rule where that
    # would mask everything else until the finding is resolved.
    include_apply_configured: bool = True

    reservations = Bundle("reservations")
    refunded = Bundle("refunded")

    def __init__(self) -> None:
        super().__init__()
        self._tmp = Path(tempfile.mkdtemp(prefix="tt-diff-"))
        self.loop = asyncio.new_event_loop()
        self.clock = FakeClock()
        self.driver = Driver(self.loop)
        self._patch = patched_clock(self.clock)
        self._patch.__enter__()
        self.targets: list[Harnessed] = []
        self.step = 0
        self.trace: list[str] = []
        self.rule_counts: dict[str, int] = {}

    @initialize()
    def build(self) -> None:
        cfg = make_config(FAMILY, QUOTAS)
        self.targets = build_all(
            cfg,
            self.clock,
            loop=self.loop,
            tmp_path=self._tmp,
            kinds=self.kinds,
            modes=self.modes,
        )
        self._compare_capacities("initial")

    def teardown(self) -> None:
        errors: list[BaseException] = []
        for target in self.targets:
            try:
                target.cleanup()
            except BaseException as exc:
                errors.append(exc)
        self._patch.__exit__(None, None, None)
        self.loop.close()
        shutil.rmtree(self._tmp, ignore_errors=True)
        if errors:
            raise RuntimeError(f"teardown errors: {errors!r}")

    # -- comparison ----------------------------------------------------------

    def _run(self, label: str, op) -> dict[str, Outcome]:
        self.step += 1
        rule_name = label.split(" ", 1)[0]
        self.rule_counts[rule_name] = self.rule_counts.get(rule_name, 0) + 1
        _GLOBAL_RULE_COUNTS[rule_name] = _GLOBAL_RULE_COUNTS.get(rule_name, 0) + 1
        self.trace.append(f"{self.step}: t={self.clock.now:.3f} {label}")
        outcomes = {target.name: op(target) for target in self.targets}
        kinds = {outcome[0] for outcome in outcomes.values()}
        types_ = {
            outcome[1] if outcome[0] == "exc" else "ok" for outcome in outcomes.values()
        }
        reasons = {
            outcome[2] if outcome[0] == "exc" else None for outcome in outcomes.values()
        }
        if len(kinds) != 1 or len(types_) != 1 or len(reasons) != 1:
            raise AssertionError(
                f"DIVERGENCE (outcome) at step {self.step} {label}\n"
                f"{describe(outcomes)}\ntrace:\n  " + "\n  ".join(self.trace)
            )
        values = {
            repr(outcome[1]) for outcome in outcomes.values() if outcome[0] == "ok"
        }
        if len(values) > 1:
            raise AssertionError(
                f"DIVERGENCE (return value) at step {self.step} {label}\n"
                f"{describe(outcomes)}\ntrace:\n  " + "\n  ".join(self.trace)
            )
        self._compare_capacities(label)
        return outcomes

    def _compare_capacities(self, label: str) -> None:
        snapshots = {t.name: self.driver.capacities(t) for t in self.targets}
        names = list(snapshots)
        reference = snapshots[names[0]]
        for name in names[1:]:
            if not capacities_close(reference, snapshots[name]):
                raise AssertionError(
                    f"DIVERGENCE (capacity) after step {self.step} {label}\n"
                    f"{describe_capacities(snapshots)}\ntrace:\n  "
                    + "\n  ".join(self.trace)
                )

    # -- rules ---------------------------------------------------------------

    @rule(seconds=advance_seconds)
    def advance(self, seconds: float) -> None:
        self.clock.advance(seconds)
        self.step += 1
        _GLOBAL_RULE_COUNTS["advance"] = _GLOBAL_RULE_COUNTS.get("advance", 0) + 1
        self.trace.append(f"{self.step}: advance {seconds} -> t={self.clock.now:.3f}")
        self._compare_capacities(f"advance {seconds}")

    @rule(
        target=reservations,
        requests=request_amounts,
        tokens=token_amounts,
        with_reservation=st.booleans(),
    )
    def try_acquire(self, requests: float, tokens: float, with_reservation: bool):
        usage = {"requests": requests, "tokens": tokens}
        rid = f"r-{uuid.uuid4().hex}" if with_reservation else None
        outcomes = self._run(
            f"try_acquire {usage} rid={rid and rid[:8]}",
            lambda t: self.driver.acquire(
                t,
                usage,
                timeout=0.0,
                reservation_id=rid,
                reservation_lifetime_seconds=RESERVATION_LIFETIME if rid else None,
            ),
        )
        first = next(iter(outcomes.values()))
        if rid is not None and first[0] == "ok":
            return Reservation(rid, usage)
        return None

    @rule(
        target=reservations,
        requests=request_amounts,
        tokens=token_amounts,
        with_reservation=st.booleans(),
    )
    def consume(self, requests: float, tokens: float, with_reservation: bool):
        usage = {"requests": requests, "tokens": tokens}
        rid = f"c-{uuid.uuid4().hex}" if with_reservation else None
        outcomes = self._run(
            f"consume {usage} rid={rid and rid[:8]}",
            lambda t: self.driver.consume(
                t,
                usage,
                reservation_id=rid,
                reservation_lifetime_seconds=RESERVATION_LIFETIME if rid else None,
            ),
        )
        first = next(iter(outcomes.values()))
        if rid is not None and first[0] == "ok":
            return Reservation(rid, usage)
        return None

    @rule(
        target=refunded,
        reservation=consumes(reservations),
        ratio_requests=actual_ratio,
        ratio_tokens=actual_ratio,
    )
    def refund(self, reservation, ratio_requests: float, ratio_tokens: float):
        if reservation is None:
            return None
        actual = {
            "requests": reservation.usage["requests"] * ratio_requests,
            "tokens": reservation.usage["tokens"] * ratio_tokens,
        }
        self._run(
            f"refund {reservation!r} actual={actual}",
            lambda t: self.driver.refund_for_buckets(
                t,
                reservation.usage,
                actual,
                bucket_ids=ALL_BUCKET_IDS,
                reservation_id=reservation.reservation_id,
                reservation_model_family=FAMILY,
                reservation_bucket_ids=ALL_BUCKET_IDS,
                reservation_reserved_usage=reservation.usage,
            ),
        )
        return reservation

    @rule(reservation=refunded)
    def refund_again(self, reservation) -> None:
        if reservation is None:
            return
        self._run(
            f"refund_again {reservation!r}",
            lambda t: self.driver.refund_for_buckets(
                t,
                reservation.usage,
                reservation.usage,
                bucket_ids=ALL_BUCKET_IDS,
                reservation_id=reservation.reservation_id,
                reservation_model_family=FAMILY,
                reservation_bucket_ids=ALL_BUCKET_IDS,
                reservation_reserved_usage=reservation.usage,
            ),
        )

    @rule(requests=request_amounts, tokens=token_amounts)
    def refund_unknown(self, requests: float, tokens: float) -> None:
        usage = {"requests": requests, "tokens": tokens}
        rid = f"u-{uuid.uuid4().hex}"
        self._run(
            f"refund_unknown {usage}",
            lambda t: self.driver.refund_for_buckets(
                t,
                usage,
                usage,
                bucket_ids=ALL_BUCKET_IDS,
                reservation_id=rid,
                reservation_model_family=FAMILY,
                reservation_bucket_ids=ALL_BUCKET_IDS,
                reservation_reserved_usage=usage,
            ),
        )

    @rule(requests=request_amounts, tokens=token_amounts, ratio=actual_ratio)
    def refund_without_reservation(
        self, requests: float, tokens: float, ratio: float
    ) -> None:
        reserved = {"requests": requests, "tokens": tokens}
        actual = {"requests": requests * ratio, "tokens": tokens * ratio}
        self._run(
            f"refund_plain reserved={reserved} actual={actual}",
            lambda t: self.driver.refund(t, reserved, actual),
        )

    @rule(bucket=bucket_choice, value=max_values)
    def set_max(self, bucket, value: float) -> None:
        metric, window = bucket
        self._run(
            f"set_max_capacity {metric}/{window}s={value}",
            lambda t: self.driver.set_max_capacity(t, metric, window, value),
        )

    @precondition(lambda self: self.include_apply_configured)
    @rule(bucket=bucket_choice, value=max_values)
    def apply_configured(self, bucket, value: float) -> None:
        metric, window = bucket
        self._run(
            f"apply_configured_max_capacity {metric}/{window}s={value}",
            lambda t: self.driver.apply_configured_max_capacity(
                t, metric, window, value
            ),
        )


_SETTINGS = settings(
    max_examples=int(__import__("os").environ.get("TT_DIFF_EXAMPLES", "25")),
    stateful_step_count=int(__import__("os").environ.get("TT_DIFF_STEPS", "40")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    print_blob=True,
)


class _AllSix(DifferentialMachine):
    include_apply_configured = False


class _MemoryVsSqlite(DifferentialMachine):
    kinds = ("memory", "sqlite")


def _report_rule_counts(label: str) -> None:
    total = sum(_GLOBAL_RULE_COUNTS.values())
    cells = ", ".join(f"{k}={v}" for k, v in sorted(_GLOBAL_RULE_COUNTS.items()))
    print(f"\n[differential {label}] {total} rule executions: {cells}")
    _GLOBAL_RULE_COUNTS.clear()


def test_differential_all_six_backends() -> None:
    require_redis()
    _GLOBAL_RULE_COUNTS.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _AllSix.TestCase.settings = _SETTINGS
        try:
            _AllSix.TestCase().runTest()
        finally:
            _report_rule_counts("all-six")


def test_differential_memory_vs_sqlite() -> None:
    _GLOBAL_RULE_COUNTS.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _MemoryVsSqlite.TestCase.settings = _SETTINGS
        try:
            _MemoryVsSqlite.TestCase().runTest()
        finally:
            _report_rule_counts("memory-vs-sqlite")
