from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping  # noqa: TC003
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import Field

from token_throttle._dto import StrictDTO
from token_throttle._interfaces._models import (  # noqa: TC001
    BucketId,
    CapacityReservation,
    FrozenUsage,
    ReservationAuthoritySnapshot,
)

DiagnosticBackendType = Literal["memory", "redis", "custom", "unknown"]
DiagnosticConsistency = Literal["eventually_consistent"]
DiagnosticIssueSeverity = Literal["info", "warning", "error"]
DiagnosticBucketStatus = Literal[
    "ok",
    "fresh_start",
    "missing",
    "partial_missing",
    "corrupt",
    "unavailable",
]
DiagnosticOverrideSource = Literal["none", "limiter", "backend", "both"]
DiagnosticReservationState = Literal[
    "pending_acquire",
    "in_flight",
    "delivery_cleanup",
]
DiagnosticWaitState = Literal["waiting_for_capacity", "pending_backend"]

DIAGNOSE_RESERVATION_ID_SAMPLE_LIMIT = 100
DIAGNOSE_WAITER_SAMPLE_LIMIT = 50


class DiagnosticIssue(StrictDTO):
    severity: DiagnosticIssueSeverity = Field(
        description="Operational severity of the diagnostic issue."
    )
    component: str = Field(
        description=(
            "Component that produced the issue, for example 'limiter', "
            "'memory_backend', 'redis_backend', or 'custom_backend'."
        )
    )
    message: str = Field(
        description="Human-readable problem statement safe for logs and health output."
    )
    model_family: str | None = Field(
        default=None,
        description="Model family involved, when the issue is family-specific.",
    )
    metric: str | None = Field(
        default=None,
        description="Metric involved, when the issue is bucket-specific.",
    )
    per_seconds: int | None = Field(
        default=None,
        description="Quota window involved, when the issue is bucket-specific.",
    )


class DiagnosticBucketKey(StrictDTO):
    model_family: str = Field(description="Resolved rate-limit model family.")
    metric: str = Field(description="Usage metric, for example 'tokens' or 'requests'.")
    per_seconds: int = Field(description="Quota window in seconds.")


class BucketDiagnostic(StrictDTO):
    model_family: str = Field(description="Resolved rate-limit model family.")
    metric: str = Field(description="Usage metric.")
    per_seconds: int = Field(description="Quota window in seconds.")
    backend_type: DiagnosticBackendType = Field(
        description="Backend that supplied this bucket state."
    )
    current_capacity: float | None = Field(
        description=(
            "Capacity available at as_of_monotonic after applying token-bucket "
            "refill. Can be negative after record_usage or negative refund. "
            "None means the backend could not safely compute a value."
        )
    )
    configured_limit: float = Field(
        description="Static quota.limit from the current PerModelConfig snapshot."
    )
    runtime_override: float | None = Field(
        default=None,
        description=(
            "Active set_max_capacity override for this bucket, if one is active."
        ),
    )
    override_source: DiagnosticOverrideSource = Field(
        default="none",
        description=(
            "'limiter' means only local limiter bookkeeping observed the override; "
            "'backend' means backend state observed it; 'both' means both agree."
        ),
    )
    effective_max_capacity: float = Field(
        description="runtime_override when present, otherwise configured_limit."
    )
    configured_to_effective_gap: float = Field(
        description="effective_max_capacity - configured_limit."
    )
    refill_rate_per_second: float = Field(
        description="effective_max_capacity / per_seconds."
    )
    status: DiagnosticBucketStatus = Field(
        description="Whether current_capacity is authoritative or degraded."
    )
    as_of_monotonic: float = Field(
        description="Local monotonic timestamp at which this bucket snapshot was read."
    )


class RuntimeOverrideDiagnostic(StrictDTO):
    model_family: str = Field(description="Model family with an active override.")
    metric: str = Field(description="Overridden metric.")
    per_seconds: int = Field(description="Overridden quota window.")
    configured_limit: float = Field(description="Static configured limit.")
    override_capacity: float = Field(description="Runtime override capacity.")
    effective_max_capacity: float = Field(description="Effective max capacity in use.")
    configured_to_override_gap: float = Field(
        description="override_capacity - configured_limit."
    )
    source: DiagnosticOverrideSource = Field(
        description="Where the active override was observed."
    )


class ReservationGroupDiagnostic(StrictDTO):
    model_family: str = Field(description="Reservation model family.")
    metric: str | None = Field(
        default=None,
        description=(
            "Metric represented by this group. None is used for unlimited "
            "reservations or reservations with no metric-specific usage."
        ),
    )
    state: DiagnosticReservationState | Literal["all"] = Field(
        description="Reservation state represented by this group."
    )
    outstanding_count: int = Field(description="Reservations in this group.")
    total_reserved_usage: float | None = Field(
        default=None,
        description=(
            "Sum of reserved usage for this metric across the group. None when "
            "metric is None or usage is unavailable."
        ),
    )
    oldest_age_seconds: float | None = Field(
        default=None,
        description=(
            "Age of the oldest reservation in this group at reservations.as_of_monotonic."
        ),
    )
    oldest_reservation_id: str | None = Field(
        default=None,
        description="Oldest reservation id in this group, if known.",
    )
    reservation_ids: tuple[str, ...] = Field(
        description="Oldest-first bounded sample of reservation ids."
    )
    reservation_id_sample_limit: int = Field(
        description="Maximum ids included in reservation_ids for this group."
    )
    reservation_ids_truncated: bool = Field(
        description="True when more ids exist than the bounded sample includes."
    )


class InFlightReservationsDiagnostic(StrictDTO):
    as_of_monotonic: float = Field(
        description="Local monotonic timestamp for this reservation snapshot."
    )
    total_count: int = Field(
        description="in_flight + pending_acquire + delivery_cleanup reservation count."
    )
    in_flight_count: int = Field(description="Reservations delivered to callers.")
    pending_acquire_count: int = Field(
        description="Reservations currently inside acquire before delivery."
    )
    delivery_cleanup_count: int = Field(
        description=(
            "Consumed reservations being delivered or cleaned up after cancellation."
        )
    )
    oldest_age_seconds: float | None = Field(
        default=None,
        description="Oldest age across all tracked reservations, if known.",
    )
    groups: tuple[ReservationGroupDiagnostic, ...] = Field(
        description="Per-family/per-metric reservation groups."
    )


class WaitBucketDiagnostic(StrictDTO):
    model_family: str = Field(description="Model family whose acquire is blocked.")
    metric: str = Field(description="Blocked metric.")
    per_seconds: int = Field(description="Blocked quota window.")
    current_capacity: float | None = Field(
        description="Capacity observed for the blocked bucket, if available."
    )
    required_capacity: float = Field(description="Usage requested for this metric.")
    deficit: float = Field(description="max(0, required_capacity - current_capacity).")
    refill_rate_per_second: float | None = Field(
        default=None,
        description="Refill rate used to estimate expected_refill_seconds.",
    )
    expected_refill_seconds: float | None = Field(
        default=None,
        description=(
            "Estimated seconds until this bucket can satisfy the request, or "
            "None when capacity/rate is unavailable."
        ),
    )
    effective_max_capacity: float | None = Field(
        default=None,
        description="Effective max capacity for the blocked bucket, if available.",
    )


class WaiterDiagnostic(StrictDTO):
    reservation_id: str | None = Field(
        default=None,
        description="Pending reservation id for the waiting acquire, if known.",
    )
    model_family: str = Field(description="Resolved model family for the waiter.")
    model: str | None = Field(
        default=None, description="Request model alias, if known."
    )
    request_id: str | None = Field(
        default=None,
        description=(
            "Caller-supplied request_id, if acquire_capacity_for_request provided one."
        ),
    )
    state: DiagnosticWaitState = Field(description="Where the acquire is waiting.")
    usage: FrozenUsage = Field(description="Usage the waiter is trying to reserve.")
    wait_started_monotonic: float | None = Field(
        default=None,
        description="Local monotonic timestamp when capacity waiting began.",
    )
    wait_age_seconds: float | None = Field(
        default=None,
        description="Current wait duration at waits.as_of_monotonic.",
    )
    timeout_remaining_seconds: float | None = Field(
        default=None,
        description="Caller timeout budget remaining, if known and finite.",
    )
    primary_bottleneck: WaitBucketDiagnostic | None = Field(
        default=None,
        description="Deficient bucket with the largest expected_refill_seconds.",
    )
    blocked_buckets: tuple[WaitBucketDiagnostic, ...] = Field(
        description="All deficient buckets known for this waiter."
    )
    as_of_monotonic: float = Field(
        description="Local monotonic timestamp for this waiter observation."
    )


class CurrentWaitsDiagnostic(StrictDTO):
    as_of_monotonic: float = Field(
        description="Local monotonic timestamp for this wait snapshot."
    )
    total_waiter_count: int = Field(description="Number of currently waiting acquires.")
    waiters: tuple[WaiterDiagnostic, ...] = Field(
        description="Bounded sample of current waiters."
    )
    waiter_sample_limit: int = Field(description="Maximum waiters included.")
    waiters_truncated: bool = Field(
        description="True when additional waiters exist outside the sample."
    )


class MemoryBackendHealthDiagnostic(StrictDTO):
    model_family_count: int = Field(
        description="Known model families using memory backends."
    )
    bucket_count: int = Field(description="Total known in-memory buckets.")
    acquired_reservation_id_count: int = Field(
        description="Total backend-local acquired reservation ids."
    )
    refund_dedup_count: int = Field(
        description="Total backend-local refunded reservation dedup ids."
    )
    refund_dedup_cap: int = Field(description="Configured dedup-id cap.")


class RedisBackendHealthDiagnostic(StrictDTO):
    model_family_count: int = Field(
        description="Known model families using Redis backends."
    )
    bucket_count: int = Field(description="Known Redis buckets for these families.")
    connection_pool_class: str | None = Field(
        default=None,
        description="redis-py connection pool class name, if observable.",
    )
    connection_pool_max_connections: int | None = Field(
        default=None,
        description="connection_pool.max_connections, if exposed by redis-py.",
    )
    connection_pool_in_use_connections: int | None = Field(
        default=None,
        description="Best-effort in-use connection count, if observable.",
    )
    connection_pool_available_connections: int | None = Field(
        default=None,
        description="Best-effort idle/available connection count, if observable.",
    )
    pool_counts_observed_with_private_attrs: bool = Field(
        description=(
            "True when in-use/available counts came from redis-py private attrs."
        )
    )
    local_marker_count_estimate: int = Field(
        description="Local limiter estimate of outstanding Redis acquired markers."
    )
    local_refund_dedup_count_estimate: int = Field(
        description="Local limiter estimate of committed refund dedup entries."
    )


class CustomBackendHealthDiagnostic(StrictDTO):
    introspection_supported: bool = Field(
        description="Whether the custom backend supplied backend introspection."
    )
    backend_class_names: tuple[str, ...] = Field(
        description="Concrete backend class names observed by the limiter."
    )


class BackendHealthDiagnostic(StrictDTO):
    backend_type: DiagnosticBackendType = Field(description="Primary backend kind.")
    memory: MemoryBackendHealthDiagnostic | None = Field(default=None)
    redis: RedisBackendHealthDiagnostic | None = Field(default=None)
    custom: CustomBackendHealthDiagnostic | None = Field(default=None)


class RateLimiterDiagnostic(StrictDTO):
    schema_version: Literal[1] = Field(
        default=1,
        description="Diagnostic schema version. This document defines version 1.",
    )
    limiter_type: Literal["async", "sync"] = Field(
        description="Which public limiter produced the diagnostic."
    )
    limiter_instance_id: str = Field(description="Limiter instance id.")
    backend_type: DiagnosticBackendType = Field(description="Primary backend kind.")
    consistency: DiagnosticConsistency = Field(
        default="eventually_consistent",
        description="Cross-section consistency model for this diagnostic.",
    )
    generated_at_unix_seconds: float = Field(
        description="Wall-clock time when diagnose() assembled the top-level DTO."
    )
    as_of_monotonic: float = Field(
        description="Local monotonic timestamp when diagnose() began assembling data."
    )
    closed: bool = Field(description="Whether the limiter is closed.")
    closing: bool = Field(description="Whether the limiter is in close/drain.")
    model_family_count: int = Field(description="Known model families in this limiter.")
    bucket_count: int = Field(description="Total bucket diagnostics included.")
    buckets: tuple[BucketDiagnostic, ...] = Field(
        description="Per-bucket state for every known limited family bucket."
    )
    runtime_overrides: tuple[RuntimeOverrideDiagnostic, ...] = Field(
        description="Active runtime overrides derived from bucket diagnostics."
    )
    reservations: InFlightReservationsDiagnostic = Field(
        description="Local outstanding reservation state."
    )
    waits: CurrentWaitsDiagnostic = Field(
        description="Current acquire waiters and bottlenecks."
    )
    backend_health: BackendHealthDiagnostic = Field(
        description="Backend-specific health markers."
    )
    issues: tuple[DiagnosticIssue, ...] = Field(
        default=(),
        description="Best-effort degradation or backend introspection errors.",
    )


class BackendIntrospectionDiagnostic(StrictDTO):
    model_family: str = Field(description="Model family owned by this backend.")
    backend_type: DiagnosticBackendType = Field(
        description="Backend that supplied this introspection."
    )
    as_of_monotonic: float = Field(
        description="Local monotonic timestamp for this backend observation."
    )
    buckets: tuple[BucketDiagnostic, ...] = Field(
        description="Bucket diagnostics for this per-family backend."
    )
    waits: tuple[WaiterDiagnostic, ...] = Field(
        default=(),
        description="Current waiter diagnostics known to this backend instance.",
    )
    memory_health: MemoryBackendHealthDiagnostic | None = Field(default=None)
    redis_health: RedisBackendHealthDiagnostic | None = Field(default=None)
    issues: tuple[DiagnosticIssue, ...] = Field(default=())


@dataclass(frozen=True, slots=True)
class DiagnosticWaiterState:
    waiter_id: str
    reservation_id: str | None
    model_family: str
    model: str | None
    request_id: str | None
    state: DiagnosticWaitState
    usage: FrozenUsage
    wait_started_monotonic: float
    timeout_deadline_monotonic: float | None
    blocked_buckets: tuple[WaitBucketDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class BackendBucketLimit:
    effective_max_capacity: float
    refill_rate_per_second: float


@dataclass(frozen=True, slots=True)
class LimiterSnapshot:
    limiter_type: Literal["async", "sync"]
    limiter_instance_id: str
    backend_type: DiagnosticBackendType
    generated_at_unix_seconds: float
    as_of_monotonic: float
    closed: bool
    closing: bool
    backends: Mapping[str, object]
    quotas: dict[str, dict[BucketId, float]]
    local_runtime_overrides: dict[str, dict[BucketId, float]]
    pending_acquire_reservations: set[str]
    acquire_delivery_cleanup_reservations: set[str]
    in_flight_reservation_ids: set[str]
    in_flight_reservation_family: dict[str, str]
    reservation_snapshots: dict[str, ReservationAuthoritySnapshot]
    committed_refund_dedup_count: int


def sorted_bucket_diagnostics(
    buckets: tuple[BucketDiagnostic, ...] | list[BucketDiagnostic],
) -> tuple[BucketDiagnostic, ...]:
    return tuple(
        sorted(buckets, key=lambda b: (b.model_family, b.metric, b.per_seconds))
    )


def _override_is_active(configured_limit: float, effective_limit: float) -> bool:
    return not math.isclose(
        configured_limit,
        effective_limit,
        rel_tol=1e-12,
        abs_tol=0.0,
    )


def make_bucket_diagnostic(  # noqa: PLR0913
    *,
    model_family: str,
    metric: str,
    per_seconds: int,
    backend_type: DiagnosticBackendType,
    current_capacity: float | None,
    configured_limit: float,
    effective_max_capacity: float,
    override_source: DiagnosticOverrideSource,
    status: DiagnosticBucketStatus,
    as_of_monotonic: float,
) -> BucketDiagnostic:
    runtime_override = (
        effective_max_capacity
        if override_source != "none"
        or _override_is_active(configured_limit, effective_max_capacity)
        else None
    )
    return BucketDiagnostic(
        model_family=model_family,
        metric=metric,
        per_seconds=int(per_seconds),
        backend_type=backend_type,
        current_capacity=current_capacity,
        configured_limit=float(configured_limit),
        runtime_override=runtime_override,
        override_source=override_source,
        effective_max_capacity=float(effective_max_capacity),
        configured_to_effective_gap=float(effective_max_capacity - configured_limit),
        refill_rate_per_second=float(effective_max_capacity / int(per_seconds)),
        status=status,
        as_of_monotonic=as_of_monotonic,
    )


def unavailable_bucket_diagnostic(  # noqa: PLR0913
    *,
    model_family: str,
    bucket_id: BucketId,
    configured_limit: float,
    local_override: float | None,
    backend_type: DiagnosticBackendType,
    as_of_monotonic: float,
    status: DiagnosticBucketStatus = "unavailable",
) -> BucketDiagnostic:
    metric, per_seconds = bucket_id
    effective = configured_limit if local_override is None else local_override
    source: DiagnosticOverrideSource = "none" if local_override is None else "limiter"
    return make_bucket_diagnostic(
        model_family=model_family,
        metric=metric,
        per_seconds=per_seconds,
        backend_type=backend_type,
        current_capacity=None,
        configured_limit=configured_limit,
        effective_max_capacity=effective,
        override_source=source,
        status=status,
        as_of_monotonic=as_of_monotonic,
    )


def wait_bucket_diagnostics(
    *,
    model_family: str,
    usage: FrozenUsage,
    capacities: dict[BucketId, float],
    limits: dict[BucketId, BackendBucketLimit],
) -> tuple[WaitBucketDiagnostic, ...]:
    blocked: list[WaitBucketDiagnostic] = []
    for (metric, per_seconds), current_capacity in capacities.items():
        required = usage.get(metric)
        if required is None:
            continue
        deficit = max(0.0, float(required) - float(current_capacity))
        if deficit <= 0.0:
            continue
        limit = limits.get((metric, int(per_seconds)))
        refill_rate = None if limit is None else limit.refill_rate_per_second
        expected = (
            None
            if refill_rate is None or refill_rate <= 0.0
            else float(deficit / refill_rate)
        )
        blocked.append(
            WaitBucketDiagnostic(
                model_family=model_family,
                metric=metric,
                per_seconds=int(per_seconds),
                current_capacity=float(current_capacity),
                required_capacity=float(required),
                deficit=deficit,
                refill_rate_per_second=refill_rate,
                expected_refill_seconds=expected,
                effective_max_capacity=(
                    None if limit is None else limit.effective_max_capacity
                ),
            )
        )
    return tuple(sorted(blocked, key=_wait_bucket_sort_key))


def _wait_bucket_sort_key(bucket: WaitBucketDiagnostic) -> tuple[int, float, str, int]:
    expected = bucket.expected_refill_seconds
    unknown = 1 if expected is None else 0
    return (unknown, -(expected or 0.0), bucket.metric, bucket.per_seconds)


def waiter_diagnostic_from_state(
    state: DiagnosticWaiterState,
    *,
    as_of_monotonic: float,
) -> WaiterDiagnostic:
    wait_age = max(0.0, as_of_monotonic - state.wait_started_monotonic)
    timeout_remaining = (
        None
        if state.timeout_deadline_monotonic is None
        else max(0.0, state.timeout_deadline_monotonic - as_of_monotonic)
    )
    primary = state.blocked_buckets[0] if state.blocked_buckets else None
    return WaiterDiagnostic(
        reservation_id=state.reservation_id,
        model_family=state.model_family,
        model=state.model,
        request_id=state.request_id,
        state=state.state,
        usage=state.usage,
        wait_started_monotonic=state.wait_started_monotonic,
        wait_age_seconds=wait_age,
        timeout_remaining_seconds=timeout_remaining,
        primary_bottleneck=primary,
        blocked_buckets=state.blocked_buckets,
        as_of_monotonic=as_of_monotonic,
    )


def build_current_waits(
    waiters: list[WaiterDiagnostic],
    *,
    as_of_monotonic: float,
) -> CurrentWaitsDiagnostic:
    ordered = sorted(
        waiters,
        key=lambda waiter: (
            waiter.wait_started_monotonic is None,
            waiter.wait_started_monotonic or math.inf,
            waiter.reservation_id or "",
        ),
    )
    sampled = tuple(ordered[:DIAGNOSE_WAITER_SAMPLE_LIMIT])
    return CurrentWaitsDiagnostic(
        as_of_monotonic=as_of_monotonic,
        total_waiter_count=len(ordered),
        waiters=sampled,
        waiter_sample_limit=DIAGNOSE_WAITER_SAMPLE_LIMIT,
        waiters_truncated=len(ordered) > len(sampled),
    )


@dataclass(frozen=True, slots=True)
class _ReservationRow:
    reservation_id: str
    model_family: str
    state: DiagnosticReservationState
    usage: FrozenUsage | None
    age_seconds: float | None
    sort_time: float


def build_reservations(
    snapshot: LimiterSnapshot,
    *,
    waiter_usage_by_reservation_id: dict[str, FrozenUsage],
) -> InFlightReservationsDiagnostic:
    rows = _reservation_rows(snapshot, waiter_usage_by_reservation_id)
    ages = [row.age_seconds for row in rows if row.age_seconds is not None]
    groups = _reservation_groups(rows)
    return InFlightReservationsDiagnostic(
        as_of_monotonic=snapshot.as_of_monotonic,
        total_count=len(
            snapshot.in_flight_reservation_ids
            | snapshot.pending_acquire_reservations
            | snapshot.acquire_delivery_cleanup_reservations
        ),
        in_flight_count=len(snapshot.in_flight_reservation_ids),
        pending_acquire_count=len(snapshot.pending_acquire_reservations),
        delivery_cleanup_count=len(snapshot.acquire_delivery_cleanup_reservations),
        oldest_age_seconds=max(ages) if ages else None,
        groups=groups,
    )


def _reservation_rows(
    snapshot: LimiterSnapshot,
    waiter_usage_by_reservation_id: dict[str, FrozenUsage],
) -> list[_ReservationRow]:
    all_ids = (
        snapshot.in_flight_reservation_ids
        | snapshot.pending_acquire_reservations
        | snapshot.acquire_delivery_cleanup_reservations
    )
    rows: list[_ReservationRow] = []
    for reservation_id in all_ids:
        if reservation_id in snapshot.pending_acquire_reservations:
            state: DiagnosticReservationState = "pending_acquire"
        elif reservation_id in snapshot.acquire_delivery_cleanup_reservations:
            state = "delivery_cleanup"
        else:
            state = "in_flight"
        authority = snapshot.reservation_snapshots.get(reservation_id)
        family = (
            authority.model_family
            if authority is not None
            else snapshot.in_flight_reservation_family.get(reservation_id, "unknown")
        )
        usage = (
            authority.usage
            if authority is not None
            else waiter_usage_by_reservation_id.get(reservation_id)
        )
        age_seconds, sort_time = _reservation_age_and_sort_time(
            snapshot,
            authority,
        )
        rows.append(
            _ReservationRow(
                reservation_id=reservation_id,
                model_family=family,
                state=state,
                usage=usage,
                age_seconds=age_seconds,
                sort_time=sort_time,
            )
        )
    return rows


def _reservation_age_and_sort_time(
    snapshot: LimiterSnapshot,
    authority: ReservationAuthoritySnapshot | None,
) -> tuple[float | None, float]:
    if authority is None or authority.created_at_seconds is None:
        return None, math.inf
    age = max(0.0, snapshot.generated_at_unix_seconds - authority.created_at_seconds)
    return age, authority.created_at_seconds


def _reservation_groups(
    rows: list[_ReservationRow],
) -> tuple[ReservationGroupDiagnostic, ...]:
    groups: dict[
        tuple[str, str | None, DiagnosticReservationState | Literal["all"]],
        list[_ReservationRow],
    ] = defaultdict(list)
    for row in rows:
        metrics = tuple(row.usage) if row.usage else (None,)
        for metric in metrics:
            groups[(row.model_family, metric, row.state)].append(row)
            groups[(row.model_family, metric, "all")].append(row)

    diagnostics: list[ReservationGroupDiagnostic] = []
    for (model_family, metric, state), group_rows in groups.items():
        ordered = sorted(
            group_rows, key=lambda row: (row.sort_time, row.reservation_id)
        )
        sampled_ids = tuple(
            row.reservation_id for row in ordered[:DIAGNOSE_RESERVATION_ID_SAMPLE_LIMIT]
        )
        ages = [row.age_seconds for row in group_rows if row.age_seconds is not None]
        total_usage = None
        if metric is not None:
            total_usage = sum(
                float(row.usage[metric])
                for row in group_rows
                if row.usage is not None and metric in row.usage
            )
        diagnostics.append(
            ReservationGroupDiagnostic(
                model_family=model_family,
                metric=metric,
                state=state,
                outstanding_count=len(group_rows),
                total_reserved_usage=total_usage,
                oldest_age_seconds=max(ages) if ages else None,
                oldest_reservation_id=ordered[0].reservation_id if ordered else None,
                reservation_ids=sampled_ids,
                reservation_id_sample_limit=DIAGNOSE_RESERVATION_ID_SAMPLE_LIMIT,
                reservation_ids_truncated=len(ordered) > len(sampled_ids),
            )
        )
    return tuple(
        sorted(
            diagnostics,
            key=lambda group: (
                group.model_family,
                group.metric or "",
                _reservation_state_sort_key(group.state),
            ),
        )
    )


def _reservation_state_sort_key(
    state: DiagnosticReservationState | Literal["all"],
) -> int:
    return {
        "all": 0,
        "pending_acquire": 1,
        "in_flight": 2,
        "delivery_cleanup": 3,
    }[state]


def reconcile_buckets(
    *,
    snapshot: LimiterSnapshot,
    backend_results: list[BackendIntrospectionDiagnostic],
    issues: list[DiagnosticIssue],
) -> tuple[tuple[BucketDiagnostic, ...], tuple[RuntimeOverrideDiagnostic, ...]]:
    by_key: dict[tuple[str, BucketId], BucketDiagnostic] = {}
    for result in backend_results:
        for bucket in result.buckets:
            by_key[(bucket.model_family, (bucket.metric, int(bucket.per_seconds)))] = (
                bucket
            )

    reconciled: list[BucketDiagnostic] = []
    for model_family, quotas in snapshot.quotas.items():
        for bucket_id, configured_limit in quotas.items():
            backend_bucket = by_key.pop((model_family, bucket_id), None)
            local_override = snapshot.local_runtime_overrides.get(model_family, {}).get(
                bucket_id
            )
            if backend_bucket is None:
                reconciled.append(
                    unavailable_bucket_diagnostic(
                        model_family=model_family,
                        bucket_id=bucket_id,
                        configured_limit=configured_limit,
                        local_override=local_override,
                        backend_type=snapshot.backend_type,
                        as_of_monotonic=snapshot.as_of_monotonic,
                        status="missing",
                    )
                )
                issues.append(
                    DiagnosticIssue(
                        severity="warning",
                        component="limiter",
                        message="configured bucket was not returned by backend introspection",
                        model_family=model_family,
                        metric=bucket_id[0],
                        per_seconds=bucket_id[1],
                    )
                )
                continue
            reconciled.append(
                _reconcile_bucket(
                    backend_bucket,
                    configured_limit=configured_limit,
                    local_override=local_override,
                    issues=issues,
                )
            )

    for backend_bucket in by_key.values():
        reconciled.append(backend_bucket)
        issues.append(
            DiagnosticIssue(
                severity="info",
                component="limiter",
                message="backend returned a bucket not present in limiter quota snapshot",
                model_family=backend_bucket.model_family,
                metric=backend_bucket.metric,
                per_seconds=backend_bucket.per_seconds,
            )
        )

    buckets = sorted_bucket_diagnostics(reconciled)
    runtime_override_diagnostics: list[RuntimeOverrideDiagnostic] = []
    for bucket in buckets:
        if bucket.runtime_override is None:
            continue
        runtime_override_diagnostics.append(
            RuntimeOverrideDiagnostic(
                model_family=bucket.model_family,
                metric=bucket.metric,
                per_seconds=bucket.per_seconds,
                configured_limit=bucket.configured_limit,
                override_capacity=bucket.runtime_override,
                effective_max_capacity=bucket.effective_max_capacity,
                configured_to_override_gap=float(
                    bucket.runtime_override - bucket.configured_limit
                ),
                source=bucket.override_source,
            )
        )
    runtime_overrides = tuple(runtime_override_diagnostics)
    return buckets, runtime_overrides


def _reconcile_bucket(
    backend_bucket: BucketDiagnostic,
    *,
    configured_limit: float,
    local_override: float | None,
    issues: list[DiagnosticIssue],
) -> BucketDiagnostic:
    backend_override = backend_bucket.runtime_override
    if local_override is None and backend_override is None:
        effective = configured_limit
        source: DiagnosticOverrideSource = "none"
    elif local_override is None:
        assert backend_override is not None  # noqa: S101
        effective = float(backend_override)
        source = "backend"
    elif backend_override is None:
        effective = float(local_override)
        source = "limiter"
    else:
        effective = float(backend_override)
        source = "both"
        if not math.isclose(
            float(local_override),
            float(backend_override),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            issues.append(
                DiagnosticIssue(
                    severity="warning",
                    component="limiter",
                    message=(
                        "limiter and backend runtime max-capacity overrides differ; "
                        "using backend value as effective"
                    ),
                    model_family=backend_bucket.model_family,
                    metric=backend_bucket.metric,
                    per_seconds=backend_bucket.per_seconds,
                )
            )
    return make_bucket_diagnostic(
        model_family=backend_bucket.model_family,
        metric=backend_bucket.metric,
        per_seconds=backend_bucket.per_seconds,
        backend_type=backend_bucket.backend_type,
        current_capacity=backend_bucket.current_capacity,
        configured_limit=configured_limit,
        effective_max_capacity=effective,
        override_source=source,
        status=backend_bucket.status,
        as_of_monotonic=backend_bucket.as_of_monotonic,
    )


def build_backend_health(
    *,
    snapshot: LimiterSnapshot,
    backend_results: list[BackendIntrospectionDiagnostic],
    unsupported_backend_class_names: tuple[str, ...],
) -> BackendHealthDiagnostic:
    memory_health = _aggregate_memory_health(backend_results)
    redis_health = _aggregate_redis_health(snapshot, backend_results)
    custom_health = None
    if unsupported_backend_class_names or snapshot.backend_type not in {
        "memory",
        "redis",
    }:
        custom_health = CustomBackendHealthDiagnostic(
            introspection_supported=not unsupported_backend_class_names,
            backend_class_names=unsupported_backend_class_names
            or tuple(
                sorted(
                    {type(backend).__name__ for backend in snapshot.backends.values()}
                )
            ),
        )
    return BackendHealthDiagnostic(
        backend_type=snapshot.backend_type,
        memory=memory_health,
        redis=redis_health,
        custom=custom_health,
    )


def _aggregate_memory_health(
    backend_results: list[BackendIntrospectionDiagnostic],
) -> MemoryBackendHealthDiagnostic | None:
    healths = [
        result.memory_health for result in backend_results if result.memory_health
    ]
    if not healths:
        return None
    return MemoryBackendHealthDiagnostic(
        model_family_count=sum(health.model_family_count for health in healths),
        bucket_count=sum(health.bucket_count for health in healths),
        acquired_reservation_id_count=sum(
            health.acquired_reservation_id_count for health in healths
        ),
        refund_dedup_count=sum(health.refund_dedup_count for health in healths),
        refund_dedup_cap=max(health.refund_dedup_cap for health in healths),
    )


def _aggregate_redis_health(
    snapshot: LimiterSnapshot,
    backend_results: list[BackendIntrospectionDiagnostic],
) -> RedisBackendHealthDiagnostic | None:
    healths = [result.redis_health for result in backend_results if result.redis_health]
    if not healths:
        return None
    first = healths[0]
    marker_estimate = len(
        snapshot.in_flight_reservation_ids | snapshot.pending_acquire_reservations
    )
    return RedisBackendHealthDiagnostic(
        model_family_count=sum(health.model_family_count for health in healths),
        bucket_count=sum(health.bucket_count for health in healths),
        connection_pool_class=first.connection_pool_class,
        connection_pool_max_connections=first.connection_pool_max_connections,
        connection_pool_in_use_connections=first.connection_pool_in_use_connections,
        connection_pool_available_connections=(
            first.connection_pool_available_connections
        ),
        pool_counts_observed_with_private_attrs=any(
            health.pool_counts_observed_with_private_attrs for health in healths
        ),
        local_marker_count_estimate=marker_estimate,
        local_refund_dedup_count_estimate=snapshot.committed_refund_dedup_count,
    )


def build_rate_limiter_diagnostic(
    *,
    snapshot: LimiterSnapshot,
    backend_results: list[BackendIntrospectionDiagnostic],
    unsupported_backend_class_names: tuple[str, ...],
    issues: list[DiagnosticIssue],
) -> RateLimiterDiagnostic:
    buckets, runtime_overrides = reconcile_buckets(
        snapshot=snapshot,
        backend_results=backend_results,
        issues=issues,
    )
    waiters = [waiter for result in backend_results for waiter in result.waits]
    waiter_usage_by_id = {
        waiter.reservation_id: waiter.usage
        for waiter in waiters
        if waiter.reservation_id is not None
    }
    reservations = build_reservations(
        snapshot,
        waiter_usage_by_reservation_id=waiter_usage_by_id,
    )
    backend_health = build_backend_health(
        snapshot=snapshot,
        backend_results=backend_results,
        unsupported_backend_class_names=unsupported_backend_class_names,
    )
    issues_sorted = tuple(
        sorted(
            issues,
            key=lambda issue: (
                issue.severity,
                issue.component,
                issue.model_family or "",
                issue.metric or "",
                issue.per_seconds or 0,
                issue.message,
            ),
        )
    )
    return RateLimiterDiagnostic(
        limiter_type=snapshot.limiter_type,
        limiter_instance_id=snapshot.limiter_instance_id,
        backend_type=snapshot.backend_type,
        generated_at_unix_seconds=snapshot.generated_at_unix_seconds,
        as_of_monotonic=snapshot.as_of_monotonic,
        closed=snapshot.closed,
        closing=snapshot.closing,
        model_family_count=len(snapshot.backends),
        bucket_count=len(buckets),
        buckets=buckets,
        runtime_overrides=runtime_overrides,
        reservations=reservations,
        waits=build_current_waits(waiters, as_of_monotonic=time.monotonic()),
        backend_health=backend_health,
        issues=issues_sorted,
    )


def backend_type_from_name(value: object) -> DiagnosticBackendType:
    if value in {"memory", "redis", "custom", "unknown"}:
        return cast("DiagnosticBackendType", value)
    return "unknown"


def backend_type_for_object(value: object) -> DiagnosticBackendType:
    module = type(value).__module__
    if "._memory." in module:
        return "memory"
    if "._redis." in module:
        return "redis"
    if value is None:
        return "unknown"
    return "custom"


def capacity_reservation_from_authority(
    authority: ReservationAuthoritySnapshot,
) -> CapacityReservation:
    return authority.to_reservation()
