# Custom Backend Contract

token-throttle treats custom backends as structural protocols. A backend does
not need nominal inheritance for type checking, but it must satisfy the full
`RateLimiterBackend` or `SyncRateLimiterBackend` protocol.
Subclassing the published protocol remains useful because it provides
conservative default implementations for optional hooks.

Run the conformance helper against custom backends before shipping:

```python
from token_throttle import (
    conformance_test_for,
    sync_conformance_test_for,
)
from token_throttle.conformance import ConformanceTiming


async def test_async_backend_contract(async_builder):
    await conformance_test_for(async_builder)


def test_sync_backend_contract(sync_builder):
    sync_conformance_test_for(
        sync_builder,
        timing=ConformanceTiming(operation_deadline_seconds=20.0),
    )
```

For synchronous test suites that still need to validate an async backend, use
`run_conformance_test_for()`. It wraps `conformance_test_for()` in
`asyncio.run(...)`:

```python
from token_throttle import MemoryBackendBuilder, run_conformance_test_for
from token_throttle.conformance import ConformanceTiming


def test_async_backend_contract_from_sync_suite() -> None:
    run_conformance_test_for(
        MemoryBackendBuilder(),
        timing=ConformanceTiming(operation_deadline_seconds=20.0),
    )
```

Use isolated backend state for these tests, such as a disposable Redis key
prefix, test database, or in-memory backend instance. A conformance failure
raises `BackendConformanceError`.

The helper invokes builder cleanup hooks in a `try`/`finally` boundary:
`conformance_test_for()` calls `aclose()` and then `close()` when those methods
exist, and `sync_conformance_test_for()` calls `close()`. If you need to manage
cleanup outside the helper, pass a small wrapper builder that does not expose
those methods.

## Conformance Timing

The default helper deadlines are:

- builder operations: 5 seconds
- backend operations: 10 seconds
- prompt no-wait checks: 1 second
- bounded wait probes: 5 seconds

Override them per test with `ConformanceTiming`:

```python
await conformance_test_for(
    async_builder,
    timing=ConformanceTiming(
        builder_deadline_seconds=10.0,
        operation_deadline_seconds=20.0,
        prompt_deadline_seconds=2.0,
        wait_budget_seconds=10.0,
    ),
)
```

For slow CI runners, set `TOKEN_THROTTLE_CONFORMANCE_TIMING_SCALE` to multiply
all defaults, for example `TOKEN_THROTTLE_CONFORMANCE_TIMING_SCALE=2`. When
`timing=` is passed, the env var is ignored; set every desired field on the
dataclass. Scales below ~0.1 may cause correct backends to fail conformance
because internal probe deadlines have an implicit floor.

## Async Protocol

`RateLimiterBackendBuilderInterface` must provide:

- `build(cfg, *, callbacks=None) -> RateLimiterBackend`: construct state for one
  validated `PerModelConfig`.
- `aclose() -> None`: async cleanup hook for shared resources.
- `close() -> None`: sync cleanup hook used by `RateLimiter.close()`.

`RateLimiterBackend` must provide:

- `async def await_for_capacity(usage, *, timeout=None, reservation_id=None, reservation_lifetime_seconds=None) -> float | None`
- `async def consume_capacity(usage, *, reservation_id=None, reservation_lifetime_seconds=None) -> float | None`
- `async def refund_capacity(reserved_usage, actual_usage) -> None`
- `async def refund_capacity_for_buckets(reserved_usage, actual_usage, *, bucket_ids=None, reservation_id=None, reservation_model_family=None, reservation_bucket_ids=None, reservation_reserved_usage=None) -> bool`
- `supports_durable_refund_dedup() -> bool`
- `supports_acquire_marker_authority() -> bool`
- `async def set_max_capacity(metric, per_seconds, value) -> None`
- `async def apply_configured_max_capacity(metric, per_seconds, value) -> None`
- `supports_metric_set_change() -> bool`
- `async def prepare_reconfigured_backend(new_backend, cfg) -> RateLimiterBackend`

The sync protocol uses the same semantics, with `wait_for_capacity()` replacing
`await_for_capacity()` and synchronous return values throughout.

## Capacity Semantics

`await_for_capacity()` / `wait_for_capacity()` must be all-or-nothing. If one
metric lacks capacity, no metric may be consumed. `timeout=0` is a try-acquire:
return immediately when capacity is available, otherwise raise `TimeoutError`.
For positive timeouts, the timeout bounds capacity waiting, not callback
dispatch or backend operation latency. Direct backend calls must include the
full configured metric key set; use `0` for metrics that should not be consumed.

`consume_capacity()` must not wait for capacity. It may make bucket capacity
negative so debt is recovered by normal refill. This is the backend entry point
used by record-usage style flows.

`refund_capacity()` must credit `reserved_usage - actual_usage`, cap positive
refunds at each bucket's current `max_capacity`, and preserve negative debt
instead of clamping it to zero. If `actual_usage` exceeds `reserved_usage`, emit
a `RuntimeWarning` and apply a negative refund. Negative or non-finite actual
usage is invalid and must raise `ValueError`.

`set_max_capacity()` is the explicit runtime override API. It changes the live
bucket maximum and recalculates the refill rate so a full refill still takes
one quota window. `apply_configured_max_capacity()` is the config-rebuild path;
backends that do not need separate persistence semantics can delegate to
`set_max_capacity()`.

## Error Taxonomy

Backends should raise:

- `TimeoutError` when bounded capacity waiting expires.
- `ValueError` for invalid usage, unknown active metrics, unsupported bucket ids,
  and usage greater than a bucket's `max_capacity`.
- `DuplicateRefundError` when the backend can prove a reservation was already
  refunded or already acquired.
- `UnknownReservationError` when a marker-authoritative backend cannot prove
  the presented reservation was acquired.
- `AcquireRefundFailedError` from public limiter acquire delivery when capacity
  was reserved, delivery was interrupted, and the fallback refund failed.
- `BackendConformanceError` only from conformance tests, not from normal backend
  operations.

## Acquire Delivery and Fallback Refund Failure Contract

`AcquireRefundFailedError` is a regular `Exception`, not an
`asyncio.CancelledError`. Catch it directly when callers need to recover a
reservation after interrupted acquire delivery.

The error exposes:

- `.reservation`: the delivered reservation that still needs operator attention
  or explicit use.
- `.interrupted_by`: the original interruption, such as cancellation, when
  available.
- `.refund_error`: the refund failure, when available.

Public limiter code chains the raised `AcquireRefundFailedError` with
`__cause__` so normal exception tooling can inspect the triggering failure.

## Metric-Set Reconfiguration

`supports_metric_set_change()` must return a plain synchronous `bool`.

Leave it `False` unless the backend can preserve live state for surviving bucket
ids when callable configs add or remove metrics. To return `True`, the backend
must either store state in stable external storage keyed by metric/window, or
override `prepare_reconfigured_backend()` to migrate or share in-process state
with the rebuilt backend. Returning `True` while inheriting the default no-op
hook is a contract violation because it can silently reset consumption state.

## Marker Authority for Acquired Reservations

`supports_acquire_marker_authority()` must return a plain synchronous `bool`.

Return `True` only when refunds can prove that the reservation was acquired by
this backend or by a cooperating process using the same shared backend state.
Redis does this with durable acquire markers. Local memory backends and opaque
custom backends should normally return `False`.

If this method returns `True`, the backend must override
`refund_capacity_for_buckets()`. The default implementation only delegates to
`refund_capacity()` and cannot verify reservation authority. Public limiters
reject backends that claim marker authority while inheriting that default path.

When marker authority is supported, `await_for_capacity()` / `consume_capacity()`
must create the marker atomically with capacity consumption whenever
`reservation_id` is supplied. `refund_capacity_for_buckets()` must verify and
consume that marker before crediting capacity. A missing marker should raise
`UnknownReservationError`; a consumed marker with a known refund tombstone should
raise `DuplicateRefundError`.

Authoritative refunds must fail closed when any marker metadata kwarg is omitted:
`reservation_model_family`, `reservation_bucket_ids`, or
`reservation_reserved_usage`. These kwargs remain optional in the structural
protocol for compatibility with non-authoritative backends, but a backend that
claims `supports_acquire_marker_authority()` must not silently default them to
caller-supplied refund data.

The refund target `bucket_ids` is also part of the authority check. It may be a
surviving subset of the reservation bucket ids after reconfiguration, but the
positional `reserved_usage` must match the corresponding projection of
`reservation_reserved_usage`. Do not credit arbitrary caller-supplied buckets
after only validating the marker metadata.

## Reservation Lifetime and Durable Marker TTLs

`reservation_lifetime_seconds` is meaningful when `reservation_id` is supplied.
Durable marker backends must expire acquire markers and refund-dedup tombstones
only after the reservation lifetime is no longer refundable. Redis builders
validate their TTL knobs so acquire-marker TTL and refund-dedup TTL are both
greater than `max_reservation_lifetime_seconds * 2`.

Custom durable backends should enforce the same invariant:

- a live marker must outlast the maximum reservation lifetime
- a refund-dedup record must outlast marker expiry and expected retry windows
- omitting `reservation_lifetime_seconds` while requesting durable marker
  authority should fail closed instead of creating immortal markers

Memory-style backends may ignore the TTL parameter because their markers are
process-local and are not durable across restarts.

## Observability Callback Emit Points

Backends receive `RateLimiterCallbacks` or `SyncRateLimiterCallbacks` from their
builder. They must invoke callback slots outside backend locks where possible:

- `on_missing_consumption_data`: when a bucket is first observed without stored
  consumption data and full quota is assumed.
- `on_capacity_consumed`: after capacity has been committed by
  `await_for_capacity()`, `wait_for_capacity()`, or `consume_capacity()`.
- `on_wait_start`: when a blocking acquire determines it must wait.
- `after_wait_end_consumption`: after a waited acquire successfully commits.
- `on_capacity_refunded`: after refund capacity has been committed.

Callback payloads must use redacted model-family, usage, capacity, and timing
data. Do not include Redis URLs, credentials, plaintext lock names, prompt text,
responses, or API keys.

The conformance helper validates callback emission and callback payload shape.
It does not require structured `DEBUG` log emission.

Callbacks are best-effort for ordinary `Exception` subclasses: token-throttle
emits a `RuntimeWarning` and continues. Critical exceptions propagate instead.
Lifecycle and backend callbacks propagate `asyncio.CancelledError`,
`concurrent.futures.CancelledError`, `KeyboardInterrupt`, `SystemExit`,
`GeneratorExit`, `MemoryError`, and `RecursionError`. Backend callback dispatch
also propagates `AcquireRefundFailedError`.

`GeneratorExit` is treated as a lifecycle-critical cleanup signal. On the public
async limiter path, CPython may surface a callback-raised `GeneratorExit` to the
awaiting caller as `RuntimeError("coroutine ignored GeneratorExit")`; use an
application exception instead of `GeneratorExit` for callback control flow.
`MemoryError` and `RecursionError` also propagate because they indicate severe
process-health or programming failures (out-of-memory or runaway recursion) that
should not be hidden behind callback warning suppression.

## Backend Method Critical Exceptions

Backend methods that raise lifecycle-critical exceptions propagate those
exceptions raw to callers. This applies to `await_for_capacity()`,
`wait_for_capacity()`, `consume_capacity()`, `refund_capacity()`,
`refund_capacity_for_buckets()`, `set_max_capacity()`, and the async/sync
configured max-capacity hooks.

The same rule applies during interrupted-acquire cleanup. If an acquire has
already committed backend capacity but is interrupted before the reservation is
delivered, token-throttle attempts a cleanup refund. A cleanup refund failure
caused by `asyncio.CancelledError`, `concurrent.futures.CancelledError`,
`KeyboardInterrupt`, `SystemExit`, `GeneratorExit`, `MemoryError`, or
`RecursionError` escapes raw. Severe failures therefore escape all recovery
layers, symmetric with the callback contract above.

`AcquireRefundFailedError` remains the recovery envelope for ordinary
`Exception` subclasses that are not lifecycle-critical. Interrupted-acquire
cleanup does not wrap critical-tuple backend failures in
`AcquireRefundFailedError`.

Durable shared-state backends should also emit structured `DEBUG` logs under
the same logger families used by Redis:

- `token_throttle.acquire` for acquire marker reads, writes, and deletes.
- `token_throttle.refund` for refund marker GET/DEL and refund-dedup writes.
- `token_throttle.lock` for distributed lock acquire, release, and extension.

Each structured record should provide a `token_throttle_event` logging attribute.
Hash or otherwise redact raw lock names and deployment prefixes. A typical
stdlib logging template is:

```python
logger.debug(
    "refund_marker_deleted",
    extra={
        "token_throttle_event": {
            "event_type": "refund_marker_deleted",
            "reservation_id": reservation_id,
            "bucket_id": bucket_id,
            "model_family": redacted_model_family,
        }
    },
)
```

## Public Limiter Snapshot State

`RateLimiter.snapshot_state()` and `SyncRateLimiter.snapshot_state()` are owned
by the public limiters rather than custom backends. Backend authors still affect
the output through builder identity and marker behavior.

The snapshot always includes:

- `in_flight_reservations`
- `model_families`
- `backend_type`

Redis backends also expose best-effort local estimates:

- `marker_count_estimate`
- `refund_dedup_count_estimate`

Custom backends must not rely on snapshot output for authority decisions. It is
a redacted health surface, not a backend inventory API.

## Conformance Scope

The bundled helpers check:

- structural protocol shape for async and sync builders/backends
- per-build isolation and runtime checking of every build result
- sync/async return contracts, including awaitable misuse on sync paths
- basic capacity acquisition, prompt try-acquire behavior, and invalid usage
  rejection
- all-or-nothing multi-metric consumption
- refund semantics, overuse warnings, and max-capacity update behavior
- callback emission and documented callback payload keys/types
- marker authority positive behavior for backends that claim support
- marker authority negative behavior for unknown, duplicate, forged-scope,
  mismatched-metadata, and omitted-metadata refunds, including no-credit
  side-effect checks
- marker authority claim consistency, including rejection of default refund
  hooks for authoritative backends
- durable refund dedup claim consistency and duplicate-refund behavior
- metric-set-change claim consistency and reconfiguration behavior
- public limiter round-trip behavior across backend builders
- the `AcquireRefundFailedError` shape exposed by public limiters, including
  `.reservation`, `.interrupted_by`, `.refund_error`, and exception chaining
- conformance-harness handling for the canonical lifecycle-critical exception
  taxonomy, including `MemoryError` and `RecursionError`

The helpers do not check:

- performance regressions beyond bounded helper deadlines
- structured `token_throttle_event` `DEBUG` log emission
- third-party storage durability or write-ahead guarantees
- every callback severe-exception path in your backend implementation;
  manually test that callback dispatch propagates `asyncio.CancelledError`,
  `concurrent.futures.CancelledError`, `KeyboardInterrupt`, `SystemExit`,
  `GeneratorExit`, `MemoryError`, `RecursionError`, and backend callback
  `AcquireRefundFailedError`
- every backend-method severe-exception path; manually inject those
  exceptions into `await_for_capacity()` / `wait_for_capacity()`,
  `consume_capacity()`, `refund_capacity()`, `refund_capacity_for_buckets()`,
  `set_max_capacity()`, and configured max-capacity hooks, including
  interrupted-acquire cleanup, and assert they escape raw rather than being
  wrapped in `AcquireRefundFailedError`
- marker TTL expiry, durable marker garbage collection, or resistance to forged
  `CapacityReservation.created_at_seconds`; test those with backend-specific
  timing/storage instrumentation

KNOWN UNKNOWN: post-write probes for third-party durable backends are still not
portable. The conformance helper verifies observable behavior through the public
backend protocol, but it cannot prove that an external storage write reached
durable media without backend-specific instrumentation.
