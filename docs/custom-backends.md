# Custom Backend Contract

token-throttle v5 treats custom backends as structural protocols. A backend no
longer needs nominal inheritance for type checking, but it must satisfy the
full `RateLimiterBackend` or `SyncRateLimiterBackend` protocol. Subclassing the
published protocol remains useful because it provides conservative default
implementations for optional hooks.

Run the conformance helper against custom backends before shipping:

```python
from token_throttle import conformance_test_for, sync_conformance_test_for


async def test_async_backend_contract(async_builder):
    await conformance_test_for(async_builder)


def test_sync_backend_contract(sync_builder):
    sync_conformance_test_for(sync_builder)
```

Use isolated backend state for these tests, such as a disposable Redis key
prefix, test database, or in-memory backend instance. A conformance failure
raises `BackendConformanceError`.

## Async Protocol

`RateLimiterBackendBuilderInterface` must provide:

- `build(cfg, *, callbacks=None) -> RateLimiterBackend`: construct state for one
  validated `PerModelConfig`.
- `aclose() -> None`: async cleanup hook for shared resources.
- `close() -> None`: sync cleanup hook used by `RateLimiter.close()`.

`RateLimiterBackend` must provide:

- `await_for_capacity(usage, *, timeout=None, reservation_id=None, reservation_lifetime_seconds=None) -> float | None`
- `consume_capacity(usage, *, reservation_id=None, reservation_lifetime_seconds=None) -> float | None`
- `refund_capacity(reserved_usage, actual_usage) -> None`
- `refund_capacity_for_buckets(reserved_usage, actual_usage, *, bucket_ids=None, reservation_id=None, reservation_model_family=None, reservation_bucket_ids=None, reservation_reserved_usage=None) -> bool`
- `supports_durable_refund_dedup() -> bool`
- `supports_acquire_marker_authority() -> bool`
- `set_max_capacity(metric, per_seconds, value) -> None`
- `apply_configured_max_capacity(metric, per_seconds, value) -> None`
- `supports_metric_set_change() -> bool`
- `prepare_reconfigured_backend(new_backend, cfg) -> RateLimiterBackend`

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
- `BackendConformanceError` only from conformance tests, not from normal backend
  operations.

## Metric-Set Reconfiguration

`supports_metric_set_change()` must return a plain synchronous `bool`.

Leave it `False` unless the backend can preserve live state for surviving bucket
ids when callable configs add or remove metrics. To return `True`, the backend
must either store state in stable external storage keyed by metric/window, or
override `prepare_reconfigured_backend()` to migrate or share in-process state
with the rebuilt backend. Returning `True` while inheriting the default no-op
hook is a contract violation because it can silently reset consumption state.

## Marker Authority (FIX-48)

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

## TTL Hooks (FIX-53)

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

## Observability Emit Points (FIX-54)

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

Durable shared-state backends should also emit structured `DEBUG` logs under
the same logger families used by Redis:

- `token_throttle.acquire` for acquire marker reads, writes, and deletes.
- `token_throttle.refund` for refund marker GET/DEL and refund-dedup writes.
- `token_throttle.lock` for distributed lock acquire, release, and extension.

Each structured record should provide a `token_throttle_event` logging attribute
with `event_type`, `reservation_id`, `bucket_id`, and operation-specific fields.
Hash or otherwise redact raw lock names and deployment prefixes.

## Snapshot State (FIX-54)

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

The bundled helpers check protocol shape, basic capacity semantics,
all-or-nothing consumption, invalid usage rejection, refund warnings,
`set_max_capacity()` / `apply_configured_max_capacity()`, callback emission, and
marker-authority claim consistency.

KNOWN UNKNOWN: post-write probes for third-party durable backends are still not
portable. The conformance helper verifies observable behavior through the public
backend protocol, but it cannot prove that an external storage write reached
durable media without backend-specific instrumentation.
