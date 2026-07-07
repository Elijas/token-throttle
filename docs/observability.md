# Observability reference

token-throttle stays framework-agnostic: it exposes logging, callbacks, and a
small health snapshot, but does not depend on Prometheus, OpenTelemetry, or any
metrics SDK. Wire these surfaces to your own collectors.

The [README](../README.md#observability) shows the `snapshot_state()` and
lifecycle-callback examples. This reference documents the debug loggers,
lifecycle event fields, callback timeouts, and the full PII surface.

## Logging and debug records

Every `RateLimiter` and `SyncRateLimiter` logs the `token_throttle` package
version at `INFO` during initialization. Existing callback loggers use the
`token_throttle` logger by default through `create_logging_callbacks()` and
`create_sync_logging_callbacks()`. Redis internals also emit structured
`DEBUG` records under:

- `token_throttle.acquire` for acquire marker reads/writes/deletes.
- `token_throttle.refund` for refund marker GET/DEL and refund-dedup writes.
- `token_throttle.lock` for Redis lock acquire, release, and extension events.

Redis debug records include a `token_throttle_event` logging attribute with
`event_type`, `reservation_id`, `bucket_id`, and operation-specific fields. For
example, a stdlib handler can read `record.token_throttle_event` and turn it
into counters or spans.

## Health snapshot

For Redis backends, marker and refund-dedup counts from `snapshot_state()` are
best-effort local estimates from limiter bookkeeping, not a cross-process Redis
inventory. The snapshot intentionally omits Redis URLs, credentials, and Redis
key prefixes.

## Diagnostics (`diagnose()`)

`await limiter.diagnose()` (`limiter.diagnose()` on `SyncRateLimiter`) returns
a `RateLimiterDiagnostic` DTO: a richer, point-in-time snapshot than
`snapshot_state()`. Unlike `snapshot_state()`, it performs bounded backend I/O
to reconcile local bookkeeping against backend-reported bucket state; it never
mutates capacity.

It reports, per model family and bucket: current and effective capacity, the
configured limit, and any active runtime override with its source (`limiter`,
`backend`, or `both`); in-flight/pending/delivery-cleanup reservation counts
grouped by family and metric; current acquire waiters and each one's primary
capacity bottleneck; backend health for memory, Redis, and custom backends;
and a severity-sorted `issues` list covering best-effort degradation or
introspection failures.

Reach for `diagnose()` when investigating a stuck or slow acquire, unexplained
capacity drift, a suspected override mismatch between limiter and backend
bookkeeping, or a custom backend that may not support introspection. Reach for
the lighter `snapshot_state()` for routine health-check polling.

```python
# (fragment — see the README Any provider example for standalone context)
diagnostic = await limiter.diagnose()
for issue in diagnostic.issues:
    print(issue.severity, issue.component, issue.message)
if diagnostic.waits.waiters:
    stuck = diagnostic.waits.waiters[0]
    print(stuck.model_family, stuck.primary_bottleneck)
```

The full DTO tree (`RateLimiterDiagnostic`, `BucketDiagnostic`,
`InFlightReservationsDiagnostic`, `CurrentWaitsDiagnostic`,
`BackendHealthDiagnostic`, `DiagnosticIssue`, and related types) is importable
from `token_throttle`; every field carries its own docstring. Custom backends
can opt into richer per-backend sections by implementing the optional
`introspect()` method from `BackendIntrospectable` /
`SyncBackendIntrospectable`; backends that omit it still get a
`custom_backend` info-level issue rather than a failure.

## Lifecycle events

Lifecycle events include `event_type`, `reservation_id`, optional `request_id`
from `acquire_capacity_for_request(..., request_id="...")`, `model_family`,
`model_alias`, `bucket_ids`, `usage`, and `timestamp`. Existing wait, consume,
refund, and missing-data callbacks keep their original keyword signatures.

Wire the additive `on_lifecycle_event` callback on `RateLimiterCallbacks` to
receive these events without changing existing callback signatures:

```python
# (fragment — see the README Any provider example for standalone context)
from token_throttle import LifecycleEvent, RateLimiterCallbacks

async def on_lifecycle_event(*, event: LifecycleEvent) -> None:
    metrics.increment(
        f"token_throttle.{event.event_type}",
        tags={
            "model_family": event.model_family,
            "model_alias": event.model_alias,
        },
    )

limiter = RateLimiter(
    get_config,
    backend=backend,
    callbacks=RateLimiterCallbacks(on_lifecycle_event=on_lifecycle_event),
)
```

## Structured errors

For alerting or retry routing, public token-throttle exception classes expose a
stable `reason` attribute where callers need structured error handling.

## Callback timeouts

User callbacks are bounded separately by `callback_timeout` on `RateLimiter`
and `SyncRateLimiter` (default: 30 seconds per callback). When a callback
exceeds that limit, token-throttle logs a warning, skips the callback result,
and does not fail the acquire/refund call. The callback is abandoned rather than
cancelled: it may keep running to completion in the background, and any error it
raises after the deadline is logged. Pass `callback_timeout=None` to
restore unbounded callback execution. Timeout-wrapped sync callbacks run in a
helper thread with the caller's `contextvars` context copied into that thread.

The two paths differ at shutdown. An abandoned async callback is still a real
task, and `asyncio.run()` cancels leftover tasks at shutdown and then waits for
them, so a timed-out callback that also swallows cancellation can block event
loop shutdown indefinitely. Abandoned sync callbacks run in daemon helper
threads and never block interpreter exit.

## PII surface

- User-controlled fields: request `model`, lifecycle `model_alias`, optional
  `request_id`, custom usage metric names, `model_family` when supplied by your
  config, and Redis `key_prefix` configured by the application.
- Potentially sensitive fields: `request_id` if it contains customer or trace
  identifiers; `model_alias` and `model_family` if your naming scheme embeds
  tenant, deployment, or account data; usage values if request size is
  sensitive in your environment.
- Not logged or returned by `snapshot_state()`: Redis URLs, credentials, Redis
  client objects, and plaintext key prefixes.
- Never included by token-throttle observability surfaces: prompt text,
  messages, responses, API keys, or request payload bodies. A custom
  `usage_counter` or your own callback code may log those separately, so audit
  application code that you attach to callbacks.
