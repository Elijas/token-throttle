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

## Lifecycle events

Lifecycle events include `event_type`, `reservation_id`, optional `request_id`
from `acquire_capacity_for_request(..., request_id="...")`, `model_family`,
`model_alias`, `bucket_ids`, `usage`, and `timestamp`. Existing wait, consume,
refund, and missing-data callbacks keep their original keyword signatures.

## Structured errors

For alerting or retry routing, public token-throttle exception classes expose a
stable `reason` attribute where callers need structured error handling.

## Callback timeouts

User callbacks are bounded separately by `callback_timeout` on `RateLimiter`
and `SyncRateLimiter` (default: 30 seconds per callback). When a callback
exceeds that limit, token-throttle logs a warning, skips the callback result,
and does not fail the acquire/refund call. Pass `callback_timeout=None` to
restore unbounded callback execution. Timeout-wrapped sync callbacks run in a
helper thread with the caller's `contextvars` context copied into that thread.

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
