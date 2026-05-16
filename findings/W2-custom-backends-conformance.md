# W2 Custom Backends Conformance

Decision: Path A. The existing ABC surface already encoded behavioral authority
claims that nominal inheritance could not validate: marker authority,
durable-refund claims, metric-set migration, callback emission, and refund
projection all fail far from the custom backend implementation when they are
wrong. Because a major-version bump is allowed, formalizing the backend and
builder surfaces as runtime-checkable `Protocol` classes plus a reusable
conformance helper is worth the v5 API commitment. Docs-only would leave the
same subtle failure mode in place.

## What changed

- Converted `RateLimiterBackend`, `SyncRateLimiterBackend`, and both builder
  interfaces from ABCs to runtime-checkable `Protocol` classes while preserving
  inherited default hook implementations for subclass-based backends.
- Added public `BackendConformanceError`.
- Added public helpers:
  - `conformance_test_for(async_builder)`
  - `sync_conformance_test_for(sync_builder)`
  - `run_conformance_test_for(async_builder)`
- Added `token_throttle/conformance.py`, which checks protocol shape, capacity
  semantics, all-or-nothing consumption, invalid usage rejection, refund
  warnings, max-capacity updates, callback emission, and authority-claim
  consistency.
- Added `tests/conformance/test_backend_conformance.py` covering the helper
  against built-in memory backends and marker-authority liar fixtures.
- Added `docs/custom-backends.md` with the backend contract, FIX-48 marker
  authority anchor, FIX-53 TTL hook anchor, FIX-54 observability emit points,
  FIX-54 snapshot-state contract, and error taxonomy.
- Linked the custom backend docs from `README.md` and added a v4.x to v5.0.0
  migration note.

## Compatibility

This is v5-breaking API work. Existing subclass-based backends keep the default
hook bodies, but structural custom backends must now implement the full protocol
surface, including cleanup hooks and optional authority hooks, to type-check.
No local version bump was made; release workflow should handle that.

## Verification

- `uv run pytest tests/conformance/test_backend_conformance.py -q`
- `uv run pytest tests/unit/ -q --tb=no`
- `uv run mypy token_throttle/__init__.py`
- `uv run ruff check .`
- `uv run ruff format --check .`

## Known Unknowns

KNOWN UNKNOWN: post-write probes for third-party durable backends are still not
portable. The conformance helper verifies public protocol behavior, but it
cannot prove that a backend-specific external storage write reached durable
media without backend-specific instrumentation.

KNOWN UNKNOWN: Redis-like custom backends may need stronger conformance checks
for duplicate refunds, unknown markers, and TTL expiry once a backend-agnostic
test harness for durable shared state exists.
