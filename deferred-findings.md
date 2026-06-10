# FIX-22 Deferred Long-Tail Findings

These R4 long-tail items were verified but left out of the code patch because
closing them cleanly would change public API shape or require broader Redis
corruption/error taxonomy work than this closure bundle should take on.

| Finding | Deferral |
|---|---|
| L01:F23 | Removing the private-but-reachable `UsageQuotas(..., _allow_empty_quotas=True)` constructor escape hatch is a public API break. Current guidance points users to `UsageQuotas.unlimited()` and empty `UsageQuotas([])` already raises clearly. |
| L17:Q03 | Redis corrupted-value diagnostics still raise `ValueError` from capacity parsing. Adding key-specific Redis namespace remediation hints requires threading key context through lower-level decode/parse helpers and belongs with a broader Redis corruption taxonomy. |

**Update (2026-06-10):** the unreleased lock-contention contract change
(`BackendLockContentionError` replacing raw `redis.exceptions.LockError`) makes
the next release a major version. That removes the "public API break" cost that
deferred L01:F23 — if the next release ships as a major, removing the
`UsageQuotas(..., _allow_empty_quotas=True)` escape hatch can ride along at
near-zero marginal cost.
