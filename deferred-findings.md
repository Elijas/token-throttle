# FIX-22 Deferred Long-Tail Findings

These R4 long-tail items were verified but left out of the code patch because
closing them cleanly would change public API shape or require broader Redis
corruption/error taxonomy work than this closure bundle should take on.

| Finding | Deferral |
|---|---|
| L01:F23 | Removing the private-but-reachable `UsageQuotas(..., _allow_empty_quotas=True)` constructor escape hatch is a public API break. Current guidance points users to `UsageQuotas.unlimited()` and empty `UsageQuotas([])` already raises clearly. |
| L17:Q03 | Redis corrupted-value diagnostics still raise `ValueError` from capacity parsing. Adding key-specific Redis namespace remediation hints requires threading key context through lower-level decode/parse helpers and belongs with a broader Redis corruption taxonomy. |
