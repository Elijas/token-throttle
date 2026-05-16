# W1 R8 Deferred Documentation Findings

## Item 1: Redis ambiguous-commit realism

Added the deferred real-proxy decision and rationale to the testing methodology
docs: FIX-45 verifies reconciliation with deterministic fake-client lost-`EVAL`
reply tests, while real TCP proxy ACK-drop and real server crash-mid-`EVAL`
coverage remain operator validation.

- `DEVELOPMENT.md:35`
- `tests/unit/test_fix_45_redis_marker_reconciliation.py:1`

## Item 2: Real-server topology matrix

Extended the Redis topology docs to say R7 covered `fakeredis` plus local
vanilla Redis 7.x only. Redis 6.0/6.1 are outside the supported 6.2+ range, and
Sentinel failover behavior, KeyDB, Dragonfly, client-side sharding, and low
`maxmemory` / low `maxclients` deployments require operator-side validation.

- `README.md:347`
- `MIGRATION.md:64`

## Item 3: Production sizing assumptions

README already had capacity-planning tables and staging-validation guidance.
Added the missing explicit disclaimer that the numbers are from short local R7
runs and estimates, not sustained production validation, and that a maintained
production-load benchmark suite is deferred.

- `README.md:398`
- `DEVELOPMENT.md:276`

## Item 4: OpenAI / token-counter accuracy

Added the OpenAI counter accuracy disclaimer in the README and token-counter
source: estimates are bounded by `tiktoken` plus local heuristics, and live
billing reconciliation is deferred to periodic operator sanity checks.

- `README.md:141`
- `token_throttle/_factories/_openai/_token_counter.py:1`
- `token_throttle/_factories/_openai/_token_counter.py:81`

## Item 5: Hostile-tenant Redis isolation

README already stated that `key_prefix` is namespace-only and not
hostile-tenant fairness. Added the migration-guide decision text spelling out
that hostile tenants require deployment-layer isolation such as separate Redis
instances or quota-aware infrastructure, plus a README pointer to that
migration decision.

- `MIGRATION.md:223`
- `README.md:362`

## Item 6: Full Redis Cluster redesign

Extended the Cluster migration section and inline reject-at-construction code
comments with the rationale: Cluster support needs hash-tagged public key
shape, per-shard Lua, Cluster client handling, and cross-shard transaction
semantics, so partial hash-tag changes are not enough.

- `MIGRATION.md:58`
- `token_throttle/_rate_limiter.py:99`
- `token_throttle/_sync_rate_limiter.py:97`
