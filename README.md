# token-throttle

[![PyPI Version](https://img.shields.io/pypi/v/token-throttle?color=43cd0f&style=flat&label=pypi)](https://pypi.org/project/token-throttle)
[![Python Versions](https://img.shields.io/pypi/pyversions/token-throttle?color=43cd0f&style=flat&label=python)](https://pypi.org/project/token-throttle)
[![PyPI Downloads](https://img.shields.io/pypi/dm/token-throttle?color=43cd0f&style=flat&label=downloads)](https://pypistats.org/packages/token-throttle)
[![stability-beta](https://img.shields.io/badge/stability-beta-33bbff.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#beta)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-43cd0f.svg?style=flat&label=license)](LICENSE)
[![Maintained: yes](https://img.shields.io/badge/yes-43cd0f.svg?style=flat&label=maintained)](https://github.com/Elijas/token-throttle/issues)
[![CI](https://github.com/Elijas/token-throttle/actions/workflows/ci.yml/badge.svg)](https://github.com/Elijas/token-throttle/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Elijas/token-throttle/graph/badge.svg)](https://codecov.io/gh/Elijas/token-throttle)
[![Linter: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Multi-resource rate limiting for LLM APIs.** Reserve tokens before you call, refund what you don't use, stay under the limit across workers.

Works with any LLM provider and any client library — token-throttle limits the _rate_, not the _client_.

> **v2.0.0** is a breaking release. See [MIGRATION.md](MIGRATION.md) for the upgrade guide.

Public constants and type aliases are documented in [docs/api.md](docs/api.md).

```bash
pip install "token-throttle[redis,tiktoken]>=7.0.1,<7.1.0"   # OpenAI + Redis (recommended)
pip install "token-throttle[redis]>=7.0.1,<7.1.0"            # Any provider + Redis
pip install "token-throttle>=7.0.1,<7.1.0"                   # Any provider + in-memory
```

## Quickstart

### Memory quickstart (zero-service)

Copy-paste runnable.

```python
import asyncio

from token_throttle import MemoryBackendBuilder, PerModelConfig, Quota, RateLimiter, UsageQuotas


async def main() -> None:
    limiter = RateLimiter(
        PerModelConfig(
            quotas=UsageQuotas(
                [
                    Quota(metric="requests", limit=60, per_seconds=60),
                    Quota(metric="tokens", limit=90_000, per_seconds=60),
                ]
            )
        ),
        backend=MemoryBackendBuilder(),
    )

    reservation = await limiter.acquire_capacity(
        model="demo-model",
        usage={"requests": 1, "tokens": 1_000},
    )

    # Replace this block with your provider call.
    actual_usage = {"requests": 1, "tokens": 425}

    await limiter.refund_capacity(
        reservation=reservation,
        actual_usage=actual_usage,
    )

    second_reservation = await limiter.acquire_capacity(
        model="demo-model",
        usage={"requests": 1, "tokens": 250},
    )
    await limiter.refund_capacity(
        reservation=second_reservation,
        actual_usage={"requests": 1, "tokens": 250},
    )

    state = limiter.snapshot_state()
    assert state["in_flight_reservations"] == 0
    assert state["model_families"] == 1

    await limiter.aclose()
    print("reserved 1000 tokens, refunded 575 unused tokens")



asyncio.run(main())
```

### OpenAI (built-in helpers)

Install token-throttle's Redis and tokenizer extras plus the OpenAI client:

```bash
pip install "token-throttle[redis,tiktoken]" openai
```

```python
import asyncio

import redis.asyncio as redis
from openai import AsyncOpenAI
from token_throttle import create_openai_redis_rate_limiter


async def main() -> None:
    redis_client = redis.from_url("redis://localhost:6379")
    client = AsyncOpenAI()
    limiter = create_openai_redis_rate_limiter(
        redis_client,
        key_prefix="my-service-prod",
        rpm=10_000,
        tpm=2_000_000,
    )

    request = {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hi"}],
    }

    reservation = await limiter.acquire_capacity_for_request(**request)
    try:
        response = await client.chat.completions.create(**request)
    except Exception:
        await limiter.refund_capacity(
            reservation=reservation,
            actual_usage={"requests": 1, "input_tokens": 0, "output_tokens": 0},
        )
        raise
    else:
        await limiter.refund_capacity_from_response(reservation, response)
    finally:
        await limiter.aclose()
        await redis_client.aclose()


asyncio.run(main())
```

`OpenAIUsageCounter` supports text-only OpenAI payloads that use the real API
fields `input` or `messages`. The plural `inputs` field is not an OpenAI request
field and is rejected. Image, audio, and file inputs are unsupported; pass usage
manually for those.

Token estimates are bounded by `tiktoken`'s model encodings plus local
best-effort heuristics for chat/message overhead, tools, functions, schemas,
and output budgets. They have not been reconciled against live OpenAI billing
dashboards across a request corpus; periodically compare reserved tokens with
your actual billing and pass usage manually where local counting is not
representative.

### Any provider (manual usage)

```python
from token_throttle import PerModelConfig, Quota, RateLimiter, RedisBackendBuilder, UsageQuotas

limiter = RateLimiter(
    lambda model: PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric="requests", limit=1_000, per_seconds=60),
                Quota(metric="input_tokens", limit=80_000, per_seconds=60),
                Quota(metric="output_tokens", limit=20_000, per_seconds=60),
            ]
        ),
    ),
    backend=RedisBackendBuilder(redis_client, key_prefix="my-service-prod"),
)

# Works with Anthropic, Gemini, local models: anything with known usage.
reservation = await limiter.acquire_capacity(
    model="claude-sonnet-4-20250514",
    usage={"requests": 1, "input_tokens": 500, "output_tokens": 4_000},
)

response = await call_your_llm(...)  # Use whatever client you want.

await limiter.refund_capacity(
    actual_usage={"requests": 1, "input_tokens": 480, "output_tokens": 1_200},
    reservation=reservation,
)
# Unused 2,800 output tokens returned to the pool
```

## Why token-throttle

**The problem:** You're running parallel LLM calls (batch processing, agents, multiple services sharing a key). Simple rate limiters waste throughput because they reserve worst-case tokens and never give them back. You hit 429s or crawl at half capacity.

**The solution:** Reserve before you call, refund after. Actual usage is tracked, not estimated maximums.

| Feature | Details |
|---------|---------|
| **Multi-resource limits** | Limit requests, tokens, input/output tokens — simultaneously, each with its own quota |
| **Multiple time windows** | e.g., 1,000 req/min AND 10,000 req/day on the same resource |
| **Reserve & refund** | Reserve max expected usage upfront, refund the difference after the call completes |
| **Distributed** | Redis backend with atomic locks — safe across workers and processes |
| **Per-model quotas** | Different limits per model via `model_family`; the built-in OpenAI helper auto-groups date-suffixed variants (e.g. gpt-4o-20241203 → gpt-4o) |
| **Pluggable** | Bring your own backend (ships with Redis and in-memory). Sync and async APIs |
| **Observability** | Callbacks for wait-start, wait-end, consume, refund, and missing-state events |

## How it works

token-throttle implements a [token bucket](https://en.wikipedia.org/wiki/Token_bucket) algorithm (capacity refills linearly over time, capped at the quota limit).

- **Acquire** — blocks until enough capacity is available, then atomically reserves it
- **Call** — make your API request with any client
- **Refund** — report actual usage; unused tokens return to the pool immediately

The Redis backend uses sorted locking to prevent deadlocks when acquiring multiple resource buckets simultaneously.

### Reservation lifecycle

A `CapacityReservation` is an internal accounting token, not a durable portable
credential. Refund it on the same limiter lifetime that issued it, after the API
call finishes. If your config changes before refund, token-throttle refunds only
the surviving buckets that still correspond to the reservation; buckets removed
by a callable-config rebuild are skipped to avoid crediting unrelated capacity.

Unlimited reservations are no-ops on refund. They are trusted in-process
objects, so do not deserialize, pickle, or accept reservations across trust
boundaries as proof that a caller was rate-limited. For queue-and-retry
workflows, reserve immediately before dispatching the external request rather
than storing reservations in a long-lived queue.

v2.0.0 is a clean break from v1.4.x reservation compatibility. Every
`CapacityReservation` requires a non-empty `limiter_instance_id`; legacy
v1.4.x reservations without it are rejected. Drain in-flight reservations
before upgrading and do not run mixed v1.4.x/v2.0.0 fleets.

## Configuration

### Quotas

```python
from token_throttle import Quota, UsageQuotas, SecondsIn

quotas = UsageQuotas([
    Quota(metric="requests", limit=2_000, per_seconds=SecondsIn.MINUTE),
    Quota(metric="tokens", limit=3_000_000, per_seconds=SecondsIn.MINUTE),
    Quota(metric="requests", limit=10_000_000, per_seconds=SecondsIn.DAY),
])
```

`per_seconds` accepts integer seconds. Use `SecondsIn.MINUTE` (60), `SecondsIn.HOUR` (3600), `SecondsIn.DAY` (86400), or any integer.

### Per-model configuration

```python
def get_config(model_name: str) -> PerModelConfig:
    if model_name.startswith("gpt"):
        return PerModelConfig(
            quotas=UsageQuotas([
                Quota(metric="requests", limit=10_000, per_seconds=60),
                Quota(metric="tokens", limit=2_000_000, per_seconds=60),
            ]),
            usage_counter=OpenAIUsageCounter(),  # text-only: counts payload + instructions/tools/schema + output budget
            model_family=openai_model_family_getter(model_name),
        )
    # ... other providers

limiter = RateLimiter(
    get_config,
    backend=RedisBackendBuilder(redis_client, key_prefix="my-service-prod"),
)
```

Models that share a `model_family` must also share the same live quota definition. If two model names need different limits, give them different `model_family` values instead of reusing one family name.

`model_family` defaults to the request model name. A typo such as
`"gpt-4o-mini-prod"` therefore creates a distinct family. Limiters fail closed
once mandatory in-process caps are reached: by default 10,000 model families,
100 metrics per family, 10,000 model aliases, and 100,000 in-flight
reservations. Key-segment length is also capped by default: model families and
aliases at 256 characters, metrics at 64 characters. Tune these constructor
arguments lower for tighter deployments and validate model names in your
application if they come from users or configuration.

Long-lived dynamic deployments should periodically call
`limiter.clear_unused_model_families(unused_for_seconds)` (or the sync method
with the same name) from an operator-controlled maintenance path. It evicts idle
in-process family caches and skips families with in-flight reservations. Redis
bucket keys expire separately through the Redis bucket TTL.

To disable rate limiting for a model while keeping the same API surface, return
an unlimited config:

```python
PerModelConfig(
    quotas=UsageQuotas.unlimited(),
    model_family="paid-tier",
)
```

Unlimited configs still validate direct usage values for `acquire_capacity()`
but do not require usage keys to match quota names because there are no quotas.
For `acquire_capacity_for_request()`, a configured `usage_counter` may still run
for telemetry; any `extra_usage` keys are accepted and then discarded with the
unlimited reservation. If you toggle a model between limited and unlimited,
keep `extra_usage` keys compatible with the limited quota metrics.

`OpenAIUsageCounter` handles text-only OpenAI requests. It counts `input` or
`messages`, plus prompt-bearing request context such as `instructions`,
tool/function definitions, and structured output schemas. The plural `inputs`
field is rejected because it is not an OpenAI request field. Image/audio/file
inputs are still unsupported; pass usage manually for those.

Custom `usage_counter` callables receive the same kwargs you pass to
`acquire_capacity_for_request()`. They can accept `**request` for the whole
payload or only the named request fields they use; fixed-signature counters do
not need to accept unrelated kwargs like `model`.

### Backends

```python
# Distributed (multiple workers/processes)
from token_throttle import RedisBackendBuilder
backend = RedisBackendBuilder(redis_client, key_prefix="my-service-prod")

# Single process (no Redis needed)
from token_throttle import MemoryBackendBuilder
backend = MemoryBackendBuilder()
```

Both backends are available in sync (`SyncRedisBackendBuilder`, `SyncMemoryBackendBuilder`) and async variants.

Redis builders and Redis OpenAI factories require a non-empty `key_prefix`.
All Redis keys are scoped as `{key_prefix}:rate_limiting:...`; choose a stable
deployment-scoped value and share it across workers that intentionally share
quota state. Use different prefixes for unrelated deployments sharing one Redis
deployment. The prefix and user-controlled key segments cannot contain
`:`, `{`, `}`, whitespace, or control characters.

#### Redis topology support

token-throttle supports standalone Redis and Sentinel-managed Redis primary
deployments. It does not support Redis Cluster or client-side sharded Redis.
The Redis backend uses multi-key Lua scripts for atomic acquire-marker and
refund updates; in Redis Cluster those keys can span hash slots and fail at
runtime. `RateLimiter` and `SyncRateLimiter` reject redis-py `RedisCluster`
clients during construction with a `ValueError` instead of failing later during
`EVAL`.

Do not add caller-controlled Redis hash tags to key prefixes, model families,
metrics, reservation ids, or other key segments. Public validators reject `{`
and `}` in Redis key segments, and Cluster support is intentionally unsupported.

Redis backends require Redis server 6.2 or newer and a Redis user that can run
`GET`, `EXISTS`, `SET`, `DEL`, `EXPIRE`, `TIME`, and Lua scripting commands
used by redis-py locks and token-throttle acquire/refund transactions.

R7 validation used `fakeredis` plus local vanilla Redis 7.x. Redis 6.0/6.1 are
outside the supported version range, and the R7 matrix did not validate
Sentinel failover behavior, KeyDB, Dragonfly, client-side sharding, or low
`maxmemory` / low `maxclients` configurations. KeyDB and Dragonfly may work as
Redis-compatible servers, but they are untested and not officially supported;
validate topology and resource limits in your environment.

#### Multi-tenant deployments

`key_prefix` provides namespace isolation only. It keeps one tenant's Redis
keys from colliding with another tenant's keys, but it is not resource
isolation. Tenants that share a Redis server still share Redis CPU, memory,
`maxclients`, command scheduling, Lua script execution time, network bandwidth,
and eviction policy.

Do not rely on `key_prefix` for hostile-tenant fairness. A hostile or runaway
tenant on shared Redis can starve benign tenants by exhausting connections,
memory, CPU, or Lua scheduling. For hostile-tenant scenarios, use separate
Redis instances per tenant, or place Redis behind infrastructure that enforces
hardware-level CPU, memory, connection, and network quotas per tenant.
See `MIGRATION.md` for the v4 multi-tenant isolation decision.

For bounded Redis deployments, prefer `redis.asyncio.BlockingConnectionPool`
or `redis.BlockingConnectionPool` and size `max_connections` to at least
`max_concurrent_acquires` plus headroom for Redis lock acquire/release, `TIME`,
and pipeline commands. A pool below 10 connections triggers a runtime warning
because it is usually too small for production traffic.

Redis bucket state expires by default after 7 days of inactivity. Configure
`bucket_ttl_seconds` on Redis builders or Redis OpenAI factories to choose a
different positive TTL. The TTL is refreshed whenever bucket state is read or
written; Redis schema-version registry keys are intentionally long-lived and do
not expire.

Redis refunds also write a cross-process idempotency key:
`{key_prefix}:rate_limiting:refund_dedup:{reservation_id}`. The TTL defaults to
7 days and can be changed with `refund_dedup_ttl_seconds` on Redis backend
builders or Redis OpenAI factories. Memory backends keep only process-local
refund dedup state and cannot safely refund reservations after a cold restart.

#### Performance and capacity planning

R7 performance testing identified two important ceilings for 10k RPS-class
deployments:

- A single hot Redis bucket can become the throughput ceiling before 10k RPS
  because all callers contend on the same Redis lock and Lua-scripted state.
- The async memory backend is process-local and can also hit an in-process
  scheduling/lock ceiling before 10k RPS; it is not a horizontal scaling
  substitute for Redis.

Exact throughput and p99 latency depend on Redis CPU, network RTT, Python
runtime, concurrency, quota shape, and how concentrated traffic is on one
model family. Treat 10k RPS as a workload that requires staging benchmarks with
your real quota mix. These numbers are based on short local runs and sizing
estimates, not sustained-load production validation; a maintained production
benchmark suite is deferred. As a starting planning table:

| Sustained acquire/refund rate | Expected p99 driver | Operational guidance |
| ---: | --- | --- |
| 100 RPS | Redis RTT plus Python scheduling | Default Redis client pools usually work; still set timeouts and monitor waits. |
| 1k RPS | Lock contention, Redis CPU, pool wait | Use `BlockingConnectionPool`, provision Redis CPU headroom, and benchmark p99 under peak concurrency. |
| 10k RPS | Hot buckets, Lua scheduling, key churn | Avoid concentrating all traffic in one model family/window; use dedicated Redis capacity and load-test before production. |

Redis backends create short-lived per-reservation keys in addition to bucket
state. Size Redis from traffic rate and TTLs, not only from the number of model
families:

- Acquire marker keys, rough count:
  `acquires_per_second * max_reservation_lifetime_seconds`.
- Refund dedup keys, rough count:
  `refunds_per_second * refund_dedup_ttl_seconds`.

The Redis default TTLs are intentionally conservative for correctness and
compatibility: `bucket_ttl_seconds=604800` and
`refund_dedup_ttl_seconds=604800` (7 days). With no explicit
`max_reservation_lifetime_seconds`, Redis derives the reservation lifetime from
the shorter Redis TTL: just below
`min(bucket_ttl_seconds, refund_dedup_ttl_seconds) / 2`. At 10k acquires/sec,
the default derived marker lifetime is about 302,400 seconds, which can imply
roughly 3.0 billion marker keys. Tune these values for sustained high-RPS
deployments.

A practical starting point is to choose the smallest reservation lifetime that
covers normal request latency, retry delay, and shutdown drain time, then choose
`bucket_ttl_seconds` and `refund_dedup_ttl_seconds` longer than twice that
lifetime. Redis enforces this invariant:
`bucket_ttl_seconds > max_reservation_lifetime_seconds * 2` and
`refund_dedup_ttl_seconds > max_reservation_lifetime_seconds * 2`. The margin
keeps bucket state, acquire markers, and refund tombstones alive for the full
window in which a reservation can still be refunded.

Example budgets for a tuned deployment, assuming about 0.5-1.0 KB per marker or
dedup key after Redis object overhead and leaving operational headroom:

| Traffic | Example knobs | Approx keys | Suggested Redis memory budget |
| --- | --- | ---: | ---: |
| 1k acquire/refund RPS | `max_reservation_lifetime_seconds=300`, `refund_dedup_ttl_seconds=600`, `bucket_ttl_seconds=900` | 300k markers + 600k dedup keys | 1-2 GB |
| 10k acquire/refund RPS | `max_reservation_lifetime_seconds=300`, `refund_dedup_ttl_seconds=600`, `bucket_ttl_seconds=900` | 3M markers + 6M dedup keys | 10-20 GB |

Validate with `INFO memory`, `DBSIZE` or keyspace scans in staging because Redis
memory per key depends on key-prefix length, allocator behavior, and value size.

Sample Redis monitoring points:

```text
INFO commandstats   # eval/evalsha, set, get, del latency and call volume
INFO clients        # connected_clients, blocked_clients, maxclients pressure
INFO memory         # used_memory, mem_fragmentation_ratio, evicted_keys
INFO stats          # instantaneous_ops_per_sec, rejected_connections
LATENCY LATEST      # server-side latency spikes
SLOWLOG GET 128     # slow Lua scripts or lock commands
```

Application-side monitoring should track acquire wait duration, timeout count,
callback errors, `snapshot_state()["in_flight_reservations"]`, Redis pool wait
time, and p50/p95/p99 latency for acquire and refund calls.

Custom backends implement `RateLimiterBackend` or `SyncRateLimiterBackend`.
Required operations are capacity wait/consume/refund and `set_max_capacity`.
Optional extension points include `refund_capacity_for_buckets`,
`apply_configured_max_capacity`, `supports_metric_set_change`, and
`prepare_reconfigured_backend`. See [`docs/custom-backends.md`](docs/custom-backends.md)
for the protocol contract and conformance helper.

Leave `supports_metric_set_change()` as `False` unless bucket additions/removals
can preserve live state for surviving metrics. To return `True`, either keep
bucket state in stable external storage keyed by metric/window, or override
`prepare_reconfigured_backend()` to migrate in-process state into the rebuilt
backend. Returning `True` with a no-op migration can silently reset accounting.

### Dynamic rate limits

Adjust bucket limits at runtime without rebuilding the limiter — useful for
adaptive rate limiting (e.g., reacting to `x-ratelimit-*` response headers):

```python
# After at least one acquire/record call for this model:
await limiter.set_max_capacity(
    model="gpt-4o",
    metric="tokens",
    per_seconds=60,
    value=5000,
)
```

For Redis backends the new limit is written to Redis, so all processes
sharing the same Redis see the change within ~1 second. This persisted Redis
value is an explicit runtime override; static quota changes from your config do
not rewrite it automatically.

If a callable config removes a bucket and later re-adds it, the re-added
bucket starts from the static quota in the current config. Runtime overrides
from earlier `set_max_capacity()` calls do not survive a remove-and-readd;
call `set_max_capacity()` again if you want the override restored.

If you want to change the static configured quota, update the callable config
and let the limiter rebuild on the next acquire/refund. `set_max_capacity()` is
an explicit runtime override, not a config edit. Config rotations concurrent
with `set_max_capacity()` are ordered by whichever backend update completes
last; a later config rebuild resolves back to the static quota unless you
reapply the override.

### Timeout

By default, `acquire_capacity` blocks until enough capacity is available.
Use `timeout` to fail fast or cap the capacity wait:

```python
# Non-blocking: check if capacity is available without waiting
try:
    reservation = await limiter.acquire_capacity(
        model="gpt-4o",
        usage={"requests": 1, "tokens": 500},
        timeout=0,  # Fail immediately if no capacity
    )
except TimeoutError:
    # Handle: retry later, use cheaper model, skip, etc.
    pass

# Bounded wait: wait up to 5 seconds
reservation = await limiter.acquire_capacity(
    model="gpt-4o",
    usage={"requests": 1, "tokens": 500},
    timeout=5.0,  # Raise TimeoutError after 5s waiting for capacity
)
```

`timeout` is not a total wall-clock deadline: backend operation latency
(including Redis round trips) is outside this budget.

User callbacks are bounded separately by `callback_timeout` on `RateLimiter`
and `SyncRateLimiter` (default: 30 seconds per callback). When a callback
exceeds that limit, token-throttle logs a warning, skips the callback result,
and does not fail the acquire/refund call. Pass `callback_timeout=None` to
restore unbounded callback execution. Timeout-wrapped sync callbacks run in a
helper thread with the caller's `contextvars` context copied into that thread.

## Observability

token-throttle stays framework-agnostic: it exposes logging, callbacks, and a
small health snapshot, but does not depend on Prometheus, OpenTelemetry, or any
metrics SDK. Wire these surfaces to your own collectors.

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

Use `snapshot_state()` for a redacted point-in-time health check:

```python
state = limiter.snapshot_state()
# {
#     "in_flight_reservations": 3,
#     "model_families": 2,
#     "backend_type": "redis",
#     "marker_count_estimate": 3,
#     "refund_dedup_count_estimate": 120,
# }
```

For Redis backends, marker and refund-dedup counts are best-effort local
estimates from limiter bookkeeping, not a cross-process Redis inventory. The
snapshot intentionally omits Redis URLs, credentials, and Redis key prefixes.

For request correlation without changing existing callback signatures, use the
additive lifecycle callback:

```python
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

Lifecycle events include `event_type`, `reservation_id`, optional `request_id`
from `acquire_capacity_for_request(..., request_id="...")`, `model_family`,
`model_alias`, `bucket_ids`, `usage`, and `timestamp`. Existing wait, consume,
refund, and missing-data callbacks keep their original keyword signatures.
Public token-throttle exception classes expose a stable `reason` attribute
where callers need structured error handling.

PII surface:

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

## Sync API

```python
from token_throttle import SyncRateLimiter, SyncMemoryBackendBuilder

limiter = SyncRateLimiter(get_config, backend=SyncMemoryBackendBuilder())

reservation = limiter.acquire_capacity(model="gpt-4.1", usage={"requests": 1, "tokens": 500})
response = call_llm_sync(...)
limiter.refund_capacity(actual_usage={"requests": 1, "tokens": 320}, reservation=reservation)
```

## Concurrency Model

Create one limiter per process and, for the async API, one limiter per event
loop. `RateLimiter` and `SyncRateLimiter` own in-process locks, lifecycle
state, and backend builders; they are not pickleable and should be constructed
inside each worker process after `fork()` or `spawn()`. By default they check
process affinity on every public method and raise if a limiter is reused in a
different PID. Pass `pid_check=False` only when you deliberately accept the
unsupported risk of divergent in-memory state.

Use `RateLimiter` from async code and `SyncRateLimiter` from synchronous code.
Calling `SyncRateLimiter.acquire_capacity()` while an event loop is running
blocks that loop; token-throttle emits a `RuntimeWarning` once per process.

Both limiter types can own their close lifecycle through context managers:

```python
async with RateLimiter(get_config, backend=MemoryBackendBuilder()) as limiter:
    reservation = await limiter.acquire_capacity({"tokens": 500}, model="gpt-4.1")
    await limiter.refund_capacity({"tokens": 320}, reservation)
```

```python
with SyncRateLimiter(get_config, backend=SyncMemoryBackendBuilder()) as limiter:
    reservation = limiter.acquire_capacity({"tokens": 500}, model="gpt-4.1")
    limiter.refund_capacity({"tokens": 320}, reservation)
```

## Links

- Originally a rewrite of [openlimit](https://github.com/shobrook/openlimit)

![GitHub Repo stars](https://img.shields.io/github/stars/elijas/token-throttle?style=flat&color=fcfcfc&labelColor=white&logo=github&logoColor=black&label=stars)
