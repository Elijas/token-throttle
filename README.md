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

```bash
pip install "token-throttle[redis,tiktoken]>=9.1.0,<10.0.0"   # OpenAI + Redis (recommended)
pip install "token-throttle[redis]>=9.1.0,<10.0.0"            # Any provider + Redis
pip install "token-throttle>=9.1.0,<10.0.0"                   # Any provider + in-memory
```

Requires Python 3.12+. The Redis backend requires Redis 6.2+, standalone or Sentinel only — not Redis Cluster or client-side sharding (see [docs/operations.md](docs/operations.md)).

token-throttle follows strict semver: breaking changes ship only as major versions, and several recent majors were correctness hardening found through fault-injection testing rather than churn. Pin an exact major range (as shown above) and read [MIGRATION.md](MIGRATION.md) before upgrading — see it for the v2/v5/v6/v7/v8/v9 contract changes. Public constants and type aliases: [docs/api.md](docs/api.md).

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

    async with limiter.reserve(
        {"requests": 1, "tokens": 1_000},
        model="demo-model",
    ) as handle:
        # Replace this block with your provider call.
        handle.set_actual_usage({"requests": 1, "tokens": 425})

    state = limiter.snapshot_state()
    assert state["in_flight_reservations"] == 0
    assert state["model_families"] == 1

    await limiter.aclose()
    print("reserved 1000 tokens, refunded 575 unused tokens")



asyncio.run(main())
```

`reserve()` (and `SyncRateLimiter.reserve()`) is the recommended wrapper over
the acquire -> call -> refund cycle. It refunds the unused remainder on exit; on
an exception it refunds `usage_on_error` (or, if omitted, the full reservation)
and re-raises. `handle.reservation` exposes the underlying `CapacityReservation`;
if the block exits without `set_actual_usage` it conservatively refunds the full
reserved usage and emits a `RuntimeWarning`. Full contract: the `reserve()`
docstring and [docs/operations.md](docs/operations.md#reservation-lifecycle-and-durability).
For explicit acquire/refund control, see the [Any-provider quickstart](#any-provider-manual-usage)
below.

### OpenAI (built-in helpers)

`create_openai_redis_rate_limiter` (and its sync/memory variants) is a
convenience adapter over `RateLimiter`: it wires up `OpenAIUsageCounter`, a
best-effort local token counter, so you don't have to write your own
`usage_counter` for OpenAI requests. Use the "Any provider" pattern below (or
a custom `usage_counter`) when you need exact provider-side counts instead.

Install token-throttle's Redis and tokenizer extras plus the OpenAI client:

```bash
pip install "token-throttle[redis,tiktoken]>=9.1.0,<10.0.0" openai
```

```python
# (fragment — needs a live Redis + OPENAI_API_KEY; see the Memory quickstart to run end-to-end)
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
        "max_tokens": 512,  # output budget; without one, output tokens aren't reserved — see docs/configuration.md
    }

    reservation = await limiter.acquire_capacity_for_request(**request)
    try:
        response = await client.chat.completions.create(**request)
    except Exception:
        await limiter.refund_capacity(
            reservation=reservation,
            actual_usage={"requests": 1, "tokens": 0},  # zero-token refund on error is an approximation; reconcile against billing
        )
        raise
    else:
        await limiter.refund_capacity_from_response(reservation, response)
    finally:
        await limiter.aclose()
        await redis_client.aclose()


asyncio.run(main())
```

This uses the manual acquire/refund pattern instead of `reserve()`: `refund_capacity_from_response` already inspects the OpenAI response to compute actual usage and refund it, so it does the bookkeeping `reserve()` would otherwise provide.

`OpenAIUsageCounter` counts text-only OpenAI payloads (`input` or `messages`,
plus chat/tool/schema/output overhead via `tiktoken`). The plural `inputs` field
and image/audio/file inputs are unsupported — pass usage manually for those.
Unrecognized model names raise a clear error directing you to pass a custom
`get_encoding_func`. Estimates are best-effort and not reconciled against live
billing, so compare reserved tokens with actual usage periodically. Full
contract: [docs/configuration.md](docs/configuration.md#usage-counters).

### Any provider (manual usage)

The manual acquire -> refund pattern that `reserve()` wraps, with explicit error handling:

```python
import asyncio

from token_throttle import MemoryBackendBuilder, PerModelConfig, Quota, RateLimiter, UsageQuotas


async def call_your_llm() -> dict[str, int]:
    return {"requests": 1, "input_tokens": 480, "output_tokens": 1_200}


async def main() -> None:
    limiter = RateLimiter(
        PerModelConfig(
            quotas=UsageQuotas(
                [
                    Quota(metric="requests", limit=1_000, per_seconds=60),
                    Quota(metric="input_tokens", limit=80_000, per_seconds=60),
                    Quota(metric="output_tokens", limit=20_000, per_seconds=60),
                ]
            ),
        ),
        backend=MemoryBackendBuilder(),
    )

    reservation = await limiter.acquire_capacity(
        model="claude-sonnet-4-20250514",
        usage={"requests": 1, "input_tokens": 500, "output_tokens": 4_000},
    )

    try:
        actual_usage = await call_your_llm()
    except Exception:
        # Refund a stranded reservation on failure so unused capacity isn't
        # locked out until the quota window refills.
        await limiter.refund_capacity(
            actual_usage={"requests": 1, "input_tokens": 0, "output_tokens": 0},
            reservation=reservation,
        )
        raise
    else:
        await limiter.refund_capacity(actual_usage=actual_usage, reservation=reservation)
    finally:
        await limiter.aclose()
    print("unused 20 input tokens and 2800 output tokens returned to the pool")


asyncio.run(main())
```

## Why token-throttle

**The problem:** You're running parallel LLM calls (batch processing, agents, multiple services sharing a key). Simple rate limiters waste throughput because they reserve worst-case tokens and never give them back. You hit 429s or crawl at half capacity.

**The solution:** Reserve before you call, refund after. Actual usage is tracked, not estimated maximums.

Why not a semaphore or a generic rate limiter? Those cap request count or assume a fixed worst-case token cost and never return unused capacity, so you either overprovision (crawl at half throughput) or underprovision (429s). token-throttle reserves your declared maximum, then refunds the delta from actual usage — across multiple simultaneous resources (requests, tokens, input/output), coordinated across processes via Redis.

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

Reserve before the call, refund after — on the **same limiter** that issued the
reservation, immediately around the external request (not from a long-lived
queue). A `CapacityReservation` is a trusted in-process accounting token, not a
portable credential: don't pickle it or pass it across trust boundaries.
Durability semantics, config-change behavior, and the v2.0.0 compatibility break
are covered in [docs/operations.md](docs/operations.md#reservation-lifecycle-and-durability).

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
# (fragment — see Quotas example for context)
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

Limiters fail closed at sensible in-process caps (model families, metrics,
aliases, in-flight reservations) and support unlimited configs, custom
`usage_counter` callables, and idle-family eviction for long-lived deployments.
See [docs/configuration.md](docs/configuration.md#per-model-configuration).

### Backends

```python
# (fragment — see Memory quickstart for standalone context)
# Distributed (multiple workers/processes)
from token_throttle import RedisBackendBuilder
backend = RedisBackendBuilder(redis_client, key_prefix="my-service-prod")

# Single process (no Redis needed)
from token_throttle import MemoryBackendBuilder
backend = MemoryBackendBuilder()
```

Both backends are available in sync (`SyncRedisBackendBuilder`, `SyncMemoryBackendBuilder`) and async variants.

Custom backends implement `RateLimiterBackend` or `SyncRateLimiterBackend`. See
[docs/custom-backends.md](docs/custom-backends.md) for the protocol contract and
conformance helper.

Redis builders and Redis OpenAI factories require a non-empty `key_prefix`.
All Redis keys are scoped as `{key_prefix}:rate_limiting:...`; choose a stable
deployment-scoped value and share it across workers that intentionally share
quota state. Use different prefixes for unrelated deployments sharing one Redis
deployment. The prefix and user-controlled key segments cannot contain
`:`, `{`, `}`, whitespace, or control characters.

### Running in production (Redis)

Distributed deployments have operational considerations worth reading before you
ship: supported Redis topologies (standalone and Sentinel — **not** Redis
Cluster or client-side sharding), multi-tenant isolation limits, connection-pool
sizing, key TTLs, capacity planning for high-RPS fleets, and how the library
behaves on Redis data loss and raw connection errors. See
[docs/operations.md](docs/operations.md).

### Dynamic rate limits

Adjust bucket limits at runtime without rebuilding the limiter — useful for
adaptive rate limiting (e.g., reacting to `x-ratelimit-*` response headers):

```python
# (fragment — see Any provider example for standalone context)
# After the limiter has initialized this model with an acquire call:
await limiter.set_max_capacity(
    model="gpt-4o",
    metric="tokens",
    per_seconds=60,
    value=5000,
)
```

For Redis backends the new limit is written to Redis, so all processes
sharing the same Redis see the change within ~1 second.

Runtime-override semantics — Redis propagation, remove-and-readd behavior, and
ordering against concurrent config rotations — are covered in
[docs/configuration.md](docs/configuration.md#dynamic-rate-limits).

### Timeout

By default, `acquire_capacity` blocks until enough capacity is available.
Use `timeout` to fail fast or cap the capacity wait:

```python
# (fragment — see Any provider example for standalone context)
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

User callbacks are bounded separately by `callback_timeout` (default 30s); see
[docs/observability.md](docs/observability.md#callback-timeouts).

## Observability

token-throttle stays framework-agnostic: it exposes logging, callbacks, and a
small health snapshot, but does not depend on Prometheus, OpenTelemetry, or any
metrics SDK. Wire these surfaces to your own collectors.

Use `snapshot_state()` for a redacted point-in-time health check:

```python
# (fragment — see Any provider example for standalone context)
state = limiter.snapshot_state()
# {
#     "in_flight_reservations": 3,
#     "model_families": 2,
#     "backend_type": "redis",
#     "marker_count_estimate": 3,
#     "refund_dedup_count_estimate": 120,
# }
```

For request correlation without changing existing callback signatures, wire the
additive `on_lifecycle_event` callback on `RateLimiterCallbacks`. Full example
and event-field reference: [docs/observability.md](docs/observability.md#lifecycle-events).

Debug loggers (`token_throttle.acquire` / `.refund` / `.lock`), lifecycle event
fields, `snapshot_state()` estimate semantics, callback timeouts, and the full
PII surface are documented in [docs/observability.md](docs/observability.md).

## Sync API

```python
from token_throttle import (
    PerModelConfig,
    Quota,
    SyncMemoryBackendBuilder,
    SyncRateLimiter,
    UsageQuotas,
)

limiter = SyncRateLimiter(
    PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric="requests", limit=60, per_seconds=60),
                Quota(metric="tokens", limit=90_000, per_seconds=60),
            ]
        )
    ),
    backend=SyncMemoryBackendBuilder(),
)

try:
    reservation = limiter.acquire_capacity(
        model="demo-model",
        usage={"requests": 1, "tokens": 500},
    )
    limiter.refund_capacity(
        actual_usage={"requests": 1, "tokens": 320},
        reservation=reservation,
    )
    assert limiter.snapshot_state()["in_flight_reservations"] == 0
finally:
    limiter.close()
```

## Concurrency Model

Create one limiter per process and, for the async API, one per event loop. Use
`RateLimiter` from async code and `SyncRateLimiter` from sync code; both are not
pickleable and must be constructed inside each worker after `fork()`/`spawn()`.
Both support context-manager close:

```python
# (fragment — see Memory quickstart for standalone context)
async with RateLimiter(get_config, backend=MemoryBackendBuilder()) as limiter:
    reservation = await limiter.acquire_capacity({"requests": 1, "tokens": 500}, model="gpt-4.1")
    await limiter.refund_capacity({"requests": 1, "tokens": 320}, reservation)
```

Process-affinity (`pid_check`) semantics, fork/spawn requirements including the
`redis_client`, the blocking-event-loop warning, and graceful-shutdown draining
are covered in [docs/operations.md](docs/operations.md#concurrency-model).

## Documentation

- [docs/api.md](docs/api.md) — public constants and type aliases
- [docs/configuration.md](docs/configuration.md) — per-model caps, unlimited configs, custom usage counters, dynamic limits
- [docs/operations.md](docs/operations.md) — reservation durability, concurrency model, Redis topology, multi-tenant isolation, capacity planning, application-facing errors
- [docs/observability.md](docs/observability.md) — logging, lifecycle events, health snapshots, PII surface
- [docs/custom-backends.md](docs/custom-backends.md) — implement your own backend
- [MIGRATION.md](MIGRATION.md) — breaking-change upgrade guides
- [CHANGELOG.md](CHANGELOG.md) — release history

## Links

- Originally a rewrite of [openlimit](https://github.com/shobrook/openlimit)

![GitHub Repo stars](https://img.shields.io/github/stars/elijas/token-throttle?style=flat&color=fcfcfc&labelColor=white&logo=github&logoColor=black&label=stars)
