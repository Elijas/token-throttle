# token-throttle

[![PyPI Version](https://img.shields.io/badge/v1.5.0-version?color=43cd0f&style=flat&label=pypi)](https://pypi.org/project/token-throttle)
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
pip install "token-throttle[redis,tiktoken]>=1.5.0,<1.6.0"   # OpenAI + Redis (recommended)
pip install "token-throttle[redis]>=1.5.0,<1.6.0"            # Any provider + Redis
pip install "token-throttle>=1.5.0,<1.6.0"                   # Any provider + in-memory
```

## Quickstart

### OpenAI (built-in helpers)

```python
import redis.asyncio as redis
from openai import AsyncOpenAI
from token_throttle import create_openai_redis_rate_limiter

redis_client = redis.from_url("redis://localhost:6379")
client = AsyncOpenAI()
limiter = create_openai_redis_rate_limiter(
    redis_client,
    key_prefix="my-service-prod",
    rpm=10_000,
    tpm=2_000_000,
)

# 1. Reserve capacity (blocks until available)
request = dict(model="gpt-4.1", messages=[{"role": "user", "content": "Hi"}])
reservation = await limiter.acquire_capacity_for_request(**request)

# 2. Make the API call
response = await client.chat.completions.create(**request)

# 3. Refund unused tokens
await limiter.refund_capacity_from_response(reservation, response)
# Or, when you already separated the usage object:
# await limiter.refund_capacity_from_response(reservation, usage=response.usage)
```

### Any provider (manual usage)

```python
from token_throttle import RateLimiter, Quota, UsageQuotas, RedisBackendBuilder
from token_throttle import PerModelConfig

limiter = RateLimiter(
    lambda model: PerModelConfig(
        quotas=UsageQuotas([
            Quota(metric="requests", limit=1_000, per_seconds=60),
            Quota(metric="input_tokens", limit=80_000, per_seconds=60),
            Quota(metric="output_tokens", limit=20_000, per_seconds=60),
        ]),
    ),
    backend=RedisBackendBuilder(redis_client, key_prefix="my-service-prod"),
)

# Works with Anthropic, Gemini, local models — anything
reservation = await limiter.acquire_capacity(
    model="claude-sonnet-4-20250514",
    usage={"requests": 1, "input_tokens": 500, "output_tokens": 4_000},
)

response = await call_your_llm(...)  # Use whatever client you want

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

`OpenAIUsageCounter` handles text-only OpenAI requests. It counts `input`,
`inputs`, or `messages`, plus prompt-bearing request context such as
`instructions`, tool/function definitions, and structured output schemas.
Image/audio/file inputs are still unsupported; pass usage manually for those.

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
DB or Redis Cluster. The prefix and user-controlled key segments cannot contain
`:`, `{`, `}`, whitespace, or control characters.

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
builders. Memory backends keep only process-local refund dedup state and cannot
safely refund reservations after a cold restart.

Custom backends implement `RateLimiterBackend` or `SyncRateLimiterBackend`.
Required operations are capacity wait/consume/refund and `set_max_capacity`.
Optional extension points include `refund_capacity_for_buckets`,
`apply_configured_max_capacity`, `supports_metric_set_change`, and
`prepare_reconfigured_backend`.

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
restore unbounded callback execution.

## Sync API

```python
from token_throttle import SyncRateLimiter, SyncMemoryBackendBuilder

limiter = SyncRateLimiter(get_config, backend=SyncMemoryBackendBuilder())

reservation = limiter.acquire_capacity(model="gpt-4.1", usage={"requests": 1, "tokens": 500})
response = call_llm_sync(...)
limiter.refund_capacity(actual_usage={"requests": 1, "tokens": 320}, reservation=reservation)
```

## Links

- Originally a rewrite of [openlimit](https://github.com/shobrook/openlimit)

![GitHub Repo stars](https://img.shields.io/github/stars/elijas/token-throttle?style=flat&color=fcfcfc&labelColor=white&logo=github&logoColor=black&label=stars)
