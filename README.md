# token-throttle

[![PyPI Version](https://img.shields.io/badge/v1.2.0-version?color=43cd0f&style=flat&label=pypi)](https://pypi.org/project/token-throttle)
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
pip install "token-throttle[redis,tiktoken]>=1.2.0,<1.3.0"   # OpenAI + Redis (recommended)
pip install "token-throttle[redis]>=1.2.0,<1.3.0"            # Any provider + Redis
pip install "token-throttle>=1.2.0,<1.3.0"                   # Any provider + in-memory
```

## Quickstart

### OpenAI (built-in helpers)

```python
from openai import AsyncOpenAI
from token_throttle import create_openai_redis_rate_limiter

client = AsyncOpenAI()
limiter = create_openai_redis_rate_limiter(
    redis_client, rpm=10_000, tpm=2_000_000,
)

# 1. Reserve capacity (blocks until available)
request = dict(model="gpt-4.1", messages=[{"role": "user", "content": "Hi"}])
reservation = await limiter.acquire_capacity_for_request(**request, extra_usage=None)

# 2. Make the API call
response = await client.chat.completions.create(**request)

# 3. Refund unused tokens
await limiter.refund_capacity_from_response(reservation, response)
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
    backend=RedisBackendBuilder(redis_client),
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
            usage_counter=OpenAIUsageCounter(),  # auto-counts tokens from messages
            model_family=openai_model_family_getter(model_name),
        )
    # ... other providers

limiter = RateLimiter(get_config, backend=RedisBackendBuilder(redis_client))
```

### Backends

```python
# Distributed (multiple workers/processes)
from token_throttle import RedisBackendBuilder
backend = RedisBackendBuilder(redis_client)

# Single process (no Redis needed)
from token_throttle import MemoryBackendBuilder
backend = MemoryBackendBuilder()
```

Both backends are available in sync (`SyncRedisBackendBuilder`, `SyncMemoryBackendBuilder`) and async variants.

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
sharing the same Redis see the change within ~1 second.

### Timeout

By default, `acquire_capacity` blocks until enough capacity is available.
Use `timeout` to fail fast or cap the wait:

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
    timeout=5.0,  # Raise TimeoutError after 5s
)
```

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
