# token-throttle

[![PyPI Version](https://img.shields.io/badge/v0.4.2-version?color=43cd0f&style=flat&label=pypi)](https://pypi.org/project/token-throttle)
[![PyPI Downloads](https://img.shields.io/pypi/dm/token-throttle?color=43cd0f&style=flat&label=downloads)](https://pypistats.org/packages/token-throttle)
[![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-43cd0f.svg?style=flat&label=license)](LICENSE)
[![Maintained: yes](https://img.shields.io/badge/yes-43cd0f.svg?style=flat&label=maintained)](https://github.com/Elijas/token-throttle/issues)
[![CI](https://github.com/Elijas/token-throttle/actions/workflows/ci.yml/badge.svg)](https://github.com/Elijas/token-throttle/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Elijas/token-throttle/graph/badge.svg)](https://codecov.io/gh/Elijas/token-throttle)
[![Linter: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

```python
def get_quota_per_model(model_name: str):
    if model_name == "gemini-2.5-pro":
        return UsageQuotas([
            Quota(metric="requests", limit=2_000, per_seconds=60),
            Quota(metric="tokens", limit=3_000_000, per_seconds=60),
            Quota(metric="requests", limit=10_000_000, per_seconds=86400),
        ])
    if model_name == "gemini-2.5-flash":
        return ...
    

gemini_rate_limiter = RateLimiter(
    get_quota_per_model,
    backend=RedisBackendBuilder(redis_client),
)
```

**Simple Multi-Resource Rate Limiting That Saves Unused Tokens.**

Rate limit API requests across different resources and workers without wasting your quota. Reserve tokens upfront, get refunds for what you don't use, and avoid over-limiting.

- Limits requests across multiple services (like OpenAI, Anthropic)
- Works across multiple servers/workers
- Returns unused tokens to your quota automatically
- Prevents hitting API rate limits while maximizing throughput
- Robust against race-conditions through the use Redis-locked atomic operations. Note: you can bring your own backend if you don't want to use Redis.
- Implements the [generic cell rate algorithm,](https://en.wikipedia.org/wiki/Generic_cell_rate_algorithm) a variant of the leaky bucket pattern with a millisecond precision.

Note: the API may unexpectedly change with future minor versions, therefore install with:

```bash
pip install "token-throttle[redis,tiktoken]>=0.4.2,<0.5.0"
```

Found this useful? Star the repo on GitHub to show support and follow for updates. Also, find me on Discord if you have questions or would like to join a discussion!

![GitHub Repo stars](https://img.shields.io/github/stars/elijas/token-throttle?style=flat&color=fcfcfc&labelColor=white&logo=github&logoColor=black&label=stars)
&nbsp;<a href="https://discord.gg/hCppPqm6"><img alt="Discord server invite" src="https://img.shields.io/discord/1119368998161752075?logo=discord&logoColor=white&style=flat&color=fcfcfc&labelColor=7289da" height="20"></a>

### Introduction

This is a tool I built as a rewrite of [openlimit](https://github.com/shobrook/openlimit/issues/20#issuecomment-2782677483) after [not finding any good Python solutions for rate limiting](https://gist.github.com/justinvanwinkle/d9f04950083c4554835c1a35f9d22dad), especially the ones that would be token-aware and had unused token refund capability.

- Rate-limit multiple resources such as requests and tokens and apples and bananas at the same time
  - This is needed because different APIs have different resource rules, e,g, Anthropic counts request and completion tokens separately.
  - While this was originally intended for LLM APIs, it's fully customizable: you can limit bananas per 32-second-time-windows and apples per 2-minute-window simultaneously. You can also connect (through Dependency Injection) your own backend if you don't want Redis.
- Rate-limit multiple resource consumers (such as LLM calling applications that are using the same API key and model).
- Rate-limit same resource across multiple time-frames
- Rate-limit each resource on it's own set of quotas
- Reserve usage while the request is being completed, and then refund/adjust according to actual usage after the request completes
- Refund unused resources (such as unused tokens).

Treat this as an early preview (no unit tests or extensive testing) but it was stable and worked correctly for my use cases.

### Illustrating use case

- Imagine you have a single API key to your provider.
- Only up to 10% of the key's throughput capacity is used by a continuously running production service.
- You want to run a massively parallelized LLM data processing workflow with the same key.
- But you need to do it without bringing down the production (or your workflow) with 429 Too Many Requests errors.
- Also, leaving latency on the table is not a good solution; you want the batch results as soon as possible.

> [!NOTE]
> Note, this example uses [BAML](https://github.com/BoundaryML/baml) to call LLMs, but you can use absolutely anything, because all you need is just a way to retrieve tokens used in the request and actual tokens used in the response. Also note that (optionally) you can use already existing utilities in `token-throttle` to calculate/extract these two values automatically from OpenAI-compatible requests and responses.

```python
from baml_client import b
from baml_py import Collector
token_counter = Collector()
b = b.with_options(token_counter)

limiter = create_limiter([
    # Let's say your production only uses up to 10% of
    # Then this should be set to 90% of your capacity
    Quota(metric="requests", limit=90_000, per_seconds=60),
    Quota(metric="tokens", limit=90_000_000, per_seconds=60),
], backend=redis)

async def massively_parallelized():
    input_tokens = inp_tok(await b.request.ExtractResume(...))
    # e.g. max_tokens value of the request
    # e.g. or 95th percentile of usual b.ExtractResume() consumption
    output_tokens = 10_000

    # Safe against race-condition and many clients
    # because it uses Redis locks and atomic operations
    reservation = await limiter.acquire_capacity(
        model="gpt-4.1"
        usage={
            "requests": 1,
            "tokens": input_tokens + output_tokens,

            # Anthropic input and output tokens
            # have separate rate limits:
            #   "input_tokens": input_tokens
            #   "output_tokens": output_tokens

        },
    )

    # Request only continues here only after the capacity
    # has been reserved to not be consumed by any other LLM calls
    c = Collector()
    b = b.with_options(collector=c)
    resume = await b.ExtractResume(...)
    actual_usage = {
        "requests": get_total_tokens(c),
        "tokens": 1,
    }
    await limiter.refund_capacity(actual_usage, reservation)
    # Now two things happened:
    # 1. Actual usage recorded
    #    (e.g. got a capacity refund for unused output tokens)
    #    (e.g. a negative refund is also possible if actual usage exceeded the expected one)
    # 2. timestamp of the usage was moved to be the last token generated

```

### Features

Here are the key features of `token-throttle`, explained:

- **Multi-Resource Limiting:**

  - Simultaneously enforce limits on multiple distinct resource types for a single operation (e.g., limit both the number of API requests _and_ the number of tokens consumed within those requests).
  - Define simultaneous different quotas for different resources (e.g., 60 requests/minute AND 1,000 requests/day AND 1,000,000 tokens/minute).

- **Accurate Capacity Management & Refunding:**

  - Implements a reserve-then-adjust mechanism (`acquire_capacity` followed by `refund_capacity`).
  - Initially reserves the maximum potential usage for an operation.
  - Allows refunding unused capacity _or_ accounting for overuse if the actual usage differs from the reservation, ensuring limits are accurately enforced based on _actual_ consumption.

- **Asyncio Native:**

  - Built from the ground up using `asyncio` for non-blocking operation, ideal for high-throughput applications interacting with external APIs.

- **Flexible Time Windows:**

  - Define quotas over various time periods (e.g., per second, per minute, per hour, per day) using the `per_seconds` parameter in `Quota` (integer seconds).
  - Enforce limits across multiple windows concurrently for the same resource (e.g., limit requests per minute _and_ requests per day).

- **Correctness & Atomicity:**

  - Designed to avoid common race conditions found in simpler rate limiters, especially when used with distributed backends like Redis.
  - The provided Redis backend uses locks and appropriate commands to guarantee atomic updates to capacity across multiple workers/processes.

- **Pluggable Backend Architecture:**

  - Core logic is separated from the storage mechanism via `RateLimiterBackend` and `RateLimiterBackendBuilderInterface` interfaces.
  - Ships with a robust `RedisBackend` for distributed rate limiting.
  - Allows implementing custom backends (e.g., in-memory for single process, other databases) if needed.

- **Configurable Per "Model" or Endpoint:**

  - Apply different sets of `UsageQuotas` to different logical entities (referred to as `model` or `model_family` internally, e.g., specific API endpoints, different LLM versions sharing a quota).
    - i.e. This allows for gpt-4o and gpt-4o-mini automatically have separate quotas, while gpt-4o-20241203 and gpt-4o-20241024 are just aliases of each other but are counted in the same quota bucket instance (i.e. belong to the same model_family).
  - Supports dynamic configuration lookups via a `PerModelConfigGetter` callable.

- **Extensible Usage Counting:**

  - Define custom logic (`UsageCounter`) to calculate the resource usage of a given request _before_ it happens (e.g., estimate token count for an LLM request based on input messages).

- **Observability Hooks:**
  - Provides callbacks (`RateLimiterCallbacks`) for monitoring key events like starting to wait for capacity, consuming capacity, refunding capacity, and detecting missing state in the backend. Includes `loguru` integration helpers.

### Getting started

For out of the box experience just do `limiter = create_openai_redis_rate_limiter()`, and use it as in the [example-1](https://github.com/shobrook/openlimit/issues/20#issuecomment-2782677483) or [example-2](https://gist.github.com/justinvanwinkle/d9f04950083c4554835c1a35f9d22dad). Otherwise, copy the function and customize it to your needs.
