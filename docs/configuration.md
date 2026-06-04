# Configuration reference

This reference covers advanced configuration and runtime tuning: per-model
in-process caps, unlimited configs, the `usage_counter` contract, and runtime
limit overrides. Start with the [README](../README.md#configuration) for the
shorter setup path.

See also [docs/api.md](api.md) for public constants and type aliases, and
[docs/custom-backends.md](custom-backends.md) for implementing your own backend.

## Per-model configuration

A callable config maps each model name to a `PerModelConfig`. The README shows
the basic shape; this section covers the in-process caps and idle-family
maintenance that matter for long-lived or dynamic deployments.

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

### Unlimited configs

To disable rate limiting for a model while keeping the same API surface, return
an unlimited config from your config callable:

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

### Usage counters

`OpenAIUsageCounter` handles text-only OpenAI requests. It counts `input` or
`messages`, plus prompt-bearing request context such as `instructions`,
tool/function definitions, and structured output schemas. The plural `inputs`
field is rejected because it is not an OpenAI request field. Image/audio/file
inputs are still unsupported; pass usage manually for those.

Token estimates are bounded by `tiktoken`'s model encodings plus local
best-effort heuristics for chat/message overhead, tools, functions, schemas,
and output budgets. They have not been reconciled against live OpenAI billing
dashboards across a request corpus; periodically compare reserved tokens with
your actual billing and pass usage manually where local counting is not
representative.

Custom `usage_counter` callables receive the same kwargs you pass to
`acquire_capacity_for_request()`. They can accept `**request` for the whole
payload or only the named request fields they use; fixed-signature counters do
not need to accept unrelated kwargs like `model`. Custom counters are trusted
application code: nested request objects are passed by reference, so counters
should avoid mutating them. Return a plain `dict` or another well-behaved
mapping of metric names to finite numeric values.

## Dynamic rate limits

Adjust bucket limits at runtime without rebuilding the limiter — useful for
adaptive rate limiting (e.g., reacting to `x-ratelimit-*` response headers):

```python
# After the limiter has initialized this model with an acquire call:
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
