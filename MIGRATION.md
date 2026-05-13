# Migrating from v1.4.x to v2.0.0

v2.0.0 keeps the strict runtime validation introduced before this release.
Do not rely on construction-time coercion during the upgrade. Run the
migration helper against your stored configuration dictionaries first, fix all
reported issues, then deploy the new version.

## 1. Preflight Config Dictionaries

```python
from token_throttle.migration import validate_config_for_v2_0

errors = validate_config_for_v2_0(your_config)
if errors:
    for error in errors:
        print(
            f"{error.field_path}: {error.value!r} -> "
            f"{error.reason}; {error.suggested_fix}"
        )
```

The helper is read-only: it does not mutate input and does not coerce values.
It reports values that v1.4.x may have accepted but v2.0.0 rejects, including:

- quoted numeric limits such as `"1000"`; use `1000`
- float time windows such as `60.0`; use `60`
- whitespace, `:`, `{`, or `}` in metrics, model families, and Redis prefixes
- bytes values where plain strings are required

## 2. Drain Reservations

Drain or refund in-flight reservations before upgrading. Legacy serialized
reservations may have `limiter_instance_id=None`; v2.0.0 reports this as a
migration issue because those reservations cannot provide the same ownership
signal as new reservations.

## 3. Add Redis Key Prefixes

Redis backend builders and OpenAI Redis factories require a deployment-scoped
`key_prefix`. Pick a stable prefix per deployment or tenant, for example
`"prod-api"` or `"tenant-a"`. The same prefix must be used by every process
that should share rate-limit state.

## 4. Review Callback Construction

`RateLimiterCallbacks(...)` and `SyncRateLimiterCallbacks(...)` now merge
user-provided slots with factory defaults. Update code that assumed a partially
specified callback bundle disabled every default callback.

## 5. Refactor DTO Subclasses

`Quota`, `PerModelConfig`, and `CapacityReservation` are strict DTOs, not
extension points. Replace subclass-based customization with composition,
factory functions, or explicit `PerModelConfig` construction before upgrading.
