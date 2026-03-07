import importlib

from token_throttle._interfaces._callbacks import (
    OnCapacityConsumedCallback,
    OnCapacityRefundedCallback,
    OnMissingConsumptionDataCallback,
    OnWaitEndCallback,
    OnWaitStartCallback,
    RateLimiterCallbacks,
    create_loguru_callbacks,
)
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    PerModelConfigGetter,
    RateLimiterBackendBuilderInterface,
    UsageCounter,
)
from token_throttle._interfaces._models import (
    BucketId,
    Capacities,
    CapacityReservation,
    FrozenUsage,
    MetricName,
    PerSeconds,
    Quota,
    SecondsIn,
    Usage,
    UsageQuotas,
    frozen_usage,
)
from token_throttle._rate_limiter import RateLimiter

__version__ = "0.4.2"

# Lazy imports: these pull in redis or tiktoken at import time.
# Deferred via __getattr__ so `import token_throttle` works without optional deps.
_LAZY_IMPORTS: dict[str, str] = {
    # redis backend
    "LOCK_TIMEOUT_SECONDS": "token_throttle._limiter_backends._redis._backend",
    "CapacitiesGetterResult": "token_throttle._limiter_backends._redis._backend",
    "RedisBackend": "token_throttle._limiter_backends._redis._backend",
    "RedisBackendBuilder": "token_throttle._limiter_backends._redis._backend",
    # redis bucket
    "CalculatedCapacity": "token_throttle._limiter_backends._redis._bucket",
    # openai factory (imports redis at module level)
    "create_openai_redis_rate_limiter": "token_throttle._factories._openai._openai_rate_limiter",
    "openai_model_family_getter": "token_throttle._factories._openai._openai_rate_limiter",
    # openai token counter (tiktoken lazy inside, but part of optional openai extra)
    "EncodingGetter": "token_throttle._factories._openai._token_counter",
    "OpenAIUsageCounter": "token_throttle._factories._openai._token_counter",
    "count_chat_input_tokens": "token_throttle._factories._openai._token_counter",
    "get_encoding": "token_throttle._factories._openai._token_counter",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LOCK_TIMEOUT_SECONDS",
    "BucketId",
    "CalculatedCapacity",
    "Capacities",
    "CapacitiesGetterResult",
    "CapacityReservation",
    "EncodingGetter",
    "FrozenUsage",
    "MetricName",
    "OnCapacityConsumedCallback",
    "OnCapacityRefundedCallback",
    "OnMissingConsumptionDataCallback",
    "OnWaitEndCallback",
    "OnWaitStartCallback",
    "OpenAIUsageCounter",
    "PerModelConfig",
    "PerModelConfigGetter",
    "PerSeconds",
    "Quota",
    "RateLimiter",
    "RateLimiterBackendBuilderInterface",
    "RateLimiterCallbacks",
    "RedisBackend",
    "RedisBackendBuilder",
    "SecondsIn",
    "Usage",
    "UsageCounter",
    "UsageQuotas",
    "count_chat_input_tokens",
    "create_loguru_callbacks",
    "create_openai_redis_rate_limiter",
    "frozen_usage",
    "get_encoding",
    "openai_model_family_getter",
]
