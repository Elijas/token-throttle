import importlib
import importlib.util

from token_throttle._interfaces._callbacks import (
    OnCapacityConsumedCallback,
    OnCapacityRefundedCallback,
    OnMissingConsumptionDataCallback,
    OnWaitEndCallback,
    OnWaitStartCallback,
    RateLimiterCallbacks,
    SyncOnCapacityConsumedCallback,
    SyncOnCapacityRefundedCallback,
    SyncOnMissingConsumptionDataCallback,
    SyncOnWaitEndCallback,
    SyncOnWaitStartCallback,
    SyncRateLimiterCallbacks,
    create_logging_callbacks,
    create_loguru_callbacks,
    create_sync_logging_callbacks,
    create_sync_loguru_callbacks,
)
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    PerModelConfigGetter,
    RateLimiterBackend,
    RateLimiterBackendBuilderInterface,
    SyncRateLimiterBackend,
    SyncRateLimiterBackendBuilderInterface,
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
from token_throttle._sync_rate_limiter import SyncRateLimiter

__version__ = "1.4.0"

# Lazy imports: these pull in redis or tiktoken at import time.
# Deferred via __getattr__ so `import token_throttle` works without optional deps.
_LAZY_IMPORTS: dict[str, str] = {
    # shared capacity math (canonical location)
    "CalculatedCapacity": "token_throttle._capacity",
    # memory backend
    "MemoryBackend": "token_throttle._limiter_backends._memory._backend",
    "MemoryBackendBuilder": "token_throttle._limiter_backends._memory._backend",
    "MemoryBucket": "token_throttle._limiter_backends._memory._bucket",
    # sync memory backend
    "SyncMemoryBackend": "token_throttle._limiter_backends._memory._sync_backend",
    "SyncMemoryBackendBuilder": "token_throttle._limiter_backends._memory._sync_backend",
    # sync redis backend
    "SyncRedisBackend": "token_throttle._limiter_backends._redis._sync_backend",
    "SyncRedisBackendBuilder": "token_throttle._limiter_backends._redis._sync_backend",
    "SyncRedisBucket": "token_throttle._limiter_backends._redis._sync_bucket",
    # redis bucket
    "RedisBucket": "token_throttle._limiter_backends._redis._bucket",
    # redis backend
    "LOCK_TIMEOUT_SECONDS": "token_throttle._limiter_backends._redis._backend",
    "CapacitiesGetterResult": "token_throttle._limiter_backends._redis._backend",
    "RedisBackend": "token_throttle._limiter_backends._redis._backend",
    "RedisBackendBuilder": "token_throttle._limiter_backends._redis._backend",
    # openai redis factory (imports redis at module level)
    "create_openai_redis_rate_limiter": "token_throttle._factories._openai._openai_rate_limiter",
    # redis-independent openai helper
    "openai_model_family_getter": "token_throttle._factories._openai._model_family",
    # openai token counter (tiktoken lazy inside, but part of optional openai extra)
    "EncodingGetter": "token_throttle._factories._openai._token_counter",
    "OpenAIUsageCounter": "token_throttle._factories._openai._token_counter",
    "count_chat_input_tokens": "token_throttle._factories._openai._token_counter",
    "get_encoding": "token_throttle._factories._openai._token_counter",
}

_HAS_REDIS = importlib.util.find_spec("redis") is not None


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(globals()))


_REDIS_ALL = [
    "LOCK_TIMEOUT_SECONDS",
    "CapacitiesGetterResult",
    "RedisBackend",
    "RedisBackendBuilder",
    "RedisBucket",
    "SyncRedisBackend",
    "SyncRedisBackendBuilder",
    "SyncRedisBucket",
    "create_openai_redis_rate_limiter",
]

__all__ = [
    "BucketId",
    "CalculatedCapacity",
    "Capacities",
    "CapacityReservation",
    "EncodingGetter",
    "FrozenUsage",
    "MemoryBackend",
    "MemoryBackendBuilder",
    "MemoryBucket",
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
    "RateLimiterBackend",
    "RateLimiterBackendBuilderInterface",
    "RateLimiterCallbacks",
    "SecondsIn",
    "SyncMemoryBackend",
    "SyncMemoryBackendBuilder",
    "SyncOnCapacityConsumedCallback",
    "SyncOnCapacityRefundedCallback",
    "SyncOnMissingConsumptionDataCallback",
    "SyncOnWaitEndCallback",
    "SyncOnWaitStartCallback",
    "SyncRateLimiter",
    "SyncRateLimiterBackend",
    "SyncRateLimiterBackendBuilderInterface",
    "SyncRateLimiterCallbacks",
    "Usage",
    "UsageCounter",
    "UsageQuotas",
    "count_chat_input_tokens",
    "create_logging_callbacks",
    "create_loguru_callbacks",
    "create_sync_logging_callbacks",
    "create_sync_loguru_callbacks",
    "frozen_usage",
    "get_encoding",
    "openai_model_family_getter",
]

if _HAS_REDIS:
    __all__ += _REDIS_ALL  # noqa: PLE0605 - dynamic __all__ to avoid ImportError without redis
