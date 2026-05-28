import importlib
import importlib.util
from typing import TYPE_CHECKING

from token_throttle._diagnostic import (
    BackendHealthDiagnostic,
    BackendIntrospectionDiagnostic,
    BucketDiagnostic,
    CurrentWaitsDiagnostic,
    CustomBackendHealthDiagnostic,
    DiagnosticBucketKey,
    DiagnosticIssue,
    InFlightReservationsDiagnostic,
    MemoryBackendHealthDiagnostic,
    RateLimiterDiagnostic,
    RedisBackendHealthDiagnostic,
    ReservationGroupDiagnostic,
    RuntimeOverrideDiagnostic,
    WaitBucketDiagnostic,
    WaiterDiagnostic,
)
from token_throttle._exceptions import (
    AcquireRefundFailedError,
    BackendConformanceError,
    CardinalityLimitExceededError,
    DuplicateRefundError,
    UnknownReservationError,
)
from token_throttle._interfaces._callbacks import (
    LifecycleEvent,
    OnCapacityConsumedCallback,
    OnCapacityRefundedCallback,
    OnLifecycleEventCallback,
    OnMissingConsumptionDataCallback,
    OnWaitEndCallback,
    OnWaitStartCallback,
    RateLimiterCallbacks,
    SyncOnCapacityConsumedCallback,
    SyncOnCapacityRefundedCallback,
    SyncOnLifecycleEventCallback,
    SyncOnMissingConsumptionDataCallback,
    SyncOnWaitEndCallback,
    SyncOnWaitStartCallback,
    SyncRateLimiterCallbacks,
    create_logging_callbacks,
    create_sync_logging_callbacks,
)
from token_throttle._interfaces._interfaces import (
    BackendIntrospectable,
    PerModelConfig,
    PerModelConfigGetter,
    RateLimiterBackend,
    RateLimiterBackendBuilderInterface,
    SyncBackendIntrospectable,
    SyncRateLimiterBackend,
    SyncRateLimiterBackendBuilderInterface,
    UsageCounter,
)
from token_throttle._interfaces._models import (
    MAX_ALIAS_LENGTH,
    MAX_KEY_PREFIX_LENGTH,
    MAX_METRIC_LENGTH,
    MAX_MODEL_FAMILY_LENGTH,
    MAX_RESERVATION_ID_LENGTH,
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
from token_throttle._validation import MAX_TOTAL_KEY_LENGTH
from token_throttle.conformance import (
    conformance_test_for,
    run_conformance_test_for,
    sync_conformance_test_for,
)
from token_throttle.migration import (
    ConfigMigrationIssue,
    async_cleanup_legacy_buckets,
    cleanup_legacy_buckets,
    validate_config_for_v2_0,
)

__version__ = "8.0.1"

if TYPE_CHECKING:
    from token_throttle._capacity import CalculatedCapacity
    from token_throttle._factories._openai._model_family import (
        openai_model_family_getter,
    )
    from token_throttle._factories._openai._openai_rate_limiter import (
        create_openai_redis_rate_limiter,
    )
    from token_throttle._factories._openai._openai_sync_rate_limiter import (
        create_openai_redis_sync_rate_limiter,
    )
    from token_throttle._factories._openai._token_counter import (
        EncodingGetter,
        OpenAIUsageCounter,
        count_chat_input_tokens,
        get_encoding,
    )
    from token_throttle._limiter_backends._memory._backend import (
        MemoryBackend,
        MemoryBackendBuilder,
    )
    from token_throttle._limiter_backends._memory._bucket import MemoryBucket
    from token_throttle._limiter_backends._memory._sync_backend import (
        SyncMemoryBackend,
        SyncMemoryBackendBuilder,
    )
    from token_throttle._limiter_backends._redis._backend import (
        LOCK_TIMEOUT_SECONDS,
        CapacitiesGetterResult,
        RedisBackend,
        RedisBackendBuilder,
    )
    from token_throttle._limiter_backends._redis._bucket import RedisBucket
    from token_throttle._limiter_backends._redis._sync_backend import (
        SyncRedisBackend,
        SyncRedisBackendBuilder,
    )
    from token_throttle._limiter_backends._redis._sync_bucket import SyncRedisBucket

    _STATIC_LAZY_EXPORTS = (
        LOCK_TIMEOUT_SECONDS,
        CapacitiesGetterResult,
        RedisBackend,
        RedisBackendBuilder,
        RedisBucket,
        SyncRedisBackend,
        SyncRedisBackendBuilder,
        SyncRedisBucket,
        create_openai_redis_rate_limiter,
        create_openai_redis_sync_rate_limiter,
    )

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
    "create_openai_redis_sync_rate_limiter": "token_throttle._factories._openai._openai_sync_rate_limiter",
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
    "create_openai_redis_sync_rate_limiter",
]

__all__ = [
    "MAX_ALIAS_LENGTH",
    "MAX_KEY_PREFIX_LENGTH",
    "MAX_METRIC_LENGTH",
    "MAX_MODEL_FAMILY_LENGTH",
    "MAX_RESERVATION_ID_LENGTH",
    "MAX_TOTAL_KEY_LENGTH",
    "AcquireRefundFailedError",
    "BackendConformanceError",
    "BackendHealthDiagnostic",
    "BackendIntrospectable",
    "BackendIntrospectionDiagnostic",
    "BucketDiagnostic",
    "BucketId",
    "CalculatedCapacity",
    "Capacities",
    "CapacityReservation",
    "CardinalityLimitExceededError",
    "ConfigMigrationIssue",
    "CurrentWaitsDiagnostic",
    "CustomBackendHealthDiagnostic",
    "DiagnosticBucketKey",
    "DiagnosticIssue",
    "DuplicateRefundError",
    "EncodingGetter",
    "FrozenUsage",
    "InFlightReservationsDiagnostic",
    "LifecycleEvent",
    "MemoryBackend",
    "MemoryBackendBuilder",
    "MemoryBackendHealthDiagnostic",
    "MemoryBucket",
    "MetricName",
    "OnCapacityConsumedCallback",
    "OnCapacityRefundedCallback",
    "OnLifecycleEventCallback",
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
    "RateLimiterDiagnostic",
    "RedisBackendHealthDiagnostic",
    "ReservationGroupDiagnostic",
    "RuntimeOverrideDiagnostic",
    "SecondsIn",
    "SyncBackendIntrospectable",
    "SyncMemoryBackend",
    "SyncMemoryBackendBuilder",
    "SyncOnCapacityConsumedCallback",
    "SyncOnCapacityRefundedCallback",
    "SyncOnLifecycleEventCallback",
    "SyncOnMissingConsumptionDataCallback",
    "SyncOnWaitEndCallback",
    "SyncOnWaitStartCallback",
    "SyncRateLimiter",
    "SyncRateLimiterBackend",
    "SyncRateLimiterBackendBuilderInterface",
    "SyncRateLimiterCallbacks",
    "UnknownReservationError",
    "Usage",
    "UsageCounter",
    "UsageQuotas",
    "WaitBucketDiagnostic",
    "WaiterDiagnostic",
    "async_cleanup_legacy_buckets",
    "cleanup_legacy_buckets",
    "conformance_test_for",
    "count_chat_input_tokens",
    "create_logging_callbacks",
    "create_sync_logging_callbacks",
    "frozen_usage",
    "get_encoding",
    "openai_model_family_getter",
    "run_conformance_test_for",
    "sync_conformance_test_for",
    "validate_config_for_v2_0",
]

if _HAS_REDIS:
    __all__ += _REDIS_ALL  # dynamic __all__ to avoid ImportError without redis  # noqa: PLE0605
