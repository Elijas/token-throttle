try:
    import redis.asyncio as redis
except ImportError as exc:
    raise ImportError(
        'The "redis" package is required for the OpenAI Redis rate limiter backend. '
        'Install it with: pip install "token-throttle[redis]"'
    ) from exc

from token_throttle._factories._openai._model_family import openai_model_family_getter
from token_throttle._factories._openai._token_counter import OpenAIUsageCounter
from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    create_logging_callbacks,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, SecondsIn, UsageQuotas
from token_throttle._limiter_backends._redis._backend import RedisBackendBuilder
from token_throttle._rate_limiter import RateLimiter


def create_openai_redis_rate_limiter(
    redis_client: redis.Redis,
    *,
    rpm: int,
    tpm: int,
    callbacks: RateLimiterCallbacks | None = None,
) -> RateLimiter:
    return RateLimiter(
        lambda model_name: PerModelConfig(
            quotas=UsageQuotas(
                [
                    Quota(metric="requests", limit=rpm, per_seconds=SecondsIn.MINUTE),
                    Quota(metric="tokens", limit=tpm, per_seconds=SecondsIn.MINUTE),
                ],
            ),
            usage_counter=OpenAIUsageCounter(),
            model_family=openai_model_family_getter(model_name),
        ),
        backend=RedisBackendBuilder(redis_client),
        callbacks=callbacks
        or create_logging_callbacks(
            wait_start=None,
            wait_end_consumption=None,
            capacity_consumed=None,
            capacity_refunded=None,
            missing_consumption_data="INFO",
        ),
    )
