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
    _merge_rate_limiter_callbacks,
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
    """
    Build an async OpenAI rate limiter backed by Redis.

    ``redis_client`` must be a ``redis.asyncio.Redis`` instance. ``rpm`` and
    ``tpm`` are per-minute limits for requests and tokens, grouped by
    ``openai_model_family_getter`` so dated model variants share a family.
    User-provided callbacks merge slot-by-slot with the factory defaults:
    non-None user callbacks win, while None slots inherit the default INFO
    logger for missing consumption data.

    The helper is intentionally minute-window-only. For hourly/daily windows,
    custom metrics, or non-OpenAI usage shapes, construct ``RateLimiter`` with
    explicit ``Quota`` objects instead.
    """
    if not isinstance(redis_client, redis.Redis):
        raise TypeError(
            f"redis_client must be a redis.asyncio.Redis (async) instance "
            f"(got {type(redis_client).__name__}). For a sync client, use "
            f"create_openai_redis_sync_rate_limiter instead."
        )
    if isinstance(rpm, bool) or not isinstance(rpm, int):
        raise TypeError(f"rpm must be an int (got {type(rpm).__name__})")
    if isinstance(tpm, bool) or not isinstance(tpm, int):
        raise TypeError(f"tpm must be an int (got {type(tpm).__name__})")
    if callbacks is not None and not isinstance(callbacks, RateLimiterCallbacks):
        raise TypeError(
            f"callbacks must be a RateLimiterCallbacks instance or None "
            f"(got {type(callbacks).__name__})"
        )
    Quota(metric="requests", limit=rpm, per_seconds=SecondsIn.MINUTE)
    Quota(metric="tokens", limit=tpm, per_seconds=SecondsIn.MINUTE)

    default_callbacks = create_logging_callbacks(
        wait_start=None,
        wait_end_consumption=None,
        capacity_consumed=None,
        capacity_refunded=None,
        missing_consumption_data="INFO",
    )

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
        callbacks=_merge_rate_limiter_callbacks(callbacks, default_callbacks),
    )
