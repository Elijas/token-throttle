try:
    import redis as _sync_redis
except ImportError as exc:
    raise ImportError(
        'The "redis" package is required for the OpenAI Redis sync rate limiter. '
        'Install it with: pip install "token-throttle[redis]"'
    ) from exc

from token_throttle._factories._openai._model_family import openai_model_family_getter
from token_throttle._factories._openai._token_counter import OpenAIUsageCounter
from token_throttle._interfaces._callbacks import (
    SyncRateLimiterCallbacks,
    create_sync_logging_callbacks,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, SecondsIn, UsageQuotas
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)
from token_throttle._sync_rate_limiter import SyncRateLimiter


def create_openai_redis_sync_rate_limiter(
    redis_client: _sync_redis.Redis,
    *,
    rpm: int,
    tpm: int,
    callbacks: SyncRateLimiterCallbacks | None = None,
) -> SyncRateLimiter:
    """
    Build a synchronous OpenAI rate limiter backed by Redis.

    This is the sync mirror of ``create_openai_redis_rate_limiter``. Use it
    when your code is not running on an event loop (CLI tools, sync request
    handlers, batch scripts).

    Parameters
    ----------
    redis_client:
        A ``redis.Redis`` (sync) instance. For an async event-loop client,
        use ``create_openai_redis_rate_limiter`` instead.
    rpm:
        Requests-per-minute limit, applied per resolved model_family.
    tpm:
        Tokens-per-minute limit, applied per resolved model_family.
    callbacks:
        Optional ``SyncRateLimiterCallbacks``. If ``None``, a default logging
        callbacks bundle is installed with ``missing_consumption_data="INFO"``.
        If provided, the user's instance is used verbatim — the factory does
        NOT auto-merge user fields with defaults, so passing a single callback
        will silently drop the default observability logger.

    Notes
    -----
    * The time window is hardcoded to 60 seconds. For per-hour or per-day
      caps, drop to the manual ``SyncRateLimiter(...)`` path with your own
      ``Quota(per_seconds=...)``.
    * ``model_family`` resolution is automatic via
      ``openai_model_family_getter`` (strips date suffixes and ``openai/``
      prefix).
    * Validation of ``rpm`` / ``tpm`` / ``redis_client`` / ``callbacks`` is
      eager — invalid arguments raise at construction time, not at first
      acquire.

    """
    if not isinstance(redis_client, _sync_redis.Redis):
        raise TypeError(
            f"redis_client must be a redis.Redis (sync) instance "
            f"(got {type(redis_client).__name__}). For an async client, use "
            f"create_openai_redis_rate_limiter instead."
        )
    if isinstance(rpm, bool) or not isinstance(rpm, int):
        raise TypeError(f"rpm must be an int (got {type(rpm).__name__})")
    if isinstance(tpm, bool) or not isinstance(tpm, int):
        raise TypeError(f"tpm must be an int (got {type(tpm).__name__})")
    if callbacks is not None and not isinstance(callbacks, SyncRateLimiterCallbacks):
        raise TypeError(
            f"callbacks must be a SyncRateLimiterCallbacks instance or None "
            f"(got {type(callbacks).__name__})"
        )
    Quota(metric="requests", limit=rpm, per_seconds=SecondsIn.MINUTE)
    Quota(metric="tokens", limit=tpm, per_seconds=SecondsIn.MINUTE)

    return SyncRateLimiter(
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
        backend=SyncRedisBackendBuilder(redis_client),
        callbacks=callbacks
        if callbacks is not None
        else create_sync_logging_callbacks(
            wait_start=None,
            wait_end_consumption=None,
            capacity_consumed=None,
            capacity_refunded=None,
            missing_consumption_data="INFO",
        ),
    )
