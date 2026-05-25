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
    _merge_sync_rate_limiter_callbacks,
    create_sync_logging_callbacks,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, SecondsIn, UsageQuotas
from token_throttle._limiter_backends._redis._keys import (
    DEFAULT_REFUND_DEDUP_TTL_SECONDS,
)
from token_throttle._limiter_backends._redis._sync_backend import (
    SyncRedisBackendBuilder,
)
from token_throttle._limiter_backends._redis._ttl import DEFAULT_BUCKET_TTL_SECONDS
from token_throttle._sync_rate_limiter import SyncRateLimiter


def create_openai_redis_sync_rate_limiter(  # noqa: PLR0913
    redis_client: _sync_redis.Redis,
    *,
    key_prefix: str,
    rpm: int,
    tpm: int,
    bucket_ttl_seconds: int = DEFAULT_BUCKET_TTL_SECONDS,
    max_reservation_lifetime_seconds: float | None = None,
    refund_dedup_ttl_seconds: int | None = None,
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
    key_prefix:
        Required deployment-scoped Redis namespace. All Redis keys use
        ``{key_prefix}:rate_limiting:...`` so unrelated deployments sharing
        Redis do not collide.
    rpm:
        Requests-per-minute limit, applied per resolved model_family.
    tpm:
        Tokens-per-minute limit, applied per resolved model_family.
    callbacks:
        Optional ``SyncRateLimiterCallbacks``. User-provided callbacks merge
        slot-by-slot with the factory defaults: non-None user callbacks win,
        while None slots inherit the default INFO logger for missing
        consumption data.
    bucket_ttl_seconds:
        Required Redis bucket-state expiry in seconds. The expiry is refreshed
        when bucket state is read or written.
    max_reservation_lifetime_seconds:
        Optional bound on how long an acquired reservation remains refundable.
        None preserves the Redis backend default derived from Redis TTLs.
    refund_dedup_ttl_seconds:
        Redis TTL for successful refund idempotency keys. None preserves the
        library Redis backend default.

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
            f"redis_client expected redis.Redis, got "
            f"{type(redis_client).__name__}; for async use redis.asyncio.Redis "
            "with create_openai_redis_rate_limiter"
        )
    if isinstance(rpm, bool) or not isinstance(rpm, int):
        raise TypeError(f"rpm must be an int (got {type(rpm).__name__})")
    if isinstance(tpm, bool) or not isinstance(tpm, int):
        raise TypeError(f"tpm must be an int (got {type(tpm).__name__})")
    if callbacks is not None and type(callbacks) is not SyncRateLimiterCallbacks:
        raise TypeError(
            f"callbacks must be a SyncRateLimiterCallbacks instance or None "
            f"(got {type(callbacks).__name__})"
        )
    Quota(metric="requests", limit=rpm, per_seconds=SecondsIn.MINUTE)
    Quota(metric="tokens", limit=tpm, per_seconds=SecondsIn.MINUTE)

    default_callbacks = create_sync_logging_callbacks(
        wait_start=None,
        wait_end_consumption=None,
        capacity_consumed=None,
        capacity_refunded=None,
        missing_consumption_data="INFO",
    )

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
        backend=SyncRedisBackendBuilder(
            redis_client,
            key_prefix=key_prefix,
            bucket_ttl_seconds=bucket_ttl_seconds,
            refund_dedup_ttl_seconds=(
                DEFAULT_REFUND_DEDUP_TTL_SECONDS
                if refund_dedup_ttl_seconds is None
                else refund_dedup_ttl_seconds
            ),
        ),
        callbacks=_merge_sync_rate_limiter_callbacks(callbacks, default_callbacks),
        max_reservation_lifetime_seconds=max_reservation_lifetime_seconds,
    )
