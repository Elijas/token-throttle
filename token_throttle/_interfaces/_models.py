import math
import unicodedata
import uuid
from collections import defaultdict
from collections.abc import Iterator, Mapping
from enum import Enum
from typing import ClassVar, Self

from frozendict import frozendict
from pydantic import Field, ValidationInfo, field_validator

from token_throttle._capacity import MIN_MAX_CAPACITY
from token_throttle._dto import StrictDTO
from token_throttle._exceptions import CardinalityLimitExceededError

_UNLIMITED_FLAG = "__rate_limiting_disabled__"
"""
Sentinel ``model_family`` value used by ``_unlimited_reservation``.

Lives in this module (not ``_validation``) so ``CapacityReservation``'s
``is_unlimited`` field validator can reference it without an import cycle.
``_validation`` re-exports it for back-compat with any external imports.
"""


MAX_MODEL_FAMILY_LENGTH = 256
MAX_METRIC_LENGTH = 64
MAX_ALIAS_LENGTH = 256
MAX_KEY_PREFIX_LENGTH = 128
MAX_RESERVATION_ID_LENGTH = 128


def _is_bool_like(value: object) -> bool:
    """Reject Python bool values without duck-typing numeric lookalikes."""
    return type(value) is bool


def _default_max_length_for_field(field_name: str) -> int | None:
    if field_name in {"metric", "bucket_id metric"}:
        return MAX_METRIC_LENGTH
    if field_name == "model_family":
        return MAX_MODEL_FAMILY_LENGTH
    if field_name in {"model_name", "alias"}:
        return MAX_ALIAS_LENGTH
    if field_name == "key_prefix":
        return MAX_KEY_PREFIX_LENGTH
    if field_name == "reservation_id":
        return MAX_RESERVATION_ID_LENGTH
    return None


def _validate_key_segment(
    value: object,
    /,
    *,
    field_name: str,
    allow_none: bool = False,
    max_length: int | None = None,
) -> str | None:
    """Validate Redis-key path segments used for metrics and model families."""
    if value is None and allow_none:
        return None
    if type(value) is not str:
        none_suffix = " or None" if allow_none else ""
        raise ValueError(
            f"{field_name} must be a str{none_suffix} (got {type(value).__name__})"
        )
    if not value:
        raise ValueError(f"{field_name} must not be empty")

    normalized = unicodedata.normalize("NFC", value)
    max_length = (
        _default_max_length_for_field(field_name) if max_length is None else max_length
    )
    if max_length is not None and len(normalized) > max_length:
        raise CardinalityLimitExceededError(
            f"{field_name} must be at most {max_length} characters "
            f"(got {len(normalized)})"
        )
    if not normalized.strip():
        raise ValueError(f"{field_name} must not be whitespace-only")
    if normalized != normalized.strip():
        raise ValueError(f"{field_name} must not contain leading/trailing whitespace")
    if any(char.isspace() for char in normalized):
        raise ValueError(f"{field_name} must not contain whitespace")
    if any(not char.isprintable() for char in normalized):
        raise ValueError(f"{field_name} must not contain non-printable characters")
    if any(unicodedata.category(char).startswith("C") for char in normalized):
        raise ValueError(f"{field_name} must not contain Unicode control characters")
    if ":" in normalized:
        raise ValueError(
            f"{field_name} must not contain ':' (used as Redis key separator)"
        )
    if any(char in normalized for char in "{}"):
        raise ValueError(
            f"{field_name} must not contain '{{' or '}}' "
            "(used as Redis Cluster hash tag delimiters)"
        )
    return normalized


class SecondsIn(int, Enum):
    MINUTE = 60
    HOUR = 3600
    DAY = 86400


class Quota(StrictDTO):
    """
    Exact-type immutable rate-limit quota (frozen Pydantic DTO).

    v2.0.0 contract: ``Quota`` is a data-transfer object, not a subclass
    extension point. Construction, assignment, copy, pickle restore,
    ``model_copy()``, and ``model_construct()`` all preserve the same
    validators; ``model_construct()`` is disabled.

    ``frozen=True`` prevents mutation via normal attribute assignment.
    Direct ``object.__setattr__`` or ``__dict__`` writes bypass this at
    the Python level — this is a CPython limitation, not a library bug.
    Do not mutate instances after construction; hash stability depends
    on immutability.
    """

    DEFAULT_SECONDS: ClassVar[int] = SecondsIn.MINUTE
    metric: str = Field(
        description=(
            "Metric name used in Redis key segments. Must be non-empty, NFC "
            "normalized, printable, contain no whitespace/control characters, "
            "and cannot contain ':', '{', or '}'; the portable recommended "
            "character set is ^[A-Za-z0-9_./-]+$."
        )
    )
    limit: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="Maximum capacity available within the time window.",
    )
    per_seconds: int = Field(
        default=DEFAULT_SECONDS,
        gt=0,  # Greater than 0
        description="Time window in seconds. Default: 60 (1 minute). E.g. For requests per minute, set to 60. For requests per hour, set to 3600.",
    )

    @field_validator("limit", "per_seconds", mode="before")
    @classmethod
    def _reject_boolean(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if _is_bool_like(value):
            raise ValueError(f"{info.field_name} must not be a boolean")
        if info.field_name == "limit" and not isinstance(value, (int, float)):
            raise ValueError(f"limit must be int or float (got {type(value).__name__})")
        return value

    @field_validator("limit")
    @classmethod
    def _reject_subnormal_limit(cls, value: float) -> float:
        if value < MIN_MAX_CAPACITY:
            raise ValueError(
                f"limit must be greater than or equal to {MIN_MAX_CAPACITY!r}"
            )
        return value

    @field_validator("per_seconds", mode="before")
    @classmethod
    def _reject_non_numeric_per_seconds(cls, value: object) -> object:
        if isinstance(value, (int, float)) or _is_bool_like(value):
            return value
        raise ValueError(
            f"per_seconds must be int or float (got {type(value).__name__})"
        )

    @field_validator("metric", mode="before")
    @classmethod
    def _reject_empty_metric(cls, value: object) -> object:
        return _validate_key_segment(value, field_name="metric")


class UsageQuotas:
    """
    Exact-type collection of per-metric quotas; empty only via ``unlimited()``.

    v2.0.0 contract: ``UsageQuotas`` is a data-transfer collection, not a
    subclass extension point. Security-sensitive callers accept the exact
    class and exact ``Quota`` instances only.
    """

    def __init__(
        self,
        quotas: list[Quota],
        /,
        *,
        _allow_empty_quotas: bool = False,
        _freeze: bool = False,
    ) -> None:
        self._frozen = False
        self._metrics: defaultdict[str, dict[int, Quota]] = defaultdict(dict)
        if not _allow_empty_quotas and not quotas:
            raise ValueError(
                "Empty quota list provided. No rate limiting will be applied. "
                "If this is intentional, use UsageQuotas.unlimited() instead."
            )
        for quota in quotas:
            self.add_metric(quota)
        if _freeze:
            self._freeze()

    @classmethod
    def unlimited(cls, *, _freeze: bool = False) -> Self:
        """Return an explicit no-limit quota set for disabled rate limiting."""
        return cls([], _allow_empty_quotas=True, _freeze=_freeze)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False) and name in {"_frozen", "_metrics"}:
            raise TypeError("UsageQuotas snapshot is frozen")
        super().__setattr__(name, value)

    def _freeze(self) -> None:
        self._metrics = frozendict(
            {
                metric: frozendict(quotas_by_window)
                for metric, quotas_by_window in self._metrics.items()
            }
        )
        self._frozen = True

    def frozen_snapshot(self) -> Self:
        """Return a frozen exact-type copy for immutable DTO composition."""
        quotas = list(self)
        return type(self)(
            quotas,
            _allow_empty_quotas=self.is_unlimited,
            _freeze=True,
        )

    @property
    def is_unlimited(self) -> bool:
        return not bool(self._metrics)

    def add_metric(self, quota: Quota) -> None:
        if self._frozen:
            raise TypeError("UsageQuotas snapshot is frozen")
        if type(quota) is not Quota:
            raise ValueError(
                f"Each quota must be a Quota instance (got {type(quota).__name__})"
            )
        quota = quota.revalidate()
        if (
            quota.metric in self._metrics
            and quota.per_seconds in self._metrics[quota.metric]
        ):
            existing = self._metrics[quota.metric][quota.per_seconds]
            raise ValueError(
                f"Metric {quota.metric} with {quota.per_seconds} seconds already "
                f"exists (existing limit={existing.limit}, new limit={quota.limit}).",
            )
        self._metrics[quota.metric][quota.per_seconds] = quota

    def __iter__(self) -> Iterator[Quota]:
        for quotas in self._metrics.values():
            yield from quotas.values()

    @property
    def names(self) -> list[str]:
        return list(self._metrics.keys())

    def get_quotas(self, item: str) -> list[Quota]:
        return list(self._metrics.get(item, {}).values())


MetricName = str
Usage = Mapping[MetricName, float]
FrozenUsage = frozendict[MetricName, float]

PerSeconds = int
BucketId = tuple[MetricName, PerSeconds]
Capacities = frozendict[BucketId, float]


def _coerce_usage_value(
    metric: str,
    amount: object,
    *,
    label: str = "Usage value",
) -> float:
    if _is_bool_like(amount):
        raise ValueError(f"{label} for {metric} must not be a boolean")
    if not isinstance(amount, (int, float)):
        raise ValueError(  # noqa: TRY004 - public validators raise ValueError.
            f"{label} for {metric} must be finite and be int or float "
            f"(got {type(amount).__name__})"
        )
    try:
        value = float(amount)
    except OverflowError as exc:
        raise ValueError(
            f"{label} for {metric} too large to fit in IEEE 754 double (got {amount!r})"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} for {metric} must be finite (got {amount!r})"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} for {metric} must be finite (got {amount!r})")
    return value


def frozen_usage(usage: Usage) -> FrozenUsage:
    """Convert usage to a frozendict."""
    if not isinstance(usage, Mapping):
        raise ValueError(  # noqa: TRY004
            f"usage must be a mapping (got {type(usage).__name__})"
        )
    converted: dict[MetricName, float] = {}
    for metric, amount in usage.items():
        if not isinstance(metric, str) or not metric:
            raise ValueError(
                f"Usage metric key must be a non-empty string (got {metric!r})"
            )
        converted[metric] = _coerce_usage_value(metric, amount)
    return frozendict(converted)


class CapacityReservation(StrictDTO):
    """
    Exact-type frozen reservation returned by acquire/record operations.

    v2.0.0 contract: ``CapacityReservation`` is a data-transfer object, not a
    subclass extension point. Construction, assignment, copy, pickle restore,
    ``model_copy()``, and ``model_construct()`` all preserve the same
    validators; ``model_construct()`` is disabled.

    Reservations bind to the limiter/backend workflow that issued them and
    should be refunded while that limiter is still alive. They are not durable
    cross-process credentials; do not accept serialized reservations across
    trust boundaries as proof that capacity was acquired.

    Reservations require ``limiter_instance_id``. Legacy v1.4.x serialized
    reservations that omit it are rejected in v2.0.0; drain in-flight
    reservations before upgrading mixed fleets.

    ``is_unlimited=True`` is a trusted in-process sentinel produced by
    unlimited configs. Refunding such a reservation is a no-op.

    Same ``frozen=True`` caveat as ``Quota``: ``object.__setattr__``
    and ``__dict__`` writes bypass Pydantic's freeze at the CPython level.
    Mutating ``is_unlimited`` via this vector would bypass metering.
    """

    reservation_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        max_length=MAX_RESERVATION_ID_LENGTH,
    )
    usage: FrozenUsage
    model_family: str = Field(
        description=(
            "Model family used in Redis key segments. Must be non-empty, NFC "
            "normalized, printable, contain no whitespace/control characters, "
            "and cannot contain ':', '{', or '}'; the portable recommended "
            "character set is ^[A-Za-z0-9_./-]+$."
        )
    )
    bucket_ids: frozenset[BucketId] | None = Field(
        default=None,
        description=(
            "Exact buckets reserved at acquire time. Iterable inputs are "
            "materialized into a frozenset, so generators are consumed."
        ),
    )
    model: str | None = None
    is_unlimited: bool = False
    limiter_instance_id: str = Field(
        ...,
        description=(
            "UUID of the limiter instance that issued this reservation. "
            "Required in v2.0.0; legacy v1.4.x reservations without this "
            "field are no longer accepted."
        ),
    )

    @field_validator("reservation_id", mode="before")
    @classmethod
    def _reject_empty_reservation_id(cls, value: object) -> object:
        return _validate_key_segment(value, field_name="reservation_id")

    @field_validator("usage", mode="before")
    @classmethod
    def _normalize_usage(cls, value: object) -> object:
        if isinstance(value, Mapping):
            normalized_usage = frozen_usage(value)
            for metric, amount in normalized_usage.items():
                if not math.isfinite(amount):
                    raise ValueError(
                        f"Reserved usage value for {metric} must be finite (got {amount!r})"
                    )
                if amount < 0:
                    raise ValueError(
                        f"Reserved usage value for {metric} must be non-negative"
                    )
            return normalized_usage
        return value

    @field_validator("model_family", mode="before")
    @classmethod
    def _reject_empty_model_family(cls, value: object) -> object:
        return _validate_key_segment(value, field_name="model_family")

    @field_validator("limiter_instance_id", mode="before")
    @classmethod
    def _reject_empty_limiter_instance_id(cls, value: object) -> object:
        return _validate_key_segment(
            value,
            field_name="limiter_instance_id",
        )

    @field_validator("is_unlimited", mode="after")
    @classmethod
    def _require_sentinel_when_unlimited(
        cls,
        value: bool,  # noqa: FBT001  # Pydantic-driven validator signature.
        info: ValidationInfo,
    ) -> bool:
        """
        Couple ``is_unlimited=True`` to its semantic invariant.

        Rejects any reservation that flips the unlimited flag without also
        matching the canonical shape produced by the library's
        ``_unlimited_reservation`` factory: sentinel ``model_family``,
        empty ``usage``, ``bucket_ids is None``. Closes the V05/V10/V14
        bypass family by failing both ``model_validate`` and
        ``model_validate_json`` on hand-constructed or forged
        reservations whose flag does not match their shape.
        """
        if not value:
            return value
        family = info.data.get("model_family")
        usage = info.data.get("usage")
        bucket_ids = info.data.get("bucket_ids")
        if family != _UNLIMITED_FLAG:
            raise ValueError(
                f"is_unlimited=True requires model_family == {_UNLIMITED_FLAG!r}; "
                f"got {family!r}"
            )
        if usage:
            raise ValueError(
                "is_unlimited=True requires empty usage (got non-empty mapping)"
            )
        if bucket_ids is not None:
            raise ValueError(
                "is_unlimited=True requires bucket_ids=None (got non-None set)"
            )
        return value

    @field_validator("bucket_ids", mode="before")
    @classmethod
    def _normalize_bucket_ids(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, (set, frozenset, list, tuple)):
            try:
                value = tuple(value)
            except TypeError:
                return value

        normalized: set[BucketId] = set()
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:  # noqa: PLR2004
                raise ValueError("Each bucket_id must be a (metric, per_seconds) pair")
            metric, per_seconds = item
            metric = _validate_key_segment(metric, field_name="bucket_id metric")
            if _is_bool_like(per_seconds):
                raise ValueError("bucket_id per_seconds must not be a boolean")
            if not isinstance(per_seconds, int) or per_seconds <= 0:
                raise ValueError("bucket_id per_seconds must be a positive integer")
            normalized.add((metric, int(per_seconds)))
        return frozenset(normalized)

    def get_usage(self) -> FrozenUsage:
        return self.usage
