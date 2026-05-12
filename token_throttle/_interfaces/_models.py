import math
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterator, Mapping
from enum import Enum
from typing import ClassVar, Self

from frozendict import frozendict
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

_UNLIMITED_FLAG = "__rate_limiting_disabled__"
"""
Sentinel ``model_family`` value used by ``_unlimited_reservation``.

Lives in this module (not ``_validation``) so ``CapacityReservation``'s
``is_unlimited`` field validator can reference it without an import cycle.
``_validation`` re-exports it for back-compat with any external imports.
"""


def _is_bool_like(value: object) -> bool:
    """Reject Python bool values without duck-typing numeric lookalikes."""
    return type(value) is bool


class SecondsIn(int, Enum):
    MINUTE = 60
    HOUR = 3600
    DAY = 86400


class Quota(BaseModel):
    """
    Immutable rate-limit quota (frozen Pydantic model).

    ``frozen=True`` prevents mutation via normal attribute assignment.
    Direct ``object.__setattr__`` or ``__dict__`` writes bypass this at
    the Python level — this is a CPython limitation, not a library bug.
    Do not mutate instances after construction; hash stability depends
    on immutability.
    """

    DEFAULT_SECONDS: ClassVar[int] = SecondsIn.MINUTE
    metric: str
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
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    @field_validator("limit", "per_seconds", mode="before")
    @classmethod
    def _reject_boolean(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if _is_bool_like(value):
            raise ValueError(f"{info.field_name} must not be a boolean")
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
        if isinstance(value, str) and not value:
            raise ValueError("metric must not be empty")
        if isinstance(value, str) and not value.strip():
            raise ValueError("metric must not be whitespace-only")
        if isinstance(value, str) and ":" in value:
            raise ValueError(
                "metric must not contain ':' (used as Redis key separator)"
            )
        return value


class UsageQuotas:
    def __init__(
        self,
        quotas: list[Quota],
        /,
        *,
        _allow_empty_quotas: bool = False,
    ) -> None:
        self._metrics: defaultdict[str, dict[int, Quota]] = defaultdict(dict)
        if not _allow_empty_quotas and not quotas:
            warnings.warn(
                "Empty quota list provided. No rate limiting will be applied. "
                "If this is intentional, use UsageQuotas.unlimited() instead.",
                UserWarning,
                stacklevel=2,
            )
        for quota in quotas:
            self.add_metric(quota)

    @classmethod
    def unlimited(cls) -> Self:
        return cls([], _allow_empty_quotas=True)

    @property
    def is_unlimited(self) -> bool:
        return not bool(self._metrics)

    def add_metric(self, quota: Quota) -> None:
        if not isinstance(quota, Quota):
            raise ValueError(  # noqa: TRY004
                f"Each quota must be a Quota instance (got {type(quota).__name__})"
            )
        if (
            quota.metric in self._metrics
            and quota.per_seconds in self._metrics[quota.metric]
        ):
            raise ValueError(
                f"Metric {quota.metric} with {quota.per_seconds} seconds already exists.",
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
    try:
        value = float(amount)
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


class CapacityReservation(BaseModel):
    """
    Frozen reservation returned by acquire/record operations.

    Same ``frozen=True`` caveat as ``Quota``: ``object.__setattr__``
    and ``__dict__`` writes bypass Pydantic's freeze at the CPython level.
    Mutating ``is_unlimited`` via this vector would bypass metering.
    """

    reservation_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    usage: FrozenUsage
    model_family: str
    bucket_ids: frozenset[BucketId] | None = None
    model: str | None = None
    is_unlimited: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

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
            if not isinstance(metric, str) or not metric:
                raise ValueError("bucket_id metric must be a non-empty string")
            if _is_bool_like(per_seconds):
                raise TypeError("bucket_id per_seconds must not be a boolean")
            try:
                parsed_per_seconds = float(per_seconds)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "bucket_id per_seconds must be a positive integer"
                ) from exc
            if (
                not math.isfinite(parsed_per_seconds)
                or parsed_per_seconds <= 0
                or not parsed_per_seconds.is_integer()
            ):
                raise ValueError("bucket_id per_seconds must be a positive integer")
            normalized.add((metric, int(parsed_per_seconds)))
        return frozenset(normalized)

    def get_usage(self) -> FrozenUsage:
        return self.usage
