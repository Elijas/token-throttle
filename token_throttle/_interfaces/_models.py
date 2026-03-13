import math
import warnings
from collections import defaultdict
from collections.abc import Iterator, Mapping
from enum import Enum
from typing import ClassVar, Self

from frozendict import frozendict
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class SecondsIn(int, Enum):
    MINUTE = 60
    HOUR = 3600
    DAY = 86400


class Quota(BaseModel):
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
        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004
                f"{info.field_name} must not be a boolean"
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
    if isinstance(amount, bool):
        raise ValueError(  # noqa: TRY004
            f"{label} for {metric} must not be a boolean"
        )
    try:
        return float(amount)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} for {metric} must be finite (got {amount!r})"
        ) from exc


def frozen_usage(usage: Usage) -> FrozenUsage:
    """Convert usage to a frozendict."""
    converted: dict[MetricName, float] = {}
    for metric, amount in usage.items():
        converted[metric] = _coerce_usage_value(metric, amount)
    return frozendict(converted)


class CapacityReservation(BaseModel):
    usage: Usage
    model_family: str
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    @field_validator("usage", mode="before")
    @classmethod
    def _normalize_usage(cls, value: object) -> object:
        if isinstance(value, Mapping):
            normalized_usage = dict(frozen_usage(value))
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

    def get_usage(self) -> FrozenUsage:
        return frozen_usage(self.usage)
