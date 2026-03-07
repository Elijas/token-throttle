from collections import defaultdict
from collections.abc import Iterator, Mapping
from enum import Enum
from typing import ClassVar, Self

from frozendict import frozendict
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class SecondsIn(int, Enum):
    MINUTE = 60
    HOUR = 3600
    DAY = 86400


class Quota(BaseModel):
    DEFAULT_SECONDS: ClassVar[int] = SecondsIn.MINUTE
    metric: str
    limit: float
    per_seconds: float = Field(
        default=DEFAULT_SECONDS,
        gt=0,  # Greater than 0
        description="Time window in seconds. Default: 60 (1 minute). E.g. For requests per minute, set to 60. For requests per hour, set to 3600.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class UsageQuotas:
    def __init__(
        self,
        quotas: list[Quota],
        /,
        *,
        _allow_empty_quotas: bool = False,
    ) -> None:
        self._metrics: defaultdict[str, dict[float, Quota]] = defaultdict(dict)
        if not _allow_empty_quotas and not quotas:
            logger.warning(
                "Empty quota list provided. No rate limiting will be applied. "
                "If this is intentional, use UsageQuotas.unlimited() instead.",
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
        return list(self._metrics[item].values())


MetricName = str
Usage = Mapping[MetricName, float]
FrozenUsage = frozendict[MetricName, float]

PerSeconds = float
BucketId = tuple[MetricName, PerSeconds]
Capacities = frozendict[BucketId, float]


def frozen_usage(usage: Usage) -> FrozenUsage:
    """Convert usage to a frozendict."""
    return frozendict({k: float(v) for k, v in usage.items()})


class CapacityReservation(BaseModel):
    usage: Usage
    model_family: str
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    def get_usage(self) -> FrozenUsage:
        return frozendict(self.usage)
