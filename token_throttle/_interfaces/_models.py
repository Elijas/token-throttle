import math
import unicodedata
import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal, Self, cast, overload

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
MAX_QUOTAS_PER_USAGE_QUOTAS = 1000
MAX_PER_SECONDS = 2**31 - 1


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


@overload
def _validate_key_segment(
    value: object,
    /,
    *,
    field_name: str,
    allow_none: Literal[False] = False,
    max_length: int | None = None,
) -> str: ...


@overload
def _validate_key_segment(
    value: object,
    /,
    *,
    field_name: str,
    allow_none: Literal[True],
    max_length: int | None = None,
) -> str | None: ...


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
    if any(unicodedata.category(char).startswith("C") for char in normalized):
        raise ValueError(f"{field_name} must not contain Unicode control characters")
    if any(not char.isprintable() for char in normalized):
        raise ValueError(f"{field_name} must not contain non-printable characters")
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
    """Common quota windows expressed in seconds."""

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

    Example usage:
    ```python
    from token_throttle import Quota

    quota = Quota(metric="tokens", limit=90_000, per_seconds=60)
    assert quota.metric == "tokens"
    ```
    """

    DEFAULT_SECONDS: ClassVar[int] = 60
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
        if _is_bool_like(value):
            return value
        if isinstance(value, SecondsIn):
            return int(value)
        if type(value) is int:
            return value
        if isinstance(value, int):
            raise ValueError(  # noqa: TRY004 - public validators raise ValueError.
                "per_seconds must be an exact int number of seconds "
                f"(got {type(value).__name__}); use a plain int such as 60"
            )
        if isinstance(value, float):
            return value
        raise ValueError(
            f"per_seconds must be int or float (got {type(value).__name__})"
        )

    @field_validator("per_seconds")
    @classmethod
    def _reject_huge_per_seconds(cls, value: int) -> int:
        if value > MAX_PER_SECONDS:
            raise ValueError(
                f"per_seconds must be <= {MAX_PER_SECONDS} seconds "
                f"(got {value!r}); choose a smaller quota window"
            )
        return value

    @field_validator("metric", mode="before")
    @classmethod
    def _reject_empty_metric(cls, value: object) -> object:
        return _validate_key_segment(value, field_name="metric")


class UsageQuotas:
    """
    Exact-type collection of per-metric quotas; empty only via ``unlimited()``.

    v2.0.0 contract: ``UsageQuotas`` is a data-transfer collection, not a
    subclass extension point. Security-sensitive callers accept the exact
    class and exact ``Quota`` instances only. Iterable inputs are materialized
    with a hard cap of 1000 entries to prevent unbounded generator consumption
    at validation boundaries.

    Example usage:
    ```python
    from token_throttle import Quota, UsageQuotas

    quotas = UsageQuotas([Quota(metric="requests", limit=60)])
    assert quotas.names == ["requests"]
    ```
    """

    def __init__(
        self,
        quotas: Iterable[Quota],
        /,
        *,
        _freeze: bool = False,
    ) -> None:
        self._frozen = False
        self._metrics: Mapping[str, Mapping[int, Quota]] = defaultdict(dict)
        quotas = self._materialize_quotas(quotas)
        if not quotas:
            raise ValueError(
                "Empty quota list provided. No rate limiting will be applied. "
                "If this is intentional, use UsageQuotas.unlimited() instead."
            )
        for quota in quotas:
            self.add_metric(quota)
        if _freeze:
            self._freeze()

    @classmethod
    def _construct_empty(cls, *, _freeze: bool = False) -> Self:
        """
        Build the empty (unlimited) quota set without a public escape hatch.

        Internal factory backing ``unlimited()`` and ``frozen_snapshot()``. It
        bypasses the empty-quota guard in ``__init__`` without exposing a
        reachable constructor kwarg; the public constructor always rejects an
        empty quota list. Callers outside this package must use
        ``UsageQuotas.unlimited()``.
        """
        instance = cls.__new__(cls)
        instance._frozen = False  # noqa: SLF001 - factory seeds its own instance state
        instance._metrics = defaultdict(dict)  # noqa: SLF001
        if _freeze:
            instance._freeze()  # noqa: SLF001
        return instance

    @classmethod
    def unlimited(cls, *, _freeze: bool = False) -> Self:
        """Return an explicit no-limit quota set for disabled rate limiting."""
        return cls._construct_empty(_freeze=_freeze)

    @staticmethod
    def _materialize_quotas(quotas: Iterable[Quota]) -> list[Quota]:
        if isinstance(quotas, Mapping):
            quota_iterable: Iterable[Quota] = quotas.values()
        elif isinstance(quotas, Iterable):
            quota_iterable = quotas
        else:
            raise ValueError(  # noqa: TRY004 - public validators raise ValueError.
                "quotas must be an iterable of Quota instances "
                f"(got {type(quotas).__name__})"
            )
        materialized: list[Quota] = []
        for index, quota in enumerate(quota_iterable, start=1):
            if index > MAX_QUOTAS_PER_USAGE_QUOTAS:
                raise CardinalityLimitExceededError(
                    "UsageQuotas accepts at most "
                    f"{MAX_QUOTAS_PER_USAGE_QUOTAS} entries; split the limiter "
                    "configuration or reduce the quota set"
                )
            materialized.append(quota)
        return materialized

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
        if self.is_unlimited:
            return self._construct_empty(_freeze=True)
        return type(self)(list(self), _freeze=True)

    @property
    def is_unlimited(self) -> bool:
        """Whether this quota set disables rate limiting."""
        return not bool(self._metrics)

    def add_metric(self, quota: Quota) -> None:
        """Add one exact ``Quota`` to this mutable quota set."""
        if self._frozen:
            raise TypeError("UsageQuotas snapshot is frozen")
        metrics = cast("defaultdict[str, dict[int, Quota]]", self._metrics)
        if type(quota) is not Quota:
            raise ValueError(
                f"Each quota must be a Quota instance (got {type(quota).__name__})"
            )
        quota = quota.revalidate()
        if quota.metric in metrics and quota.per_seconds in metrics[quota.metric]:
            existing = metrics[quota.metric][quota.per_seconds]
            raise ValueError(
                f"Metric {quota.metric} with {quota.per_seconds} seconds already "
                f"exists (existing limit={existing.limit}, new limit={quota.limit}).",
            )
        metrics[quota.metric][quota.per_seconds] = quota

    def __iter__(self) -> Iterator[Quota]:
        """Iterate over configured quotas."""
        for quotas in self._metrics.values():
            yield from quotas.values()

    @property
    def names(self) -> list[str]:
        """Metric names configured in this quota set."""
        return list(self._metrics.keys())

    def get_quotas(self, item: str) -> list[Quota]:
        """Return all quota windows for one metric name."""
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
    except Exception as exc:
        raise ValueError(
            f"{label} for {metric} must be finite (got {amount!r})"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} for {metric} must be finite (got {amount!r})")
    return value


def _materialize_usage_items(
    usage: Mapping[object, object],
) -> list[tuple[object, object]]:
    try:
        raw_items = usage.items()
    except Exception as exc:
        raise ValueError(
            "usage must yield consistent metric/value pairs "
            f"(got {type(exc).__name__}: {exc})"
        ) from exc

    materialized: list[tuple[object, object]] = []
    try:
        for raw_item in raw_items:
            try:
                metric, amount = raw_item
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"usage must yield metric/value pairs (got {raw_item!r})"
                ) from exc
            materialized.append((metric, amount))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "usage must yield consistent metric/value pairs "
            f"(got {type(exc).__name__}: {exc})"
        ) from exc
    return materialized


def frozen_usage(usage: object) -> FrozenUsage:
    """Convert usage to a frozendict."""
    if not isinstance(usage, Mapping):
        raise ValueError(  # noqa: TRY004
            f"usage must be a mapping (got {type(usage).__name__})"
        )
    converted: dict[MetricName, float] = {}
    seen_metrics: set[str] = set()
    for metric, amount in _materialize_usage_items(usage):
        if not isinstance(metric, str) or not metric:
            raise ValueError(
                f"Usage metric key must be a non-empty string (got {metric!r})"
            )
        normalized_metric = _validate_key_segment(metric, field_name="metric")
        if normalized_metric in seen_metrics:
            raise ValueError("usage must not contain duplicate metric keys")
        seen_metrics.add(normalized_metric)
        converted[normalized_metric] = _coerce_usage_value(normalized_metric, amount)
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

    Example usage:
    ```python
    from token_throttle import CapacityReservation

    reservation = CapacityReservation(
        usage={"tokens": 500},
        model_family="gpt-4o",
        limiter_instance_id="limiter-1",
    )
    assert reservation.usage["tokens"] == 500.0
    ```
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
    created_at_seconds: float | None = Field(
        default=None,
        description=(
            "Wall-clock Unix timestamp recorded by the limiter when the "
            "reservation is issued. None is accepted for backward-compatible "
            "deserialization, but bounded reservation lifetimes require it."
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

    @field_validator("created_at_seconds", mode="before")
    @classmethod
    def _validate_created_at_seconds(cls, value: object) -> object:
        if value is None:
            return None
        if _is_bool_like(value) or not isinstance(value, (int, float)):
            raise ValueError("created_at_seconds must be finite")
        value_float = float(value)
        if not math.isfinite(value_float):
            raise ValueError("created_at_seconds must be finite")
        return value_float

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
            if not isinstance(value, Iterable):
                return value
            value = tuple(value)

        normalized: set[BucketId] = set()
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:  # noqa: PLR2004
                raise ValueError("Each bucket_id must be a (metric, per_seconds) pair")
            metric, per_seconds = item
            metric = _validate_key_segment(metric, field_name="bucket_id metric")
            if _is_bool_like(per_seconds):
                raise ValueError("bucket_id per_seconds must not be a boolean")
            if type(per_seconds) is not int or per_seconds <= 0:
                raise ValueError("bucket_id per_seconds must be a positive integer")
            if per_seconds > MAX_PER_SECONDS:
                raise ValueError(
                    f"bucket_id per_seconds must be <= {MAX_PER_SECONDS} seconds"
                )
            normalized.add((metric, int(per_seconds)))
        return frozenset(normalized)

    def get_usage(self) -> FrozenUsage:
        """Return the immutable usage reserved by this reservation."""
        return self.usage


@dataclass(frozen=True, slots=True)
class ReservationAuthoritySnapshot:
    """Internal immutable authority for refunding an issued reservation."""

    reservation_id: str
    usage: FrozenUsage
    model_family: str
    bucket_ids: frozenset[BucketId] | None
    model: str | None
    is_unlimited: bool
    limiter_instance_id: str
    created_at_seconds: float | None

    @classmethod
    def from_reservation(
        cls,
        reservation: CapacityReservation,
    ) -> "ReservationAuthoritySnapshot":
        reservation = reservation.revalidate()
        return cls(
            reservation_id=reservation.reservation_id,
            usage=reservation.usage,
            model_family=reservation.model_family,
            bucket_ids=reservation.bucket_ids,
            model=reservation.model,
            is_unlimited=reservation.is_unlimited,
            limiter_instance_id=reservation.limiter_instance_id,
            created_at_seconds=reservation.created_at_seconds,
        )

    def to_reservation(self) -> CapacityReservation:
        """Convert the internal snapshot back to a public reservation DTO."""
        return CapacityReservation(
            reservation_id=self.reservation_id,
            usage=self.usage,
            model_family=self.model_family,
            bucket_ids=self.bucket_ids,
            model=self.model,
            is_unlimited=self.is_unlimited,
            limiter_instance_id=self.limiter_instance_id,
            created_at_seconds=self.created_at_seconds,
        )
