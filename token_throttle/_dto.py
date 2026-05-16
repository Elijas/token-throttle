from __future__ import annotations

from copy import deepcopy as _deepcopy
from typing import TYPE_CHECKING, Any, Self, override

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic.config import ExtraValues

STRICT_DTO_CONFIG: ConfigDict = ConfigDict(
    arbitrary_types_allowed=True,
    extra="forbid",
    frozen=True,
    revalidate_instances="always",
    strict=True,
    validate_assignment=True,
)


class StrictDTO(BaseModel):
    """Base for exact-type public DTOs with composable validation."""

    model_config = STRICT_DTO_CONFIG

    @classmethod
    @override
    def model_validate(
        cls,
        obj: object,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        from_attributes: bool | None = None,
        context: object | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        _ = strict
        return super().model_validate(
            obj,
            strict=True,
            extra=extra,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    @override
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: object | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        _ = strict
        return super().model_validate_json(
            json_data,
            strict=True,
            extra=extra,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def model_construct(cls, *_args: object, **_kwargs: object) -> Self:
        raise TypeError(
            f"{cls.__name__}.model_construct is disabled in v2.0.0; use model_validate."
        )

    @staticmethod
    def _state_fields_for_validation(state: dict[str, object]) -> dict[str, object]:
        raw_fields = state.get("__dict__", state)
        fields = dict(raw_fields) if isinstance(raw_fields, dict) else {}
        extra = state.get("__pydantic_extra__")
        if isinstance(extra, dict):
            fields.update(extra)
        return fields

    def _dump_for_revalidation(self) -> dict[str, object]:
        dump = {
            field_name: self.__dict__[field_name]
            for field_name in type(self).model_fields
            if field_name in self.__dict__
        }
        extra_fields = set(self.__dict__) - set(type(self).model_fields)
        for field_name in extra_fields:
            dump[field_name] = self.__dict__[field_name]
        return dump

    def _state_for_copy(self) -> dict[str, object]:
        state = self.__getstate__()
        if not isinstance(state, dict):  # pragma: no cover - pydantic invariant
            raise TypeError(
                f"{type(self).__name__}.__getstate__ returned "
                f"{type(state).__name__}, expected dict"
            )
        return state

    def revalidate(self) -> Self:
        """Return a freshly validated copy of this exact DTO."""
        return type(self).model_validate(self._dump_for_revalidation())

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        dump = self._dump_for_revalidation()
        if update is not None:
            dump.update(update)
        if deep:
            dump = _deepcopy(dump)
        return type(self).model_validate(dump)

    def __setstate__(self, state: dict[str, object]) -> None:
        validated = type(self).model_validate(self._state_fields_for_validation(state))
        super().__setstate__(validated.__getstate__())

    def __copy__(self) -> Self:
        copied = type(self).__new__(type(self))
        copied.__setstate__(self._state_for_copy())
        return copied

    def __deepcopy__(self, memo: dict[int, object] | None = None) -> Self:
        copied = type(self).__new__(type(self))
        copied.__setstate__(_deepcopy(self._state_for_copy(), memo=memo))
        return copied
