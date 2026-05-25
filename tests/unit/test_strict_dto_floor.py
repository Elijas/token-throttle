"""Regression coverage for dependency floors used by StrictDTO."""

from token_throttle._dto import StrictDTO


class _ExtraDTO(StrictDTO):
    value: int


def test_model_validate_accepts_extra_keyword() -> None:
    dto = _ExtraDTO.model_validate({"value": 1, "extra_value": 2}, extra="allow")

    assert dto.value == 1
    assert dto.model_extra == {"extra_value": 2}
