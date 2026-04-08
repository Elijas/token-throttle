import re


def openai_model_family_getter(model: str, /) -> str:
    # Strip provider prefix if present, then collapse date/snapshot suffixes.
    # Matches -MMDD (e.g. -0613), -YYYYMMDD (e.g. -20241203), and
    # -YYYY-MM-DD (e.g. -2024-04-09), with optional -preview before
    # and/or after the date component.
    # Single/triple-digit version numbers (-1, -002) are preserved.
    model = model.removeprefix("openai/")
    result = re.sub(
        r"((-preview)?-(?:\d{8}|\d{4}(?:-\d{2}){0,2})(-preview)?|-preview)$",
        "",
        model,
    )
    return result or model
