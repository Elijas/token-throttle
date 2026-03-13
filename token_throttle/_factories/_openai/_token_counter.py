import math
import typing
from typing import Protocol, cast, runtime_checkable

if typing.TYPE_CHECKING:
    from tiktoken import Encoding

from frozendict import frozendict

from token_throttle._interfaces._models import FrozenUsage

_OUTPUT_BUDGET_KEYS = (
    "max_output_tokens",
    "max_completion_tokens",
    "max_tokens",
)
_UNSUPPORTED_CONTENT_PART_TYPES = frozenset(
    {
        "input_audio",
        "input_file",
        "input_image",
    },
)
_UNSUPPORTED_CONTENT_FIELDS = (
    "audio",
    "audio_url",
    "file",
    "file_id",
    "file_url",
    "image",
    "image_url",
)


@runtime_checkable
class EncodingGetter(Protocol):
    def __call__(self, model_name: str) -> "Encoding": ...


class OpenAIUsageCounter:
    def __init__(self, get_encoding_func: EncodingGetter | None = None):
        self._get_encoding = get_encoding_func or get_encoding

    def __call__(self, model: str, **request) -> FrozenUsage:
        encoding = self._get_encoding(model)
        reserved_output_tokens = _get_reserved_output_tokens(request)

        if "input" in request:
            input_ = request["input"]
            if isinstance(input_, str):
                tokens = len(encoding.encode(input_)) + reserved_output_tokens
                return frozendict({"tokens": tokens, "requests": 1})
            tokens = (
                count_structured_input_tokens(encoding, input_)
                + reserved_output_tokens
            )
            return frozendict({"tokens": tokens, "requests": 1})

        if "inputs" in request:
            if not all(isinstance(i, str) for i in request["inputs"]):
                raise ValueError("All values in 'inputs' must be of type str")
            tokens = (
                sum(len(encoding.encode(i)) for i in request["inputs"])
                + reserved_output_tokens
            )
            return frozendict({"tokens": tokens, "requests": 1})

        if "messages" in request:
            messages = request["messages"]
            if not isinstance(messages, list) or not all(
                isinstance(m, dict) for m in messages
            ):
                raise ValueError("All messages must be dicts")
            tokens = count_chat_input_tokens(
                encoding,
                messages=cast("list[dict[str, object]]", messages),
            ) + reserved_output_tokens
            return frozendict({"tokens": tokens, "requests": 1})

        raise ValueError("Request must contain 'input', 'inputs', or 'messages'")


def get_encoding(model_name: str) -> "Encoding":
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError(
            'The "tiktoken" package is required for OpenAI token counting. '
            'Install it with: pip install "token-throttle[tiktoken]"'
        ) from exc

    model_name = model_name.removeprefix("openai/")
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        pass

    substring_to_encoding = {
        "gpt-4o-mini": "o200k_base",
        "gpt-4o": "o200k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        "text-davinci-002": "p50k_base",
        "text-davinci-003": "p50k_base",
        "davinci": "r50k_base",
        "codex": "p50k_base",
    }
    for model_name_substring, encoding_name in substring_to_encoding.items():
        if model_name_substring == model_name:
            return tiktoken.get_encoding(encoding_name)
    for model_name_substring, encoding_name in sorted(
        substring_to_encoding.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if model_name_substring in model_name:
            return tiktoken.get_encoding(encoding_name)
    return tiktoken.encoding_for_model(model_name)


def count_structured_input_tokens(
    encoding: "Encoding",
    input_: object,
) -> int:
    """Count tokens for OpenAI Responses-style structured input payloads."""
    if isinstance(input_, dict):
        if _looks_like_message(input_):
            return count_chat_input_tokens(encoding, messages=[input_])
        return _count_text_fragments(
            encoding,
            input_,
            invalid_error=(
                "The value of 'input' must be of type str or a list/dict of "
                "structured input items"
            ),
        )
    if isinstance(input_, list):
        if not all(isinstance(item, dict) for item in input_):
            raise ValueError(
                "The value of 'input' must be of type str or a list/dict of "
                "structured input items"
            )
        items = cast("list[dict[str, object]]", input_)
        if all(_looks_like_message(item) for item in items):
            return count_chat_input_tokens(encoding, messages=items)
        return _count_text_fragments(
            encoding,
            items,
            invalid_error=(
                "The value of 'input' must be of type str or a list/dict of "
                "structured input items"
            ),
        )
    raise ValueError(
        "The value of 'input' must be of type str or a list/dict of "
        "structured input items"
    )


def _looks_like_message(value: dict[str, object]) -> bool:
    return "role" in value or "content" in value or "name" in value


def _get_reserved_output_tokens(request: dict[str, object]) -> int:
    budgets: list[int] = []
    for key in _OUTPUT_BUDGET_KEYS:
        raw_value = request.get(key)
        if raw_value is None:
            continue
        budgets.append(_parse_non_negative_int(raw_value, key))
    return max(budgets, default=0)


def _parse_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"'{field_name}' must be a finite non-negative integer")

    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or not parsed.is_integer():
        raise ValueError(f"'{field_name}' must be a finite non-negative integer")
    return int(parsed)


def _unsupported_content_part_error(part_type: str) -> ValueError:
    return ValueError(
        f"Structured content part type '{part_type}' is not supported by "
        "OpenAIUsageCounter; pass usage manually for non-text inputs."
    )


def _get_unsupported_content_part_name(value: dict[str, object]) -> str | None:
    part_type = value.get("type")
    if isinstance(part_type, str) and part_type in _UNSUPPORTED_CONTENT_PART_TYPES:
        return part_type
    for field in _UNSUPPORTED_CONTENT_FIELDS:
        if field in value:
            return field
    return None


def _count_text_fragments(
    encoding: "Encoding",
    value: object,
    *,
    invalid_error: str,
) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        # bool before int: isinstance(True, int) is True in Python.
        # Serialize as JSON would ("true"/"false") and count tokens.
        return len(encoding.encode(str(value).lower()))
    if isinstance(value, str):
        return len(encoding.encode(value))
    if isinstance(value, (int, float)):
        return len(encoding.encode(str(value)))
    if isinstance(value, list):
        return sum(
            _count_text_fragments(
                encoding,
                item,
                invalid_error=invalid_error,
            )
            for item in value
        )
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(invalid_error)
        unsupported_part_name = _get_unsupported_content_part_name(value)
        if unsupported_part_name is not None:
            raise _unsupported_content_part_error(unsupported_part_name)
        part_type = value.get("type")
        if isinstance(part_type, str):
            if "text" in value:
                text = value["text"]
                if not isinstance(text, str):
                    raise ValueError(invalid_error)
                return len(encoding.encode(text))
            if "content" in value:
                return _count_text_fragments(
                    encoding,
                    value["content"],
                    invalid_error=invalid_error,
                )
        return sum(
            _count_text_fragments(
                encoding,
                nested_value,
                invalid_error=invalid_error,
            )
            for nested_value in value.values()
        )
    raise ValueError(invalid_error)


def count_chat_input_tokens(
    encoding: "Encoding",
    messages: list[dict[str, object]],
    **_,
) -> int:
    """Calculate tokens for a chat completion request."""
    num_tokens = 0

    for message in messages:
        if not all(isinstance(key, str) for key in message):
            raise ValueError("All message dict keys must be strings")
        num_tokens += 4  # <im_start>{role/name}\n{content}<im_end>\n

        for key, value in message.items():
            num_tokens += _count_text_fragments(
                encoding,
                value,
                invalid_error="All keys and values in messages must be of type str",
            )

            if key == "name":  # If there's a name, the role is omitted
                num_tokens -= 1

    num_tokens += 2  # <im_start>assistant
    return num_tokens
