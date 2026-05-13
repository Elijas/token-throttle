from __future__ import annotations

import asyncio
import json
import math
import threading
import typing
from typing import Protocol, cast, runtime_checkable

if typing.TYPE_CHECKING:
    from tiktoken import Encoding

from frozendict import frozendict

from token_throttle._interfaces._models import FrozenUsage, _is_bool_like

_OUTPUT_BUDGET_KEYS = (
    "max_output_tokens",
    "max_completion_tokens",
    "max_tokens",
)
_OUTPUT_MULTIPLIER_KEYS = ("n", "best_of")
_REQUEST_PAYLOAD_KEYS = ("input", "messages")
_REQUEST_CONTEXT_KEYS = (
    "instructions",
    "tools",
    "functions",
    "response_format",
    "text",
)
# Fields whose values are JSON schemas on the wire (tool/function definitions,
# structured-output response schemas). Counting these via a recursive walk over
# keys+values misses the JSON structural tokens ({, }, [, ], :, ",", whitespace)
# which can be the majority of the wire-format cost. Serialize with json.dumps
# and encode the full string so the counter matches what the API actually sees.
_JSON_SERIALIZED_CONTEXT_KEYS = frozenset({"tools", "functions", "response_format"})
_UNSUPPORTED_CONTENT_PART_TYPES = frozenset(
    {
        "input_audio",
        "input_file",
        "input_image",
        "output_audio",
        "output_image",
        "computer_use_screenshot",
    },
)
# NOTE: Responses-API content-part types like "mcp_tool_result" and
# "mcp_list_tools" are not denied here because their payloads can contain
# countable text.  "reasoning" items likewise carry text summaries.  Only
# types whose content is fundamentally non-textual (binary, image, audio)
# belong in the deny set.  New non-text types should be added as the API
# evolves.
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
    def __call__(self, model_name: str) -> Encoding: ...


class OpenAIUsageCounter:
    """
    Estimate OpenAI request usage for text, chat, tool, and function payloads.

    Supported chat message shapes include string/None message fields, string
    content parts such as ``{"type": "input_text", "text": "..."}``, and
    JSON-serializable ``tool_calls`` / ``function_call`` payloads. Non-text
    image, file, and audio content is rejected because token cost cannot be
    inferred locally.
    """

    def __init__(self, get_encoding_func: EncodingGetter | None = None):
        self._get_encoding = get_encoding_func or get_encoding
        self._encoding_cache: dict[str, Encoding] = {}
        self._encoding_cache_lock = threading.RLock()

    def __call__(self, model: str, **request) -> FrozenUsage:
        self._validate_model(model)
        encoding = self._get_cached_encoding(model)
        return self._count_with_encoding(model, encoding, request)

    async def count_request_async(self, model: str, **request) -> FrozenUsage:
        self._validate_model(model)
        encoding = await self._get_cached_encoding_async(model)
        return self._count_with_encoding(model, encoding, request)

    async def warmup_models(self, models: list[str]) -> None:
        """Pre-load tokenizers in executor threads during async app startup."""
        for model in models:
            self._validate_model(model)
        await asyncio.gather(
            *(self._get_cached_encoding_async(model) for model in models)
        )

    def _get_cached_encoding(self, model: str) -> Encoding:
        with self._encoding_cache_lock:
            encoding = self._encoding_cache.get(model)
        if encoding is not None:
            return encoding

        encoding = self._get_encoding(model)
        with self._encoding_cache_lock:
            return self._encoding_cache.setdefault(model, encoding)

    async def _get_cached_encoding_async(self, model: str) -> Encoding:
        with self._encoding_cache_lock:
            encoding = self._encoding_cache.get(model)
        if encoding is not None:
            return encoding

        encoding = await asyncio.to_thread(self._get_encoding, model)
        with self._encoding_cache_lock:
            return self._encoding_cache.setdefault(model, encoding)

    def _count_with_encoding(
        self,
        model: str,
        encoding: Encoding,
        request: dict[str, object],
    ) -> FrozenUsage:
        self._validate_model(model)
        _validate_max_kwargs(request)
        reserved_output_tokens = _get_reserved_output_tokens(request)
        payload_key = _get_request_payload_key(request)
        request_context_tokens = _count_request_context_tokens(encoding, request)

        if payload_key == "input":
            input_ = request["input"]
            if isinstance(input_, str):
                tokens = (
                    len(encoding.encode(input_))
                    + request_context_tokens
                    + reserved_output_tokens
                )
                return frozendict({"tokens": tokens, "requests": 1})
            # List of strings — valid for OpenAI Embeddings API (e.g. input=["hello", "world"]).
            if isinstance(input_, list) and all(isinstance(i, str) for i in input_):
                tokens = (
                    sum(len(encoding.encode(i)) for i in input_)
                    + request_context_tokens
                    + reserved_output_tokens
                )
                return frozendict({"tokens": tokens, "requests": 1})
            # OpenAI Embeddings also accepts pre-tokenized payloads such as
            # input=[1, 2, 3] or input=[[1, 2], [3, 4]].
            pretokenized_tokens = _count_pretokenized_input_tokens(input_)
            if pretokenized_tokens is not None:
                return frozendict(
                    {
                        "tokens": (
                            pretokenized_tokens
                            + request_context_tokens
                            + reserved_output_tokens
                        ),
                        "requests": 1,
                    }
                )
            tokens = (
                count_structured_input_tokens(encoding, input_)
                + request_context_tokens
                + reserved_output_tokens
            )
            return frozendict({"tokens": tokens, "requests": 1})

        if payload_key == "messages":
            messages = request["messages"]
            if not isinstance(messages, list):
                raise ValueError(
                    f"messages must be a list of dicts (got {type(messages).__name__})"
                )
            if not all(isinstance(m, dict) for m in messages):
                raise ValueError("All messages must be dicts")
            tokens = (
                count_chat_input_tokens(
                    encoding,
                    messages=cast("list[dict[str, object]]", messages),
                )
                + request_context_tokens
                + reserved_output_tokens
            )
            return frozendict({"tokens": tokens, "requests": 1})

        raise ValueError("Request must contain 'input' or 'messages'")

    @staticmethod
    def _validate_model(model: object) -> None:
        if not isinstance(model, str):
            raise TypeError(
                f"model must be a non-empty string (got {type(model).__name__})"
            )
        if not model:
            raise ValueError("model must be a non-empty string")


def _get_request_payload_key(request: dict[str, object]) -> str:
    payload_keys = [key for key in _REQUEST_PAYLOAD_KEYS if key in request]
    if not payload_keys:
        raise ValueError("Request must contain 'input' or 'messages'")
    if len(payload_keys) > 1:
        raise ValueError("Exactly one of 'input' or 'messages' must be provided")
    return payload_keys[0]


def get_encoding(model_name: str) -> Encoding:
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
    encoding: Encoding,
    input_: object,
) -> int:
    """Count tokens for OpenAI Responses-style structured input payloads."""
    invalid_error = (
        "The value of 'input' must be of type str or a list/dict of "
        f"structured input items (got {type(input_).__name__})"
    )
    if isinstance(input_, dict):
        if _looks_like_message(input_):
            return count_chat_input_tokens(encoding, messages=[input_])
        unsupported = _get_unsupported_content_part_name(
            input_
        ) or _check_nested_unsupported_content(input_)
        if unsupported is not None:
            raise _unsupported_content_part_error(unsupported)
        return _count_json_serialized_tokens(
            encoding,
            input_,
            invalid_error=invalid_error,
        )
    if isinstance(input_, list):
        if not all(isinstance(item, dict) for item in input_):
            raise ValueError(invalid_error)
        items = cast("list[dict[str, object]]", input_)
        message_items = [item for item in items if _looks_like_message(item)]
        if len(message_items) == len(items):
            return count_chat_input_tokens(encoding, messages=items)
        if not message_items:
            for item in items:
                unsupported = _get_unsupported_content_part_name(
                    item
                ) or _check_nested_unsupported_content(item)
                if unsupported is not None:
                    raise _unsupported_content_part_error(unsupported)
            return _count_json_serialized_tokens(
                encoding,
                items,
                invalid_error=invalid_error,
            )
        structured_items = [item for item in items if not _looks_like_message(item)]
        for item in structured_items:
            unsupported = _get_unsupported_content_part_name(
                item
            ) or _check_nested_unsupported_content(item)
            if unsupported is not None:
                raise _unsupported_content_part_error(unsupported)
        return count_chat_input_tokens(
            encoding,
            messages=message_items,
        ) + _count_json_serialized_tokens(
            encoding,
            structured_items,
            invalid_error=invalid_error,
        )
    raise ValueError(invalid_error)


def _looks_like_message(value: dict[str, object]) -> bool:
    return "role" in value


def _get_reserved_output_tokens(request: dict[str, object]) -> int:
    budgets: list[int] = []
    for key in _OUTPUT_BUDGET_KEYS:
        raw_value = request.get(key)
        if raw_value is None:
            continue
        budgets.append(_parse_non_negative_int(raw_value, key))
    return max(budgets, default=0) * _get_output_multiplier(request)


def _get_output_multiplier(request: dict[str, object]) -> int:
    multipliers: list[int] = []
    for key in _OUTPUT_MULTIPLIER_KEYS:
        raw_value = request.get(key)
        if raw_value is None:
            continue
        multipliers.append(_parse_non_negative_int(raw_value, key))
    return max(multipliers, default=1)


def _validate_max_kwargs(request: dict[str, object]) -> None:
    unknown_max_keys = [
        key
        for key in request
        if key.startswith("max_") and key not in _OUTPUT_BUDGET_KEYS
    ]
    if unknown_max_keys:
        known = ", ".join(_OUTPUT_BUDGET_KEYS)
        raise ValueError(
            f"Unknown OpenAI max_* token budget field(s): {unknown_max_keys}. "
            f"Expected one of: {known}."
        )


def _count_request_context_tokens(
    encoding: Encoding,
    request: dict[str, object],
) -> int:
    total = 0
    for key in _REQUEST_CONTEXT_KEYS:
        raw_value = request.get(key)
        if raw_value is None:
            continue
        invalid_error = f"Unsupported value for request field '{key}'"
        if key in _JSON_SERIALIZED_CONTEXT_KEYS:
            total += _count_json_serialized_tokens(
                encoding,
                raw_value,
                invalid_error=invalid_error,
            )
        else:
            total += _count_request_context_fragments(
                encoding,
                raw_value,
                invalid_error=invalid_error,
            )
    return total


def _count_json_serialized_tokens(
    encoding: Encoding,
    value: object,
    *,
    invalid_error: str,
) -> int:
    try:
        serialized = json.dumps(value)
    except TypeError as exc:
        raise ValueError(invalid_error) from exc
    return len(encoding.encode(serialized))


def _count_request_context_fragments(
    encoding: Encoding,
    value: object,
    *,
    invalid_error: str,
) -> int:
    if value is None:
        return 0
    if _is_bool_like(value):
        return len(encoding.encode(str(bool(value)).lower()))
    if isinstance(value, str):
        return len(encoding.encode(value))
    if isinstance(value, int | float):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(invalid_error)
        return len(encoding.encode(str(value)))
    if isinstance(value, list):
        return sum(
            _count_request_context_fragments(
                encoding,
                item,
                invalid_error=invalid_error,
            )
            for item in value
        )
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(invalid_error)
        return sum(
            len(encoding.encode(key))
            + _count_request_context_fragments(
                encoding,
                nested_value,
                invalid_error=invalid_error,
            )
            for key, nested_value in value.items()
        )
    raise ValueError(invalid_error)


def _parse_non_negative_int(value: object, field_name: str) -> int:
    if _is_bool_like(value) or not isinstance(value, int):
        raise ValueError(f"'{field_name}' must be a finite non-negative integer")

    if value < 0:
        raise ValueError(f"'{field_name}' must be a finite non-negative integer")
    return value


def _is_token_id(value: object) -> bool:
    # Ambiguity: any non-negative int matches. A list like [1, 2, 3] could be
    # text content rather than pre-tokenized IDs. Without API-level context
    # (e.g. an explicit "encoding" flag), there is no way to distinguish the
    # two; this heuristic is the best available approach.
    return isinstance(value, int) and not _is_bool_like(value) and value >= 0


def _count_pretokenized_input_tokens(input_: object) -> int | None:
    if not isinstance(input_, list):
        return None
    if all(_is_token_id(token) for token in input_):
        return len(input_)
    if all(isinstance(item, list) for item in input_) and all(
        _is_token_id(token) for item in input_ for token in item
    ):
        return sum(len(item) for item in input_)
    return None


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


def _check_nested_unsupported_content(value: object) -> str | None:
    """
    Walk a value tree and return the first unsupported content part type, if any.

    Only checks ``type``-based matches (not field-level matches like
    ``image_url``) to avoid false positives on string fields that happen
    to share a name with an unsupported content field.
    """
    if isinstance(value, dict):
        part_type = value.get("type")
        if isinstance(part_type, str) and part_type in _UNSUPPORTED_CONTENT_PART_TYPES:
            return part_type
        for nested in value.values():
            result = _check_nested_unsupported_content(nested)
            if result is not None:
                return result
    elif isinstance(value, list):
        for item in value:
            result = _check_nested_unsupported_content(item)
            if result is not None:
                return result
    return None


def _count_text_fragments(
    encoding: Encoding,
    value: object,
    *,
    invalid_error: str,
    content_part_context: bool = False,
    coerce_scalars: bool = False,
) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(encoding.encode(value))
    if coerce_scalars and _is_bool_like(value):
        return len(encoding.encode(str(bool(value)).lower()))
    if coerce_scalars and isinstance(value, int | float):
        return len(encoding.encode(str(value)))
    if isinstance(value, list):
        return sum(
            _count_text_fragments(
                encoding,
                item,
                invalid_error=invalid_error,
                content_part_context=content_part_context,
                coerce_scalars=coerce_scalars,
            )
            for item in value
        )
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(invalid_error)
        if content_part_context:
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
                    content_part_context=content_part_context,
                    coerce_scalars=coerce_scalars,
                )
            return sum(
                _count_text_fragments(
                    encoding,
                    nested_value,
                    invalid_error=invalid_error,
                    content_part_context=content_part_context
                    and nested_key == "content",
                    coerce_scalars=coerce_scalars and nested_key != "content",
                )
                for nested_key, nested_value in value.items()
            )
    raise ValueError(invalid_error)


def count_chat_input_tokens(
    encoding: Encoding,
    messages: list[dict[str, object]],
    **_,
) -> int:
    """Calculate tokens for a chat completion request."""
    num_tokens = 0

    for message in messages:
        if not all(isinstance(key, str) for key in message):
            raise ValueError("All message dict keys must be strings")
        # Per-message frame for current chat models (gpt-4, gpt-4o,
        # gpt-3.5-turbo) from the OpenAI token-count cookbook.
        num_tokens += 3

        for key, value in message.items():
            if key in ("tool_calls", "function_call"):
                num_tokens += _count_json_serialized_tokens(
                    encoding,
                    value,
                    invalid_error="All keys and values in messages must be of type str",
                )
            else:
                num_tokens += _count_text_fragments(
                    encoding,
                    value,
                    invalid_error="All keys and values in messages must be of type str",
                    content_part_context=key == "content",
                    coerce_scalars=key != "content",
                )

            if key == "name":
                num_tokens += 1

    # Trailing assistant prime: <|start|>assistant<|message|>.
    num_tokens += 3
    return num_tokens
