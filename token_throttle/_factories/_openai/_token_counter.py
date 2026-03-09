import typing
from typing import Protocol, cast, runtime_checkable

if typing.TYPE_CHECKING:
    from tiktoken import Encoding

from frozendict import frozendict

from token_throttle._interfaces._models import FrozenUsage


@runtime_checkable
class EncodingGetter(Protocol):
    def __call__(self, model_name: str) -> "Encoding": ...


class OpenAIUsageCounter:
    def __init__(self, get_encoding_func: EncodingGetter | None = None):
        self._get_encoding = get_encoding_func or get_encoding

    def __call__(self, model: str, **request) -> FrozenUsage:
        encoding = self._get_encoding(model)

        if "input" in request:
            if not isinstance(request["input"], str):
                raise ValueError("The value of 'input' must be of type str")
            tokens = len(encoding.encode(request["input"]))
            return frozendict({"tokens": tokens, "requests": 1})

        if "inputs" in request:
            if not all(isinstance(i, str) for i in request["inputs"]):
                raise ValueError("All values in 'inputs' must be of type str")
            tokens = sum(len(encoding.encode(i)) for i in request["inputs"])
            return frozendict({"tokens": tokens, "requests": 1})

        if "messages" in request:
            if not all(isinstance(m, dict) for m in request["messages"]):
                raise ValueError("All messages must be dicts")
            if not all(
                isinstance(k, str) and isinstance(v, str)
                for message in request["messages"]
                for k, v in message.items()
            ):
                raise ValueError("All keys and values in messages must be of type str")
            messages = cast("list[dict[str, str]]", request["messages"])
            tokens = count_chat_input_tokens(
                encoding,
                messages=messages,
            )
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
    for model_name_substring, encoding_name in substring_to_encoding.items():
        if model_name_substring in model_name:
            return tiktoken.get_encoding(encoding_name)
    return tiktoken.encoding_for_model(model_name)


def count_chat_input_tokens(
    encoding: "Encoding",
    messages: list[dict[str, str]],
    **_,
) -> int:
    """Calculate tokens for a chat completion request."""
    num_tokens = 0

    for message in messages:
        num_tokens += 4  # <im_start>{role/name}\n{content}<im_end>\n

        for key, value in message.items():
            num_tokens += len(encoding.encode(value))

            if key == "name":  # If there's a name, the role is omitted
                num_tokens -= 1

    num_tokens += 2  # <im_start>assistant
    return num_tokens
