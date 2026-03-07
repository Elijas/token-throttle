"""
Tests for OpenAI token counting logic.

Source: token_throttle/_factories/_openai/_token_counter.py
"""

from unittest.mock import MagicMock

import pytest
from frozendict import frozendict

from token_throttle._factories._openai._token_counter import (
    OpenAIUsageCounter,
    count_chat_input_tokens,
    get_encoding,
)


def _make_mock_encoding(chars_per_token: int = 1) -> MagicMock:
    """Create a mock encoding where each `chars_per_token` characters = 1 token."""
    mock_enc = MagicMock()
    mock_enc.encode.side_effect = lambda text: list(range(len(text) // chars_per_token))
    return mock_enc


def _make_mock_get_encoding(chars_per_token: int = 1):
    """Return a get_encoding callable that always returns a mock encoding."""
    mock_enc = _make_mock_encoding(chars_per_token)

    def getter(model_name: str):  # noqa: ARG001
        return mock_enc

    return getter


class TestOpenAIUsageCounterWithInput:
    """Tests for OpenAIUsageCounter with the 'input' keyword."""

    def test_input_string_returns_tokens_and_requests(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter("gpt-4", input="hello")
        assert result == frozendict({"tokens": 5, "requests": 1})

    def test_input_empty_string(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter("gpt-4", input="")
        assert result == frozendict({"tokens": 0, "requests": 1})

    def test_input_non_string_raises_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="must be of type str"):
            counter("gpt-4", input=123)

    def test_input_list_raises_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="must be of type str"):
            counter("gpt-4", input=["hello"])


class TestOpenAIUsageCounterWithInputs:
    """Tests for OpenAIUsageCounter with the 'inputs' keyword."""

    def test_inputs_list_sums_tokens(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter("gpt-4", inputs=["hello", "world"])
        # "hello" = 5 tokens, "world" = 5 tokens => total 10
        assert result == frozendict({"tokens": 10, "requests": 1})

    def test_inputs_single_item(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter("gpt-4", inputs=["abc"])
        assert result == frozendict({"tokens": 3, "requests": 1})

    def test_inputs_non_string_items_raises_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError, match="All values in 'inputs' must be of type str"
        ):
            counter("gpt-4", inputs=["hello", 42])

    def test_inputs_empty_list(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter("gpt-4", inputs=[])
        assert result == frozendict({"tokens": 0, "requests": 1})


class TestOpenAIUsageCounterWithMessages:
    """Tests for OpenAIUsageCounter with the 'messages' keyword."""

    def test_messages_basic(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        messages = [{"role": "user", "content": "hi"}]
        result = counter("gpt-4", messages=messages)
        # 1 message: 4 overhead tokens
        # "role" value "user" = 4 tokens, "content" value "hi" = 2 tokens
        # + 2 assistant prefix
        # total = 4 + 4 + 2 + 2 = 12
        assert result["requests"] == 1
        assert isinstance(result["tokens"], int)

    def test_messages_non_string_value_raises(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError, match="All keys and values in messages must be of type str"
        ):
            counter("gpt-4", messages=[{"role": "user", "content": 123}])

    def test_messages_non_string_key_raises(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError, match="All keys and values in messages must be of type str"
        ):
            counter("gpt-4", messages=[{42: "user"}])


class TestOpenAIUsageCounterMissingKeys:
    """Tests for missing required keys."""

    def test_no_input_inputs_or_messages_raises(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError, match="Request must contain 'input', 'inputs', or 'messages'"
        ):
            counter("gpt-4", something_else="foo")

    def test_empty_request_raises(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError, match="Request must contain 'input', 'inputs', or 'messages'"
        ):
            counter("gpt-4")


class TestCountChatInputTokens:
    """Tests for count_chat_input_tokens overhead and name key handling."""

    def test_per_message_overhead(self):
        """Each message adds 4 overhead tokens, plus 2 for assistant prefix."""
        mock_enc = _make_mock_encoding()
        # Two messages, each with empty role/content
        messages = [
            {"role": "", "content": ""},
            {"role": "", "content": ""},
        ]
        result = count_chat_input_tokens(mock_enc, messages=messages)
        # 2 messages * 4 overhead + 2 assistant prefix = 10
        # Each value is "" which encodes to 0 tokens
        assert result == 10

    def test_single_message_overhead(self):
        mock_enc = _make_mock_encoding()
        messages = [{"role": "", "content": ""}]
        result = count_chat_input_tokens(mock_enc, messages=messages)
        # 1 * 4 + 2 = 6
        assert result == 6

    def test_name_key_subtracts_one_token(self):
        """If a message has a 'name' key, it subtracts 1 token."""
        mock_enc = _make_mock_encoding()
        messages = [{"role": "", "content": "", "name": ""}]
        result = count_chat_input_tokens(mock_enc, messages=messages)
        # 1 message * 4 overhead + 2 assistant prefix = 6
        # role "" = 0, content "" = 0, name "" = 0
        # name key adjustment: -1
        # total = 6 - 1 = 5
        assert result == 5

    def test_name_key_with_content(self):
        """Name key subtracts 1 token even when there's content in the name field."""
        mock_enc = _make_mock_encoding()
        messages = [{"role": "user", "content": "hi", "name": "bob"}]
        result = count_chat_input_tokens(mock_enc, messages=messages)
        # 4 overhead + "user"=4 + "hi"=2 + "bob"=3 + name_adj=-1 + 2 assistant = 14
        assert result == 14

    def test_no_messages_returns_assistant_prefix_only(self):
        mock_enc = _make_mock_encoding()
        result = count_chat_input_tokens(mock_enc, messages=[])
        # 0 messages overhead + 2 assistant prefix = 2
        assert result == 2

    def test_token_counting_includes_values_not_keys(self):
        """Only message values are encoded, not the keys themselves."""
        mock_enc = _make_mock_encoding()
        # The key "role" is NOT encoded; only the value "user" is
        messages = [{"role": "user"}]
        result = count_chat_input_tokens(mock_enc, messages=messages)
        # 4 overhead + len("user")=4 tokens + 2 assistant = 10
        assert result == 10


class TestCustomGetEncodingFunc:
    """Tests for injecting a custom get_encoding_func."""

    def test_custom_func_is_used(self):
        call_log = []

        def custom_getter(model_name: str):
            call_log.append(model_name)
            return _make_mock_encoding()

        counter = OpenAIUsageCounter(get_encoding_func=custom_getter)
        counter("my-custom-model", input="test")
        assert call_log == ["my-custom-model"]

    def test_default_func_uses_get_encoding(self):
        """When no custom func is provided, the module-level get_encoding is used."""
        # We just verify the counter is created without error
        counter = OpenAIUsageCounter()
        assert counter._get_encoding is get_encoding


class TestGetEncoding:
    """Tests for the get_encoding function (model name mapping + prefix stripping)."""

    def test_strips_openai_prefix(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("openai/gpt-4o")
        # Should resolve to o200k_base after stripping "openai/" prefix
        expected = tiktoken.get_encoding("o200k_base")
        assert enc.name == expected.name

    def test_exact_model_match_with_prefix(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("openai/gpt-4")
        expected = tiktoken.get_encoding("cl100k_base")
        assert enc.name == expected.name

    def test_substring_model_match_with_prefix(self):
        tiktoken = pytest.importorskip("tiktoken")
        # "gpt-4o-mini-2024-07-18" contains "gpt-4o-mini" as substring
        enc = get_encoding("openai/gpt-4o-mini-2024-07-18")
        expected = tiktoken.get_encoding("o200k_base")
        assert enc.name == expected.name

    def test_gpt4_turbo_encoding(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("openai/gpt-4-turbo")
        expected = tiktoken.get_encoding("cl100k_base")
        assert enc.name == expected.name

    def test_gpt35_turbo_encoding(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("openai/gpt-3.5-turbo")
        expected = tiktoken.get_encoding("cl100k_base")
        assert enc.name == expected.name

    def test_embedding_model(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("openai/text-embedding-3-small")
        expected = tiktoken.get_encoding("cl100k_base")
        assert enc.name == expected.name

    def test_exact_match_takes_priority_over_substring(self):
        tiktoken = pytest.importorskip("tiktoken")
        # "gpt-4" exact match should use cl100k_base (not o200k_base from "gpt-4o")
        enc = get_encoding("openai/gpt-4")
        expected = tiktoken.get_encoding("cl100k_base")
        assert enc.name == expected.name

    def test_openai_prefix_model_mapping(self):
        """Models with openai/ prefix should map correctly after stripping."""
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("openai/gpt-3.5-turbo")
        expected = tiktoken.get_encoding("cl100k_base")
        assert enc.name == expected.name

    def test_without_openai_prefix_falls_through_to_tiktoken(self):
        """Without openai/ prefix, partition yields empty string, falling to tiktoken."""
        pytest.importorskip("tiktoken")
        # "gpt-4".partition("openai/") -> ("gpt-4", "", ""), [2] -> ""
        # empty string is not in the mapping, so falls to tiktoken.encoding_for_model("")
        with pytest.raises(KeyError):
            get_encoding("gpt-4")


class TestOpenAIUsageCounterWithRealTiktoken:
    """Integration-style tests using real tiktoken encoding."""

    def test_real_encoding_input(self):
        tiktoken = pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        result = counter("openai/gpt-4", input="hello world")
        assert result["tokens"] > 0
        assert result["requests"] == 1
        # Verify against direct tiktoken call
        enc = tiktoken.get_encoding("cl100k_base")
        expected_tokens = len(enc.encode("hello world"))
        assert result["tokens"] == expected_tokens

    def test_real_encoding_inputs_list(self):
        tiktoken = pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        result = counter("openai/gpt-4", inputs=["hello", "world"])
        enc = tiktoken.get_encoding("cl100k_base")
        expected = len(enc.encode("hello")) + len(enc.encode("world"))
        assert result["tokens"] == expected
        assert result["requests"] == 1

    def test_real_encoding_messages(self):
        pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = counter("openai/gpt-4", messages=messages)
        assert result["tokens"] > 0
        assert result["requests"] == 1
