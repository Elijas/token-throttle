"""
Tests for OpenAI token counting logic.

Source: token_throttle/_factories/_openai/_token_counter.py
"""

import sys
from unittest.mock import MagicMock, patch

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

    def getter(model_name: str):
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

    def test_input_single_item_string_list(self):
        """input=["hello"] is valid per the OpenAI Embeddings API."""
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter("gpt-4", input=["hello"])
        assert result == frozendict({"tokens": 5, "requests": 1})

    def test_input_multi_string_list(self):
        """input=["hello", "world"] is valid per the OpenAI Embeddings API."""
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter("gpt-4", input=["hello", "world"])
        # "hello" = 5 tokens, "world" = 5 tokens => total 10
        assert result == frozendict({"tokens": 10, "requests": 1})

    def test_input_structured_message_list_counts_text(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter(
            "gpt-4",
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            ],
        )
        assert result == frozendict({"tokens": 12, "requests": 1})

    def test_input_mixed_message_and_structured_item_preserves_message_overhead(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        message_item = {"role": "user", "content": "hi"}
        structured_item = {
            "type": "function_call_output",
            "call_id": "abc",
            "output": "ok",
        }

        message_tokens = counter("gpt-4", input=[message_item])["tokens"]
        structured_tokens = counter("gpt-4", input=[structured_item])["tokens"]
        result = counter("gpt-4", input=[message_item, structured_item])

        assert result == frozendict(
            {"tokens": message_tokens + structured_tokens, "requests": 1}
        )

    def test_input_mixed_multiple_messages_and_structured_item_counts_once(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        message_items = [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "hi"},
        ]
        structured_item = {
            "type": "function_call_output",
            "call_id": "abc",
            "output": "ok",
        }

        message_tokens = counter("gpt-4", input=message_items)["tokens"]
        structured_tokens = counter("gpt-4", input=[structured_item])["tokens"]
        result = counter(
            "gpt-4",
            input=[message_items[0], structured_item, message_items[1]],
        )

        assert result == frozendict(
            {"tokens": message_tokens + structured_tokens, "requests": 1}
        )

    def test_input_reserves_max_output_tokens(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter("gpt-4", input="hi", max_output_tokens=50)
        assert result == frozendict({"tokens": 52, "requests": 1})

    def test_input_instructions_are_counted(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        base = counter("gpt-4", input="hi")
        instructions = "system prompt here"

        result = counter("gpt-4", input="hi", instructions=instructions)

        assert result == frozendict(
            {"tokens": base["tokens"] + len(instructions), "requests": 1}
        )

    def test_input_image_part_raises_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="input_image"):
            counter(
                "gpt-4",
                input=[{"type": "input_image", "image_url": "https://example.com/a"}],
            )

    def test_input_image_field_without_type_raises_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="image_url"):
            counter(
                "gpt-4",
                input=[{"image_url": "https://example.com/a"}],
            )

    def test_input_function_call_output_allows_nested_image_url_fields(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())

        result = counter(
            "gpt-4",
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "abc",
                    "output": {
                        "image_url": "https://example.com/a.png",
                        "status": "ok",
                    },
                }
            ],
        )

        assert result["requests"] == 1
        assert result["tokens"] > 0

    def test_input_function_call_output_allows_nested_file_fields(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())

        result = counter(
            "gpt-4",
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "abc",
                    "output": {
                        "file": "report.pdf",
                        "status": "ok",
                    },
                }
            ],
        )

        assert result["requests"] == 1
        assert result["tokens"] > 0


class TestOpenAIUsageCounterWithPretokenizedInput:
    """Embeddings-style token-id arrays should count without text encoding."""

    def test_input_token_id_list_counts_items(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter("text-embedding-3-small", input=[101, 102, 103])
        assert result == frozendict({"tokens": 3, "requests": 1})

    def test_input_nested_token_id_lists_counts_all_items(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter(
            "text-embedding-3-small",
            input=[[101, 102], [201, 202, 203]],
        )
        assert result == frozendict({"tokens": 5, "requests": 1})

    def test_input_token_ids_reject_boolean_values(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="must be of type str"):
            counter("text-embedding-3-small", input=[True, 102])


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
        with pytest.raises(ValueError, match="'inputs' must be a list of strings"):
            counter("gpt-4", inputs=["hello", 42])

    def test_inputs_bare_string_raises_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="'inputs' must be a list of strings"):
            counter("gpt-4", inputs="hello")

    def test_inputs_integer_raises_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="'inputs' must be a list of strings"):
            counter("gpt-4", inputs=42)

    def test_inputs_none_raises_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="'inputs' must be a list of strings"):
            counter("gpt-4", inputs=None)

    def test_inputs_dict_raises_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="'inputs' must be a list of strings"):
            counter("gpt-4", inputs={"key": "value"})

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

    def test_messages_numeric_content_counted_as_string(self):
        """Numeric content values are converted to string for token counting."""
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter("gpt-4", messages=[{"role": "user", "content": 123}])
        assert result["requests"] == 1
        assert result["tokens"] > 0

    def test_messages_unsupported_type_value_raises(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError, match="All keys and values in messages must be of type str"
        ):
            counter("gpt-4", messages=[{"role": "user", "content": object()}])

    def test_messages_non_string_key_raises(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="keys must be strings"):
            counter("gpt-4", messages=[{42: "user"}])

    def test_messages_non_string_key_error_mentions_keys(self):
        """Error message for non-string keys should mention 'keys', not 'values'."""
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="keys") as exc_info:
            counter("gpt-4", messages=[{42: "user"}])
        # The error should NOT claim values are the problem
        assert "values" not in str(exc_info.value).lower()

    def test_messages_content_parts_are_supported(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter(
            "gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            ],
        )
        assert result == frozendict({"tokens": 12, "requests": 1})

    def test_messages_reserve_max_tokens(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter(
            "gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
        )
        assert result == frozendict({"tokens": 62, "requests": 1})

    def test_messages_reserve_max_completion_tokens(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter(
            "gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=50,
        )
        assert result == frozendict({"tokens": 62, "requests": 1})

    def test_messages_with_tool_calls_are_supported(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        base = counter("gpt-4", messages=[{"role": "assistant", "content": ""}])
        result = counter(
            "gpt-4",
            messages=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"x":1}',
                            },
                        }
                    ],
                }
            ],
        )
        assert result["requests"] == 1
        assert result["tokens"] > base["tokens"]

    def test_messages_tools_are_counted(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        base = counter("gpt-4", messages=[{"role": "user", "content": "hi"}])

        result = counter(
            "gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Find matching records",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "q": {
                                    "type": "string",
                                    "description": "Search query",
                                }
                            },
                        },
                    },
                }
            ],
        )

        assert result["requests"] == 1
        assert result["tokens"] > base["tokens"]

    def test_messages_response_format_is_counted(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        base = counter("gpt-4", messages=[{"role": "user", "content": "hi"}])

        result = counter(
            "gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "Brief answer",
                            }
                        },
                    },
                },
            },
        )

        assert result["requests"] == 1
        assert result["tokens"] > base["tokens"]

    def test_messages_tools_allow_schema_fields_named_image_url(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        base = counter("gpt-4", messages=[{"role": "user", "content": "hi"}])

        result = counter(
            "gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "save_image",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "image_url": {
                                    "type": "string",
                                    "description": "Remote image URL",
                                }
                            },
                        },
                    },
                }
            ],
        )

        assert result["requests"] == 1
        assert result["tokens"] > base["tokens"]

    def test_messages_with_image_content_raise_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="input_image"):
            counter(
                "gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/a",
                            }
                        ],
                    }
                ],
            )

    def test_messages_with_image_field_without_type_raise_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="image_url"):
            counter(
                "gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "image_url": "https://example.com/a",
                            }
                        ],
                    }
                ],
            )


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


class TestOpenAIUsageCounterAmbiguousPayloads:
    """Only one of input / inputs / messages may be supplied per request."""

    @pytest.mark.parametrize(
        "request_kwargs",
        [
            {"input": "hi", "messages": [{"role": "user", "content": "hello"}]},
            {"input": "hi", "inputs": ["hello"]},
            {"inputs": ["hello"], "messages": [{"role": "user", "content": "hello"}]},
            {
                "input": "hi",
                "inputs": ["hello"],
                "messages": [{"role": "user", "content": "hello"}],
            },
        ],
    )
    def test_multiple_payload_keys_raise_value_error(self, request_kwargs):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError,
            match="Exactly one of 'input', 'inputs', or 'messages' must be provided",
        ):
            counter("gpt-4", **request_kwargs)


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

    def test_name_key_adds_one_token(self):
        """If a message has a 'name' key, it adds 1 token."""
        mock_enc = _make_mock_encoding()
        messages = [{"role": "", "content": "", "name": ""}]
        result = count_chat_input_tokens(mock_enc, messages=messages)
        # 1 message * 4 overhead + 2 assistant prefix = 6
        # role "" = 0, content "" = 0, name "" = 0
        # name key adjustment: +1
        # total = 6 + 1 = 7
        assert result == 7

    def test_name_key_with_content(self):
        """Name key adds 1 token even when there's content in the name field."""
        mock_enc = _make_mock_encoding()
        messages = [{"role": "user", "content": "hi", "name": "bob"}]
        result = count_chat_input_tokens(mock_enc, messages=messages)
        # 4 overhead + "user"=4 + "hi"=2 + "bob"=3 + name_adj=+1 + 2 assistant = 16
        assert result == 16

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


class TestGetEncodingImportError:
    """Cover lines 58-59: ImportError when tiktoken is missing."""

    def test_raises_import_error_without_tiktoken(self):
        # get_encoding() does `import tiktoken` lazily; setting the module
        # to None in sys.modules causes Python to raise ImportError.
        with (
            patch.dict(sys.modules, {"tiktoken": None}),
            pytest.raises(ImportError, match="tiktoken"),
        ):
            get_encoding("gpt-4")


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

    def test_without_openai_prefix_works(self):
        """Bare model names (without openai/ prefix) resolve correctly."""
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("gpt-4")
        expected = tiktoken.get_encoding("cl100k_base")
        assert enc.name == expected.name

    def test_bare_gpt4o_works(self):
        """Bare gpt-4o resolves to o200k_base."""
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("gpt-4o")
        expected = tiktoken.get_encoding("o200k_base")
        assert enc.name == expected.name

    def test_fallback_to_encoding_for_model(self):
        """Cover line 87: unknown model falls through to tiktoken.encoding_for_model.

        We need a model name that: (1) has no exact match in the substring map,
        (2) does NOT contain any map key as a substring, (3) tiktoken recognizes.
        'o1' and 'ada' both satisfy these — neither contains 'gpt-4', 'davinci', etc.
        """
        tiktoken = pytest.importorskip("tiktoken")
        # 'ada' is known by tiktoken (maps to r50k_base) but is NOT a substring
        # of any key in our map, and no map key appears in 'ada'.
        enc = get_encoding("ada")
        expected = tiktoken.encoding_for_model("ada")
        assert enc.name == expected.name

    def test_prefers_tiktoken_resolution_for_gpt41_models(self):
        tiktoken = pytest.importorskip("tiktoken")
        for model_name in ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]:
            enc = get_encoding(model_name)
            expected = tiktoken.encoding_for_model(model_name)
            assert enc.name == expected.name


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


class TestMalformedMessages:
    """Non-dict messages must raise ValueError, not AttributeError."""

    def test_string_messages_raise_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="All messages must be dicts"):
            counter("gpt-4", messages=["not a dict"])

    def test_int_messages_raise_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="All messages must be dicts"):
            counter("gpt-4", messages=[42])

    def test_mixed_dict_and_non_dict_raises(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="All messages must be dicts"):
            counter("gpt-4", messages=[{"role": "user", "content": "hi"}, "bad"])


class TestNumericValuesInMessages:
    """Numeric metadata in message dicts should not crash the token counter."""

    def test_tool_call_with_int_index_does_not_crash(self):
        """tool_calls with index (int) field should not raise ValueError."""
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter(
            "gpt-4",
            messages=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"x":1}',
                            },
                        }
                    ],
                }
            ],
        )
        assert result["requests"] == 1
        assert result["tokens"] > 0

    def test_message_with_float_value_does_not_crash(self):
        """Float values in message dicts should be handled gracefully."""
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter(
            "gpt-4",
            messages=[{"role": "user", "content": "hi", "score": 0.95}],
        )
        assert result["requests"] == 1
        assert result["tokens"] > 0

    def test_message_with_bool_value_does_not_crash(self):
        """Bool values in message dicts should be handled gracefully."""
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter(
            "gpt-4",
            messages=[{"role": "user", "content": "hi", "refusal": False}],
        )
        assert result["requests"] == 1
        assert result["tokens"] > 0
