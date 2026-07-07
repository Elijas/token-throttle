"""
Tests for OpenAI token counting logic.

Source: token_throttle/_factories/_openai/_token_counter.py
"""

import json
import sys
import warnings
from unittest.mock import MagicMock, patch

import pytest
from frozendict import frozendict

from token_throttle._factories._openai import _token_counter as _token_counter_module
from token_throttle._factories._openai._token_counter import (
    OpenAIUsageCounter,
    count_chat_input_tokens,
    get_encoding,
)


def _make_mock_encoding(chars_per_token: int = 1) -> MagicMock:
    """Create a mock encoding where each `chars_per_token` characters = 1 token."""
    mock_enc = MagicMock()
    mock_enc.encode.side_effect = lambda text, **_kwargs: list(
        range(len(text) // chars_per_token)
    )
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

    def test_bytes_input_error_names_actual_type(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match=r"got bytes"):
            counter("gpt-4", input=b"hello")

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


class TestOpenAIUsageCounterRejectsInputsPlural:
    """'inputs' (plural) is not a real OpenAI API key — only 'input' (singular) is valid."""

    def test_inputs_alone_rejected(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError, match="must contain 'input', 'messages', or 'prompt'"
        ):
            counter("gpt-4", inputs=["hello", "world"])


class TestOpenAIUsageCounterWithMessages:
    """Tests for OpenAIUsageCounter with the 'messages' keyword."""

    def test_messages_basic(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        messages = [{"role": "user", "content": "hi"}]
        result = counter("gpt-4", messages=messages)
        # 1 message: 3 overhead tokens
        # "role" value "user" = 4 tokens, "content" value "hi" = 2 tokens
        # + 3 assistant prefix
        # total = 3 + 4 + 2 + 3 = 12
        assert result["requests"] == 1
        assert isinstance(result["tokens"], int)

    def test_messages_numeric_content_raises_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError, match="All keys and values in messages must be of type str"
        ):
            counter("gpt-4", messages=[{"role": "user", "content": 123}])

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

    def test_no_input_or_messages_raises(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError, match="Request must contain 'input', 'messages', or 'prompt'"
        ):
            counter("gpt-4", something_else="foo")

    def test_empty_request_raises(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError, match="Request must contain 'input', 'messages', or 'prompt'"
        ):
            counter("gpt-4")

    def test_prompt_none_alone_still_raises(self):
        """prompt=None does not make a request a stored-prompt-only request."""
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError, match="Request must contain 'input', 'messages', or 'prompt'"
        ):
            counter("gpt-4", prompt=None)

    def test_inputs_plural_not_recognised(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError, match="Request must contain 'input', 'messages', or 'prompt'"
        ):
            counter("gpt-4", inputs=["hello", "world"])


class TestOpenAIUsageCounterAmbiguousPayloads:
    """Only one of input / messages may be supplied per request."""

    def test_input_and_messages_raise_value_error(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError,
            match="Exactly one of 'input' or 'messages' must be provided",
        ):
            counter(
                "gpt-4",
                input="hi",
                messages=[{"role": "user", "content": "hello"}],
            )


class TestCountChatInputTokens:
    """Tests for count_chat_input_tokens overhead and name key handling."""

    def test_per_message_overhead(self):
        """Each message adds 3 overhead tokens, plus 3 for assistant prefix."""
        mock_enc = _make_mock_encoding()
        # Two messages, each with empty role/content
        messages = [
            {"role": "", "content": ""},
            {"role": "", "content": ""},
        ]
        result = count_chat_input_tokens(mock_enc, messages=messages)
        # 2 messages * 3 overhead + 3 assistant prefix = 9
        # Each value is "" which encodes to 0 tokens
        assert result == 9

    def test_single_message_overhead(self):
        mock_enc = _make_mock_encoding()
        messages = [{"role": "", "content": ""}]
        result = count_chat_input_tokens(mock_enc, messages=messages)
        # 1 * 3 + 3 = 6
        assert result == 6

    def test_name_key_adds_one_token(self):
        """If a message has a 'name' key, it adds 1 token."""
        mock_enc = _make_mock_encoding()
        messages = [{"role": "", "content": "", "name": ""}]
        result = count_chat_input_tokens(mock_enc, messages=messages)
        # 1 message * 3 overhead + 3 assistant prefix = 6
        # role "" = 0, content "" = 0, name "" = 0
        # name key adjustment: +1
        # total = 6 + 1 = 7
        assert result == 7

    def test_name_key_with_content(self):
        """Name key adds 1 token even when there's content in the name field."""
        mock_enc = _make_mock_encoding()
        messages = [{"role": "user", "content": "hi", "name": "bob"}]
        result = count_chat_input_tokens(mock_enc, messages=messages)
        # 3 overhead + "user"=4 + "hi"=2 + "bob"=3 + name_adj=+1 + 3 assistant = 16
        assert result == 16

    def test_no_messages_returns_assistant_prefix_only(self):
        mock_enc = _make_mock_encoding()
        result = count_chat_input_tokens(mock_enc, messages=[])
        # 0 messages overhead + 3 assistant prefix = 3
        assert result == 3

    def test_token_counting_includes_values_not_keys(self):
        """Only message values are encoded, not the keys themselves."""
        mock_enc = _make_mock_encoding()
        # The key "role" is NOT encoded; only the value "user" is
        messages = [{"role": "user"}]
        result = count_chat_input_tokens(mock_enc, messages=messages)
        # 3 overhead + len("user")=4 tokens + 3 assistant = 10
        assert result == 10

    @pytest.mark.parametrize("num_messages", [1, 2, 3, 10])
    def test_matches_openai_cookbook_formula(self, num_messages):
        """Matches the num_tokens_from_messages formula from the OpenAI
        cookbook for gpt-4 / gpt-4o / gpt-3.5-turbo current models:
          tokens_per_message = 3
          tokens_per_name = 1
          +3 for assistant priming
        """
        tiktoken = pytest.importorskip("tiktoken")
        enc = tiktoken.encoding_for_model("gpt-4")
        messages = [
            {"role": "user", "content": f"message number {i}"}
            for i in range(num_messages)
        ]

        result = count_chat_input_tokens(enc, messages=messages)

        tokens_per_message = 3
        cookbook = 3  # trailing assistant prime
        for message in messages:
            cookbook += tokens_per_message
            for value in message.values():
                cookbook += len(enc.encode(value))

        assert result == cookbook

    def test_cookbook_formula_with_name_field(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = tiktoken.encoding_for_model("gpt-4")
        messages = [{"role": "user", "content": "hi", "name": "alice"}]

        result = count_chat_input_tokens(enc, messages=messages)

        # 3 per-message + values + 1 for name + 3 trailing
        expected = (
            3
            + len(enc.encode("user"))
            + len(enc.encode("hi"))
            + len(enc.encode("alice"))
            + 1
            + 3
        )
        assert result == expected


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
    """Tests for the get_encoding function.

    get_encoding has no hardcoded model-family fallback table: resolution is
    delegated entirely to ``tiktoken.encoding_for_model``, with only the
    ``openai/`` provider-prefix stripping as custom logic. These tests pin
    that delegation (via tiktoken's own resolution) rather than a bespoke
    substring-matching table.
    """

    def test_strips_openai_prefix(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("openai/gpt-4o")
        expected = tiktoken.encoding_for_model("gpt-4o")
        assert enc.name == expected.name

    def test_exact_model_match_with_prefix(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("openai/gpt-4")
        expected = tiktoken.encoding_for_model("gpt-4")
        assert enc.name == expected.name

    def test_dated_model_snapshot_with_prefix(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("openai/gpt-4o-mini-2024-07-18")
        expected = tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18")
        assert enc.name == expected.name

    def test_gpt4_turbo_encoding(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("openai/gpt-4-turbo")
        expected = tiktoken.encoding_for_model("gpt-4-turbo")
        assert enc.name == expected.name

    def test_gpt35_turbo_encoding(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("openai/gpt-3.5-turbo")
        expected = tiktoken.encoding_for_model("gpt-3.5-turbo")
        assert enc.name == expected.name

    def test_embedding_model(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("openai/text-embedding-3-small")
        expected = tiktoken.encoding_for_model("text-embedding-3-small")
        assert enc.name == expected.name

    def test_gpt4_is_not_confused_with_gpt4o(self):
        tiktoken = pytest.importorskip("tiktoken")
        # "gpt-4" and "gpt-4o" use different encodings; make sure prefix
        # stripping doesn't accidentally conflate the two.
        enc = get_encoding("openai/gpt-4")
        expected = tiktoken.encoding_for_model("gpt-4")
        assert enc.name == expected.name
        assert enc.name != tiktoken.encoding_for_model("gpt-4o").name

    def test_without_openai_prefix_works(self):
        """Bare model names (without openai/ prefix) resolve correctly."""
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("gpt-4")
        expected = tiktoken.encoding_for_model("gpt-4")
        assert enc.name == expected.name

    def test_bare_gpt4o_works(self):
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("gpt-4o")
        expected = tiktoken.encoding_for_model("gpt-4o")
        assert enc.name == expected.name

    def test_delegates_to_tiktoken_for_models_it_knows(self):
        """'ada' is resolved directly by tiktoken.encoding_for_model."""
        tiktoken = pytest.importorskip("tiktoken")
        enc = get_encoding("ada")
        expected = tiktoken.encoding_for_model("ada")
        assert enc.name == expected.name

    def test_prefers_tiktoken_resolution_for_gpt41_models(self):
        tiktoken = pytest.importorskip("tiktoken")
        for model_name in ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]:
            enc = get_encoding(model_name)
            expected = tiktoken.encoding_for_model(model_name)
            assert enc.name == expected.name


class TestGetEncodingUnknownModelRaisesGuidedError:
    """When tiktoken cannot resolve a model, get_encoding must raise a clear,
    actionable ValueError instead of letting tiktoken's raw KeyError escape.

    Regression coverage for dot-named model releases (e.g. gpt-5.1, gpt-5.2)
    that the installed tiktoken does not recognize, and for legacy model
    names that used to be silently (and often incorrectly) resolved by the
    now-removed hardcoded substring-to-encoding fallback table.
    """

    def test_unknown_dotted_model_raises_value_error_not_key_error(self):
        pytest.importorskip("tiktoken")
        with pytest.raises(ValueError, match="get_encoding_func") as exc_info:
            get_encoding("gpt-5.1")
        assert "gpt-5.1" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, KeyError)

    def test_stale_fallback_table_entry_now_raises(self):
        """'codex' was hardcoded in the deleted fallback table but is not
        resolvable by tiktoken; it must now raise the guided error instead of
        silently returning a (possibly wrong) encoding.
        """
        pytest.importorskip("tiktoken")
        with pytest.raises(ValueError, match="get_encoding_func"):
            get_encoding("codex")

    def test_counter_call_raises_value_error_for_unknown_model(self):
        """OpenAIUsageCounter()(model, ...) must surface the guided ValueError,
        not a raw tiktoken KeyError.
        """
        pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        with pytest.raises(ValueError, match="get_encoding_func"):
            counter("gpt-5.1", messages=[{"role": "user", "content": "hi"}])

    async def test_count_request_async_raises_value_error_for_unknown_model(self):
        pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        with pytest.raises(ValueError, match="get_encoding_func"):
            await counter.count_request_async(
                "gpt-5.1", messages=[{"role": "user", "content": "hi"}]
            )

    async def test_warmup_models_raises_value_error_for_unknown_model(self):
        pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        with pytest.raises(ValueError, match="get_encoding_func"):
            await counter.warmup_models(["gpt-5.1"])


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

    def test_real_encoding_inputs_plural_rejected(self):
        pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        with pytest.raises(
            ValueError, match="must contain 'input', 'messages', or 'prompt'"
        ):
            counter("openai/gpt-4", inputs=["hello", "world"])

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


class TestRequestContextJsonSerialization:
    """tools/functions/response_format must be counted via JSON serialization
    so that structural tokens ({, }, [, ], :, ",", spaces) are not dropped.
    """

    def test_tools_delta_matches_json_dumps(self):
        """Regression for under-counting of ~65%: counter must include JSON
        structural characters for tools schemas.
        """
        tiktoken = pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        enc = tiktoken.encoding_for_model("gpt-4")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        messages = [{"role": "user", "content": "hi"}]
        base = counter(model="gpt-4", messages=messages)["tokens"]
        with_tools = counter(model="gpt-4", messages=messages, tools=tools)["tokens"]
        tools_delta = with_tools - base
        json_tokens = len(enc.encode(json.dumps(tools)))

        assert tools_delta >= json_tokens, (
            f"counter under-counts tools: delta={tools_delta}, json={json_tokens}"
        )
        assert tools_delta - json_tokens <= 5

    def test_response_format_delta_matches_json_dumps(self):
        tiktoken = pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        enc = tiktoken.encoding_for_model("gpt-4")

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Brief answer",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Model confidence 0-1",
                        },
                    },
                    "required": ["summary"],
                },
            },
        }
        messages = [{"role": "user", "content": "hi"}]
        base = counter(model="gpt-4", messages=messages)["tokens"]
        with_rf = counter(
            model="gpt-4", messages=messages, response_format=response_format
        )["tokens"]
        rf_delta = with_rf - base
        json_tokens = len(enc.encode(json.dumps(response_format)))

        assert rf_delta >= json_tokens
        assert rf_delta - json_tokens <= 5

    def test_text_structured_output_delta_matches_json_dumps(self):
        """Regression: the Responses API's `text={"format": {...}}`
        structured-output config must be JSON-serialized like
        `response_format`, not walked as plain text fragments (which drops
        JSON structural tokens and undercounts by ~62%).
        """
        tiktoken = pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        enc = tiktoken.encoding_for_model("gpt-4")

        text_config = {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Brief answer",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Model confidence 0-1",
                        },
                    },
                    "required": ["summary"],
                },
            },
        }
        messages = [{"role": "user", "content": "hi"}]
        base = counter(model="gpt-4", messages=messages)["tokens"]
        with_text = counter(model="gpt-4", messages=messages, text=text_config)[
            "tokens"
        ]
        text_delta = with_text - base
        json_tokens = len(enc.encode(json.dumps(text_config)))

        assert text_delta >= json_tokens, (
            f"counter under-counts text: delta={text_delta}, json={json_tokens}"
        )
        assert text_delta - json_tokens <= 5

    def test_text_and_response_format_counts_are_comparable(self):
        """The same JSON schema delivered via the Responses API `text` field
        or the Chat Completions `response_format` field must produce
        comparable token counts. Before the fix, `text` skipped JSON
        serialization and undercounted the identical schema by ~62%.
        """
        pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()

        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Brief answer"},
                "confidence": {
                    "type": "number",
                    "description": "Model confidence 0-1",
                },
            },
            "required": ["summary"],
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        }
        text_config = {
            "format": {"type": "json_schema", "name": "answer", "schema": schema},
        }

        messages = [{"role": "user", "content": "hi"}]
        base = counter(model="gpt-4", messages=messages)["tokens"]
        rf_delta = (
            counter(model="gpt-4", messages=messages, response_format=response_format)[
                "tokens"
            ]
            - base
        )
        text_delta = (
            counter(model="gpt-4", messages=messages, text=text_config)["tokens"] - base
        )

        assert text_delta > 0
        # Same schema content, just nested one level differently between the
        # two OpenAI wire shapes — deltas should be close, not off by ~62%.
        assert abs(rf_delta - text_delta) <= 10

    def test_functions_delta_matches_json_dumps(self):
        """'functions' (legacy key) should also be JSON-serialized."""
        tiktoken = pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        enc = tiktoken.encoding_for_model("gpt-4")

        functions = [
            {
                "name": "lookup",
                "description": "Find matching records",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }
        ]
        messages = [{"role": "user", "content": "hi"}]
        base = counter(model="gpt-4", messages=messages)["tokens"]
        with_fns = counter(model="gpt-4", messages=messages, functions=functions)[
            "tokens"
        ]
        fns_delta = with_fns - base
        json_tokens = len(enc.encode(json.dumps(functions)))

        assert fns_delta >= json_tokens
        assert fns_delta - json_tokens <= 5

    def test_instructions_plain_string_uses_fragment_path(self):
        """`instructions` is typically a plain string; it must not gain extra
        quote tokens from JSON serialization (regression guard for the
        non-JSON path still being used for that field).
        """
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        base = counter("gpt-4", input="hi")
        instructions = "system prompt here"

        result = counter("gpt-4", input="hi", instructions=instructions)

        assert result == frozendict(
            {"tokens": base["tokens"] + len(instructions), "requests": 1}
        )

    def test_non_serializable_tool_value_raises_value_error(self):
        """Non-JSON-serializable values raise ValueError (not TypeError from
        deep inside json.dumps).
        """
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="tools"):
            counter(
                "gpt-4",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"fn": object()}],
            )


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

    def test_tuple_messages_error_names_outer_container(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match=r"messages must be a list.*got tuple"):
            counter("gpt-4", messages=({"role": "user", "content": "hi"},))


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


class TestToolCallsJsonSerialization:
    """tool_calls in assistant messages must be counted via JSON serialization
    so that structural tokens ({, }, [, ], :, ",", spaces) are not dropped.
    """

    def test_tool_calls_delta_matches_json_dumps(self):
        """Regression for under-counting of ~41-58%: counter must include JSON
        structural characters for tool_calls in messages.
        """
        tiktoken = pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        enc = tiktoken.encoding_for_model("gpt-4")

        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "London", "units": "celsius"}',
                },
            }
        ]
        base = counter(
            model="gpt-4",
            messages=[{"role": "assistant", "content": ""}],
        )["tokens"]
        with_tc = counter(
            model="gpt-4",
            messages=[{"role": "assistant", "tool_calls": tool_calls}],
        )["tokens"]
        tc_delta = with_tc - base
        json_tokens = len(enc.encode(json.dumps(tool_calls)))

        assert tc_delta >= json_tokens, (
            f"tool_calls under-counted: delta={tc_delta}, json={json_tokens}"
        )
        assert tc_delta - json_tokens <= 5

    def test_multiple_tool_calls_counted_via_json_dumps(self):
        tiktoken = pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        enc = tiktoken.encoding_for_model("gpt-4")

        tool_calls = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": f"fn_{i}",
                    "arguments": json.dumps({"key": f"value_{i}"}),
                },
            }
            for i in range(3)
        ]
        base = counter(
            model="gpt-4",
            messages=[{"role": "assistant", "content": ""}],
        )["tokens"]
        with_tc = counter(
            model="gpt-4",
            messages=[{"role": "assistant", "tool_calls": tool_calls}],
        )["tokens"]
        tc_delta = with_tc - base
        json_tokens = len(enc.encode(json.dumps(tool_calls)))

        assert tc_delta >= json_tokens
        assert tc_delta - json_tokens <= 5


class TestResponsesApiStructuredItemCounting:
    """Responses-API structured items (function_call, etc.) must not be
    misclassified as chat messages, and must be JSON-serialized for counting.
    """

    def test_function_call_item_not_classified_as_message(self):
        """function_call items have 'name' but no 'role'; they must not be
        routed through the chat message counting path.
        """
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        function_call_item = {
            "type": "function_call",
            "name": "get_weather",
            "call_id": "call_abc",
            "arguments": '{"city": "London"}',
        }

        result = counter("gpt-4", input=[function_call_item])
        assert result["requests"] == 1
        assert result["tokens"] > 0

    def test_function_call_counted_via_json_serialization(self):
        tiktoken = pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        enc = tiktoken.encoding_for_model("gpt-4")

        function_call_item = {
            "type": "function_call",
            "name": "get_weather",
            "call_id": "call_abc",
            "arguments": '{"city": "London"}',
        }

        result = counter("gpt-4", input=[function_call_item])
        json_tokens = len(enc.encode(json.dumps([function_call_item])))

        assert result["tokens"] == json_tokens

    def test_mixed_messages_and_function_calls_counted_correctly(self):
        pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()

        message = {"role": "user", "content": "call the weather function"}
        function_call = {
            "type": "function_call",
            "name": "get_weather",
            "call_id": "call_abc",
            "arguments": '{"city": "Paris"}',
        }

        msg_tokens = counter("gpt-4", input=[message])["tokens"]
        fc_tokens = counter("gpt-4", input=[function_call])["tokens"]
        combined = counter("gpt-4", input=[message, function_call])["tokens"]

        assert combined == msg_tokens + fc_tokens

    def test_single_structured_dict_counted_via_json_serialization(self):
        tiktoken = pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        enc = tiktoken.encoding_for_model("gpt-4")

        item = {
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": '{"temp": 22}',
        }

        result = counter("gpt-4", input=item)
        json_tokens = len(enc.encode(json.dumps(item)))

        assert result["tokens"] == json_tokens


class TestChatPredictionCounting:
    """Chat Predicted Outputs `prediction.content` is real billed token volume
    and must contribute to the reserve, scaling with the content size.
    """

    def _messages(self) -> list[dict[str, object]]:
        return [{"role": "user", "content": "regenerate the file"}]

    def test_prediction_content_string_reserves_more_than_without(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        messages = self._messages()
        base = counter("gpt-4", messages=messages)["tokens"]

        small = counter(
            "gpt-4",
            messages=messages,
            prediction={"type": "content", "content": "x" * 10},
        )["tokens"]
        large = counter(
            "gpt-4",
            messages=messages,
            prediction={"type": "content", "content": "x" * 200},
        )["tokens"]

        assert small > base
        # Mock encoding is 1 char == 1 token, so the delta must track content
        # size: the extra 190 chars of prediction content are reserved.
        assert large - small >= 190

    def test_prediction_content_parts_are_counted(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        messages = self._messages()
        base = counter("gpt-4", messages=messages)["tokens"]

        result = counter(
            "gpt-4",
            messages=messages,
            prediction={
                "type": "content",
                "content": [{"type": "text", "text": "y" * 100}],
            },
        )["tokens"]

        assert result - base >= 100

    def test_prediction_never_undercounts_billed_content(self):
        """The reserve must include at least the content's real token count;
        Predicted-Outputs content is billed (accepted/rejected prediction
        tokens show up in usage), so under-counting it would leak budget.
        """
        tiktoken = pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        enc = tiktoken.encoding_for_model("gpt-4")
        messages = self._messages()

        content = "def regenerate() -> int:\n    return 42\n" * 5
        base = counter(model="gpt-4", messages=messages)["tokens"]
        with_prediction = counter(
            model="gpt-4",
            messages=messages,
            prediction={"type": "content", "content": content},
        )["tokens"]

        assert with_prediction - base >= len(enc.encode(content))


class TestResponsesPromptCounting:
    """Responses stored-prompt `prompt.variables` carry client-visible text
    that must be counted; `id`/`version` are opaque references that add nothing.
    """

    def test_prompt_string_variables_scale_with_size(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        base = counter("gpt-4", input="hi")["tokens"]

        small = counter(
            "gpt-4",
            input="hi",
            prompt={"id": "pmpt_abc", "variables": {"topic": "x" * 10}},
        )["tokens"]
        large = counter(
            "gpt-4",
            input="hi",
            prompt={"id": "pmpt_abc", "variables": {"topic": "x" * 200}},
        )["tokens"]

        assert small > base
        assert large - small >= 190

    def test_prompt_content_part_variables_are_counted(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        base = counter("gpt-4", input="hi")["tokens"]

        result = counter(
            "gpt-4",
            input="hi",
            prompt={
                "id": "pmpt_abc",
                "variables": {
                    "detail": {"type": "input_text", "text": "y" * 100},
                },
            },
        )["tokens"]

        assert result - base >= 100

    def test_prompt_id_and_version_alone_add_nothing(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())

        vars_only = counter(
            "gpt-4",
            input="hi",
            prompt={"variables": {"topic": "billing"}},
        )["tokens"]
        with_id_version = counter(
            "gpt-4",
            input="hi",
            prompt={
                "id": "pmpt_abc123",
                "version": "7",
                "variables": {"topic": "billing"},
            },
        )["tokens"]

        # Opaque id/version references are not billed prompt text, so they must
        # not change the reserve.
        assert with_id_version == vars_only

    def test_prompt_without_variables_adds_nothing(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        base = counter("gpt-4", input="hi")["tokens"]

        result = counter(
            "gpt-4",
            input="hi",
            prompt={"id": "pmpt_abc123", "version": "7"},
        )["tokens"]

        assert result == base

    def test_prompt_non_dict_raises(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="prompt"):
            counter("gpt-4", input="hi", prompt="just a string")


class TestStoredPromptOnlyRequests:
    """The Responses API accepts `prompt={"id": ...}` with no `input`; the
    counter must accept the same shape instead of failing the acquire.
    """

    def test_prompt_only_with_id_counts_zero_payload_tokens(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter("gpt-4", prompt={"id": "pmpt_abc123"})
        assert result == frozendict({"tokens": 0, "requests": 1})

    def test_prompt_only_with_id_and_version_counts_zero_payload_tokens(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter("gpt-4", prompt={"id": "pmpt_abc123", "version": "7"})
        assert result == frozendict({"tokens": 0, "requests": 1})

    def test_prompt_only_variables_are_counted(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter(
            "gpt-4",
            prompt={"id": "pmpt_abc", "variables": {"topic": "x" * 40}},
        )
        assert result == frozendict({"tokens": 40, "requests": 1})

    def test_prompt_only_reserves_output_budget(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = counter("gpt-4", prompt={"id": "pmpt_abc"}, max_output_tokens=50)
        assert result == frozendict({"tokens": 50, "requests": 1})

    def test_prompt_only_counts_other_request_context(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        instructions = "be brief"
        result = counter(
            "gpt-4",
            prompt={"id": "pmpt_abc"},
            instructions=instructions,
        )
        assert result == frozendict({"tokens": len(instructions), "requests": 1})

    def test_prompt_only_invalid_prompt_type_still_raises(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(ValueError, match="prompt"):
            counter("gpt-4", prompt="pmpt_abc123")

    def test_prompt_alongside_input_and_messages_still_ambiguous(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.raises(
            ValueError,
            match="Exactly one of 'input' or 'messages' must be provided",
        ):
            counter(
                "gpt-4",
                input="hi",
                messages=[{"role": "user", "content": "hello"}],
                prompt={"id": "pmpt_abc"},
            )

    async def test_prompt_only_async_parity(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        result = await counter.count_request_async(
            "gpt-4", prompt={"id": "pmpt_abc123"}
        )
        assert result == frozendict({"tokens": 0, "requests": 1})


class TestPromptNonTextVariables:
    """Non-text prompt variable values (images, files) are valid OpenAI
    requests. They count as 0 tokens with a once-per-process warning instead
    of failing the acquire; chat/input content keeps rejecting them.
    """

    @pytest.fixture(autouse=True)
    def _fresh_warning_registry(self, monkeypatch):
        monkeypatch.setattr(
            _token_counter_module, "_warned_uncounted_prompt_parts", set()
        )

    def test_image_variable_counts_zero_and_warns(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        base = counter("gpt-4", input="hi")["tokens"]

        with pytest.warns(UserWarning, match=r"'pic'.*'input_image'"):
            result = counter(
                "gpt-4",
                input="hi",
                prompt={
                    "id": "pmpt_abc",
                    "variables": {
                        "pic": {
                            "type": "input_image",
                            "image_url": "https://example.com/a.png",
                        }
                    },
                },
            )

        assert result == frozendict({"tokens": base, "requests": 1})

    def test_file_variable_counts_zero_and_warns(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        base = counter("gpt-4", input="hi")["tokens"]

        with pytest.warns(UserWarning, match=r"'doc'.*'input_file'"):
            result = counter(
                "gpt-4",
                input="hi",
                prompt={
                    "id": "pmpt_abc",
                    "variables": {"doc": {"type": "input_file", "file_id": "file-123"}},
                },
            )

        assert result == frozendict({"tokens": base, "requests": 1})

    def test_warning_names_best_effort_contract(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        with pytest.warns(UserWarning, match="best-effort"):
            counter(
                "gpt-4",
                input="hi",
                prompt={
                    "variables": {
                        "pic": {
                            "type": "input_image",
                            "image_url": "https://example.com/a.png",
                        }
                    }
                },
            )

    def test_warning_is_emitted_once_per_process_per_part_type(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        image_prompt = {
            "variables": {
                "pic": {
                    "type": "input_image",
                    "image_url": "https://example.com/a.png",
                }
            }
        }

        with pytest.warns(UserWarning, match="input_image"):
            counter("gpt-4", input="hi", prompt=image_prompt)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            counter("gpt-4", input="hi", prompt=image_prompt)
        assert not [w for w in caught if issubclass(w.category, UserWarning)]

    def test_text_variables_still_counted_alongside_image_variable(self):
        counter = OpenAIUsageCounter(get_encoding_func=_make_mock_get_encoding())
        base = counter("gpt-4", input="hi")["tokens"]

        with pytest.warns(UserWarning, match="input_image"):
            result = counter(
                "gpt-4",
                input="hi",
                prompt={
                    "id": "pmpt_abc",
                    "variables": {
                        "pic": {
                            "type": "input_image",
                            "image_url": "https://example.com/a.png",
                        },
                        "topic": "y" * 30,
                    },
                },
            )

        assert result == frozendict({"tokens": base + 30, "requests": 1})

    def test_chat_content_images_still_rejected(self):
        """The prompt-variable leniency must not leak into chat content."""
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


_SPECIAL_TOKEN_TEXT = "docs snippet: <|endoftext|> is a special token"  # noqa: S105 — tokenizer special-token literal, not a credential


def _make_special_token_strict_get_encoding():
    """Mimic tiktoken's default behavior: `encode` raises on special-token
    literals unless the caller opts out via `disallowed_special=()`.
    """
    mock_enc = MagicMock()

    def encode(text, *, disallowed_special="all"):
        if disallowed_special == "all" and "<|endoftext|>" in text:
            raise ValueError(
                "Encountered text corresponding to disallowed special token "
                "'<|endoftext|>'."
            )
        return list(range(len(text)))

    mock_enc.encode.side_effect = encode
    return lambda _model_name: mock_enc


class TestSpecialTokenLiterals:
    """Text containing tiktoken special-token literals (e.g. pasted from LLM
    docs) is ordinary request text and must count instead of crashing the
    acquire, on every encode path in the counter.
    """

    def test_special_token_in_input_string(self):
        counter = OpenAIUsageCounter(
            get_encoding_func=_make_special_token_strict_get_encoding()
        )
        result = counter("gpt-4", input=_SPECIAL_TOKEN_TEXT)
        assert result["tokens"] == len(_SPECIAL_TOKEN_TEXT)

    def test_special_token_in_input_string_list(self):
        counter = OpenAIUsageCounter(
            get_encoding_func=_make_special_token_strict_get_encoding()
        )
        result = counter("gpt-4", input=[_SPECIAL_TOKEN_TEXT, "plain"])
        assert result["tokens"] == len(_SPECIAL_TOKEN_TEXT) + len("plain")

    def test_special_token_in_messages(self):
        counter = OpenAIUsageCounter(
            get_encoding_func=_make_special_token_strict_get_encoding()
        )
        result = counter(
            "gpt-4",
            messages=[{"role": "user", "content": _SPECIAL_TOKEN_TEXT}],
        )
        assert result["tokens"] > len(_SPECIAL_TOKEN_TEXT)

    def test_special_token_in_message_content_parts(self):
        counter = OpenAIUsageCounter(
            get_encoding_func=_make_special_token_strict_get_encoding()
        )
        result = counter(
            "gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": _SPECIAL_TOKEN_TEXT}],
                }
            ],
        )
        assert result["tokens"] > len(_SPECIAL_TOKEN_TEXT)

    def test_special_token_in_tools_json_path(self):
        counter = OpenAIUsageCounter(
            get_encoding_func=_make_special_token_strict_get_encoding()
        )
        result = counter(
            "gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "explain",
                        "description": _SPECIAL_TOKEN_TEXT,
                    },
                }
            ],
        )
        assert result["tokens"] > len(_SPECIAL_TOKEN_TEXT)

    def test_special_token_in_tool_calls_json_path(self):
        counter = OpenAIUsageCounter(
            get_encoding_func=_make_special_token_strict_get_encoding()
        )
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
                                "name": "explain",
                                "arguments": json.dumps({"text": _SPECIAL_TOKEN_TEXT}),
                            },
                        }
                    ],
                }
            ],
        )
        assert result["tokens"] > len(_SPECIAL_TOKEN_TEXT)

    def test_special_token_in_instructions_fragment_walk(self):
        counter = OpenAIUsageCounter(
            get_encoding_func=_make_special_token_strict_get_encoding()
        )
        result = counter("gpt-4", input="hi", instructions=_SPECIAL_TOKEN_TEXT)
        assert result["tokens"] == len("hi") + len(_SPECIAL_TOKEN_TEXT)

    def test_special_token_in_prediction(self):
        counter = OpenAIUsageCounter(
            get_encoding_func=_make_special_token_strict_get_encoding()
        )
        result = counter(
            "gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            prediction={"type": "content", "content": _SPECIAL_TOKEN_TEXT},
        )
        assert result["tokens"] > len(_SPECIAL_TOKEN_TEXT)

    def test_special_token_in_prompt_variables(self):
        counter = OpenAIUsageCounter(
            get_encoding_func=_make_special_token_strict_get_encoding()
        )
        result = counter(
            "gpt-4",
            input="hi",
            prompt={"id": "pmpt_abc", "variables": {"topic": _SPECIAL_TOKEN_TEXT}},
        )
        assert result["tokens"] == len("hi") + len(_SPECIAL_TOKEN_TEXT)

    def test_special_token_in_structured_input_json_path(self):
        counter = OpenAIUsageCounter(
            get_encoding_func=_make_special_token_strict_get_encoding()
        )
        result = counter(
            "gpt-4",
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "abc",
                    "output": _SPECIAL_TOKEN_TEXT,
                }
            ],
        )
        assert result["tokens"] > len(_SPECIAL_TOKEN_TEXT)

    def test_special_token_with_real_tiktoken(self):
        pytest.importorskip("tiktoken")
        counter = OpenAIUsageCounter()
        result = counter(
            "gpt-4",
            messages=[{"role": "user", "content": _SPECIAL_TOKEN_TEXT}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "f", "description": _SPECIAL_TOKEN_TEXT},
                }
            ],
            prediction={"type": "content", "content": _SPECIAL_TOKEN_TEXT},
            prompt={"id": "pmpt_abc", "variables": {"topic": _SPECIAL_TOKEN_TEXT}},
        )
        assert result["requests"] == 1
        assert result["tokens"] > 0
