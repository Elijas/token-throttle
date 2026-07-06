"""
Tokenizer / API-surface drift canary for the OpenAI token counter.

This module exists to turn silent tokenizer and API-surface drift into a
loud CI failure instead of a quietly wrong token count. See
``token_throttle/_factories/_openai/_token_counter.py``'s ``get_encoding``
docstring for the underlying design decision: there is no hardcoded
model-family fallback table, so an unknown model is tiktoken's problem to
solve (upgrade it), not a stale local guess to paper over.

Two checks, run against the LATEST unpinned openai/tiktoken releases:

  (a) Every model name the installed openai SDK currently exposes as a valid
      chat model must resolve to an encoding via ``get_encoding()``. A
      failure means tiktoken has fallen behind the openai SDK's model list
      for an already-released model.
  (b) Every top-level request param the installed openai SDK declares for
      ``chat.completions.create`` / ``responses.create`` must be either
      counted by the token counter (``_REQUEST_CONTEXT_KEYS`` and friends)
      or explicitly triaged below as non-token-bearing. A failure means the
      SDK grew a new top-level param this module has not yet classified.

Skipped unless ``CANARY=1`` is set: this only means something against the
latest unpinned releases (see ``.github/workflows/tokenizer-drift.yml``),
not the versions pinned in ``uv.lock``. A stale local tiktoken failing check
(a) is expected and is not, by itself, a bug in this repository.
"""

from __future__ import annotations

import os
import typing

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CANARY") != "1",
    reason="tokenizer-drift canary only runs with CANARY=1; see tokenizer-drift.yml",
)

pytest.importorskip("openai", reason="openai package not installed")
pytest.importorskip("tiktoken", reason="tiktoken package not installed")

from token_throttle._factories._openai._token_counter import (  # noqa: E402
    _OUTPUT_BUDGET_KEYS,
    _OUTPUT_MULTIPLIER_KEYS,
    _REQUEST_CONTEXT_KEYS,
    _REQUEST_PAYLOAD_KEYS,
    get_encoding,
)


def _typed_dict_keys(typed_dict: type) -> set[str]:
    return set(typing.get_type_hints(typed_dict, include_extras=True))


def _declared_param_keys(params_type: object) -> set[str]:
    """Union of top-level keys across a TypedDict, or a Union of TypedDicts
    (e.g. the Streaming/NonStreaming create-params variants).
    """
    members = typing.get_args(params_type) or (params_type,)
    keys: set[str] = set()
    for member in members:
        keys |= _typed_dict_keys(member)
    return keys


def _known_counted_keys() -> set[str]:
    return (
        set(_REQUEST_CONTEXT_KEYS)
        | set(_REQUEST_PAYLOAD_KEYS)
        | set(_OUTPUT_BUDGET_KEYS)
        | set(_OUTPUT_MULTIPLIER_KEYS)
    )


# Sampling/decoding/transport/bookkeeping controls that do not add prompt
# tokens. Shared between chat.completions.create and responses.create where
# both APIs declare the same param name.
_SHARED_NON_TOKEN_BEARING_KEYS = frozenset(
    {
        "model",  # consumed as OpenAIUsageCounter's positional `model` arg
        "stream",  # transport flag, added only by the Streaming param variant
        "temperature",
        "top_p",
        "top_logprobs",
        "parallel_tool_calls",
        "stream_options",
        "store",
        "metadata",
        "safety_identifier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "service_tier",
        "user",
        # "auto"/"none"/"required"/named-tool; a named choice adds a short
        # function name, in the same negligible-control ballpark as the
        # model name itself (see the counter module's "best-effort
        # heuristics" framing).
        "tool_choice",
        "moderation",  # {"model": "..."}: names which moderation model to run
        # against the content, not prompt text itself
    }
)

_CHAT_NON_TOKEN_BEARING_KEYS = _SHARED_NON_TOKEN_BEARING_KEYS | {
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "logprobs",
    "logit_bias",  # token_id -> bias map, not prompt text
    "stop",  # stop sequences are not counted as prompt tokens
    "modalities",
    "reasoning_effort",
    "verbosity",
    "audio",  # output voice/format config, not prompt text
    "web_search_options",
    "function_call",  # legacy control; same rationale as tool_choice above
}

_RESPONSES_NON_TOKEN_BEARING_KEYS = _SHARED_NON_TOKEN_BEARING_KEYS | {
    "background",
    "context_management",
    "include",
    "previous_response_id",
    "reasoning",
    "truncation",
    "conversation",  # server-side conversation id/object reference
    # Any max_* key not in _OUTPUT_BUDGET_KEYS is already loudly rejected by
    # _validate_max_kwargs rather than silently miscounted.
    "max_tool_calls",
}

# Model names the openai SDK already ships but tiktoken (0.13.0 as of
# 2026-07-06) cannot yet resolve. For each of these, get_encoding() raises
# its designed guided ValueError (upgrade tiktoken / pass get_encoding_func),
# so the gap is known and handled — re-flagging it weekly would only train
# alert-blindness. Exact names only, no wildcards: a NEW unresolvable name
# (even a new date variant of these same families) still fails the canary
# and must be triaged here deliberately. The resolve test below is
# self-cleaning in both directions: an entry that starts resolving, or that
# the SDK no longer lists, fails the canary until it is removed.
_KNOWN_PENDING_TIKTOKEN_MODELS = frozenset(
    {
        "codex-mini-latest",
        "gpt-5.1",
        "gpt-5.1-2025-11-13",
        "gpt-5.1-chat-latest",
        "gpt-5.1-codex",
        "gpt-5.1-mini",
        "gpt-5.2",
        "gpt-5.2-2025-12-11",
        "gpt-5.2-chat-latest",
        "gpt-5.2-pro",
        "gpt-5.2-pro-2025-12-11",
        "gpt-5.3-chat-latest",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-mini-2026-03-17",
        "gpt-5.4-nano",
        "gpt-5.4-nano-2026-03-17",
    }
)


def test_all_current_chat_models_resolve_via_get_encoding():
    from openai.types import ChatModel  # noqa: PLC0415

    model_names = typing.get_args(ChatModel)
    assert model_names, "expected openai.types.ChatModel to enumerate model names"

    failures = {}
    for model_name in model_names:
        try:
            get_encoding(model_name)
        except ValueError as exc:
            failures[model_name] = str(exc)

    new_failures = {
        name: msg
        for name, msg in failures.items()
        if name not in _KNOWN_PENDING_TIKTOKEN_MODELS
    }
    assert not new_failures, (
        "get_encoding() could not resolve the following current openai "
        "ChatModel name(s) -- tiktoken has fallen behind the installed "
        "openai SDK's model list. Triage each: add to "
        "_KNOWN_PENDING_TIKTOKEN_MODELS if get_encoding()'s guided error is "
        f"the intended behavior for it: {sorted(new_failures)}"
    )

    now_resolving = sorted(_KNOWN_PENDING_TIKTOKEN_MODELS - set(failures))
    assert not now_resolving, (
        "stale _KNOWN_PENDING_TIKTOKEN_MODELS entries: these now resolve "
        "(tiktoken caught up) or are no longer in the SDK's ChatModel list; "
        f"remove them: {now_resolving}"
    )


def test_chat_completions_params_are_fully_triaged():
    from openai.types.chat import completion_create_params  # noqa: PLC0415

    declared = _declared_param_keys(completion_create_params.CompletionCreateParams)
    accounted_for = _known_counted_keys() | _CHAT_NON_TOKEN_BEARING_KEYS
    unknown = declared - accounted_for
    assert not unknown, (
        "openai's chat.completions.create params include untriaged top-level "
        f"key(s): {sorted(unknown)}. Add each to _REQUEST_CONTEXT_KEYS if it "
        "carries prompt text, or to this test's non-token-bearing ignore "
        "list with a reason if it does not."
    )


def test_responses_params_are_fully_triaged():
    from openai.types.responses import response_create_params  # noqa: PLC0415

    declared = _declared_param_keys(response_create_params.ResponseCreateParams)
    accounted_for = _known_counted_keys() | _RESPONSES_NON_TOKEN_BEARING_KEYS
    unknown = declared - accounted_for
    assert not unknown, (
        "openai's responses.create params include untriaged top-level "
        f"key(s): {sorted(unknown)}. Add each to _REQUEST_CONTEXT_KEYS if it "
        "carries prompt text, or to this test's non-token-bearing ignore "
        "list with a reason if it does not."
    )
