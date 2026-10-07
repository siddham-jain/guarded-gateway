from typing import Any

import pytest

from gg.core.errors import ProviderError
from gg.providers.gemini.quirks import GeminiQuirks
from gg.providers.gemini.request import Translated, build_body, history_call_ids, safety_identifier
from gg.providers.state.signatures import DUMMY_SIGNATURE
from tests.conftest import make_request
from tests.unit.providers.gemini.support import gemini_dep

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "weather by city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            "strict": True,
        },
    },
    {"type": "function", "function": {"name": "get_time"}},
]


def call(
    call_id: str, name: str = "get_weather", arguments: str = '{"city": "Paris"}', **extra: Any
) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}, **extra}


def build(
    signatures: dict[str, str] | None = None, quirks: dict[str, Any] | None = None, **req: Any
) -> Translated:
    dep = req.pop("dep", None) or gemini_dep()
    return build_body(
        make_request(**req),
        dep,
        GeminiQuirks.model_validate(quirks or {}),
        key_id="key-1",
        signatures=signatures,
    )


def test_minimal_body_has_privacy_defaults() -> None:
    out = build()
    assert out.body == {
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        "labels": {"safety_identifier": safety_identifier("key-1", "")},
        "store": False,
    }
    assert out.ignored == ()
    assert out.dummy_signatures == 0


def test_safety_identifier_is_a_short_lowercase_hash() -> None:
    value = build(user="alice@example.com").body["labels"]["safety_identifier"]
    assert value == safety_identifier("key-1", "alice@example.com")
    assert len(value) == 48
    assert value == value.lower()
    assert "alice" not in value


def test_system_and_developer_are_hoisted_and_roles_mapped() -> None:
    out = build(
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "developer", "content": "use metric"},
            {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
        ]
    )
    assert out.body["systemInstruction"] == {"parts": [{"text": "be terse\n\nuse metric"}]}
    assert out.body["contents"] == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hello"}]},
        {"role": "user", "parts": [{"text": "weather?"}]},
    ]


def test_consecutive_same_role_turns_merge() -> None:
    out = build(messages=[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    assert out.body["contents"] == [{"role": "user", "parts": [{"text": "a"}, {"text": "b"}]}]


def test_tool_loop_merges_results_in_call_order_with_names_from_history() -> None:
    out = build(
        tools=TOOLS,
        signatures={"fc_1": "sig-1"},
        messages=[
            {"role": "user", "content": "weather in paris and the time?"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [call("fc_1"), call("fc_2", "get_time", "")],
            },
            {"role": "tool", "tool_call_id": "fc_2", "content": "12:00"},
            {"role": "tool", "tool_call_id": "fc_1", "content": '{"temp_c": 18}'},
            {"role": "user", "content": "thanks"},
        ],
    )
    assert out.body["contents"] == [
        {"role": "user", "parts": [{"text": "weather in paris and the time?"}]},
        {
            "role": "model",
            "parts": [
                {"text": "checking"},
                {
                    "functionCall": {"id": "fc_1", "name": "get_weather", "args": {"city": "Paris"}},
                    "thoughtSignature": "sig-1",
                },
                {"functionCall": {"id": "fc_2", "name": "get_time", "args": {}}},
            ],
        },
        {
            "role": "user",
            "parts": [
                {"functionResponse": {"id": "fc_1", "name": "get_weather", "response": {"temp_c": 18}}},
                {"functionResponse": {"id": "fc_2", "name": "get_time", "response": {"result": "12:00"}}},
                {"text": "thanks"},
            ],
        },
    ]
    assert out.dummy_signatures == 0


def test_signature_precedence_store_then_client_then_dummy() -> None:
    client = {"extra_content": {"google": {"thought_signature": "client-sig"}}}
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [call("a") | client]},
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "assistant", "content": None, "tool_calls": [call("b"), call("c")]},
        {"role": "tool", "tool_call_id": "b", "content": "2"},
        {"role": "tool", "tool_call_id": "c", "content": "3"},
    ]
    stored = build(tools=TOOLS, messages=messages, signatures={"a": "stored-sig"})
    model_turns = [c["parts"] for c in stored.body["contents"] if c["role"] == "model"]
    assert model_turns[0][0]["thoughtSignature"] == "stored-sig"
    assert model_turns[1][0]["thoughtSignature"] == DUMMY_SIGNATURE
    assert "thoughtSignature" not in model_turns[1][1]
    assert stored.dummy_signatures == 1
    from_client = build(tools=TOOLS, messages=messages)
    first = next(c for c in from_client.body["contents"] if c["role"] == "model")
    assert first["parts"][0]["thoughtSignature"] == "client-sig"


def test_synthesised_ids_are_not_sent_upstream() -> None:
    out = build(
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [call("call_gg_abc")]},
            {"role": "tool", "tool_call_id": "call_gg_abc", "content": "ok"},
        ],
    )
    fc = out.body["contents"][1]["parts"][0]["functionCall"]
    fr = out.body["contents"][2]["parts"][0]["functionResponse"]
    assert "id" not in fc
    assert "id" not in fr
    assert fr["name"] == "get_weather"


def test_history_call_ids_are_unique_and_ordered() -> None:
    request = make_request(
        messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [call("x"), call("y")]},
            {"role": "tool", "tool_call_id": "x", "content": "1"},
            {"role": "tool", "tool_call_id": "y", "content": "2"},
        ]
    )
    assert history_call_ids(request) == ["x", "y"]


@pytest.mark.parametrize(
    ("messages", "code"),
    [
        (
            [{"role": "user", "content": "x"}, {"role": "tool", "tool_call_id": "nope", "content": "1"}],
            "orphan_tool_result",
        ),
        (
            [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": None, "tool_calls": [call("a", arguments="[1, 2]")]},
            ],
            "invalid_tool_arguments",
        ),
        (
            [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": None, "tool_calls": [call("a", arguments="{bad")]},
            ],
            "invalid_tool_arguments",
        ),
        ([{"role": "assistant", "content": "hi"}, {"role": "user", "content": "x"}], "invalid_message_order"),
        ([{"role": "system", "content": "only system"}], "invalid_message_order"),
    ],
)
def test_invalid_history_is_a_client_error(messages: list[dict[str, Any]], code: str) -> None:
    with pytest.raises(ProviderError) as exc:
        build(messages=messages)
    assert (exc.value.kind, exc.value.code) == ("client", code)


def test_tools_and_tool_choice_modes() -> None:
    out = build(tools=TOOLS)
    assert out.body["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "get_weather",
                    "description": "weather by city",
                    "parametersJsonSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
                {"name": "get_time", "description": ""},
            ]
        }
    ]
    assert "toolConfig" not in out.body
    assert out.ignored == ("tools.function.strict",)
    for choice, config in [
        ("auto", {"mode": "AUTO"}),
        ("none", {"mode": "NONE"}),
        ("required", {"mode": "ANY"}),
        (
            {"type": "function", "function": {"name": "get_time"}},
            {"mode": "ANY", "allowedFunctionNames": ["get_time"]},
        ),
    ]:
        body = build(tools=TOOLS, tool_choice=choice).body
        assert body["toolConfig"] == {"functionCallingConfig": config}


def test_unknown_tool_type_is_a_capability_mismatch() -> None:
    with pytest.raises(ProviderError) as exc:
        build(tools=[{"type": "web_search"}])
    assert (exc.value.kind, exc.value.code, exc.value.violations) == (
        "fallback",
        "capability_mismatch",
        ("tools.web_search",),
    )


def test_generation_config_mapping() -> None:
    out = build(
        max_completion_tokens=2048,
        temperature=1.2,
        top_p=0.9,
        stop=["END", "STOP"],
        seed=7,
        presence_penalty=0.5,
        frequency_penalty=-0.5,
        logprobs=True,
        top_logprobs=3,
        reasoning_effort="medium",
    )
    assert out.body["generationConfig"] == {
        "maxOutputTokens": 2048,
        "temperature": 1.2,
        "topP": 0.9,
        "stopSequences": ["END", "STOP"],
        "seed": 7,
        "presencePenalty": 0.5,
        "frequencyPenalty": -0.5,
        "responseLogprobs": True,
        "logprobs": 3,
        "thinkingConfig": {"thinkingLevel": "medium"},
    }


def test_candidate_count_for_n() -> None:
    assert build(n=2).body["generationConfig"] == {"candidateCount": 2}


def test_small_cap_gets_thinking_headroom_and_lowest_level() -> None:
    out = build(max_completion_tokens=16, reasoning_effort="high")
    assert out.body["generationConfig"] == {
        "maxOutputTokens": 1040,
        "thinkingConfig": {"thinkingLevel": "low"},
    }
    assert out.adjustments == ("reasoning_effort:high->low", "max_output_tokens:16->1040")
    lite = gemini_dep("gemini-3.1-flash-lite", effort_levels=frozenset({"minimal", "low", "medium", "high"}))
    assert build(dep=lite, max_completion_tokens=100).body["generationConfig"] == {
        "maxOutputTokens": 1124,
        "thinkingConfig": {"thinkingLevel": "minimal"},
    }
    assert build(quirks={"min_thinking_output": 0}, max_completion_tokens=16).body["generationConfig"] == {
        "maxOutputTokens": 16
    }


def test_reasoning_effort_ignored_on_non_thinking_deployment() -> None:
    out = build(dep=gemini_dep(thinking_mode="none"), reasoning_effort="low", max_completion_tokens=16)
    assert out.body["generationConfig"] == {"maxOutputTokens": 16}
    assert out.ignored == ("reasoning_effort",)


@pytest.mark.parametrize(
    ("fmt", "response_format", "mime_type"),
    [
        (
            {"type": "json_object"},
            {"responseFormat": {"text": {"mimeType": "application/json"}}},
            {"responseMimeType": "application/json"},
        ),
        (
            {
                "type": "json_schema",
                "json_schema": {"name": "s", "schema": {"type": "object"}, "strict": True},
            },
            {"responseFormat": {"text": {"mimeType": "application/json", "schema": {"type": "object"}}}},
            {"responseMimeType": "application/json", "responseJsonSchema": {"type": "object"}},
        ),
    ],
)
def test_response_format_modes(
    fmt: dict[str, Any], response_format: dict[str, Any], mime_type: dict[str, Any]
) -> None:
    assert build(response_format=fmt).body["generationConfig"] == response_format
    assert (
        build(response_format=fmt, quirks={"structured_output": "mime_type"}).body["generationConfig"]
        == mime_type
    )


def test_images_files_and_audio_become_inline_data() -> None:
    out = build(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                    {
                        "type": "file",
                        "file": {"file_data": "data:application/pdf;base64,JVBERi0=", "filename": "a.pdf"},
                    },
                    {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
                ],
            }
        ]
    )
    assert out.body["contents"][0]["parts"] == [
        {"text": "describe"},
        {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}},
        {"inlineData": {"mimeType": "application/pdf", "data": "JVBERi0="}},
        {"inlineData": {"mimeType": "audio/wav", "data": "UklGRg=="}},
    ]


def test_remote_image_url_is_a_capability_mismatch() -> None:
    with pytest.raises(ProviderError) as exc:
        build(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}],
                }
            ]
        )
    assert exc.value.violations == ("content.image_url",)


def test_unhandled_params_are_reported_and_dropped() -> None:
    out = build(
        logit_bias={"1": 2}, metadata={"a": "b"}, parallel_tool_calls=False, service_tier="flex", foo=1
    )
    assert set(out.ignored) == {"logit_bias", "metadata", "parallel_tool_calls", "service_tier", "foo"}
    assert set(out.body) == {"contents", "labels", "store"}


def test_safety_settings_from_quirks() -> None:
    setting = {"category": "HARM_CATEGORY_JAILBREAK", "threshold": "BLOCK_ONLY_HIGH"}
    assert build(quirks={"safety_settings": [setting]}).body["safetySettings"] == [setting]


def test_body_is_deterministic() -> None:
    kwargs: dict[str, Any] = {"tools": TOOLS, "temperature": 1.0, "user": "u", "max_completion_tokens": 99}
    assert build(**kwargs).body == build(**kwargs).body
