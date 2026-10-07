"""request mapping, one case per row of research provider-anthropic.md §9.1 plus the §9.3 edge cases"""

import hashlib
from typing import Any

import orjson
import pytest

from gg.core.deployment import Deployment
from gg.core.errors import ProviderError
from gg.providers.anthropic.quirks import AnthropicQuirks
from gg.providers.anthropic.request import (
    JSON_OBJECT_INSTRUCTION,
    RequestInfo,
    Translated,
    build_body,
    history_tool_call_ids,
    sanitize_tool_id,
)
from gg.providers.catalog.capabilities import CapabilityChecker
from gg.providers.errors import capability_mismatch
from gg.providers.state.thinking import ThinkingBlocks
from tests.conftest import make_request
from tests.unit.providers.anthropic.support import haiku, opus, sonnet

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
INFO = RequestInfo(key_id="test-key")


def translate(
    dep: Deployment | None = None,
    *,
    thinking: dict[str, ThinkingBlocks] | None = None,
    quirks: dict[str, Any] | None = None,
    **req: Any,
) -> Translated:
    """the adapter path: capability check + adjustments, then the pure translator"""
    dep = dep or haiku()
    request = make_request(**req)
    checker = CapabilityChecker()
    result = checker.check(request, dep)
    if result.rejects:
        raise capability_mismatch("anthropic", dep.id, result.rejects)
    body = build_body(
        checker.apply(request, result),
        dep,
        AnthropicQuirks.model_validate(quirks or {}),
        INFO,
        thinking=thinking,
    )
    return Translated(
        body.body, (*result.ignored_params, *body.ignored), (*result.adjustments, *body.adjustments)
    )


def call(call_id: str, name: str = "get_weather", arguments: str = '{"city":"Paris"}') -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def rejects(**req: Any) -> ProviderError:
    with pytest.raises(ProviderError) as exc:
        translate(**req)
    return exc.value


def test_model_and_minimal_body() -> None:
    out = translate()
    assert out.body == {
        "model": "claude-haiku-4-5",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "metadata": {"user_id": hashlib.sha256(b"test-key:").hexdigest()},
        "stream": True,
    }
    assert list(out.body) == ["model", "max_tokens", "messages", "metadata", "stream"]


def test_system_and_developer_hoisted_in_order() -> None:
    out = translate(
        messages=[
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "developer", "content": [{"type": "text", "text": "use metric"}]},
            {"role": "user", "content": "weather?"},
        ]
    )
    assert out.body["system"] == [
        {"type": "text", "text": "be brief"},
        {"type": "text", "text": "use metric"},
    ]
    assert [m["role"] for m in out.body["messages"]] == ["user", "assistant", "user"]


def test_user_text_parts_and_consecutive_turns_merge() -> None:
    out = translate(
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": ""}]},
            {"role": "user", "content": "b"},
        ]
    )
    assert out.body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    ]


def image_request(url: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]}


def test_image_data_url_becomes_base64_source() -> None:
    out = translate(**image_request("data:image/png;base64,iVBORw0KGgo="))
    assert out.body["messages"][0]["content"] == [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
    ]


def test_image_https_url_becomes_url_source() -> None:
    out = translate(**image_request("https://example.com/cat.jpg"))
    assert out.body["messages"][0]["content"] == [
        {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.jpg"}}
    ]


@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("data:image/bmp;base64,Qk0=", "unsupported_image_type"),
        ("data:image/png,raw", "invalid_image"),
        ("data:image/png;base64," + "A" * (14 * 1024 * 1024), "image_too_large"),
        ("ftp://example.com/cat.jpg", "invalid_image_url"),
    ],
)
def test_bad_images_are_client_errors(url: str, code: str) -> None:
    err = rejects(**image_request(url))
    assert (err.kind, err.status, err.code) == ("client", 400, code)


def test_pdf_file_becomes_document_and_file_id_is_a_capability_mismatch() -> None:
    pdf = {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0="}}
    out = translate(messages=[{"role": "user", "content": [pdf]}])
    assert out.body["messages"][0]["content"] == [
        {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="},
        }
    ]
    err = rejects(messages=[{"role": "user", "content": [{"type": "file", "file": {"file_id": "file-1"}}]}])
    assert (err.kind, err.code, err.violations) == ("fallback", "capability_mismatch", ("file_id",))


def test_audio_is_a_capability_mismatch() -> None:
    audio = {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}
    err = rejects(messages=[{"role": "user", "content": [audio]}])
    assert err.code == "capability_mismatch"
    assert "audio" in err.violations


def test_assistant_text_then_tool_use_and_refusal_dropped() -> None:
    out = translate(
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "refusal": "no", "tool_calls": [call("toolu_1")]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is sunny."}, {"type": "refusal", "refusal": "x"}],
            },
            {"role": "user", "content": "thanks"},
        ],
    )
    messages = out.body["messages"]
    assert messages[1]["content"] == [
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
    ]
    assert messages[3]["content"] == [{"type": "text", "text": "It is sunny."}]


@pytest.mark.parametrize("arguments", ["{not json", "[1, 2]", '"x"'])
def test_tool_arguments_must_be_a_json_object(arguments: str) -> None:
    err = rejects(
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": None, "tool_calls": [call("c1", arguments=arguments)]},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ],
    )
    assert (err.kind, err.code) == ("client", "invalid_tool_arguments")


def test_tool_results_grouped_first_with_following_user_text() -> None:
    out = translate(
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "both cities"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [call("c1"), call("c2", arguments="")],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            {"role": "tool", "tool_call_id": "c2", "content": [{"type": "text", "text": "rain"}]},
            {"role": "user", "content": "and tomorrow?"},
        ],
    )
    assistant, results = out.body["messages"][1:]
    assert [b["type"] for b in assistant["content"]] == ["text", "tool_use", "tool_use"]
    assert assistant["content"][2]["input"] == {}
    assert results == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "sunny"},
            {"type": "tool_result", "tool_use_id": "c2", "content": [{"type": "text", "text": "rain"}]},
            {"type": "text", "text": "and tomorrow?"},
        ],
    }


def test_tool_results_lead_even_when_user_text_came_first() -> None:
    out = translate(
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": None, "tool_calls": [call("c1")]},
            {"role": "user", "content": "note"},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ],
    )
    assert [b["type"] for b in out.body["messages"][2]["content"]] == ["tool_result", "text"]


@pytest.mark.parametrize(
    ("dep", "requested", "expected"),
    [(haiku(), None, 4096), (sonnet(), None, 16000), (haiku(), 100_000, 64_000), (sonnet(), 10, 1024)],
)
def test_max_tokens_default_and_clamp(dep: Deployment, requested: int | None, expected: int) -> None:
    extra = {"max_completion_tokens": requested} if requested else {}
    assert translate(dep, **extra).body["max_tokens"] == expected


def test_sampling_on_haiku_clamps_temperature_and_drops_top_p() -> None:
    out = translate(temperature=1.5, top_p=0.5)
    assert out.body["temperature"] == 1.0
    assert "top_p" not in out.body
    assert "top_p" in out.ignored
    assert "temperature:1.5->1.0" in out.adjustments
    assert translate(top_p=0.5).body["top_p"] == 0.5


def test_sampling_on_5_5_models_is_dropped() -> None:
    out = translate(sonnet(), temperature=0, top_p=0.9)
    assert "temperature" not in out.body
    assert "top_p" not in out.body
    assert set(out.ignored) >= {"temperature", "top_p"}


def test_stop_becomes_stop_sequences_without_whitespace_entries() -> None:
    out = translate(stop=["\n\nEND", "  "])
    assert out.body["stop_sequences"] == ["\n\nEND"]
    assert "stop:dropped_whitespace" in out.adjustments
    assert "stop_sequences" not in translate(stop=[" "]).body


@pytest.mark.parametrize(("req", "violation"), [({"n": 2}, "n"), ({"logprobs": True}, "logprobs")])
def test_n_and_logprobs_are_rejected(req: dict[str, Any], violation: str) -> None:
    err = rejects(**req)
    assert err.code == "capability_mismatch"
    assert violation in err.violations


def test_tools_map_to_input_schema() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "a", "description": "A", "parameters": {"type": "object"}, "strict": True},
        },
        {"type": "function", "function": {"name": "b"}},
        {"type": "web_search"},
    ]
    out = translate(tools=tools)
    assert out.body["tools"] == [
        {"name": "a", "description": "A", "input_schema": {"type": "object"}, "strict": True},
        {"name": "b", "input_schema": {"type": "object", "properties": {}}},
    ]
    assert "tools:web_search" in out.ignored
    err = rejects(tools=[{"type": "function", "function": {"name": "ns.tool"}}])
    assert err.violations == ("tool_name",)


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        ("required", {"type": "any"}),
        ({"type": "function", "function": {"name": "get_weather"}}, {"type": "tool", "name": "get_weather"}),
    ],
)
def test_tool_choice_on_haiku(choice: Any, expected: dict[str, Any]) -> None:
    out = translate(tools=TOOLS, tool_choice=choice)
    assert out.body["tool_choice"] == expected
    assert out.body["tools"]


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "get_weather"}}])
def test_forced_tool_choice_on_5_5_is_a_capability_mismatch(choice: Any) -> None:
    with pytest.raises(ProviderError) as exc:
        translate(sonnet(), tools=TOOLS, tool_choice=choice)
    assert (exc.value.kind, exc.value.code) == ("fallback", "capability_mismatch")
    assert exc.value.violations == ("forced_tool_choice",)
    assert translate(sonnet(), tools=TOOLS, tool_choice="auto").body["tool_choice"] == {"type": "auto"}


def test_parallel_tool_calls_false_disables_parallel_use() -> None:
    out = translate(tools=TOOLS, parallel_tool_calls=False)
    assert out.body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    forced = translate(tools=TOOLS, parallel_tool_calls=False, tool_choice="required")
    assert forced.body["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}


def test_tool_choice_without_tools_is_dropped() -> None:
    out = translate(tool_choice="auto")
    assert "tool_choice" not in out.body
    assert "tool_choice" in out.ignored


def test_json_schema_maps_to_output_config_format() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": False}
    fmt = {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}}
    out = translate(sonnet(), response_format=fmt)
    assert out.body["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": schema},
    }


@pytest.mark.parametrize(
    ("schema", "violation"),
    [
        ({"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}}, "json_schema:minimum"),
        ({"type": "object", "additionalProperties": True}, "json_schema:additionalProperties"),
        ({"type": "object", "properties": {"self": {"$ref": "#"}}}, "json_schema:recursion"),
        (
            {"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(25)}},
            "json_schema:optional_params",
        ),
    ],
)
def test_json_schema_limits_name_the_keyword(schema: dict[str, Any], violation: str) -> None:
    err = rejects(response_format={"type": "json_schema", "json_schema": {"name": "s", "schema": schema}})
    assert err.code == "capability_mismatch"
    assert violation in err.violations


def test_property_named_like_a_keyword_is_not_a_violation() -> None:
    schema = {"type": "object", "properties": {"minimum": {"type": "string"}}, "required": ["minimum"]}
    out = translate(response_format={"type": "json_schema", "json_schema": {"name": "s", "schema": schema}})
    assert out.body["output_config"]["format"]["schema"] == schema


def test_json_object_becomes_a_system_instruction() -> None:
    out = translate(
        messages=[{"role": "system", "content": "be terse"}, {"role": "user", "content": "x"}],
        response_format={"type": "json_object"},
    )
    assert out.body["system"][-1] == {"type": "text", "text": JSON_OBJECT_INSTRUCTION}
    assert "response_format:json_object->instruction" in out.adjustments
    assert "output_config" not in out.body


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        (None, {"effort": "medium"}),
        ("high", {"effort": "high"}),
        ("minimal", {"effort": "low"}),
        ("max", {"effort": "max"}),
    ],
)
def test_reasoning_effort_on_5_5_models(effort: str | None, expected: dict[str, str]) -> None:
    extra = {"reasoning_effort": effort} if effort else {}
    for dep in (sonnet(), opus()):
        out = translate(dep, **extra)
        assert out.body["output_config"] == expected
        assert "thinking" not in out.body


@pytest.mark.parametrize(
    ("effort", "budget"),
    [("none", None), ("minimal", None), ("low", 1024), ("medium", 4096), ("high", 16000)],
)
def test_reasoning_effort_on_haiku_is_a_thinking_budget(effort: str, budget: int | None) -> None:
    out = translate(reasoning_effort=effort, max_completion_tokens=32000)
    if budget is None:
        assert "thinking" not in out.body
    else:
        assert out.body["thinking"] == {"type": "enabled", "budget_tokens": budget}
    assert "output_config" not in out.body


def test_haiku_budget_shrinks_below_max_tokens_or_is_dropped() -> None:
    out = translate(reasoning_effort="high", max_completion_tokens=4096)
    assert out.body["thinking"] == {"type": "enabled", "budget_tokens": 3840}
    assert "thinking.budget_tokens:16000->3840" in out.adjustments
    small = translate(reasoning_effort="low", max_completion_tokens=1024)
    assert "thinking" not in small.body
    assert "thinking:dropped_max_tokens" in small.adjustments


def test_haiku_forced_tool_choice_drops_thinking() -> None:
    out = translate(tools=TOOLS, tool_choice="required", reasoning_effort="medium")
    assert "thinking" not in out.body
    assert "thinking:dropped_forced_tool_choice" in out.adjustments


def test_thinking_budgets_are_configurable() -> None:
    out = translate(reasoning_effort="low", quirks={"thinking_budgets": {"low": 2000}})
    assert out.body["thinking"] == {"type": "enabled", "budget_tokens": 2000}


def test_user_becomes_hashed_metadata_user_id() -> None:
    out = translate(user="alice@example.com")
    assert out.body["metadata"] == {"user_id": hashlib.sha256(b"test-key:alice@example.com").hexdigest()}
    assert b"alice" not in orjson.dumps(out.body)


def test_unsupported_params_are_ignored_and_listed() -> None:
    out = translate(
        seed=1,
        logit_bias={"1": 2},
        presence_penalty=0.5,
        frequency_penalty=0.5,
        metadata={"a": "b"},
        store=True,
        service_tier="flex",
        prediction={"type": "content"},
        modalities=["text"],
    )
    assert set(out.ignored) >= {
        "seed",
        "logit_bias",
        "presence_penalty",
        "frequency_penalty",
        "metadata",
        "store",
        "service_tier",
        "prediction",
        "modalities",
    }
    assert set(out.body) == {"model", "max_tokens", "messages", "metadata", "stream"}


@pytest.mark.parametrize(
    ("messages", "code"),
    [
        ([{"role": "assistant", "content": "hi"}, {"role": "user", "content": "x"}], "invalid_message_order"),
        ([{"role": "system", "content": "only system"}], "invalid_message_order"),
        (
            [{"role": "user", "content": "x"}, {"role": "tool", "tool_call_id": "nope", "content": "1"}],
            "orphan_tool_result",
        ),
    ],
)
def test_history_shape_errors_are_client_errors(messages: list[dict[str, Any]], code: str) -> None:
    err = rejects(messages=messages)
    assert (err.kind, err.status, err.code) == ("client", 400, code)


def test_prefill_allowed_on_haiku_without_trailing_whitespace_and_rejected_on_5_5() -> None:
    messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "The answer is "}]
    out = translate(messages=messages)
    assert out.body["messages"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "The answer is"}],
    }
    with pytest.raises(ProviderError) as exc:
        translate(sonnet(), messages=messages)
    assert exc.value.violations == ("prefill",)


def test_non_conforming_tool_ids_are_mapped_deterministically() -> None:
    weird = "call:1/abc"
    mapped = sanitize_tool_id(weird)
    assert mapped == "gg_" + hashlib.sha1(weird.encode()).hexdigest()[:24]  # noqa: S324
    assert sanitize_tool_id("toolu_01A09q90qw90lq917835lq9") == "toolu_01A09q90qw90lq917835lq9"
    assert sanitize_tool_id("call_abc-123") == "call_abc-123"
    out = translate(
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": None, "tool_calls": [call(weird)]},
            {"role": "tool", "tool_call_id": weird, "content": "r"},
        ],
    )
    assert out.body["messages"][1]["content"][0]["id"] == mapped
    assert out.body["messages"][2]["content"][0]["tool_use_id"] == mapped


THINK_A: ThinkingBlocks = [{"type": "thinking", "thinking": "", "signature": "sigA"}]
THINK_B: ThinkingBlocks = [{"type": "redacted_thinking", "data": "opaque"}]
LOOP = [
    {"role": "user", "content": "x"},
    {"role": "assistant", "content": "Let me check.", "tool_calls": [call("toolu_a"), call("toolu_b")]},
    {"role": "tool", "tool_call_id": "toolu_a", "content": "1"},
    {"role": "tool", "tool_call_id": "toolu_b", "content": "2"},
]


def test_stored_thinking_is_reinjected_around_tool_use() -> None:
    out = translate(sonnet(), tools=TOOLS, messages=LOOP, thinking={"toolu_a": THINK_A, "toolu_b": THINK_B})
    content = out.body["messages"][1]["content"]
    assert [b["type"] for b in content] == ["thinking", "text", "tool_use", "redacted_thinking", "tool_use"]
    assert content[0] == THINK_A[0]
    assert content[3] == THINK_B[0]
    assert history_tool_call_ids(make_request(tools=TOOLS, messages=LOOP)) == ["toolu_a", "toolu_b"]


def test_thinking_is_not_reinjected_when_haiku_does_not_think() -> None:
    out = translate(tools=TOOLS, messages=LOOP, thinking={"toolu_a": THINK_A})
    assert [b["type"] for b in out.body["messages"][1]["content"]] == ["text", "tool_use", "tool_use"]
    thinking = translate(tools=TOOLS, messages=LOOP, thinking={"toolu_a": THINK_A}, reasoning_effort="low")
    assert thinking.body["messages"][1]["content"][0] == THINK_A[0]


def test_cache_control_needs_reuse_and_the_model_minimum() -> None:
    long = "word " * 3000
    multi = [
        {"role": "user", "content": long},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "more"},
    ]
    assert translate(sonnet(), messages=multi).body["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in translate(sonnet(), messages=[{"role": "user", "content": long}]).body
    assert "cache_control" not in translate(haiku(), messages=multi).body  # haiku minimum is 4096 tokens
    static = [{"role": "system", "content": long * 3}, {"role": "user", "content": "q"}]
    assert translate(haiku(), messages=static).body["cache_control"] == {"type": "ephemeral"}
    assert (
        "cache_control" not in translate(sonnet(), messages=multi, quirks={"cache": {"enabled": False}}).body
    )


def test_translation_is_deterministic() -> None:
    req: dict[str, Any] = {"tools": TOOLS, "messages": LOOP, "user": "u", "stop": ["x"], "temperature": 0.3}
    assert orjson.dumps(translate(**req).body) == orjson.dumps(translate(**req).body)
