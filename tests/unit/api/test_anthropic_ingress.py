from typing import Any

import orjson
import pytest

from gg.api.anthropic_ingress.request import translate_request
from gg.api.anthropic_ingress.response import error_body, to_message, usage_block
from gg.api.anthropic_ingress.stream import AnthropicStreamEncoder
from gg.api.parsing import chat_request_from_dict
from gg.core.errors import (
    AuthenticationError,
    GGError,
    GuardrailBlockedError,
    InternalError,
    InvalidRequestError,
    NotFoundError,
    PermissionDeniedError,
    QuotaExceededError,
    RateLimitedError,
    ServiceUnavailableError,
    UpstreamError,
    UpstreamTimeoutError,
)
from gg.core.schema import (
    AssistantMessage,
    ChatChunk,
    ChatResponse,
    Choice,
    ChunkChoice,
    Delta,
    FunctionCall,
    FunctionCallDelta,
    PromptTokensDetails,
    ToolCall,
    ToolCallDelta,
    Usage,
)

BASE: dict[str, Any] = {"model": "gg/auto", "max_tokens": 64}
HI: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]


def body(**extra: Any) -> dict[str, Any]:
    return {**BASE, "messages": HI, **extra}


TRANSLATIONS: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    (
        "minimal",
        body(),
        {"model": "gg/auto", "messages": HI, "max_completion_tokens": 64, "stream": False},
    ),
    (
        "system string",
        body(system="be brief"),
        {"messages": [{"role": "system", "content": "be brief"}, *HI]},
    ),
    (
        "system blocks become one system message each",
        body(system=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]),
        {"messages": [{"role": "system", "content": "a"}, {"role": "system", "content": "b"}, *HI]},
    ),
    (
        "sampling, stops, user, stream",
        body(temperature=0.2, top_p=0.9, stop_sequences=["END"], metadata={"user_id": "u1"}, stream=True),
        {"temperature": 0.2, "top_p": 0.9, "stop": ["END"], "user": "u1", "stream": True},
    ),
    (
        "text and image blocks",
        body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "AAA"},
                        },
                        {"type": "image", "source": {"type": "url", "url": "https://x.test/a.png"}},
                    ],
                }
            ]
        ),
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                        {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
                    ],
                }
            ]
        },
    ),
    (
        "pdf and text documents",
        body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "title": "r.pdf",
                            "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBE"},
                        },
                        {"type": "document", "source": {"type": "text", "data": "plain"}},
                    ],
                }
            ]
        ),
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {"file_data": "data:application/pdf;base64,JVBE", "filename": "r.pdf"},
                        },
                        {"type": "text", "text": "plain"},
                    ],
                }
            ]
        },
    ),
    (
        "tool use round trip",
        body(
            tools=[{"name": "get_weather", "description": "w", "input_schema": {"type": "object"}}],
            tool_choice={"type": "any", "disable_parallel_tool_use": True},
            messages=[
                *HI,
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "", "signature": "sig"},
                        {"type": "text", "text": "checking"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "Oslo"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "15C"},
                        {"type": "text", "text": "and tomorrow?"},
                    ],
                },
            ],
        ),
        {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object"}, "description": "w"},
                }
            ],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "messages": [
                *HI,
                {
                    "role": "assistant",
                    "content": "checking",
                    "tool_calls": [
                        {
                            "id": "toolu_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "toolu_1", "content": "15C"},
                {"role": "user", "content": [{"type": "text", "text": "and tomorrow?"}]},
            ],
        },
    ),
    (
        "tool result error and block content",
        body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t",
                            "is_error": True,
                            "content": [{"type": "text", "text": "boom"}],
                        }
                    ],
                }
            ]
        ),
        {
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "t",
                    "content": [{"type": "text", "text": "Error:"}, {"type": "text", "text": "boom"}],
                }
            ]
        },
    ),
    ("tool_choice auto", body(tool_choice={"type": "auto"}), {"tool_choice": "auto"}),
    ("tool_choice none", body(tool_choice={"type": "none"}), {"tool_choice": "none"}),
    (
        "tool_choice named",
        body(tool_choice={"type": "tool", "name": "f"}),
        {"tool_choice": {"type": "function", "function": {"name": "f"}}},
    ),
    (
        "thinking small budget",
        body(thinking={"type": "enabled", "budget_tokens": 1024}),
        {"reasoning_effort": "low"},
    ),
    (
        "thinking mid budget",
        body(thinking={"type": "enabled", "budget_tokens": 8000}),
        {"reasoning_effort": "medium"},
    ),
    (
        "thinking big budget",
        body(thinking={"type": "enabled", "budget_tokens": 32000}),
        {"reasoning_effort": "high"},
    ),
    (
        "output_config effort wins",
        body(thinking={"type": "enabled", "budget_tokens": 32000}, output_config={"effort": "low"}),
        {"reasoning_effort": "low"},
    ),
    (
        "structured output",
        body(output_config={"format": {"type": "json_schema", "schema": {"type": "object"}}}),
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": {"type": "object"}, "strict": True},
            }
        },
    ),
    (
        "mid-conversation system turn",
        body(messages=[*HI, {"role": "system", "content": "now be terse"}]),
        {"messages": [*HI, {"role": "system", "content": "now be terse"}]},
    ),
    ("gg extensions pass through", body(gg={"cache": "off"}), {"gg": {"cache": "off"}}),
]


@pytest.mark.parametrize(
    ("raw", "expected"), [(r, e) for _, r, e in TRANSLATIONS], ids=[n for n, _, _ in TRANSLATIONS]
)
def test_translate_request(raw: dict[str, Any], expected: dict[str, Any]) -> None:
    payload, _ = translate_request(raw)
    for key, value in expected.items():
        assert payload[key] == value, key
    assert "thinking" not in payload
    chat_request_from_dict(payload)


IGNORED: list[tuple[dict[str, Any], set[str]]] = [
    (body(top_k=5, service_tier="auto"), {"top_k", "service_tier"}),
    (body(system=[{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}]), {"cache_control"}),
    (
        body(messages=[*HI, {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}]}]),
        {"thinking_blocks"},
    ),
    (body(thinking={"type": "adaptive"}), set()),
]


@pytest.mark.parametrize(("raw", "ignored"), IGNORED)
def test_ignored_params_are_reported(raw: dict[str, Any], ignored: set[str]) -> None:
    payload, reported = translate_request(raw)
    assert reported == ignored
    assert "top_k" not in payload


REJECTED: list[tuple[str, dict[str, Any], str, str]] = [
    ("missing max_tokens", {"model": "m", "messages": HI}, "max_tokens", "missing_required_parameter"),
    ("unknown field", body(frequency_penalty=1), "frequency_penalty", "unknown_parameter"),
    ("temperature above anthropic range", body(temperature=1.5), "temperature", "invalid_value"),
    ("mcp servers", body(mcp_servers=[]), "mcp_servers", "unsupported_feature"),
    (
        "server tool",
        body(tools=[{"type": "web_search_20250305", "name": "web_search"}]),
        "tools[0].type",
        "unsupported_feature",
    ),
    (
        "unknown content block",
        body(messages=[{"role": "user", "content": [{"type": "search_result", "source": "x"}]}]),
        "messages[0].content[0]",
        "unsupported_feature",
    ),
    (
        "file image source",
        body(
            messages=[
                {"role": "user", "content": [{"type": "image", "source": {"type": "file", "file_id": "f"}}]}
            ]
        ),
        "messages[0].content[0].source",
        "unsupported_feature",
    ),
    ("too many stops", body(stop_sequences=["a", "b", "c", "d", "e"]), "stop_sequences", "invalid_value"),
    ("named tool without name", body(tool_choice={"type": "tool"}), "tool_choice.name", "invalid_value"),
    (
        "enabled thinking without budget",
        body(thinking={"type": "enabled"}),
        "thinking.budget_tokens",
        "invalid_value",
    ),
    (
        "only thinking content",
        body(
            messages=[
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": "s"}]}
            ]
        ),
        "messages",
        "invalid_value",
    ),
]


@pytest.mark.parametrize(
    ("raw", "param", "code"), [(r, p, c) for _, r, p, c in REJECTED], ids=[n for n, *_ in REJECTED]
)
def test_rejected_requests(raw: dict[str, Any], param: str, code: str) -> None:
    with pytest.raises(InvalidRequestError) as info:
        translate_request(raw)
    assert info.value.param == param
    assert info.value.code == code


def _response(message: AssistantMessage, finish: Any, usage: Usage | None) -> ChatResponse:
    return ChatResponse(
        id="chatcmpl-1",
        created=1,
        model="mock/echo",
        choices=(Choice(index=0, message=message, finish_reason=finish),),
        usage=usage,
    )


def test_response_text_and_usage() -> None:
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=7,
        total_tokens=107,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=60, cache_write_tokens=10),
    )
    message = to_message(_response(AssistantMessage(content="hey"), "length", usage), request_id="req_abc")
    assert message == {
        "id": "msg_abc",
        "type": "message",
        "role": "assistant",
        "model": "mock/echo",
        "content": [{"type": "text", "text": "hey"}],
        "stop_reason": "max_tokens",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 30,
            "output_tokens": 7,
            "cache_read_input_tokens": 60,
            "cache_creation_input_tokens": 10,
        },
    }


@pytest.mark.parametrize(
    ("finish", "stop"),
    [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
        ("content_filter", "refusal"),
        (None, "end_turn"),
    ],
)
def test_stop_reason_mapping(finish: Any, stop: str) -> None:
    message = to_message(_response(AssistantMessage(content="x"), finish, None), request_id="req_1")
    assert message["stop_reason"] == stop
    assert message["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_response_tool_use() -> None:
    call = ToolCall(id="call_1", function=FunctionCall(name="f", arguments='{"a":1}'))
    message = to_message(
        _response(AssistantMessage(content=None, tool_calls=(call,)), "tool_calls", None), request_id="req_1"
    )
    assert message["content"] == [{"type": "tool_use", "id": "call_1", "name": "f", "input": {"a": 1}}]


def test_response_malformed_tool_arguments_is_an_upstream_error() -> None:
    call = ToolCall(id="call_1", function=FunctionCall(name="f", arguments="[1]"))
    with pytest.raises(UpstreamError):
        to_message(_response(AssistantMessage(tool_calls=(call,)), "tool_calls", None), request_id="req_1")


def test_usage_block_without_details() -> None:
    assert usage_block(Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)) == {
        "input_tokens": 3,
        "output_tokens": 2,
    }


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (InvalidRequestError("x"), "invalid_request_error"),
        (GuardrailBlockedError("x"), "invalid_request_error"),
        (AuthenticationError("x"), "authentication_error"),
        (QuotaExceededError("x"), "billing_error"),
        (PermissionDeniedError("x"), "permission_error"),
        (NotFoundError("x"), "not_found_error"),
        (RateLimitedError("x"), "rate_limit_error"),
        (InternalError("x"), "api_error"),
        (UpstreamError("x"), "api_error"),
        (ServiceUnavailableError("x"), "overloaded_error"),
        (UpstreamTimeoutError("x"), "timeout_error"),
    ],
)
def test_error_body(error: GGError, kind: str) -> None:
    out = error_body(error, request_id="req_1")
    assert out["type"] == "error"
    assert out["error"]["type"] == kind
    assert out["error"]["message"] == "x"
    assert out["error"]["details"]["code"] == error.code
    assert out["request_id"] == "req_1"


def _chunk(delta: Delta, finish: Any = None, usage: Usage | None = None) -> ChatChunk:
    choices = (ChunkChoice(index=0, delta=delta, finish_reason=finish),) if delta or finish else ()
    return ChatChunk(id="c", created=1, model="mock/echo", choices=choices, usage=usage)


def _events(frames: list[bytes]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for frame in frames:
        event, data = frame.decode().strip().split("\n")
        payload = orjson.loads(data.removeprefix("data: "))
        assert event == "event: " + payload["type"]
        out.append((payload["type"], payload))
    return out


def test_stream_text_then_tool_calls() -> None:
    enc = AnthropicStreamEncoder(request_id="req_1", model="gg/auto")
    first = ToolCallDelta(
        index=0, id="call_a", type="function", function=FunctionCallDelta(name="f", arguments="")
    )
    frames = enc.chunk(_chunk(Delta(role="assistant", content="Hi")))
    frames += enc.chunk(_chunk(Delta(content=" there")))
    frames += enc.chunk(_chunk(Delta(tool_calls=(first,))))
    frames += enc.chunk(
        _chunk(Delta(tool_calls=(ToolCallDelta(index=0, function=FunctionCallDelta(arguments='{"a":')),)))
    )
    frames += enc.chunk(
        _chunk(Delta(tool_calls=(ToolCallDelta(index=0, function=FunctionCallDelta(arguments="1}")),)))
    )
    second = ToolCallDelta(index=1, id="call_b", function=FunctionCallDelta(name="g", arguments="{}"))
    frames += enc.chunk(_chunk(Delta(tool_calls=(second,)), finish="tool_calls"))
    frames += enc.chunk(_chunk(Delta(), usage=Usage(prompt_tokens=4, completion_tokens=9, total_tokens=13)))
    assert enc.keepalive().startswith(b"event: ping")
    frames += enc.end()
    events = _events(frames)
    assert [t for t, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start = events[0][1]["message"]
    assert start["id"] == "msg_1"
    assert start["model"] == "mock/echo"
    assert events[1][1]["content_block"] == {"type": "text", "text": ""}
    assert events[5][1] == {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "tool_use", "id": "call_a", "name": "f", "input": {}},
    }
    assert events[7][1]["delta"] == {"type": "input_json_delta", "partial_json": "1}"}
    assert events[9][1]["index"] == 2
    assert events[12][1] == {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"input_tokens": 4, "output_tokens": 9},
    }


def test_stream_empty_and_early_keepalive() -> None:
    enc = AnthropicStreamEncoder(request_id="req_1", model="gg/auto")
    assert enc.keepalive() == b": keep-alive\n\n"
    events = _events(enc.end())
    assert [t for t, _ in events] == ["message_start", "message_delta", "message_stop"]
    assert events[0][1]["message"]["model"] == "gg/auto"


def test_stream_error_event() -> None:
    enc = AnthropicStreamEncoder(request_id="req_1", model="gg/auto")
    enc.chunk(_chunk(Delta(content="partial")))
    [(kind, payload)] = _events(enc.error(GuardrailBlockedError("withheld", code="output_blocked")))
    assert kind == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["details"]["code"] == "output_blocked"
