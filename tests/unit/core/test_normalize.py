from typing import Any

import pytest

from gg.core.errors import InvalidRequestError
from gg.core.normalize import normalize_request_dict
from gg.core.schema import ChatRequest

BASE: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def test_max_tokens_becomes_max_completion_tokens() -> None:
    out = normalize_request_dict({**BASE, "max_tokens": 10})
    assert out.payload["max_completion_tokens"] == 10
    assert "max_tokens" not in out.payload
    assert [n.rule for n in out.applied] == ["max_tokens"]


def test_conflicting_max_tokens_is_rejected() -> None:
    with pytest.raises(InvalidRequestError) as exc:
        normalize_request_dict({**BASE, "max_tokens": 10, "max_completion_tokens": 20})
    assert exc.value.param == "max_tokens"


def test_legacy_function_calling_becomes_tools() -> None:
    raw = {
        "model": "m",
        "functions": [{"name": "f", "parameters": {"type": "object"}}],
        "function_call": {"name": "f"},
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "function_call": {"name": "f", "arguments": "{}"}},
            {"role": "function", "name": "f", "content": "42"},
        ],
    }
    out = normalize_request_dict(raw)
    req = ChatRequest.model_validate(out.payload)
    assert req.tools is not None
    assert req.tool_choice is not None
    assistant, tool = req.messages[1], req.messages[2]
    assert assistant.tool_calls is not None
    assert tool.role == "tool"
    assert tool.tool_call_id == assistant.tool_calls[0].id == "call_fn_1"


def test_function_message_without_call_is_rejected() -> None:
    raw = {**BASE, "messages": [{"role": "function", "name": "f", "content": "x"}]}
    with pytest.raises(InvalidRequestError):
        normalize_request_dict(raw)


def test_stop_string_becomes_list_and_empty_list_is_dropped() -> None:
    assert normalize_request_dict({**BASE, "stop": "x"}).payload["stop"] == ["x"]
    assert "stop" not in normalize_request_dict({**BASE, "stop": []}).payload


def test_stream_options_dropped_without_stream() -> None:
    out = normalize_request_dict({**BASE, "stream_options": {"include_usage": True}})
    assert "stream_options" not in out.payload


@pytest.mark.parametrize(
    "raw",
    [
        {**BASE, "max_tokens": 5, "stop": "x"},
        {**BASE, "functions": [{"name": "f"}]},
        BASE,
    ],
)
def test_normalisation_is_idempotent(raw: dict[str, Any]) -> None:
    once = normalize_request_dict(raw).payload
    twice = normalize_request_dict(once)
    assert twice.payload == once
    assert twice.applied == ()
