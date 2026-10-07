import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import ValidationError

from gg.core.schema import (
    AssistantMessage,
    ChatChunk,
    ChatRequest,
    ChatResponse,
    Choice,
    ChunkChoice,
    Delta,
    FunctionTool,
    ImagePart,
    TextPart,
    UnknownPart,
    UnknownTool,
    Usage,
    to_wire,
)
from tests.conftest import make_request


def test_minimal_request_round_trips() -> None:
    raw = {"model": "gg/auto", "messages": [{"role": "user", "content": "hi"}]}
    assert ChatRequest.model_validate(raw).upstream_payload() == raw


def test_unknown_top_level_fields_survive_passthrough() -> None:
    req = make_request(verbosity="low", prompt_cache_key="abc")
    payload = req.upstream_payload()
    assert payload["verbosity"] == "low"
    assert payload["prompt_cache_key"] == "abc"


def test_gg_extension_is_parsed_and_never_forwarded() -> None:
    req = make_request(gg={"route_threshold": 0.4, "cache": "off"})
    assert req.gg is not None
    assert req.gg.route_threshold == 0.4
    assert "gg" not in req.upstream_payload()


def test_gg_extension_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        make_request(gg={"route_treshold": 0.4})


def test_content_parts_are_discriminated_and_unknown_parts_pass_through() -> None:
    req = make_request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                    {"type": "video_url", "video_url": {"url": "https://x"}},
                ],
            }
        ]
    )
    parts = req.messages[0].content
    assert isinstance(parts, tuple)
    assert isinstance(parts[0], TextPart)
    assert isinstance(parts[1], ImagePart)
    assert isinstance(parts[2], UnknownPart)
    assert req.upstream_payload()["messages"][0]["content"][2]["video_url"] == {"url": "https://x"}


def test_tools_are_discriminated() -> None:
    req = make_request(
        tools=[
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
            {"type": "custom", "custom": {"name": "grammar"}},
        ]
    )
    assert req.tools is not None
    assert isinstance(req.tools[0], FunctionTool)
    assert isinstance(req.tools[1], UnknownTool)
    assert req.has_tools()


def test_json_schema_alias_round_trips() -> None:
    req = make_request(
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    )
    assert req.upstream_payload()["response_format"]["json_schema"]["schema"] == {"type": "object"}


@pytest.mark.parametrize(
    "message",
    [
        {"role": "tool", "content": "x"},
        {"role": "user"},
        {"role": "assistant"},
    ],
)
def test_role_rules(message: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        make_request(messages=[message])


def test_validation_errors_never_contain_prompt_text() -> None:
    secret = "my-very-secret-prompt-text"
    with pytest.raises(ValidationError) as exc:
        ChatRequest.model_validate({"model": "", "messages": [{"role": "user", "content": secret}]})
    assert secret not in str(exc.value)


def test_last_user_text_reads_latest_user_turn() -> None:
    req = make_request(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ]
    )
    assert req.last_user_text() == "second"


def test_metadata_limits() -> None:
    with pytest.raises(ValidationError):
        make_request(metadata={f"k{i}": "v" for i in range(17)})


def test_response_validates_with_openai_sdk_types() -> None:
    resp = ChatResponse(
        id="chatcmpl-1",
        created=1,
        model="m",
        choices=(Choice(index=0, message=AssistantMessage(content="hi"), finish_reason="stop"),),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    ChatCompletion.model_validate(to_wire(resp))


def test_chunk_validates_with_openai_sdk_types_and_detects_content() -> None:
    chunk = ChatChunk(
        id="chatcmpl-1",
        created=1,
        model="m",
        choices=(ChunkChoice(index=0, delta=Delta(content="hi")),),
    )
    ChatCompletionChunk.model_validate(to_wire(chunk))
    assert chunk.has_content()
    role_only = ChatChunk(
        id="c", created=1, model="m", choices=(ChunkChoice(index=0, delta=Delta(role="assistant")),)
    )
    assert not role_only.has_content()
