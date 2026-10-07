"""adapter conformance (c3 plan §8.3): the same behavioural contract for every provider profile, no network.

each profile gets a synthetic upstream shaped by its own quirks (usage placement, reasoning field), served
through httpx2.MockTransport. native adapters bring their own upstream, which replays the same openai-shaped
scenarios in their wire format.
"""

from collections.abc import Callable
from typing import Any

import httpx2
import orjson
import pytest

from gg.core.errors import ProviderError
from gg.core.schema import ChatChunk, ChatRequest
from gg.core.usage import UsageRecord
from gg.pipeline.streams import StreamAssembler
from gg.providers.http import HttpClientFactory
from gg.providers.meta import ResponseMeta
from gg.providers.mock.adapter import MockAdapter
from gg.providers.openai_compat.quirks import QuirkProfile
from gg.providers.runtime import AdapterDeps, ProviderRuntime
from tests.conftest import make_request
from tests.unit.providers.anthropic.support import AnthropicUpstream, make_anthropic_adapter
from tests.unit.providers.gemini.support import GeminiUpstream, make_gemini_adapter
from tests.unit.providers.support import (
    SECRET,
    Parts,
    collect,
    collect_until_error,
    compat_providers,
    ctx_for,
    json_response,
    make_adapter,
    make_dep,
    profile_quirks,
    sse_response,
)

# provider -> (upstream factory, adapter factory) for adapters with their own wire format
NATIVE: dict[str, tuple[Callable[[], Any], Callable[..., Any]]] = {
    "gemini": (GeminiUpstream, make_gemini_adapter),
    "anthropic": (AnthropicUpstream, make_anthropic_adapter),
}
PROFILES = [*compat_providers(), *NATIVE]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
USAGE = {
    "prompt_tokens": 20,
    "completion_tokens": 6,
    "total_tokens": 26,
    "prompt_tokens_details": {"cached_tokens": 5},
    "completion_tokens_details": {"reasoning_tokens": 2},
}


class Upstream:
    """synthetic sse in the shape a given quirk profile expects"""

    request_id = "req-up"
    auth_header = "authorization"

    def __init__(self, provider: str) -> None:
        self.q = QuirkProfile.model_validate(profile_quirks(provider))

    @property
    def json_schema_supported(self) -> bool:
        return self.q.request.response_format.json_schema != "unsupported"

    def _chunk(
        self, delta: dict[str, Any] | None = None, finish: str | None = None, **extra: Any
    ) -> dict[str, Any]:
        choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]
        return {
            "id": "up-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "conf",
            "choices": choices,
            "noise": {"unknown": True},
            **extra,
        }

    def reasoning(self, text: str) -> dict[str, Any]:
        field = self.q.reasoning.field
        if field == "content_array":
            return {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": text}]}]}
        if field == "think_tags":
            return {"content": f"<think>{text}</think>"}
        return {field if field in ("reasoning", "reasoning_content") else "reasoning_content": text}

    def body(
        self, deltas: list[dict[str, Any]], finish: str, *, done: bool = True, usage: bool = True
    ) -> bytes:
        events = [self._chunk({"role": "assistant", "content": ""})]
        events.append(self._chunk(self.reasoning("thinking")))
        events.extend(self._chunk(d) for d in deltas)
        if usage and self.q.request.stream_usage == "inject":
            events.append(self._chunk({}, finish))
            events.append(self._chunk(None, usage=USAGE))
        else:
            events.append(self._chunk({}, finish, **({"usage": USAGE} if usage else {})))
        raw = b"".join(b"data: " + orjson.dumps(e) + b"\n\n" for e in events)
        return raw + (b"data: [DONE]\n\n" if done else b"")

    def text(self, **kw: Any) -> bytes:
        return self.body([{"content": "Hello"}, {"content": " world"}], "stop", **kw)

    def assert_tool_loop(self, body: dict[str, Any]) -> None:
        messages = body["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
        assert messages[2]["tool_call_id"] == messages[1]["tool_calls"][0]["id"]

    def assert_json_schema(self, body: dict[str, Any]) -> None:
        assert body["response_format"]["type"] in ("json_schema", "json_object")


def upstream_for(provider: str) -> Any:
    native = NATIVE.get(provider)
    return native[0]() if native else Upstream(provider)


def dep(provider: str):
    return make_dep(provider, "conf-model", vision=True)


def adapter_for(provider: str, handler: Any) -> Any:
    native = NATIVE.get(provider)
    if native:
        return native[1](handler, provider=provider)
    return make_adapter(handler, provider=provider, quirks=profile_quirks(provider))


def meta_of(chunk: ChatChunk) -> ResponseMeta:
    meta = chunk.gg_meta
    assert isinstance(meta, ResponseMeta)
    return meta


def assert_usage_invariants(record: UsageRecord) -> None:
    assert record.cached_input_tokens + record.cache_write_tokens <= record.input_tokens
    assert record.reasoning_tokens <= record.output_tokens
    assert min(record.input_tokens, record.output_tokens, record.cached_input_tokens) >= 0


@pytest.fixture(params=PROFILES)
def provider(request: pytest.FixtureRequest) -> str:
    return request.param


async def test_c1_non_stream(provider: str) -> None:
    up = upstream_for(provider)
    adapter, _ = adapter_for(
        provider, lambda r: sse_response(up.text(), headers={"x-request-id": up.request_id})
    )
    request = make_request()
    response = await adapter.complete(request, dep(provider), ctx_for(request))
    assert response.choices[0].message.content == "Hello world"
    assert response.choices[0].finish_reason == "stop"
    meta = response.gg_meta
    assert isinstance(meta, ResponseMeta)
    assert meta.upstream_request_id == up.request_id
    assert meta.usage is not None
    assert meta.usage.usage_source == "reported"
    assert_usage_invariants(meta.usage)


async def test_c2_stream_shape(provider: str) -> None:
    up = upstream_for(provider)
    adapter, _ = adapter_for(provider, lambda r: sse_response(up.text(), split=5))
    chunks = await collect(adapter, make_request(), dep(provider))
    first = chunks[0].choices[0].delta
    assert first.role == "assistant"
    assert first.content
    assert chunks[-2].choices[0].finish_reason == "stop"
    assert chunks[-1].choices == ()
    assert chunks[-1].usage is not None
    assert len({(c.id, c.created, c.model) for c in chunks}) == 1
    assert all("reasoning_content" not in (c.delta.model_extra or {}) for ch in chunks for c in ch.choices)


async def test_c3_stream_equals_complete(provider: str) -> None:
    up = upstream_for(provider)
    adapter, _ = adapter_for(provider, lambda r: sse_response(up.text()))
    request = make_request()
    assembler = StreamAssembler()
    for chunk in await collect(adapter, request, dep(provider)):
        assembler.feed(chunk)
    complete = await adapter.complete(request, dep(provider), ctx_for(request))
    assert assembler.result().model_dump() == complete.model_dump()


async def test_c4_c5_tool_calls(provider: str) -> None:
    up = upstream_for(provider)
    deltas = [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":'},
                }
            ]
        },
        {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]},
        {
            "tool_calls": [
                {
                    "index": 1,
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ]
        },
    ]
    adapter, _ = adapter_for(provider, lambda r: sse_response(up.body(deltas, "tool_calls")))
    request = make_request(tools=TOOLS)
    response = await adapter.complete(request, dep(provider), ctx_for(request))
    calls = response.choices[0].message.tool_calls or ()
    assert [c.id for c in calls] == ["call_a", "call_b"]
    assert orjson.loads(calls[0].function.arguments) == {"city": "Paris"}
    assert response.choices[0].finish_reason == "tool_calls"


async def test_c6_tool_loop_request(provider: str) -> None:
    up = upstream_for(provider)
    adapter, rec = adapter_for(provider, lambda r: sse_response(up.text()))
    request = make_request(
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "sunny"},
        ],
    )
    await collect(adapter, request, dep(provider))
    up.assert_tool_loop(rec.body())


async def test_c7_json_schema(provider: str) -> None:
    up = upstream_for(provider)
    adapter, rec = adapter_for(provider, lambda r: sse_response(up.text()))
    fmt = {"type": "json_schema", "json_schema": {"name": "s", "schema": {"type": "object"}}}
    request = make_request(response_format=fmt)
    if not up.json_schema_supported:
        with pytest.raises(ProviderError, match="capability"):
            await collect(adapter, request, dep(provider))
        assert rec.requests == []
        return
    await collect(adapter, request, dep(provider))
    up.assert_json_schema(rec.body())


@pytest.mark.parametrize(
    ("upstream", "canonical"), [("length", "length"), ("content_filter", "content_filter")]
)
async def test_c8_c9_finish_reasons(provider: str, upstream: str, canonical: str) -> None:
    up = upstream_for(provider)
    adapter, _ = adapter_for(provider, lambda r: sse_response(up.body([{"content": "x"}], upstream)))
    chunks = await collect(adapter, make_request(), dep(provider))
    assert [c.finish_reason for ch in chunks for c in ch.choices if c.finish_reason] == [canonical]


@pytest.mark.parametrize(
    ("status", "headers", "kind"),
    [(429, {"retry-after": "2"}, "quota_minute"), (500, {}, "retryable"), (401, {}, "auth")],
)
async def test_c10_c11_c12_http_errors(
    provider: str, status: int, headers: dict[str, str], kind: str
) -> None:
    adapter, _ = adapter_for(provider, lambda r: json_response(status, {"error": {"message": "x"}}, headers))
    with pytest.raises(ProviderError) as exc:
        await collect(adapter, make_request(), dep(provider))
    assert exc.value.kind == kind
    assert not exc.value.committed
    if kind == "quota_minute":
        assert exc.value.retry_after_s == 2.0
    if kind == "auth":
        assert exc.value.scope == "provider"
    assert SECRET not in exc.value.message


async def stream_until_error(adapter: Any, provider: str) -> tuple[list[ChatChunk], ProviderError]:
    return await collect_until_error(adapter, make_request(), dep(provider))


async def test_c13_pre_content_error_yields_nothing(provider: str) -> None:
    raw = b"data: " + orjson.dumps({"error": {"message": "overloaded", "type": "server_error"}}) + b"\n\n"
    adapter, _ = adapter_for(provider, lambda r: sse_response(raw))
    seen, err = await stream_until_error(adapter, provider)
    assert seen == []
    assert not err.committed


async def test_c14_post_content_error_is_committed(provider: str) -> None:
    up = upstream_for(provider)
    raw = up.text(done=False, usage=False).rsplit(b"data: ", 1)[0]
    raw += b"data: " + orjson.dumps({"error": {"message": "boom", "type": "server_error"}}) + b"\n\n"
    adapter, _ = adapter_for(provider, lambda r: sse_response(raw))
    seen, err = await stream_until_error(adapter, provider)
    assert seen
    assert err.committed


async def test_c15_truncated_stream(provider: str) -> None:
    up = upstream_for(provider)
    raw = up.text(done=False, usage=False).rsplit(b"data: ", 1)[0]
    adapter, _ = adapter_for(provider, lambda r: sse_response(raw))
    seen, err = await stream_until_error(adapter, provider)
    assert seen
    assert err.committed
    assert err.code == "truncated"
    empty, _ = adapter_for(provider, lambda r: sse_response(b""))
    nothing, early = await stream_until_error(empty, provider)
    assert nothing == []
    assert not early.committed
    assert early.kind == "retryable"


async def test_c16_unknown_fields_do_not_change_output(provider: str) -> None:
    up = upstream_for(provider)
    clean = up.text()
    noisy = clean.replace(b'"noise":{"unknown":true}', b'"noise":{"unknown":true},"x_extra":[1,{"a":null}]')
    a, _ = adapter_for(provider, lambda r: sse_response(clean))
    b, _ = adapter_for(provider, lambda r: sse_response(noisy + b": comment\n\nevent: ping\ndata: {}\n\n"))
    out_a = [c.model_dump() for c in await collect(a, make_request(), dep(provider))]
    out_b = [c.model_dump() for c in await collect(b, make_request(), dep(provider))]
    assert out_a == out_b


async def test_c17_cancellation_closes_upstream(provider: str) -> None:
    body = Parts([upstream_for(provider).text()])
    adapter, _ = adapter_for(provider, lambda r: httpx2.Response(200, stream=body))
    stream = adapter.stream(make_request(), dep(provider), ctx_for())
    await anext(stream)
    await stream.aclose()
    assert body.closed


async def test_c18_upstream_body_is_deterministic(provider: str) -> None:
    up = upstream_for(provider)
    adapter, rec = adapter_for(provider, lambda r: sse_response(up.text()))
    request = make_request(tools=TOOLS, temperature=0.4, user="u", max_completion_tokens=99, seed=1)
    await collect(adapter, request, dep(provider))
    await collect(adapter, request, dep(provider))
    assert rec.requests[0].content == rec.requests[1].content
    assert b"gg" not in rec.requests[0].content.split(b'"messages"')[0]


async def test_c19_capability_reject_makes_no_call(provider: str) -> None:
    adapter, rec = adapter_for(provider, lambda r: sse_response(b""))
    with pytest.raises(ProviderError) as exc:
        await collect(adapter, make_request(tools=TOOLS), make_dep(provider, "no-tools", tools=False))
    assert exc.value.kind == "fallback"
    assert exc.value.code == "capability_mismatch"
    assert rec.requests == []


async def test_c20_key_only_in_auth_header(provider: str) -> None:
    up = upstream_for(provider)
    adapter, rec = adapter_for(provider, lambda r: sse_response(up.text()))
    await collect(adapter, make_request(), dep(provider))
    sent = rec.requests[0]
    assert SECRET not in str(sent.url)
    assert SECRET.encode() not in sent.content
    assert [k for k, v in sent.headers.items() if SECRET in v] == [up.auth_header]


def mock() -> MockAdapter:
    runtime = ProviderRuntime(name="mock", type="mock", base_url="mock://local")
    return MockAdapter(runtime, AdapterDeps(http=HttpClientFactory()))


async def test_mock_adapter_core_contract() -> None:
    mock_dep = make_dep("mock", "echo", defaults={"mock": {"text": "Hello world"}})
    request: ChatRequest = make_request()
    chunks = await collect(mock(), request, mock_dep)
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[0].choices[0].delta.content
    assert chunks[-1].choices == ()
    assert chunks[-1].usage is not None
    assembler = StreamAssembler()
    for chunk in chunks:
        assembler.feed(chunk)
    complete = await mock().complete(request, mock_dep, ctx_for(request))
    assert assembler.result().model_dump() == complete.model_dump()
    usage = meta_of(chunks[-1]).usage
    assert usage is not None
    assert_usage_invariants(usage)
    with pytest.raises(ProviderError) as exc:
        await collect(mock(), make_request(tools=TOOLS), make_dep("mock", "echo", tools=False))
    assert exc.value.code == "capability_mismatch"
