import asyncio

import httpx2
import orjson
import pytest

from gg.core.errors import ProviderError
from gg.providers.meta import ResponseMeta
from gg.providers.state.memory import InMemoryStateStore
from tests.conftest import make_request
from tests.unit.providers.support import (
    SECRET,
    Parts,
    collect,
    collect_until_error,
    ctx_for,
    fixture,
    json_response,
    make_adapter,
    make_dep,
    profile_quirks,
    sse,
    sse_response,
    text_chunk,
    text_of,
    usage_chunk,
)

TEXT = sse(
    text_chunk("", role=True),
    text_chunk("Hello"),
    text_chunk(" there"),
    text_chunk(finish="stop"),
    usage_chunk(12, 3),
)
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]


async def test_stream_headers_url_and_first_chunk_is_content() -> None:
    adapter, rec = make_adapter(
        lambda r: sse_response(TEXT, headers={"x-request-id": "up_1", "x-ratelimit-remaining-requests": "9"}),
        quirks=profile_quirks("openai"),
    )
    request = make_request(max_completion_tokens=20)
    ctx = ctx_for(request)
    chunks = await collect(adapter, request, make_dep(), ctx)
    sent = rec.requests[0]
    assert str(sent.url) == "https://upstream.test/v1/chat/completions"
    assert sent.headers["authorization"] == f"Bearer {SECRET}"
    assert sent.headers["x-client-request-id"] == "req_test"
    assert SECRET not in str(sent.url)
    first = chunks[0].choices[0].delta
    assert (first.role, first.content) == ("assistant", "Hello")
    assert text_of(chunks) == "Hello there"
    assert chunks[-2].choices[0].finish_reason == "stop"
    assert chunks[-1].choices == ()
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.prompt_tokens == 12
    assert {c.id for c in chunks} == {"chatcmpl-test"}
    head, tail = chunks[0].gg_meta, chunks[-1].gg_meta
    assert isinstance(head, ResponseMeta)
    assert isinstance(tail, ResponseMeta)
    assert head.upstream_request_id == "up_1"
    assert head.usage is None
    assert tail.usage is not None
    assert tail.usage.usage_source == "reported"
    assert tail.usage.input_tokens == 12
    assert tail.ratelimit is not None
    assert tail.ratelimit.remaining_requests == 9


async def test_split_reads_produce_the_same_chunks() -> None:
    whole, _ = make_adapter(lambda r: sse_response(TEXT))
    split, _ = make_adapter(lambda r: sse_response(TEXT, split=7))
    request = make_request()
    a = await collect(whole, request, make_dep())
    b = await collect(split, request, make_dep())
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]


async def test_complete_aggregates_stream() -> None:
    adapter, _ = make_adapter(lambda r: sse_response(TEXT))
    request = make_request()
    response = await adapter.complete(request, make_dep(), ctx_for(request))
    assert response.choices[0].message.content == "Hello there"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.total_tokens == 15
    assert isinstance(response.gg_meta, ResponseMeta)


@pytest.mark.parametrize(
    ("status", "body", "headers", "kind"),
    [
        (
            429,
            {"error": {"message": "slow down", "type": "rate_limit_error"}},
            {"retry-after": "2"},
            "quota_minute",
        ),
        (500, {"error": {"message": "boom"}}, {}, "retryable"),
        (401, {"error": {"message": "bad key"}}, {}, "auth"),
        (404, {"error": {"message": "no model"}}, {}, "fallback"),
    ],
)
async def test_http_errors_are_classified_before_commit(
    status: int, body: object, headers: dict[str, str], kind: str
) -> None:
    adapter, _ = make_adapter(lambda r: json_response(status, body, headers))
    with pytest.raises(ProviderError) as exc:
        await collect(adapter, make_request(), make_dep())
    assert exc.value.kind == kind
    assert not exc.value.committed
    assert exc.value.deployment_id == "test/test-model"


async def test_pre_content_stream_error_yields_nothing() -> None:
    body = sse(
        text_chunk("", role=True), {"error": {"message": "overloaded", "type": "server_error"}}, done=False
    )
    adapter, _ = make_adapter(lambda r: sse_response(body))
    seen, err = await collect_until_error(adapter, make_request(), make_dep())
    assert seen == []
    assert not err.committed


async def test_post_content_error_is_committed() -> None:
    adapter, _ = make_adapter(lambda r: sse_response(fixture("openai", "stream_error_midstream.sse")))
    seen, err = await collect_until_error(adapter, make_request(), make_dep())
    assert len(seen) == 1
    assert err.committed


async def test_truncated_stream_raises_truncated() -> None:
    adapter, _ = make_adapter(lambda r: sse_response(fixture("openai", "stream_cut_no_usage.sse")))
    with pytest.raises(ProviderError) as exc:
        await collect(adapter, make_request(), make_dep())
    assert (exc.value.code, exc.value.committed) == ("truncated", True)


async def test_missing_usage_is_estimated() -> None:
    body = sse(text_chunk("hi there", role=True), text_chunk(finish="stop"))
    adapter, _ = make_adapter(lambda r: sse_response(body))
    chunks = await collect(adapter, make_request(), make_dep())
    meta = chunks[-1].gg_meta
    assert isinstance(meta, ResponseMeta)
    assert meta.usage is not None
    assert meta.usage.usage_source == "estimated"
    assert meta.usage.output_tokens > 0


async def test_transport_error_is_retryable_status_zero() -> None:
    def fail(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    adapter, _ = make_adapter(fail)
    with pytest.raises(ProviderError) as exc:
        await collect(adapter, make_request(), make_dep())
    assert (exc.value.kind, exc.value.status, exc.value.code) == ("retryable", 0, "connect_error")
    local, _ = make_adapter(fail, quirks=profile_quirks("ollama"))
    with pytest.raises(ProviderError) as local_exc:
        await collect(local, make_request(), make_dep())
    assert local_exc.value.kind == "fallback"


async def test_capability_reject_makes_no_http_call() -> None:
    adapter, rec = make_adapter(lambda r: sse_response(TEXT))
    with pytest.raises(ProviderError) as exc:
        await collect(adapter, make_request(tools=TOOLS), make_dep(tools=False))
    assert exc.value.code == "capability_mismatch"
    assert exc.value.violations == ("tools",)
    assert rec.requests == []


async def test_ignored_params_reported_on_ctx_and_meta() -> None:
    adapter, _ = make_adapter(lambda r: sse_response(TEXT))
    request = make_request(temperature=0.3, verbosity="low")
    ctx = ctx_for(request)
    chunks = await collect(adapter, request, make_dep(sampling_params="none"), ctx)
    assert {"temperature", "verbosity"} <= ctx.ignored_params
    meta = chunks[0].gg_meta
    assert isinstance(meta, ResponseMeta)
    assert "temperature" in meta.ignored_params


async def test_max_in_flight_fails_fast() -> None:
    gate = asyncio.Event()

    async def slow_body():
        await gate.wait()
        yield TEXT

    class Slow(Parts):
        async def __aiter__(self):
            async for part in slow_body():
                yield part

    adapter, _ = make_adapter(lambda r: httpx2.Response(200, stream=Slow([])), max_in_flight=1)
    first = adapter.stream(make_request(), make_dep(), ctx_for())
    pending = asyncio.ensure_future(anext(first))
    await asyncio.sleep(0.01)
    with pytest.raises(ProviderError) as exc:
        await collect(adapter, make_request(), make_dep())
    assert exc.value.code == "local_busy"
    gate.set()
    await pending
    await first.aclose()
    await collect(adapter, make_request(), make_dep())


async def test_aclose_after_first_chunk_closes_upstream_body() -> None:
    body = Parts([TEXT])
    adapter, _ = make_adapter(lambda r: httpx2.Response(200, stream=body))
    stream = adapter.stream(make_request(), make_dep(), ctx_for())
    await anext(stream)
    await stream.aclose()
    assert body.closed


async def test_reasoning_round_trip_through_store() -> None:
    first = sse(
        text_chunk(None, role=True, reasoning_content=None),
        {
            **text_chunk(None),
            "choices": [{"index": 0, "delta": {"reasoning_content": "use f"}, "finish_reason": None}],
        },
        {
            **text_chunk(None),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    )
    second = sse(text_chunk("sunny", role=True), text_chunk(finish="stop"))
    bodies = iter([first, second, second])
    state = InMemoryStateStore()
    adapter, rec = make_adapter(
        lambda r: sse_response(next(bodies)), quirks=profile_quirks("deepseek"), state=state
    )
    dep = make_dep("deepseek", "deepseek-flash", effort_levels=frozenset({"none", "low", "high", "max"}))
    request = make_request(tools=TOOLS, reasoning_effort="high")
    await collect(adapter, request, dep)
    follow_up = make_request(
        tools=TOOLS,
        reasoning_effort="high",
        messages=[
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
    )
    await collect(adapter, follow_up, dep)
    sent = orjson.loads(rec.requests[1].content)
    assert sent["messages"][1]["reasoning_content"] == "use f"
    assert sent["reasoning_effort"] == "high"
    assert "thinking" not in sent
    # another tenant cannot read this key's reasoning, so deepseek gets thinking disabled instead of a 400
    other = ctx_for(follow_up)
    other.key = other.key.model_copy(update={"id": "other-key"})
    await collect(adapter, follow_up, dep, other)
    isolated = orjson.loads(rec.requests[2].content)
    assert "reasoning_content" not in isolated["messages"][1]
    assert isolated["thinking"] == {"type": "disabled"}


async def test_observer_gets_estimated_usage_after_cancel() -> None:
    finished: list[ResponseMeta] = []

    class Rec:
        def first_byte(self) -> None: ...
        def commit(self) -> None: ...
        def event(self, name: str, **attrs: object) -> None: ...
        def fail(self, err: ProviderError) -> None: ...
        def finish(self, meta: ResponseMeta) -> None:
            finished.append(meta)

    class Observer:
        def attempt(self, ctx: object, dep: object, *, stream: bool) -> Rec:
            return Rec()

    adapter, _ = make_adapter(lambda r: sse_response(TEXT))
    adapter.deps.observer = Observer()
    stream = adapter.stream(make_request(), make_dep(), ctx_for())
    await anext(stream)
    await stream.aclose()
    assert len(finished) == 1
    usage = finished[0].usage
    assert usage is not None
    assert usage.usage_source == "estimated"
    assert usage.output_tokens > 0
