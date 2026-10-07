from collections.abc import Mapping, Sequence
from typing import Any

import httpx2
import orjson
import pytest

from gg.core.clock import FakeClock
from gg.core.errors import ProviderError
from gg.providers.meta import ResponseMeta
from gg.providers.state.memory import InMemoryStateStore
from gg.providers.state.thinking import NAMESPACE, ThinkingStore
from tests.conftest import make_request
from tests.unit.providers.anthropic.support import haiku, make_anthropic_adapter, opus, sonnet
from tests.unit.providers.support import (
    SECRET,
    collect,
    collect_until_error,
    ctx_for,
    fixture,
    sse_response,
)

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
HEADERS = {
    "request-id": "req_up_1",
    "anthropic-ratelimit-requests-remaining": "41",
    "anthropic-ratelimit-tokens-remaining": "90000",
}


def respond(name: str, headers: Mapping[str, str] = HEADERS) -> Any:
    return lambda r: sse_response(fixture("anthropic", name), headers=dict(headers))


def meta_of(chunk: Any) -> ResponseMeta:
    assert isinstance(chunk.gg_meta, ResponseMeta)
    return chunk.gg_meta


async def test_request_url_headers_and_first_chunk() -> None:
    adapter, rec = make_anthropic_adapter(respond("msg_text.sse"))
    chunks = await collect(adapter, make_request(), haiku())
    sent = rec.requests[0]
    assert str(sent.url) == "https://api.anthropic.test/v1/messages"
    assert sent.headers["x-api-key"] == SECRET
    assert sent.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in sent.headers
    assert SECRET not in str(sent.url)
    assert SECRET.encode() not in sent.content
    assert rec.body()["stream"] is True
    first = chunks[0].choices[0].delta
    assert (first.role, first.content) == ("assistant", "Hello")
    assert chunks[-1].choices == ()
    assert chunks[-1].usage is not None
    head, tail = meta_of(chunks[0]), meta_of(chunks[-1])
    assert head.upstream_request_id == "req_up_1"
    assert head.served_model == "claude-haiku-4-5"
    assert tail.ratelimit is not None
    assert (tail.ratelimit.remaining_requests, tail.ratelimit.remaining_tokens) == (41, 90000)
    assert tail.usage is not None
    assert tail.usage.usage_source == "reported"


async def test_complete_aggregates_and_reports_cache_writes() -> None:
    adapter, _ = make_anthropic_adapter(respond("msg_cache_usage.sse"))
    request = make_request()
    response = await adapter.complete(request, sonnet(), ctx_for(request))
    assert response.choices[0].message.content == "cached"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 105_170
    usage = meta_of(response).usage
    assert usage is not None
    assert (usage.cached_input_tokens, usage.cache_write_tokens, usage.cache_write_1h_tokens) == (
        100_000,
        4096,
        1024,
    )
    assert usage.reasoning_tokens == 310
    assert usage.service_tier == "standard"
    assert (
        usage.cached_input_tokens + usage.cache_write_tokens + usage.cache_write_1h_tokens
        <= usage.input_tokens
    )


async def test_refusal_meta_carries_category() -> None:
    adapter, _ = make_anthropic_adapter(respond("msg_refusal_precontent.sse"))
    chunks = await collect(adapter, make_request(), opus())
    assert chunks[0].choices[0].finish_reason == "content_filter"
    assert meta_of(chunks[0]).refusal_category == "cyber"
    assert meta_of(chunks[-1]).provider_finish_reason == "refusal"


async def test_http_errors_are_classified_before_commit() -> None:
    adapter, _ = make_anthropic_adapter(
        lambda r: httpx2.Response(
            529, content=fixture("anthropic", "err_529.json"), headers={"request-id": "r9"}
        )
    )
    seen, err = await collect_until_error(adapter, make_request(), haiku())
    assert seen == []
    assert (err.kind, err.code, err.committed, err.upstream_request_id) == (
        "fallback",
        "overloaded",
        False,
        "r9",
    )
    assert err.deployment_id == "anthropic/claude-haiku-4-5"


async def test_mid_stream_errors_respect_the_commit_point() -> None:
    early, _ = make_anthropic_adapter(respond("stream_error_overloaded_precontent.sse"))
    seen, err = await collect_until_error(early, make_request(), haiku())
    assert (seen, err.committed, err.code) == ([], False, "overloaded")
    late, _ = make_anthropic_adapter(respond("stream_error_postcontent.sse"))
    seen, err = await collect_until_error(late, make_request(), haiku())
    assert seen
    assert err.committed


async def test_forced_tool_choice_on_sonnet_makes_no_call() -> None:
    adapter, rec = make_anthropic_adapter(respond("msg_text.sse"))
    with pytest.raises(ProviderError) as exc:
        await collect(adapter, make_request(tools=TOOLS, tool_choice="required"), sonnet())
    assert (exc.value.kind, exc.value.code) == ("fallback", "capability_mismatch")
    assert rec.requests == []


async def test_sonnet_effort_none_turns_off_up_front_thinking() -> None:
    adapter, rec = make_anthropic_adapter(respond("msg_text.sse"))
    await collect(adapter, make_request(reasoning_effort="none"), sonnet())
    body = rec.body()
    assert body["thinking"] == {"type": "between_tools"}
    assert body["output_config"] == {"effort": "low"}
    await collect(adapter, make_request(reasoning_effort="none"), opus())
    assert "thinking" not in rec.body()
    assert rec.body()["output_config"] == {"effort": "low"}


async def test_ignored_params_and_adjustments_reach_meta() -> None:
    adapter, _ = make_anthropic_adapter(respond("msg_text.sse"))
    request = make_request(temperature=0.2, seed=3, stop=[" "])
    ctx = ctx_for(request)
    chunks = await collect(adapter, request, sonnet(), ctx)
    meta = meta_of(chunks[0])
    assert {"temperature", "seed"} <= set(meta.ignored_params)
    assert "stop:dropped_whitespace" in meta.adjustments
    assert {"temperature", "seed"} <= ctx.ignored_params


def loop_request(model_reply_ids: Sequence[str]) -> Any:
    return make_request(
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": i,
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                    for i in model_reply_ids
                ],
            },
            *({"role": "tool", "tool_call_id": i, "content": "sunny"} for i in model_reply_ids),
        ],
    )


async def test_thinking_round_trips_to_the_same_model_only() -> None:
    store = InMemoryStateStore(clock=FakeClock())
    adapter, rec = make_anthropic_adapter(respond("msg_thinking_tool.sse"), state=store)
    first = await collect(adapter, make_request(tools=TOOLS), sonnet())
    call_id = next(t.id for ch in first for c in ch.choices for t in c.delta.tool_calls or () if t.id)
    assert call_id == "toolu_think"
    stored = await ThinkingStore(store).lookup("test-key", "claude-sonnet-5-5", [call_id])
    assert len(stored[call_id]) == 2

    final = await collect(adapter, loop_request([call_id]), sonnet())
    content = rec.body()["messages"][1]["content"]
    assert content[:2] == stored[call_id]
    assert content[2]["type"] == "tool_use"
    assert "thinking_reinjected" in meta_of(final[0]).flags

    await collect(adapter, loop_request([call_id]), opus())
    assert [b["type"] for b in rec.body()["messages"][1]["content"]] == ["tool_use"]


async def test_thinking_is_scoped_by_key() -> None:
    store = InMemoryStateStore(clock=FakeClock())
    await ThinkingStore(store).save("other-key", "claude-sonnet-5-5", {"toolu_x": [{"type": "thinking"}]})
    adapter, rec = make_anthropic_adapter(respond("msg_text.sse"), state=store)
    await collect(adapter, loop_request(["toolu_x"]), sonnet())
    assert [b["type"] for b in rec.body()["messages"][1]["content"]] == ["tool_use"]


class BrokenStore:
    async def get_many(self, namespace: str, keys: Sequence[str]) -> dict[str, bytes]:
        raise ConnectionError("redis down")

    async def put_many(self, namespace: str, items: Mapping[str, bytes], ttl_s: int) -> None:
        raise ConnectionError("redis down")


async def test_state_store_failure_fails_open() -> None:
    adapter, rec = make_anthropic_adapter(respond("msg_thinking_tool.sse"), state=BrokenStore())
    chunks = await collect(adapter, loop_request(["toolu_prev"]), sonnet())
    assert chunks[-1].usage is not None
    assert len(rec.requests) == 1


async def test_thinking_store_round_trip_and_ttl() -> None:
    clock = FakeClock()
    store = InMemoryStateStore(clock=clock)
    thinking = ThinkingStore(store, ttl_s=60)
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "redacted_thinking", "data": "d"},
    ]
    await thinking.save("k1", "claude-haiku-4-5", {"toolu_1": blocks, "toolu_empty": []})
    assert await thinking.lookup("k1", "claude-haiku-4-5", ["toolu_1", "toolu_empty", "missing"]) == {
        "toolu_1": blocks
    }
    assert await thinking.lookup("k1", "claude-sonnet-5-5", ["toolu_1"]) == {}
    assert await thinking.lookup("k2", "claude-haiku-4-5", ["toolu_1"]) == {}
    assert await thinking.lookup("k1", "claude-haiku-4-5", []) == {}
    raw = await store.get_many(NAMESPACE, ["k1:claude-haiku-4-5:toolu_1"])
    assert orjson.loads(raw["k1:claude-haiku-4-5:toolu_1"]) == blocks
    clock.advance(61)
    assert await thinking.lookup("k1", "claude-haiku-4-5", ["toolu_1"]) == {}
