from typing import Any

import orjson
import pytest

from gg.config.settings import Settings
from gg.core.clock import FakeClock
from gg.core.errors import ProviderError
from gg.providers.catalog.loader import build_catalog
from gg.providers.gemini.adapter import GeminiAdapter
from gg.providers.gemini.errors import next_pacific_midnight
from gg.providers.http import HttpClientFactory
from gg.providers.meta import ResponseMeta
from gg.providers.registry import build_adapters
from gg.providers.runtime import AdapterDeps
from gg.providers.state.memory import InMemoryStateStore
from gg.providers.state.signatures import DUMMY_SIGNATURE
from tests.conftest import make_key, make_request
from tests.unit.providers.gemini.support import gemini_dep, make_gemini_adapter
from tests.unit.providers.support import (
    SECRET,
    collect,
    collect_until_error,
    ctx_for,
    fixture,
    json_response,
    models_config,
    sse_response,
    text_of,
)

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]


def meta(chunk: Any) -> ResponseMeta:
    assert isinstance(chunk.gg_meta, ResponseMeta)
    return chunk.gg_meta


async def test_url_auth_header_and_stream_shape() -> None:
    adapter, rec = make_gemini_adapter(lambda r: sse_response(fixture("gemini", "gen_text.sse"), split=9))
    request = make_request(max_completion_tokens=2048)
    chunks = await collect(adapter, request, gemini_dep())
    sent = rec.requests[0]
    assert str(sent.url) == (
        "https://generativelanguage.test/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse"
    )
    assert sent.headers["x-goog-api-key"] == SECRET
    assert "authorization" not in sent.headers
    assert SECRET not in str(sent.url)
    assert SECRET.encode() not in sent.content
    assert rec.body()["generationConfig"] == {"maxOutputTokens": 2048}
    first = chunks[0].choices[0].delta
    assert (first.role, first.content) == ("assistant", "The capital")
    assert text_of(chunks) == "The capital of France is Paris."
    assert chunks[-2].choices[0].finish_reason == "stop"
    assert chunks[-1].choices == ()
    usage = chunks[-1].usage
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens) == (4100, 47)
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 4096
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 40
    tail = meta(chunks[-1])
    assert (tail.upstream_request_id, tail.served_model, tail.provider_finish_reason) == (
        "resp-1",
        "gemini-3.8-flash",
        "STOP",
    )
    assert tail.usage is not None
    assert tail.usage.usage_source == "reported"


async def test_two_step_tool_loop_round_trips_the_signature() -> None:
    state = InMemoryStateStore(clock=FakeClock())
    adapter, rec = make_gemini_adapter(
        lambda r: sse_response(fixture("gemini", "gen_parallel_fc.sse")), state=state
    )
    first = make_request(tools=TOOLS, messages=[{"role": "user", "content": "weather in paris and rome?"}])
    response = await adapter.complete(first, gemini_dep(), ctx_for(first))
    calls = response.choices[0].message.tool_calls or ()
    assert response.choices[0].finish_reason == "tool_calls"
    assert [c.id for c in calls] == ["fc_paris", "fc_rome"]

    # an openai client replays the calls without any signature field
    history = [
        {"role": "user", "content": "weather in paris and rome?"},
        {"role": "assistant", "content": None, "tool_calls": [c.model_dump() for c in calls]},
        {"role": "tool", "tool_call_id": "fc_paris", "content": "18C"},
        {"role": "tool", "tool_call_id": "fc_rome", "content": "24C"},
    ]
    second = make_request(tools=TOOLS, messages=history)
    chunks = await collect(adapter, second, gemini_dep(), ctx_for(second))
    model_turn = rec.body()["contents"][1]["parts"]
    assert model_turn[0]["thoughtSignature"] == "c2lnLXBhcmlz"
    assert "thoughtSignature" not in model_turn[1]
    assert "dummy_signature" not in meta(chunks[-1]).flags


async def test_foreign_history_gets_the_dummy_signature_and_a_flag() -> None:
    adapter, rec = make_gemini_adapter(lambda r: sse_response(fixture("gemini", "gen_text.sse")))
    request = make_request(
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_01",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_01", "content": "sunny"},
        ],
    )
    chunks = await collect(adapter, request, gemini_dep())
    assert rec.body()["contents"][1]["parts"][0]["thoughtSignature"] == DUMMY_SIGNATURE
    assert "dummy_signature" in meta(chunks[0]).flags
    assert "dummy_signature" in meta(chunks[-1]).flags


async def test_signatures_are_scoped_to_the_virtual_key() -> None:
    state = InMemoryStateStore(clock=FakeClock())
    adapter, rec = make_gemini_adapter(
        lambda r: sse_response(fixture("gemini", "gen_parallel_fc.sse")), state=state
    )
    first = make_request(tools=TOOLS)
    await collect(adapter, first, gemini_dep(), ctx_for(first))
    replay = make_request(
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "fc_paris",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "fc_paris", "content": "1"},
        ],
    )
    other = ctx_for(replay)
    other.key = make_key(id="other-key")
    await collect(adapter, replay, gemini_dep(), other)
    assert rec.body()["contents"][1]["parts"][0]["thoughtSignature"] == DUMMY_SIGNATURE


async def test_blocked_prompt_raises_before_anything_is_yielded() -> None:
    adapter, _ = make_gemini_adapter(lambda r: sse_response(fixture("gemini", "gen_prompt_blocked.sse")))
    seen, err = await collect_until_error(adapter, make_request(), gemini_dep())
    assert seen == []
    assert (err.kind, err.committed, err.deployment_id) == (
        "content_filter",
        False,
        "gemini/gemini-3.8-flash",
    )


async def test_safety_without_content_completes_with_content_filter() -> None:
    adapter, _ = make_gemini_adapter(lambda r: sse_response(fixture("gemini", "gen_safety_no_content.sse")))
    request = make_request()
    response = await adapter.complete(request, gemini_dep(), ctx_for(request))
    choice = response.choices[0]
    assert (choice.message.content, choice.finish_reason) == (None, "content_filter")
    assert meta(response).provider_finish_reason == "SAFETY"


async def test_mid_stream_error_and_cut_are_committed() -> None:
    adapter, _ = make_gemini_adapter(lambda r: sse_response(fixture("gemini", "stream_error_object.sse")))
    seen, err = await collect_until_error(adapter, make_request(), gemini_dep())
    assert text_of(seen) == "partial"
    assert (err.kind, err.code, err.committed) == ("retryable", "overloaded", True)
    adapter, _ = make_gemini_adapter(lambda r: sse_response(fixture("gemini", "stream_cut_no_finish.sse")))
    seen, err = await collect_until_error(adapter, make_request(), gemini_dep())
    assert seen
    assert (err.code, err.committed) == ("truncated", True)


async def test_per_day_quota_over_http() -> None:
    body = orjson.loads(fixture("gemini", "err_429_per_day.json"))
    adapter, _ = make_gemini_adapter(lambda r: json_response(429, body))
    with pytest.raises(ProviderError) as exc:
        await collect(adapter, make_request(), gemini_dep())
    assert exc.value.kind == "quota_day"
    assert exc.value.quota_reset_at == next_pacific_midnight(FakeClock().now())
    assert not exc.value.committed


async def test_capability_reject_makes_no_call() -> None:
    adapter, rec = make_gemini_adapter(lambda r: sse_response(b""))
    with pytest.raises(ProviderError) as exc:
        await collect(adapter, make_request(n=2), gemini_dep())
    assert (exc.value.kind, exc.value.code) == ("fallback", "capability_mismatch")
    assert rec.requests == []


async def test_client_history_error_makes_no_call() -> None:
    adapter, rec = make_gemini_adapter(lambda r: sse_response(b""))
    request = make_request(
        messages=[{"role": "user", "content": "x"}, {"role": "tool", "tool_call_id": "a", "content": "1"}]
    )
    with pytest.raises(ProviderError) as exc:
        await collect(adapter, request, gemini_dep())
    assert (exc.value.kind, exc.value.code) == ("client", "orphan_tool_result")
    assert rec.requests == []


async def test_repo_catalog_enables_gemini_with_a_key() -> None:
    settings = Settings(_env_file=None, providers={"gemini": {"api_key": "k"}})  # pyright: ignore[reportCallIssue]
    catalog = build_catalog(models_config(), settings, profile="dev-free")
    assert catalog.providers["gemini"].base_url == "https://generativelanguage.googleapis.com/v1beta"
    dep = catalog.get("gemini/gemini-3.8-flash")
    assert dep.enabled
    assert dep.capabilities.thinking_mode == "levels"
    assert dep.timeouts.ttft_s == 60
    http = HttpClientFactory()
    adapters = build_adapters(catalog, AdapterDeps(http=http))
    assert isinstance(adapters["gemini"], GeminiAdapter)
    assert adapters["gemini"].quirks.emit_thought_signatures
    await http.aclose()
