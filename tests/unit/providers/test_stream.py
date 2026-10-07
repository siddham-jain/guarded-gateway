from typing import Any

import orjson
import pytest

from gg.core.errors import ProviderError
from gg.core.schema import ChatChunk, Usage
from gg.providers.openai_compat.quirks import QuirkProfile
from gg.providers.openai_compat.stream import CompatStreamTranslator, ThinkSplitter, split_content_array
from gg.providers.sse import SSEParser
from tests.unit.providers.support import fixture, profile_quirks, sse, text_chunk


def translator(provider: str | dict[str, Any] = "openai") -> CompatStreamTranslator:
    quirks = profile_quirks(provider) if isinstance(provider, str) else provider
    return CompatStreamTranslator(
        chunk_id="chatcmpl-gg",
        created=7,
        model="p/m",
        quirks=QuirkProfile.model_validate(quirks),
        provider="p",
        request_id="req_1",
    )


def run(tr: CompatStreamTranslator, raw: bytes) -> list[ChatChunk]:
    parser = SSEParser()
    out: list[ChatChunk] = []
    for event in [*parser.feed(raw), *parser.flush()]:
        out.extend(tr.feed(event))
    return out


def deltas(chunks: list[ChatChunk]) -> list[dict[str, Any]]:
    return [c.delta.model_dump(exclude_none=True) for ch in chunks for c in ch.choices]


def finishes(chunks: list[ChatChunk]) -> list[str]:
    return [c.finish_reason for ch in chunks for c in ch.choices if c.finish_reason]


def content(chunks: list[ChatChunk]) -> str:
    return "".join(c.delta.content or "" for ch in chunks for c in ch.choices)


def test_openai_text_stream() -> None:
    tr = translator()
    chunks = run(tr, fixture("openai", "chat_text.sse"))
    assert content(chunks) == "Hello world"
    assert finishes(chunks) == ["stop"]
    assert all((c.id, c.created, c.model) == ("chatcmpl-gg", 7, "p/m") for c in chunks)
    assert deltas(chunks)[0]["role"] == "assistant"
    assert "obfuscation" not in chunks[0].model_dump()
    assert tr.finished
    assert tr.counts is not None
    assert (tr.counts.input, tr.counts.cached) == (19, 4)
    assert (tr.upstream_id, tr.served_model) == ("chatcmpl-1", "gpt-6-luna")


def test_openai_parallel_tools_with_several_fragments_per_chunk() -> None:
    chunks = run(translator(), fixture("openai", "chat_tool_parallel.sse"))
    calls = [t for ch in chunks for c in ch.choices for t in (c.delta.tool_calls or ())]
    assert [(t.index, t.id) for t in calls if t.id] == [(0, "call_a"), (1, "call_b")]
    args = "".join(t.function.arguments or "" for t in calls if t.index == 0 and t.function)
    assert orjson.loads(args) == {"city": "Paris"}
    assert finishes(chunks) == ["tool_calls"]


def test_openai_midstream_error_raises() -> None:
    tr = translator()
    with pytest.raises(ProviderError) as exc:
        run(tr, fixture("openai", "stream_error_midstream.sse"))
    assert exc.value.kind == "retryable"


def test_cut_stream_is_not_finished() -> None:
    tr = translator()
    run(tr, fixture("openai", "stream_cut_no_usage.sse"))
    assert not tr.finished
    assert tr.counts is None
    assert tr.output_text() == "partial answer"


def test_groq_usage_under_x_groq_and_reasoning_dropped() -> None:
    tr = translator("groq")
    chunks = run(tr, fixture("groq", "chat_text_x_groq.sse"))
    assert content(chunks) == "OK"
    assert all("reasoning_content" not in d for d in deltas(chunks))
    assert "".join(tr.reasoning_parts) == "thinking briefly"
    assert tr.counts is not None
    assert (tr.counts.input, tr.counts.reasoning) == (12, 5)


def test_reasoning_exposed_when_profile_allows() -> None:
    quirks = {**profile_quirks("groq"), "reasoning": {"field": "reasoning", "expose": True}}
    chunks = run(translator(quirks), fixture("groq", "chat_text_x_groq.sse"))
    assert any(d.get("reasoning_content") == "thinking briefly" for d in deltas(chunks))


def test_together_eos_and_flat_cached_tokens() -> None:
    tr = translator("together")
    chunks = run(tr, fixture("together", "chat_eos_flat_cached.sse"))
    assert finishes(chunks) == ["stop"]
    assert tr.counts is not None
    assert tr.counts.cached == 16


def test_deepseek_reasoning_then_tool_call_and_cache_hits() -> None:
    tr = translator("deepseek")
    chunks = run(tr, fixture("deepseek", "chat_reasoning_tool.sse"))
    assert finishes(chunks) == ["tool_calls"]
    assert tr.tool_call_ids == ["call_ds1"]
    assert "".join(tr.reasoning_parts) == "Need the weather tool."
    assert tr.counts is not None
    assert (tr.counts.cached, tr.counts.reasoning) == (32, 8)


def test_ollama_tool_without_id_gets_deterministic_id_and_tool_finish() -> None:
    chunks = run(translator("ollama"), fixture("ollama", "chat_tool_no_id.sse"))
    call = next(t for ch in chunks for c in ch.choices for t in (c.delta.tool_calls or ()))
    assert call.id is not None
    assert call.id.startswith("call_gg_")
    assert len(call.id) == 32
    assert call.function is not None
    assert orjson.loads(call.function.arguments or "") == {"city": "Paris"}
    assert finishes(chunks) == ["tool_calls"]
    again = run(translator("ollama"), fixture("ollama", "chat_tool_no_id.sse"))
    assert next(t for ch in again for c in ch.choices for t in (c.delta.tool_calls or ())).id == call.id


def test_openrouter_native_finish_and_provider() -> None:
    tr = translator("openrouter")
    chunks = run(tr, fixture("openrouter", "chat_reasoning.sse"))
    assert content(chunks) == "Done"
    assert tr.provider_finish_reason == "end_turn"
    assert tr.upstream_provider == "DeepInfra"


def test_openrouter_midstream_error_chunk() -> None:
    with pytest.raises(ProviderError) as exc:
        run(translator("openrouter"), fixture("openrouter", "stream_error_midstream.sse"))
    assert exc.value.kind == "retryable"
    assert exc.value.status == 200


def test_zai_sensitive_is_content_filter() -> None:
    chunks = run(translator("zai"), fixture("zai", "chat_sensitive.sse"))
    assert finishes(chunks) == ["content_filter"]


def test_mistral_content_arrays_split_into_reasoning_and_text() -> None:
    tr = translator("mistral")
    chunks = run(tr, fixture("mistral", "chat_content_array.sse"))
    assert content(chunks) == "Answer is 4."
    assert "".join(tr.reasoning_parts) == "Let me think."
    assert tr.counts is not None
    assert tr.counts.output == 6


def test_minimax_error_in_200_body() -> None:
    with pytest.raises(ProviderError) as exc:
        run(translator("minimax"), fixture("minimax", "chat_base_resp_error.sse"))
    assert (exc.value.kind, exc.value.code) == ("auth", "billing")


def test_moonshot_usage_inside_choice() -> None:
    tr = translator("moonshot")
    run(tr, fixture("moonshot", "chat_usage_in_choice.sse"))
    assert tr.counts is not None
    assert (tr.counts.input, tr.counts.cached) == (11, 8)


def test_xai_cumulative_usage_takes_last_and_cost() -> None:
    tr = translator("xai")
    run(tr, fixture("xai", "chat_cumulative_usage.sse"))
    assert tr.counts is not None
    assert tr.counts.output == 2
    assert tr.cost_usd == pytest.approx(1.25e-5)


def test_perplexity_think_tags_split_and_citations_passthrough() -> None:
    tr = translator("perplexity")
    chunks = run(tr, fixture("perplexity", "chat_think_citations.sse"))
    assert content(chunks) == "Paris"
    assert "".join(tr.reasoning_parts) == "searching"
    assert chunks[-1].model_extra == {"citations": ["https://example.com/a"]}
    assert tr.cost_usd == pytest.approx(0.006)


def test_unknown_finish_reason_and_error_finish() -> None:
    tr = translator()
    chunks = run(tr, sse(text_chunk("x", role=True), text_chunk(finish="weird")))
    assert finishes(chunks) == ["stop"]
    assert "unknown_finish_reason" in tr.flags
    with pytest.raises(ProviderError):
        run(translator(), sse(text_chunk("x", role=True), text_chunk(finish="error")))


def test_unknown_fields_and_events_are_ignored() -> None:
    noisy = sse(
        {**text_chunk("a", role=True), "mystery": {"x": 1}},
        {"type": "ping"},
        [1, 2],
        {
            **text_chunk(None, finish="stop"),
            "choices": [{"index": 0, "delta": {"surprise": 1}, "finish_reason": "stop"}],
        },
    )
    assert content(run(translator(), noisy)) == "a"


def test_invalid_json_is_retryable() -> None:
    with pytest.raises(ProviderError) as exc:
        run(translator(), b"data: {not json\n\n")
    assert exc.value.code == "bad_upstream_response"


def test_tail_synthesises_finish_and_usage() -> None:
    tr = translator()
    run(tr, sse(text_chunk("a", role=True)))
    tail = tr.tail(Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
    assert finishes(tail) == ["stop"]
    assert tail[-1].choices == ()
    assert "synthesized_finish" in tr.flags


def test_think_splitter_handles_tags_across_chunks() -> None:
    splitter = ThinkSplitter()
    parts = [splitter.feed(t) for t in ["a<th", "ink>r1", "</thi", "nk>b<"]]
    assert "".join(p[0] for p in parts) + splitter.flush()[0] == "ab<"
    assert "".join(p[1] for p in parts) == "r1"


def test_split_content_array_variants() -> None:
    assert split_content_array(
        ["x", {"type": "thinking", "thinking": "t"}, {"type": "text", "text": "y"}]
    ) == ("xy", "t")
