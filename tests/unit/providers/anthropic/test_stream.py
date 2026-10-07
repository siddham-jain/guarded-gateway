import orjson
import pytest

from gg.core.schema import ChatChunk
from gg.providers.anthropic.stream import AnthropicStreamTranslator
from gg.providers.anthropic.usage import AnthropicTokenCounts, counts_from_usage
from tests.unit.providers.anthropic.support import (
    anthropic_sse,
    message_end,
    message_start,
    run_translator,
    run_until_error,
    text_block,
)
from tests.unit.providers.support import fixture


def load(name: str) -> bytes:
    return fixture("anthropic", name)


def text_of(chunks: list[ChatChunk]) -> str:
    return "".join(c.delta.content or "" for ch in chunks for c in ch.choices)


def finishes(chunks: list[ChatChunk]) -> list[str]:
    return [c.finish_reason for ch in chunks for c in ch.choices if c.finish_reason]


def test_text_stream() -> None:
    tr, chunks = run_translator(load("msg_text.sse"))
    assert tr.finished
    assert text_of(chunks) == "Hello world"
    assert chunks[0].choices[0].delta.role == "assistant"
    assert finishes(chunks) == ["stop"]
    assert (tr.upstream_id, tr.served_model, tr.provider_finish_reason) == (
        "msg_01Text",
        "claude-haiku-4-5",
        "end_turn",
    )
    assert tr.counts == AnthropicTokenCounts(input=25, output=6)


def test_tool_use_with_partial_json_has_stable_indices() -> None:
    tr, chunks = run_translator(load("msg_tool_use.sse"))
    deltas = [t for ch in chunks for c in ch.choices for t in c.delta.tool_calls or ()]
    assert deltas[0].id == "toolu_01T1x1fJ34qAmk2tNTrN7Up6"
    assert deltas[0].function is not None
    assert deltas[0].function.name == "get_weather"
    assert {d.index for d in deltas} == {0}
    arguments = "".join(d.function.arguments or "" for d in deltas if d.function)
    assert orjson.loads(arguments) == {"location": "San Francisco, CA"}
    assert len(deltas) == 3  # empty partial_json fragments are skipped
    assert text_of(chunks) == "Okay, let's check"
    assert finishes(chunks) == ["tool_calls"]
    assert tr.tool_call_ids == ["toolu_01T1x1fJ34qAmk2tNTrN7Up6"]
    assert tr.thinking_writes == {}


def test_parallel_tools_get_contiguous_indices() -> None:
    _, chunks = run_translator(load("msg_parallel_tools.sse"))
    starts = [t for ch in chunks for c in ch.choices for t in c.delta.tool_calls or () if t.id]
    assert [(t.index, t.id) for t in starts] == [(0, "toolu_a"), (1, "toolu_b")]


def test_thinking_then_tool_use_is_captured_not_emitted() -> None:
    tr, chunks = run_translator(load("msg_thinking_tool.sse"))
    assert all("reasoning_content" not in (c.delta.model_extra or {}) for ch in chunks for c in ch.choices)
    assert tr.reasoning_parts == ["I need the weather", " for Paris."]
    assert tr.thinking_writes == {
        "toolu_think": [
            {
                "type": "thinking",
                "thinking": "I need the weather for Paris.",
                "signature": "EqQBCgIYAhIM1gbcDa9G",
            },
            {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix"},
        ]
    }
    assert tr.counts is not None
    assert tr.counts.reasoning == 50
    assert finishes(chunks) == ["tool_calls"]


def test_thinking_is_exposed_only_when_configured() -> None:
    tr = AnthropicStreamTranslator(
        chunk_id="c", created=1, model="m", provider="anthropic", expose_reasoning=True
    )
    _, chunks = run_translator(load("msg_thinking_tool.sse"), tr)
    exposed = [(c.delta.model_extra or {}).get("reasoning_content") for ch in chunks for c in ch.choices]
    assert [r for r in exposed if r] == ["I need the weather", " for Paris."]


def test_thinking_segments_belong_to_the_next_tool_use_only() -> None:
    def thinking(index: int, sig: str) -> list[dict[str, object]]:
        return [
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "signature_delta", "signature": sig},
            },
            {"type": "content_block_stop", "index": index},
        ]

    def tool(index: int, call_id: str) -> list[dict[str, object]]:
        block = {"type": "tool_use", "id": call_id, "name": "f", "input": {}}
        return [
            {"type": "content_block_start", "index": index, "content_block": block},
            {"type": "content_block_stop", "index": index},
        ]

    raw = anthropic_sse(
        message_start(),
        *thinking(0, "a"),
        *tool(1, "t1"),
        *tool(2, "t2"),
        *thinking(3, "b"),
        *tool(4, "t3"),
        *message_end("tool_use"),
    )
    tr, _ = run_translator(raw)
    assert {k: [b["signature"] for b in v] for k, v in tr.thinking_writes.items()} == {
        "t1": ["a"],
        "t3": ["b"],
    }


def test_refusal_before_content_is_a_content_filter_finish() -> None:
    tr, chunks = run_translator(load("msg_refusal_precontent.sse"))
    assert len(chunks) == 1
    choice = chunks[0].choices[0]
    assert choice.finish_reason == "content_filter"
    assert choice.delta.refusal == "This request was declined because it could enable cyber harm."
    assert tr.refusal_category == "cyber"
    assert tr.provider_finish_reason == "refusal"


def test_refusal_mid_stream_keeps_partial_text() -> None:
    tr, chunks = run_translator(load("msg_refusal_midstream.sse"))
    assert text_of(chunks) == "Sure, step one"
    assert finishes(chunks) == ["content_filter"]
    assert tr.refusal_category == "bio"


def test_max_tokens_is_length() -> None:
    _, chunks = run_translator(load("msg_max_tokens.sse"))
    assert finishes(chunks) == ["length"]


@pytest.mark.parametrize(
    ("reason", "finish", "flag"),
    [
        ("stop_sequence", "stop", None),
        ("model_context_window_exceeded", "length", None),
        ("pause_turn", "stop", "pause_turn"),
        ("brand_new_reason", "stop", "unknown_finish_reason"),
    ],
)
def test_other_stop_reasons(reason: str, finish: str, flag: str | None) -> None:
    tr, chunks = run_translator(anthropic_sse(message_start(), *text_block(0, "x"), *message_end(reason)))
    assert finishes(chunks) == [finish]
    assert tr.flags == ([flag] if flag else [])


def test_empty_message_still_finishes_with_role() -> None:
    _, chunks = run_translator(anthropic_sse(message_start(), *message_end("end_turn")))
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[0].choices[0].finish_reason == "stop"


def test_error_before_content_raises_overloaded_fallback() -> None:
    seen, err = run_until_error(load("stream_error_overloaded_precontent.sse"))
    assert seen == []
    assert (err.kind, err.code, err.status) == ("fallback", "overloaded", 200)


def test_error_after_content_raises_retryable() -> None:
    seen, err = run_until_error(load("stream_error_postcontent.sse"))
    assert text_of(seen) == "partial"
    assert (err.kind, err.code) == ("retryable", "upstream_error")


def test_bad_json_is_a_retryable_upstream_error() -> None:
    _, err = run_until_error(b"event: message_start\ndata: {nope\n\n")
    assert (err.kind, err.code) == ("retryable", "bad_upstream_response")


def test_ping_citations_and_unknown_events_are_ignored() -> None:
    tr, chunks = run_translator(load("stream_ping_unknown_event.sse"))
    _, clean = run_translator(load("msg_text.sse"))
    assert text_of(chunks) == text_of(clean) == "Hello world"
    assert tr.flags == []


def test_stream_without_message_stop_is_not_finished() -> None:
    raw = anthropic_sse(message_start(), *text_block(0, "x"))
    tr, _ = run_translator(raw)
    assert not tr.finished


def test_cache_usage_includes_reads_and_both_write_ttls() -> None:
    tr, _ = run_translator(load("msg_cache_usage.sse"))
    assert tr.counts == AnthropicTokenCounts(
        input=105_170, output=503, cached=100_000, cache_write=4096, reasoning=310, cache_write_1h=1024
    )
    usage = tr.counts.to_usage()
    assert usage.prompt_tokens == 105_170
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 100_000
    assert usage.prompt_tokens_details.cache_write_tokens == 5120
    assert tr.raw_usage is not None
    assert tr.raw_usage["service_tier"] == "standard"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"input_tokens": 10, "output_tokens": 2}, AnthropicTokenCounts(input=10, output=2)),
        (
            {"input_tokens": 10, "cache_creation_input_tokens": 30, "output_tokens": 2},
            AnthropicTokenCounts(input=40, output=2, cache_write=30),
        ),
        ({"output_tokens": 7}, AnthropicTokenCounts(input=0, output=7)),
        ({"input_tokens": -3, "output_tokens": True}, AnthropicTokenCounts(input=0, output=0)),
    ],
)
def test_usage_normalisation(raw: dict[str, object], expected: AnthropicTokenCounts) -> None:
    assert counts_from_usage(raw) == expected


def test_usage_absent_means_no_counts() -> None:
    assert counts_from_usage({}) is None
    tr, _ = run_translator(
        anthropic_sse(message_start(usage={}), *text_block(0, "x"), *message_end("end_turn", {}))
    )
    assert tr.counts is None
