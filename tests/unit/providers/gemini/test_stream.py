from datetime import UTC, datetime
from typing import Any

import orjson
import pytest

from gg.core.errors import ProviderError
from gg.core.schema import ChatChunk
from gg.providers.gemini.stream import GeminiStreamTranslator, usage_counts
from gg.providers.sse import SSEParser
from gg.providers.usage import TokenCounts
from tests.unit.providers.gemini.support import candidate_event, gemini_sse
from tests.unit.providers.support import fixture

NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)


def translator(**kw: Any) -> GeminiStreamTranslator:
    return GeminiStreamTranslator(
        chunk_id="chatcmpl-x",
        created=1,
        model="gemini/m",
        provider="gemini",
        request_id="req_1",
        now=NOW,
        **kw,
    )


def run(
    raw: bytes, tr: GeminiStreamTranslator | None = None
) -> tuple[GeminiStreamTranslator, list[ChatChunk]]:
    tr = tr or translator()
    parser = SSEParser()
    chunks: list[ChatChunk] = []
    for event in [*parser.feed(raw), *parser.flush()]:
        chunks.extend(tr.feed(event))
    return tr, chunks


def deltas(chunks: list[ChatChunk]) -> list[dict[str, Any]]:
    return [c.delta.model_dump(exclude_none=True) for ch in chunks for c in ch.choices]


def finishes(chunks: list[ChatChunk]) -> list[str]:
    return [c.finish_reason for ch in chunks for c in ch.choices if c.finish_reason]


def test_text_stream_with_trailing_signature_part() -> None:
    tr, chunks = run(fixture("gemini", "gen_text.sse"))
    assert deltas(chunks) == [
        {"role": "assistant", "content": "The capital"},
        {"content": " of France is Paris."},
        {},
    ]
    assert finishes(chunks) == ["stop"]
    assert tr.finished
    assert tr.output_text() == "The capital of France is Paris."
    assert (tr.upstream_id, tr.served_model, tr.provider_finish_reason) == (
        "resp-1",
        "gemini-3.8-flash",
        "STOP",
    )
    assert tr.signatures == {}
    assert tr.counts == TokenCounts(input=4100, output=47, cached=4096, reasoning=40)


def test_parallel_calls_are_whole_deltas_with_stable_indices() -> None:
    tr, chunks = run(fixture("gemini", "gen_parallel_fc.sse"))
    assert len(chunks) == 1
    choice = chunks[0].choices[0]
    assert choice.finish_reason == "tool_calls"
    calls = choice.delta.tool_calls
    assert calls is not None
    assert [(c.index, c.id, c.function and c.function.name) for c in calls] == [
        (0, "fc_paris", "get_weather"),
        (1, "fc_rome", "get_weather"),
    ]
    assert [orjson.loads(c.function.arguments or "") for c in calls if c.function] == [
        {"city": "Paris"},
        {"city": "Rome"},
    ]
    assert calls[0].model_extra == {"extra_content": {"google": {"thought_signature": "c2lnLXBhcmlz"}}}
    assert calls[1].model_extra == {}
    assert tr.signatures == {"fc_paris": "c2lnLXBhcmlz"}
    assert tr.tool_call_ids == ["fc_paris", "fc_rome"]


def test_signature_emission_can_be_turned_off() -> None:
    tr, chunks = run(fixture("gemini", "gen_parallel_fc.sse"), translator(emit_signatures=False))
    calls = chunks[0].choices[0].delta.tool_calls or ()
    assert all(not c.model_extra for c in calls)
    assert tr.signatures == {"fc_paris": "c2lnLXBhcmlz"}


def test_call_after_text_overrides_stop_with_tool_calls() -> None:
    tr, chunks = run(fixture("gemini", "gen_function_call_signature.sse"))
    assert deltas(chunks)[0] == {"role": "assistant", "content": "Let me check."}
    assert finishes(chunks) == ["tool_calls"]
    assert tr.signatures == {"fc_1": "c2lnLW9zbG8="}


def test_missing_call_ids_are_synthesised_deterministically() -> None:
    _, first = run(fixture("gemini", "gen_v25_fc_no_id.sse"))
    _, again = run(fixture("gemini", "gen_v25_fc_no_id.sse"))
    ids = [c.id for c in first[0].choices[0].delta.tool_calls or ()]
    assert ids == [c.id for c in again[0].choices[0].delta.tool_calls or ()]
    assert len(set(ids)) == 2
    assert all(i and i.startswith("call_gg_") for i in ids)


def test_thought_parts_are_dropped() -> None:
    tr, chunks = run(fixture("gemini", "gen_thought_summary.sse"))
    assert deltas(chunks) == [{"role": "assistant", "content": "42"}]
    assert tr.reasoning_parts == ["**Planning** the answer"]
    assert tr.counts == TokenCounts(input=5, output=31, reasoning=30)


def test_blocked_prompt_is_a_content_filter_error() -> None:
    with pytest.raises(ProviderError) as exc:
        run(fixture("gemini", "gen_prompt_blocked.sse"))
    assert (exc.value.kind, exc.value.status, exc.value.code) == ("content_filter", 200, "content_filter")
    assert "SAFETY" in exc.value.message


@pytest.mark.parametrize(
    ("name", "finish", "text"),
    [
        ("gen_safety_no_content.sse", "content_filter", ""),
        ("gen_safety_midstream.sse", "content_filter", "Here is how"),
        ("gen_recitation.sse", "content_filter", "It was the best of times"),
        ("gen_max_tokens_empty.sse", "length", ""),
    ],
)
def test_terminal_finish_reasons(name: str, finish: str, text: str) -> None:
    tr, chunks = run(fixture("gemini", name))
    assert finishes(chunks) == [finish]
    assert tr.output_text() == text
    assert tr.finished
    assert chunks[0].choices[0].delta.role == "assistant"


def test_max_tokens_empty_still_bills_thoughts() -> None:
    tr, _ = run(fixture("gemini", "gen_max_tokens_empty.sse"))
    assert tr.counts == TokenCounts(input=10, output=16, reasoning=16)


def test_malformed_call_before_content_is_retryable() -> None:
    with pytest.raises(ProviderError) as exc:
        run(fixture("gemini", "gen_malformed_function_call.sse"))
    assert (exc.value.kind, exc.value.code) == ("retryable", "malformed_function_call")


def test_malformed_call_after_content_is_a_flagged_stop() -> None:
    raw = gemini_sse(candidate_event([{"text": "hi"}]), candidate_event(None, "MALFORMED_FUNCTION_CALL"))
    tr, chunks = run(raw)
    assert finishes(chunks) == ["stop"]
    assert tr.flags == ["malformed_function_call"]


def test_missing_thought_signature_falls_back() -> None:
    with pytest.raises(ProviderError) as exc:
        run(fixture("gemini", "gen_missing_thought_signature.sse"))
    assert (exc.value.kind, exc.value.code) == ("fallback", "missing_thought_signature")


@pytest.mark.parametrize(
    ("reason", "flag"),
    [
        ("UNEXPECTED_TOOL_CALL", "unexpected_tool_call"),
        ("OTHER", "other"),
        ("BRAND_NEW", "unknown_finish_reason"),
    ],
)
def test_flagged_finish_reasons_stop(reason: str, flag: str) -> None:
    tr, chunks = run(gemini_sse(candidate_event([{"text": "x"}], reason)))
    assert finishes(chunks) == ["stop"]
    assert tr.flags == [flag]


def test_in_band_error_is_classified() -> None:
    with pytest.raises(ProviderError) as exc:
        run(fixture("gemini", "stream_error_object.sse"))
    assert (exc.value.kind, exc.value.status, exc.value.code) == ("retryable", 200, "overloaded")


def test_stream_cut_before_finish_is_not_finished() -> None:
    tr, chunks = run(fixture("gemini", "stream_cut_no_finish.sse"))
    assert not tr.finished
    assert finishes(chunks) == []
    assert tr.output_text() == "Once upon a time"


def test_invalid_json_is_retryable() -> None:
    with pytest.raises(ProviderError) as exc:
        run(b"data: {not json\n\n")
    assert (exc.value.kind, exc.value.code) == ("retryable", "bad_upstream_response")


def test_empty_and_unknown_events_produce_nothing() -> None:
    tr, chunks = run(b"data: {}\n\ndata: []\n\n" + gemini_sse({"candidates": [], "x": 1}))
    assert chunks == []
    assert not tr.finished


def test_candidates_map_to_choice_indices() -> None:
    event = {
        "candidates": [
            {"index": 0, "content": {"parts": [{"text": "a"}]}, "finishReason": "STOP"},
            {"index": 1, "content": {"parts": [{"text": "b"}]}, "finishReason": "MAX_TOKENS"},
        ]
    }
    tr, chunks = run(gemini_sse(event))
    assert [(c.index, c.delta.content, c.finish_reason) for c in chunks[0].choices] == [
        (0, "a", "stop"),
        (1, "b", "length"),
    ]
    assert tr.finished


def test_usage_counts_mapping() -> None:
    assert usage_counts({"totalTokenCount": 3}) is None
    counts = usage_counts(
        {
            "promptTokenCount": 100,
            "toolUsePromptTokenCount": 10,
            "cachedContentTokenCount": 200,
            "candidatesTokenCount": 5,
            "thoughtsTokenCount": 7,
        }
    )
    assert counts == TokenCounts(input=110, output=12, cached=110, reasoning=7)
