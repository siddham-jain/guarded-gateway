from typing import Any

import pytest

from gg.core.schema import ChatChunk, ChunkChoice, Delta, ToolCallDelta, Usage
from gg.providers.commit import CommitGate, merge_chunks
from gg.providers.usage import TokenCounts, counts_from_usage, estimate_prompt_tokens, estimate_text_tokens
from tests.conftest import make_request


def chunk(finish: Any = None, usage: Usage | None = None, empty: bool = False, **delta: Any) -> ChatChunk:
    choices = (
        ()
        if empty
        else (
            ChunkChoice.model_construct(index=0, delta=Delta.model_construct(**delta), finish_reason=finish),
        )
    )
    return ChatChunk.model_construct(id="c", created=1, model="m", choices=choices, usage=usage)


def test_role_and_reasoning_are_held_until_content() -> None:
    gate = CommitGate()
    assert gate.push_many([chunk(role="assistant", content="")]) == []
    assert gate.push_many([chunk(reasoning_content="think")]) == []
    out = gate.push_many([chunk(content="Hi"), chunk(content=" there")])
    assert gate.committed
    assert len(out) == 2
    first = out[0].choices[0].delta
    assert (first.role, first.content, (first.model_extra or {}).get("reasoning_content")) == (
        "assistant",
        "Hi",
        "think",
    )
    assert gate.push_many([chunk(content="!")])[0].choices[0].delta.content == "!"


def test_tool_call_and_finish_commit() -> None:
    call = ToolCallDelta.model_construct(index=0, id="call_1", type="function", function=None)
    gate = CommitGate()
    assert gate.push_many([chunk(tool_calls=(call,))])[0].choices[0].delta.tool_calls == (call,)
    finish_gate = CommitGate()
    finish_gate.push_many([chunk(role="assistant")])
    out = finish_gate.push_many([chunk(finish="stop")])
    assert out[0].choices[0].finish_reason == "stop"
    assert out[0].choices[0].delta.role == "assistant"


def test_buffer_over_limit_force_commits() -> None:
    gate = CommitGate(max_bytes=10)
    assert gate.push_many([chunk(reasoning_content="x" * 5)]) == []
    out = gate.push_many([chunk(reasoning_content="y" * 6)])
    assert gate.committed
    assert gate.forced
    assert len(out) == 1


def test_close_flushes_empty_response_and_keeps_usage_separate() -> None:
    gate = CommitGate()
    gate.push_many([chunk(role="assistant")])
    usage = Usage(prompt_tokens=1, completion_tokens=0, total_tokens=1)
    out = gate.close([chunk(finish="stop"), chunk(usage=usage, empty=True)])
    assert [bool(c.choices) for c in out] == [True, False]
    assert out[0].choices[0].finish_reason == "stop"
    assert out[1].usage == usage
    assert gate.close([chunk(content="late")])[0].choices[0].delta.content == "late"


def test_merge_keeps_tool_fragments_in_order() -> None:
    a = ToolCallDelta.model_construct(index=0, id="a")
    b = ToolCallDelta.model_construct(index=0, id=None)
    merged = merge_chunks([chunk(role="assistant", tool_calls=(a,)), chunk(tool_calls=(b,))])
    assert merged.choices[0].delta.tool_calls == (a, b)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            {
                "prompt_tokens": 19,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
            TokenCounts(19, 10, 4, 0, 3),
        ),
        (
            {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "prompt_cache_hit_tokens": 32,
                "prompt_cache_miss_tokens": 18,
            },
            TokenCounts(50, 20, 32, 0, 0),
        ),
        ({"prompt_tokens": 30, "completion_tokens": 2, "cached_tokens": 16}, TokenCounts(30, 2, 16, 0, 0)),
        (
            {
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 9, "cache_write_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
            TokenCounts(5, 1, 5, 0, 1),
        ),
        ({"input_tokens": 3, "output_tokens": 4}, TokenCounts(3, 4)),
    ],
)
def test_counts_from_usage_variants_and_invariants(raw: dict[str, Any], expected: TokenCounts) -> None:
    counts = counts_from_usage(raw)
    assert counts == expected
    assert counts is not None
    assert counts.cached + counts.cache_write <= counts.input
    assert counts.reasoning <= counts.output


def test_counts_absent_and_custom_paths() -> None:
    assert counts_from_usage({"foo": 1}) is None
    custom = counts_from_usage(
        {"prompt_tokens": 9, "completion_tokens": 1, "hits": 4}, {"cached_tokens": ("hits",)}
    )
    assert custom is not None
    assert custom.cached == 4


def test_usage_conversion() -> None:
    usage = TokenCounts(10, 5, 2, 1, 3).to_usage()
    assert usage.total_tokens == 15
    assert usage.prompt_tokens_details is not None
    assert usage.completion_tokens_details is not None
    assert (usage.prompt_tokens_details.cached_tokens, usage.prompt_tokens_details.cache_write_tokens) == (
        2,
        1,
    )
    assert usage.completion_tokens_details.reasoning_tokens == 3


def test_estimates_are_positive_and_grow() -> None:
    assert estimate_text_tokens("") == 0
    short = estimate_prompt_tokens(make_request())
    long = estimate_prompt_tokens(make_request(messages=[{"role": "user", "content": "word " * 400}]))
    assert 0 < short < long
