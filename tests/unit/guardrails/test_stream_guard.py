import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gg.core.clock import FakeClock, SystemClock
from gg.core.context import RequestContext
from gg.core.errors import GuardrailBlockedError
from gg.core.guard_types import Verdict
from gg.core.schema import (
    AssistantMessage,
    ChatChunk,
    ChatResponse,
    Choice,
    ChunkChoice,
    Delta,
    FunctionCallDelta,
    ToolCallDelta,
    Usage,
)
from gg.guardrails.fakes import FakeValues
from gg.guardrails.output.runner import OutputGuardRunner
from gg.guardrails.output.stream_guard import StreamGuard
from gg.guardrails.stages import EFFECTIVE_POLICY
from gg.guardrails.vault import GuardVault
from tests.conftest import make_ctx, make_request
from tests.unit.guardrails.support import (
    Upstream,
    chunk,
    engine,
    policy_doc,
    policy_set,
    released_text,
    text_chunks,
    write_policy,
)

FAKES = FakeValues()
KEY = FAKES.get("openai_key")
EMAIL = FAKES.get("email")


def setup(
    policy_dir: Path | None = None, *, vault: dict[str, str] | None = None, **request: Any
) -> RequestContext:
    ctx = make_ctx(FakeClock(), make_request(stream=True, **request))
    ctx.vault = GuardVault()
    for label, value in (vault or {}).items():
        ctx.vault.add(label, value)
    policies = policy_set(policy_dir) if policy_dir else policy_set()
    ctx.set(EFFECTIVE_POLICY, policies.effective(ctx.key, ctx.request))
    return ctx


def guard_for(ctx: RequestContext, *, detect: bool = True) -> StreamGuard:
    policy = ctx.get(EFFECTIVE_POLICY)
    assert policy is not None
    return StreamGuard(engine(), policy, ctx, clock=SystemClock(), detect=detect)


async def run_stream(
    ctx: RequestContext, chunks: list[ChatChunk], **kwargs: Any
) -> tuple[list[ChatChunk], Upstream]:
    upstream = Upstream(chunks)
    out = [c async for c in guard_for(ctx, **kwargs).guard(upstream.gen())]
    return out, upstream


async def collect(stream: AsyncIterator[ChatChunk], into: list[ChatChunk]) -> None:
    async for c in stream:
        into.append(c)  # noqa: PERF401 - keeps what arrived before the stream raised


def prefixes(chunks: list[ChatChunk]) -> list[str]:
    out: list[str] = []
    text = ""
    for c in chunks:
        if c.choices:
            text += c.choices[0].delta.content or ""
            out.append(text)
    return out


TEXT = (
    f"Sure, your key is {KEY} and you should keep it out of the repo, rotate it monthly and never paste it."
)


@pytest.mark.parametrize("split", range(1, len(TEXT)))
async def test_secret_split_across_two_chunks_never_leaks(split: int) -> None:
    ctx = setup()
    out, _ = await run_stream(ctx, text_chunks([TEXT[:split], TEXT[split:]]))
    assert released_text(out) == TEXT.replace(KEY, "[REDACTED:SECRET]")
    assert not any(KEY[:6] in p for p in prefixes(out))


@pytest.mark.parametrize("split", range(1, len(KEY), 7))
async def test_secret_split_across_three_chunks_with_an_empty_one(split: int) -> None:
    ctx = setup()
    start = TEXT.index(KEY)
    parts = [TEXT[: start + split], "", TEXT[start + split :]]
    out, _ = await run_stream(ctx, text_chunks(parts))
    assert released_text(out) == TEXT.replace(KEY, "[REDACTED:SECRET]")


PLACEHOLDER_TEXT = (
    "Hi Dana, I have copied [EMAIL_1] on the thread and will follow up with the finance team tomorrow."
)


@pytest.mark.parametrize("split", range(1, len(PLACEHOLDER_TEXT)))
async def test_placeholder_split_is_restored_and_never_released_partially(split: int) -> None:
    ctx = setup(vault={"EMAIL": "dana@corp.example"})
    out, _ = await run_stream(ctx, text_chunks([PLACEHOLDER_TEXT[:split], PLACEHOLDER_TEXT[split:]]))
    assert released_text(out) == PLACEHOLDER_TEXT.replace("[EMAIL_1]", "dana@corp.example")
    assert not any("[" in p for p in prefixes(out))


def blocking_policy(tmp_path: Path, **streaming: Any) -> Path:
    doc = policy_doc()
    doc["output"]["guards"][0]["action"] = "block"
    doc["output"]["streaming"].update(streaming)
    return write_policy(tmp_path, doc)


async def test_block_mid_stream_ends_gracefully_and_closes_upstream(tmp_path: Path) -> None:
    ctx = setup(blocking_policy(tmp_path))
    safe = ["This part is fine and long enough to pass the first window. ", "More fine text here. " * 10]
    chunks = text_chunks([*safe, f"now the key {KEY} ", *(["tail text "] * 20)])
    out, upstream = await run_stream(ctx, chunks)
    assert upstream.closed
    assert upstream.served < len(chunks)
    last = out[-1]
    assert last.choices[0].finish_reason == "content_filter"
    marker = (last.model_extra or {})["gg_guardrail"]
    assert marker["stage"] == "output"
    assert marker["action"] == "block"
    assert marker["policy"].startswith("default@")
    assert "guard" not in str(marker).replace("gg_guardrail", "")
    assert KEY[:6] not in released_text(out)
    assert released_text(out).startswith("This part is fine")
    assert ctx.outcome == "guard_aborted"
    assert ctx.output_verdict is not None
    assert ctx.output_verdict.verdict is Verdict.BLOCK


async def test_block_in_first_window_sends_role_then_content_filter(tmp_path: Path) -> None:
    ctx = setup(blocking_policy(tmp_path))
    out, _ = await run_stream(ctx, text_chunks([f"{KEY} is the key, that is all there is to it."]))
    (only,) = out
    assert only.choices[0].delta.role == "assistant"
    assert only.choices[0].finish_reason == "content_filter"


async def test_error_frame_mode_raises_after_committing_the_stream(tmp_path: Path) -> None:
    ctx = setup(blocking_policy(tmp_path, abort="error_frame"))
    upstream = Upstream(text_chunks([f"{KEY} is the key, that is all there is to it."]))
    got: list[ChatChunk] = []
    with pytest.raises(GuardrailBlockedError) as info:
        await collect(guard_for(ctx).guard(upstream.gen()), got)
    assert info.value.code == "guardrail_blocked"
    assert len(got) == 1
    assert got[0].choices[0].delta.role == "assistant"
    assert upstream.closed


async def test_usage_is_released_after_the_finish_chunk() -> None:
    ctx = setup()
    usage = ChatChunk(
        id="c1",
        created=1,
        model="m",
        choices=(),
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    chunks = [*text_chunks(["hello there, nothing sensitive in this reply at all."]), usage]
    out, _ = await run_stream(ctx, chunks)
    assert out[-1] is usage
    assert out[-2].choices[0].finish_reason == "stop"
    assert out[0].choices[0].delta.role == "assistant"


async def test_tool_arguments_restore_json_escaped_with_header_first() -> None:
    ctx = setup(vault={"EMAIL": 'odd"name@corp.example'})

    def tool(index: int, args: str, *, header: bool = False) -> ChatChunk:
        call = ToolCallDelta(
            index=index,
            id="call_1" if header else None,
            type="function" if header else None,
            function=FunctionCallDelta(name="send" if header else None, arguments=args),
        )
        return ChatChunk(
            id="c1", created=1, model="m", choices=(ChunkChoice(index=0, delta=Delta(tool_calls=(call,))),)
        )

    chunks = [
        tool(0, '{"to": "[EMA', header=True),
        tool(0, 'IL_1]", "body": "hi"}'),
        chunk(finish="tool_calls"),
    ]
    out, _ = await run_stream(ctx, chunks)
    calls = [tc for c in out if c.choices for tc in (c.choices[0].delta.tool_calls or ())]
    assert calls[0].id == "call_1"
    assert calls[0].function is not None
    assert calls[0].function.name == "send"
    args = "".join(tc.function.arguments or "" for tc in calls if tc.function)
    assert args == '{"to": "odd\\"name@corp.example", "body": "hi"}'
    assert out[-1].choices[0].finish_reason == "tool_calls"


async def test_tool_call_without_arguments_still_emits_its_header() -> None:
    ctx = setup()
    call = ToolCallDelta(index=0, id="call_9", type="function", function=FunctionCallDelta(name="ping"))
    chunks = [
        ChatChunk(
            id="c", created=1, model="m", choices=(ChunkChoice(index=0, delta=Delta(tool_calls=(call,))),)
        ),
        chunk(finish="tool_calls"),
    ]
    out, _ = await run_stream(ctx, chunks)
    headers = [tc for c in out for tc in (c.choices[0].delta.tool_calls or ())]
    assert headers[0].id == "call_9"


async def test_long_private_key_is_swallowed_whole() -> None:
    ctx = setup()
    pem = FAKES.get("pem_private_key")
    body_line = pem.splitlines()[2]
    text = f"Here is the key you asked for:\n{pem}\nUse it carefully please."
    parts = [text[i : i + 13] for i in range(0, len(text), 13)]
    out, _ = await run_stream(ctx, text_chunks(parts))
    released = released_text(out)
    assert body_line[:10] not in released
    assert released.count("[REDACTED:SECRET]") == 1
    assert released.endswith("Use it carefully please.")


async def test_upstream_failure_releases_vetted_text_then_raises() -> None:
    ctx = setup()
    chunks = text_chunks(
        ["Some safe text that will be released before the failure happens. ", "more", "never"]
    )
    upstream = Upstream(chunks, fail_after=2)
    got: list[ChatChunk] = []
    with pytest.raises(RuntimeError, match="upstream broke"):
        await collect(guard_for(ctx).guard(upstream.gen()), got)
    assert released_text(got) == "Some safe text that will be released before the failure happens. more"


async def test_restore_only_stream_does_not_run_detectors() -> None:
    ctx = setup(vault={"EMAIL": "dana@corp.example"})
    out, _ = await run_stream(ctx, text_chunks([f"cached {KEY} for [EMAIL", "_1] end"]), detect=False)
    assert released_text(out) == f"cached {KEY} for dana@corp.example end"


async def test_shadow_output_guard_only_records(tmp_path: Path) -> None:
    doc = policy_doc()
    doc["output"]["guards"][0]["mode"] = "shadow"
    ctx = setup(write_policy(tmp_path, doc))
    text = f"the key is {KEY} and that is the end of this short message."
    out, _ = await run_stream(ctx, text_chunks([text]))
    assert released_text(out) == text
    assert ctx.output_verdict is not None
    assert ctx.output_verdict.verdict is Verdict.ALLOW
    assert any(getattr(f, "would_verdict", None) is Verdict.REDACT for f in ctx.output_verdict.findings)


async def test_multiple_choices_are_buffered_independently() -> None:
    ctx = setup()
    chunks = [
        chunk("choice zero says hello to everyone in the room today.", role=True, index=0),
        chunk(f"choice one leaks {KEY} right here in the middle.", role=True, index=1),
        chunk(finish="stop", index=0),
        chunk(finish="stop", index=1),
    ]
    out, _ = await run_stream(ctx, chunks)
    by_choice: dict[int, str] = {}
    for c in out:
        for choice in c.choices:
            by_choice[choice.index] = by_choice.get(choice.index, "") + (choice.delta.content or "")
    assert by_choice[0] == "choice zero says hello to everyone in the room today."
    assert by_choice[1] == "choice one leaks [REDACTED:SECRET] right here in the middle."


PROPERTY_TEXT = (
    f"Contact {EMAIL} or [EMAIL_1]. Key {KEY} must stay private. "
    "Card numbers like 4111 1111 1111 1111 are test values too, and the rest is filler text."
)


@settings(max_examples=150, deadline=None)
@given(st.lists(st.integers(min_value=1, max_value=40), min_size=1, max_size=60))
def test_random_chunkings_match_the_non_stream_result_and_never_leak(sizes: list[int]) -> None:
    async def scenario() -> None:
        parts: list[str] = []
        pos = 0
        for size in sizes:
            if pos >= len(PROPERTY_TEXT):
                break
            parts.append(PROPERTY_TEXT[pos : pos + size])
            pos += size
        if pos < len(PROPERTY_TEXT):
            parts.append(PROPERTY_TEXT[pos:])
        ctx = setup(vault={"EMAIL": "dana@corp.example"})
        out, _ = await run_stream(ctx, text_chunks(parts))
        reference_ctx = setup(vault={"EMAIL": "dana@corp.example"})
        policy = reference_ctx.get(EFFECTIVE_POLICY)
        assert policy is not None
        response = ChatResponse(
            id="r",
            created=1,
            model="m",
            choices=(Choice(index=0, message=AssistantMessage(content=PROPERTY_TEXT), finish_reason="stop"),),
        )
        expected = (await OutputGuardRunner(engine()).check(response, reference_ctx, policy)).response
        assert released_text(out) == expected.choices[0].message.content
        for prefix in prefixes(out):
            assert EMAIL[:8] not in prefix
            assert KEY[:6] not in prefix
            assert "[EMA" not in prefix

    asyncio.run(scenario())
