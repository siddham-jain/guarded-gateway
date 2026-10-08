from pathlib import Path
from typing import Any

from gg.core.aio import TaskSupervisor
from gg.core.clock import FakeClock, SystemClock
from gg.core.context import RequestContext
from gg.core.guard_types import Verdict
from gg.core.schema import AssistantMessage, ChatResponse, Choice, FunctionCall, StrictModel, ToolCall
from gg.guardrails.base import GuardContext, GuardFinding
from gg.guardrails.builtin import default_registry
from gg.guardrails.fakes import FakeValues
from gg.guardrails.output.stage import OutputGuardStage
from gg.guardrails.registry import register
from gg.guardrails.stages import EFFECTIVE_POLICY
from gg.guardrails.vault import GuardVault
from gg.pipeline.stage import PipelineResult, ResultSource
from gg.pipeline.streams import ChunkStream
from tests.conftest import make_ctx, make_request
from tests.unit.guardrails.support import (
    FakeGuard,
    Upstream,
    engine,
    policy_doc,
    policy_set,
    released_text,
    text_chunks,
    write_policy,
)

FAKES = FakeValues()
KEY = FAKES.get("github_pat")


def setup(policy_dir: Path | None = None, **request: Any) -> RequestContext:
    ctx = make_ctx(FakeClock(), make_request(**request))
    ctx.vault = GuardVault()
    ctx.vault.add("EMAIL", "dana@corp.example")
    policies = policy_set(policy_dir) if policy_dir else policy_set()
    ctx.set(EFFECTIVE_POLICY, policies.effective(ctx.key, ctx.request))
    return ctx


def response(content: str | None, *, tool_args: str | None = None) -> ChatResponse:
    calls = (
        (ToolCall(id="t1", function=FunctionCall(name="send", arguments=tool_args)),) if tool_args else None
    )
    message = AssistantMessage(content=content, tool_calls=calls)
    return ChatResponse(
        id="r", created=1, model="m", choices=(Choice(index=0, message=message, finish_reason="stop"),)
    )


def returning(result: PipelineResult) -> Any:
    async def call_next(ctx: RequestContext) -> PipelineResult:
        return result

    return call_next


def stage(supervisor: TaskSupervisor | None = None) -> OutputGuardStage:
    return OutputGuardStage(engine(), clock=SystemClock(), supervisor=supervisor)


async def test_non_stream_redacts_leaks_and_restores_own_values() -> None:
    ctx = setup()
    content = f"Mail [EMAIL_1] or eve@corp.example, token {KEY}."
    result = await stage()(ctx, returning(PipelineResult(source="upstream", response=response(content))))
    assert result.response is not None
    out = result.response.choices[0].message.content
    assert out == "Mail dana@corp.example or [REDACTED:EMAIL], token [REDACTED:SECRET]."
    assert ctx.output_verdict is not None
    assert ctx.output_verdict.verdict is Verdict.REDACT
    assert {f.guard for f in ctx.output_verdict.findings} == {"secrets_out", "pii_leak"}


async def test_non_stream_tool_arguments_are_checked_and_json_restored() -> None:
    ctx = setup()
    args = f'{{"to": "[EMAIL_1]", "token": "{KEY}"}}'
    result = await stage()(
        ctx, returning(PipelineResult(source="upstream", response=response(None, tool_args=args)))
    )
    assert result.response is not None
    calls = result.response.choices[0].message.tool_calls
    assert calls is not None
    assert calls[0].function.arguments == '{"to": "dana@corp.example", "token": "[REDACTED:SECRET]"}'


async def test_non_stream_block_returns_content_filter_with_marker(tmp_path: Path) -> None:
    doc = policy_doc()
    doc["output"]["guards"][0]["action"] = "block"
    ctx = setup(write_policy(tmp_path, doc))
    result = await stage()(ctx, returning(PipelineResult(source="upstream", response=response(f"key {KEY}"))))
    assert result.response is not None
    choice = result.response.choices[0]
    assert choice.finish_reason == "content_filter"
    assert choice.message.content is None
    assert choice.message.refusal == "The response was withheld by the gateway's content policy."
    assert (result.response.model_extra or {})["gg_guardrail"]["action"] == "block"
    assert ctx.response_headers["x-gg-guardrails"] == "blocked"


async def test_cache_hits_are_restored_but_not_rechecked() -> None:
    ctx = setup()
    sources: tuple[ResultSource, ...] = ("exact_cache", "semantic_cache")
    for source in sources:
        cached = PipelineResult(source=source, response=response(f"hi [EMAIL_1], {KEY}"))
        result = await stage()(ctx, returning(cached))
        assert result.response is not None
        assert result.response.choices[0].message.content == f"hi dana@corp.example, {KEY}"


async def test_cached_stream_is_restored_only() -> None:
    ctx = setup(stream=True)
    upstream = Upstream(text_chunks(["hi [EMA", "IL_1] bye"]))
    cached = PipelineResult(source="exact_cache", stream=ChunkStream(upstream.gen()))
    result = await stage()(ctx, returning(cached))
    assert result.stream is not None
    out = [c async for c in result.stream]
    assert released_text(out) == "hi dana@corp.example bye"


async def test_upstream_stream_is_windowed_and_inner_stream_closed() -> None:
    ctx = setup(stream=True)
    upstream = Upstream(text_chunks([f"token {KEY} done and some more words to finish the reply."]))
    inner = ChunkStream(upstream.gen())
    result = await stage()(ctx, returning(PipelineResult(source="upstream", stream=inner)))
    assert result.stream is not None
    out = [c async for c in result.stream]
    await result.stream.aclose()
    assert "[REDACTED:SECRET]" in released_text(out)
    assert upstream.closed


async def test_json_response_format_switches_streams_to_buffer_mode() -> None:
    ctx = setup(stream=True, response_format={"type": "json_object"})
    upstream = Upstream(text_chunks(['{"to": "[EMAIL_1]",', ' "ok": tr', "ue}"]))
    result = await stage()(
        ctx, returning(PipelineResult(source="upstream", stream=ChunkStream(upstream.gen())))
    )
    assert result.stream is not None
    out = [c async for c in result.stream]
    assert released_text(out) == '{"to": "dana@corp.example", "ok": true}'
    assert ctx.output_verdict is not None
    assert ctx.output_verdict.verdict is Verdict.ALLOW


async def test_invalid_json_is_flagged_in_buffer_mode() -> None:
    ctx = setup(stream=True, response_format={"type": "json_object"})
    upstream = Upstream(text_chunks(["not json at all"]))
    result = await stage()(
        ctx, returning(PipelineResult(source="upstream", stream=ChunkStream(upstream.gen())))
    )
    assert result.stream is not None
    out = [c async for c in result.stream]
    assert released_text(out) == "not json at all"
    assert ctx.output_verdict is not None
    assert ctx.output_verdict.verdict is Verdict.FLAG


async def test_without_an_effective_policy_the_stage_is_a_passthrough() -> None:
    ctx = make_ctx(FakeClock(), make_request())
    original = PipelineResult(source="upstream", response=response(f"token {KEY}"))
    assert await stage()(ctx, returning(original)) is original


async def test_posthoc_hook_schedules_nothing_without_posthoc_guards(tmp_path: Path) -> None:
    supervisor = TaskSupervisor()
    doc = policy_doc()
    doc["output"]["guards"] = [g for g in doc["output"]["guards"] if g["guard"] != "grounding"]
    ctx = setup(write_policy(tmp_path, doc))
    await stage(supervisor)(ctx, returning(PipelineResult(source="upstream", response=response("fine"))))
    assert len(supervisor) == 0


async def test_posthoc_guards_run_on_the_supervisor_with_placeholder_space_text(tmp_path: Path) -> None:
    seen: list[str] = []

    class Posthoc(FakeGuard):
        async def check(self, gctx: GuardContext, /) -> GuardFinding:
            seen.append(gctx.segments[0].text)
            return await super().check(gctx)

    class NoCfg(StrictModel):
        pass

    registry = default_registry()
    register(
        registry,
        "late_check",
        NoCfg,
        lambda cfg, deps: Posthoc("late_check", tier=5, stage="output", streaming="post_hoc"),
    )
    doc = policy_doc()
    doc["output"]["guards"].append({"guard": "late_check"})
    ctx = make_ctx(FakeClock(), make_request())
    ctx.vault = GuardVault()
    ctx.vault.add("EMAIL", "dana@corp.example")
    ctx.set(
        EFFECTIVE_POLICY,
        policy_set(write_policy(tmp_path, doc), registry=registry).effective(ctx.key, ctx.request),
    )
    supervisor = TaskSupervisor()
    result = await stage(supervisor)(
        ctx, returning(PipelineResult(source="upstream", response=response("hi [EMAIL_1]")))
    )
    await supervisor.drain(1.0)
    assert seen == ["hi [EMAIL_1]"]
    assert result.response is not None
    assert result.response.choices[0].message.content == "hi dana@corp.example"
