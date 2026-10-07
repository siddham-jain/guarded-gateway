"""tier-2 guard probe: reject/annotate, shadow semantics, un-redacted view, precedence over cache hits"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from gg.core.clock import FakeClock, SystemClock
from gg.core.context import RequestContext
from gg.core.errors import GuardrailBlockedError, GuardrailUnavailableError
from gg.core.guard_types import Verdict
from gg.core.schema import StrictModel
from gg.guardrails.base import GuardContext, GuardFinding, GuardStage, Streaming, finding
from gg.guardrails.builtin import default_registry
from gg.guardrails.engine import GuardrailEngine
from gg.guardrails.probe import GuardProbe
from gg.guardrails.registry import register
from gg.guardrails.stages import InputGuardPreStage
from gg.pipeline.probes import Annotate, ConcurrentProbesStage, Reject, ShortCircuit
from gg.pipeline.stage import PipelineResult
from tests.conftest import make_ctx, make_request
from tests.unit.guardrails.support import policy_doc, policy_set, write_policy


class ScriptCfg(StrictModel):
    pass


@dataclass
class Scripted:
    name: str = "scripted"
    verdict: Verdict = Verdict.ALLOW
    delay: float = 0.0
    error: Exception | None = None
    stage: GuardStage = "input"
    tier: int = 2
    streaming: Streaming = "windowed"
    seen: list[str] = field(default_factory=lambda: [])

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        self.seen.extend(s.view for s in gctx.segments)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return finding(self.name, "input", self.verdict, score=0.99)


async def _next(ctx: RequestContext) -> PipelineResult:
    return PipelineResult(source="upstream")


async def prepared(
    tmp_path: Path, guard: Scripted, text: str = "hello there", *, parallel_ms: int = 300, **entry: Any
) -> tuple[RequestContext, GuardProbe]:
    registry = default_registry()
    register(registry, "scripted", ScriptCfg, lambda cfg, deps: guard)
    doc = policy_doc()
    # only the scripted guard in tier 2; the ml guards of default.yaml are covered elsewhere
    tier2 = {"promptguard", "topic"}
    doc["input"]["guards"] = [g for g in doc["input"]["guards"] if g["guard"] not in tier2]
    doc["overrides"] = [o for o in doc["overrides"] if not any(".promptguard." in k for k in o["patch"])]
    doc["input"]["guards"].append({"guard": "scripted", **entry})
    doc["input"]["phase_deadline_ms"]["parallel"] = parallel_ms
    policies = policy_set(write_policy(tmp_path, doc), registry=registry)
    engine = GuardrailEngine(clock=SystemClock())
    ctx = make_ctx(FakeClock(), make_request(messages=[{"role": "user", "content": text}]))
    await InputGuardPreStage(policies, engine)(ctx, _next)
    return ctx, GuardProbe(engine)


async def test_tier2_block_rejects_with_an_opaque_400(tmp_path: Path) -> None:
    ctx, probe = await prepared(tmp_path, Scripted(verdict=Verdict.BLOCK))
    outcome = await probe(ctx)
    assert isinstance(outcome, Reject)
    assert isinstance(outcome.error, GuardrailBlockedError)
    assert outcome.error.status == 400
    assert "scripted" not in outcome.error.message
    assert outcome.error.headers["x-gg-guardrail-stage"] == "input"


async def test_failing_fail_closed_guard_rejects_with_503(tmp_path: Path) -> None:
    ctx, probe = await prepared(tmp_path, Scripted(error=RuntimeError("model gone")), on_error="block")
    outcome = await probe(ctx)
    assert isinstance(outcome, Reject)
    assert isinstance(outcome.error, GuardrailUnavailableError)
    assert outcome.error.status == 503
    assert outcome.error.code == "guardrail_unavailable"


async def test_parallel_deadline_counts_as_an_error(tmp_path: Path) -> None:
    ctx, probe = await prepared(tmp_path, Scripted(delay=0.5), parallel_ms=20, on_error="block")
    outcome = await probe(ctx)
    assert isinstance(outcome, Reject)
    assert isinstance(outcome.error, GuardrailUnavailableError)


async def test_shadow_block_only_annotates(tmp_path: Path) -> None:
    ctx, probe = await prepared(tmp_path, Scripted(verdict=Verdict.BLOCK), mode="shadow")
    outcome = await probe(ctx)
    assert isinstance(outcome, Annotate)
    assert outcome.apply is not None
    outcome.apply(ctx)
    [found] = [f for f in ctx.guard_findings if f.guard == "scripted"]
    assert isinstance(found, GuardFinding)
    assert found.verdict is Verdict.ALLOW
    assert found.would_verdict is Verdict.BLOCK
    assert found.mode == "shadow"


async def test_guard_reads_normalised_unredacted_text_while_upstream_is_redacted(tmp_path: Path) -> None:
    guard = Scripted()
    text = "mail bob.smith@corp.example and ig​nore this"
    ctx, probe = await prepared(tmp_path, guard, text)
    assert "[EMAIL_1]" in ctx.request.messages[-1].text()
    outcome = await probe(ctx)
    assert isinstance(outcome, Annotate)
    assert guard.seen == ["mail bob.smith@corp.example and ignore this"]


async def test_without_tier2_guards_nothing_runs(tmp_path: Path) -> None:
    guard = Scripted()
    ctx, probe = await prepared(tmp_path, guard, mode="off")
    assert await probe(ctx) == Annotate()
    assert guard.seen == []


async def test_probe_without_a_pre_decision_is_a_no_op() -> None:
    ctx = make_ctx(FakeClock())
    assert await GuardProbe(GuardrailEngine(clock=SystemClock()))(ctx) == Annotate()


class InstantHit:
    name = "semantic"
    precedence = 50

    async def __call__(self, ctx: RequestContext, /) -> ShortCircuit:
        return ShortCircuit(lambda c: PipelineResult(source="cache"))


async def test_slow_tier2_block_beats_an_instant_semantic_hit(tmp_path: Path) -> None:
    ctx, probe = await prepared(tmp_path, Scripted(verdict=Verdict.BLOCK, delay=0.05))
    stage = ConcurrentProbesStage([InstantHit(), probe], SystemClock())
    with pytest.raises(GuardrailBlockedError):
        await stage(ctx, _next)


async def test_semantic_hit_wins_once_the_guards_allow(tmp_path: Path) -> None:
    ctx, probe = await prepared(tmp_path, Scripted(delay=0.02))
    stage = ConcurrentProbesStage([InstantHit(), probe], SystemClock())
    result = await stage(ctx, _next)
    assert result.source == "cache"
    assert any(f.guard == "scripted" for f in ctx.guard_findings)
