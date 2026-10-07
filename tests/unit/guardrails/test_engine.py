import asyncio
import time
from dataclasses import replace

import pytest

from gg.core.aio import CpuSaturatedError, TaskSupervisor
from gg.core.clock import SystemClock
from gg.core.guard_types import Verdict
from gg.guardrails.base import GuardContext, GuardFinding, Mode, OnError, Redaction, Segment, finding
from gg.guardrails.engine import BoundGuard, GuardChain, GuardrailEngine, GuardSettings, sampled
from tests.unit.guardrails.support import FakeGuard, RecordingMetrics, gctx


def bind(
    guard: FakeGuard,
    *,
    mode: Mode = Mode.ENFORCE,
    on_error: OnError = OnError.FLAG,
    timeout_s: float = 1.0,
    sample_rate: float = 1.0,
    when: bool | None = None,
) -> BoundGuard:
    predicate = None if when is None else (lambda _r, w=when: w)
    settings = GuardSettings(guard.name, mode, on_error, timeout_s, sample_rate, predicate)
    return BoundGuard(guard, settings)


def make_engine(**kwargs: object) -> GuardrailEngine:
    return GuardrailEngine(clock=SystemClock(), **kwargs)  # pyright: ignore[reportArgumentType]


async def test_tiers_run_in_order_and_guards_in_a_tier_run_together() -> None:
    a = FakeGuard("a", tier=1, delay=0.05)
    b = FakeGuard("b", tier=1, delay=0.05)
    later = FakeGuard("later", tier=2, verdict=Verdict.FLAG)
    start = time.perf_counter()
    decision = await make_engine().run(GuardChain([bind(later), bind(a), bind(b)]), gctx("x"))
    assert time.perf_counter() - start < 0.09
    assert [f.guard for f in decision.findings] == ["a", "b", "later"]
    assert decision.verdict is Verdict.FLAG


async def test_enforce_block_stops_later_tiers_and_cancels_enforce_siblings() -> None:
    blocker = FakeGuard("blocker", tier=1, verdict=Verdict.BLOCK)
    slow = FakeGuard("slow", tier=1, delay=1.0)
    later = FakeGuard("later", tier=2)
    decision = await make_engine().run(GuardChain([bind(blocker), bind(slow), bind(later)]), gctx("x"))
    assert decision.blocked
    assert not decision.error_closed
    assert slow.cancelled
    assert later.calls == 0


async def test_shadow_siblings_and_followups_keep_running_off_path() -> None:
    supervisor = TaskSupervisor()
    blocker = FakeGuard("blocker", tier=1, verdict=Verdict.BLOCK)
    shadow_sibling = FakeGuard("sibling", tier=1, delay=0.02)
    shadow_later = FakeGuard("later", tier=2)
    chain = GuardChain(
        [bind(blocker), bind(shadow_sibling, mode=Mode.SHADOW), bind(shadow_later, mode=Mode.SHADOW)]
    )
    decision = await make_engine(supervisor=supervisor).run(chain, gctx("x"))
    assert decision.blocked
    await supervisor.drain(1.0)
    assert shadow_sibling.finished
    assert not shadow_sibling.cancelled
    assert shadow_later.calls == 1


async def test_shadow_block_is_recorded_with_its_would_verdict_only() -> None:
    metrics = RecordingMetrics()
    guard = FakeGuard("shadowy", verdict=Verdict.BLOCK)
    decision = await make_engine(metrics=metrics).run(GuardChain([bind(guard, mode=Mode.SHADOW)]), gctx("x"))
    (f,) = decision.findings
    assert (f.verdict, f.would_verdict, f.mode) == (Verdict.ALLOW, Verdict.BLOCK, "shadow")
    assert decision.verdict is Verdict.ALLOW
    assert decision.would_verdict is Verdict.BLOCK
    assert metrics.decisions == [("input", "shadowy", "block", "shadow")]


@pytest.mark.parametrize(
    ("on_error", "verdict"),
    [(OnError.ALLOW, Verdict.ALLOW), (OnError.FLAG, Verdict.FLAG), (OnError.BLOCK, Verdict.BLOCK)],
)
async def test_timeouts_map_through_on_error(on_error: OnError, verdict: Verdict) -> None:
    metrics = RecordingMetrics()
    guard = FakeGuard("slow", delay=1.0)
    chain = GuardChain([bind(guard, on_error=on_error, timeout_s=0.01)])
    decision = await make_engine(metrics=metrics).run(chain, gctx("x"))
    (f,) = decision.findings
    assert f.error == "timeout"
    assert f.verdict is verdict
    assert decision.error_closed is (verdict is Verdict.BLOCK)
    assert metrics.errors == [("input", "slow", "timeout")]


async def test_exceptions_and_overload_fail_closed_only_in_enforce_mode() -> None:
    crash = FakeGuard("crash", error=ValueError("boom"))
    busy = FakeGuard("busy", error=CpuSaturatedError("full"))
    enforce = await make_engine().run(GuardChain([bind(crash, on_error=OnError.BLOCK)]), gctx("x"))
    assert enforce.findings[0].error == "exception"
    assert enforce.error_closed
    overload = await make_engine().run(GuardChain([bind(busy, on_error=OnError.BLOCK)]), gctx("x"))
    assert overload.findings[0].error == "overload"
    shadow = await make_engine().run(
        GuardChain([bind(crash, mode=Mode.SHADOW, on_error=OnError.BLOCK)]), gctx("x")
    )
    assert shadow.verdict is Verdict.ALLOW
    assert shadow.would_verdict is Verdict.BLOCK


async def test_phase_deadline_caps_guard_timeouts() -> None:
    guard = FakeGuard("slow", delay=0.5)
    chain = GuardChain([bind(guard, timeout_s=5.0, on_error=OnError.FLAG)])
    start = time.perf_counter()
    decision = await make_engine().run(chain, gctx("x"), deadline_s=0.02)
    assert time.perf_counter() - start < 0.3
    assert decision.findings[0].error == "timeout"


async def test_sampling_is_deterministic_per_request_and_counted_as_skip() -> None:
    assert sampled("req_a", "g", 0.5) == sampled("req_a", "g", 0.5)
    assert not sampled("req_a", "g", 0.0)
    assert sum(sampled(f"req_{i}", "g", 0.3) for i in range(2000)) in range(480, 720)
    metrics = RecordingMetrics()
    guard = FakeGuard("sampled", verdict=Verdict.BLOCK)
    decision = await make_engine(metrics=metrics).run(GuardChain([bind(guard, sample_rate=0.0)]), gctx("x"))
    assert guard.calls == 0
    assert decision.findings[0].skipped == "sampled_out"
    assert metrics.decisions == [("input", "sampled", "skip", "enforce")]


async def test_when_predicate_skips_guard() -> None:
    guard = FakeGuard("when", verdict=Verdict.BLOCK)
    decision = await make_engine().run(GuardChain([bind(guard, when=False)]), gctx("x"))
    assert guard.calls == 0
    assert decision.findings[0].skipped == "not_applicable"
    assert decision.verdict is Verdict.ALLOW


async def test_transforms_apply_between_tiers_and_never_reach_findings() -> None:
    seen: list[str] = []

    class Reader(FakeGuard):
        async def check(self, gctx: GuardContext, /) -> GuardFinding:
            seen.append(gctx.segments[0].text)
            return await super().check(gctx)

    original = gctx("raw\u200b")
    cleaned = replace(original.segments[0], text="raw")
    transform = FakeGuard("normalize", tier=0, replacements=(cleaned,))
    decision = await make_engine().run(
        GuardChain([bind(transform), bind(Reader("reader", tier=1))]), original
    )
    assert seen == ["raw"]
    assert decision.segments == (cleaned,)
    assert all(not f.replacements for f in decision.findings)


async def test_tier_selection_runs_only_requested_tiers() -> None:
    zero = FakeGuard("zero", tier=0)
    two = FakeGuard("two", tier=2)
    await make_engine().run(GuardChain([bind(zero), bind(two)]), gctx("x"), tiers=range(0, 2))
    assert (zero.calls, two.calls) == (1, 0)


async def test_redactions_split_enforced_and_shadow() -> None:
    class Redactor(FakeGuard):
        async def check(self, gctx: GuardContext, /) -> GuardFinding:
            return finding(self.name, "input", Verdict.REDACT, redactions=(Redaction(0, 0, 1, "EMAIL"),))

    chain = GuardChain([bind(Redactor("on")), bind(Redactor("off"), mode=Mode.SHADOW)])
    decision = await make_engine().run(chain, gctx("x"))
    assert len(decision.redactions(enforced_only=True)) == 1
    assert len(decision.redactions(enforced_only=False)) == 2
    assert decision.verdict is Verdict.REDACT


async def test_cancelled_request_cancels_running_guards() -> None:
    a = FakeGuard("a", delay=1.0)
    b = FakeGuard("b", delay=1.0)
    task = asyncio.create_task(make_engine().run(GuardChain([bind(a), bind(b)]), gctx("x")))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert a.cancelled
    assert b.cancelled


def test_segment_view_falls_back_to_text() -> None:
    seg = Segment(index=0, role="user", kind="content", msg=0, text="t")
    assert seg.view == "t"
    assert replace(seg, inspect="i").view == "i"
