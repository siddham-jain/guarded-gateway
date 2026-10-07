import asyncio

import pytest

from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.errors import GuardrailBlockedError, InvalidRequestError
from gg.core.schema import AssistantMessage, ChatResponse, Choice
from gg.pipeline.probes import Annotate, ConcurrentProbesStage, ProbeOutcome, Reject, ShortCircuit
from gg.pipeline.stage import PipelineResult
from tests.conftest import make_ctx


def _result(text: str) -> PipelineResult:
    msg = AssistantMessage(content=text)
    resp = ChatResponse(
        id="x", created=1, model="m", choices=(Choice(index=0, message=msg, finish_reason="stop"),)
    )
    return PipelineResult(source="semantic_cache", response=resp)


class FakeProbe:
    def __init__(
        self,
        name: str,
        precedence: int,
        outcome: ProbeOutcome | Exception,
        *,
        delay: float = 0.0,
        log: list[str] | None = None,
    ) -> None:
        self.name = name
        self.precedence = precedence
        self._outcome = outcome
        self._delay = delay
        self.log = log if log is not None else []
        self.cancelled = False

    async def __call__(self, ctx: RequestContext, /) -> ProbeOutcome:
        try:
            await asyncio.sleep(self._delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def annotate(log: list[str], label: str) -> Annotate:
    return Annotate(lambda ctx: log.append(label))


async def downstream(ctx: RequestContext) -> PipelineResult:
    ctx.response_headers["downstream"] = "yes"
    return _result("upstream")


def hit() -> ShortCircuit:
    return ShortCircuit(lambda ctx: _result("cached"))


async def test_all_annotate_applies_in_precedence_order(clock: FakeClock) -> None:
    applied: list[str] = []
    probes = [
        FakeProbe("router", 90, annotate(applied, "router"), delay=0.0),
        FakeProbe("guard", 10, annotate(applied, "guard"), delay=0.02),
        FakeProbe("cache", 50, annotate(applied, "cache"), delay=0.01),
    ]
    ctx = make_ctx(clock)
    result = await ConcurrentProbesStage(probes, clock)(ctx, downstream)
    assert applied == ["guard", "cache", "router"]
    assert ctx.response_headers["downstream"] == "yes"
    assert result.response is not None
    assert {"probes.guard", "probes.cache", "probes.router"} <= set(ctx.timings.durations)


async def test_guard_reject_outranks_earlier_semantic_hit(clock: FakeClock) -> None:
    guard = FakeProbe("guard", 10, Reject(GuardrailBlockedError("blocked")), delay=0.02)
    cache = FakeProbe("cache", 50, hit(), delay=0.0)
    with pytest.raises(GuardrailBlockedError):
        await ConcurrentProbesStage([cache, guard], clock)(make_ctx(clock), downstream)


async def test_semantic_hit_waits_for_guard_then_cancels_router(clock: FakeClock) -> None:
    applied: list[str] = []
    guard = FakeProbe("guard", 10, annotate(applied, "guard"), delay=0.01)
    cache = FakeProbe("cache", 50, hit(), delay=0.0)
    router = FakeProbe("router", 90, annotate(applied, "router"), delay=5)
    ctx = make_ctx(clock)
    result = await ConcurrentProbesStage([guard, cache, router], clock)(ctx, downstream)
    assert result.source == "semantic_cache"
    assert "downstream" not in ctx.response_headers
    assert router.cancelled
    assert applied == ["guard"]


async def test_guard_reject_cancels_slower_probes_without_waiting(clock: FakeClock) -> None:
    guard = FakeProbe("guard", 10, Reject(GuardrailBlockedError("blocked")), delay=0.0)
    router = FakeProbe("router", 90, Annotate(), delay=5)
    cache = FakeProbe("cache", 50, Annotate(), delay=5)
    async with asyncio.timeout(1):
        with pytest.raises(GuardrailBlockedError):
            await ConcurrentProbesStage([guard, router, cache], clock)(make_ctx(clock), downstream)
    assert router.cancelled
    assert cache.cancelled


async def test_low_precedence_reject_waits_for_authoritative_annotate(clock: FakeClock) -> None:
    guard = FakeProbe("guard", 10, Annotate(), delay=0.02)
    router = FakeProbe("router", 90, Reject(InvalidRequestError("nope")), delay=0.0)
    with pytest.raises(InvalidRequestError):
        await ConcurrentProbesStage([guard, router], clock)(make_ctx(clock), downstream)
    assert not guard.cancelled


async def test_semantic_hit_loses_to_router_reject_only_by_precedence(clock: FakeClock) -> None:
    cache = FakeProbe("cache", 50, hit(), delay=0.02)
    router = FakeProbe("router", 90, Reject(InvalidRequestError("nope")), delay=0.0)
    result = await ConcurrentProbesStage([cache, router], clock)(make_ctx(clock), downstream)
    assert result.source == "semantic_cache"


async def test_probe_exception_fails_closed(clock: FakeClock) -> None:
    guard = FakeProbe("guard", 10, RuntimeError("guard bug"), delay=0.0)
    router = FakeProbe("router", 90, Annotate(), delay=5)
    with pytest.raises(RuntimeError, match="guard bug"):
        await ConcurrentProbesStage([guard, router], clock)(make_ctx(clock), downstream)
    assert router.cancelled


async def test_authoritative_reject_wins_over_later_probe_bug(clock: FakeClock) -> None:
    guard = FakeProbe("guard", 10, Reject(GuardrailBlockedError("blocked")), delay=0.02)
    router = FakeProbe("router", 90, RuntimeError("router bug"), delay=0.0)
    with pytest.raises(GuardrailBlockedError):
        await ConcurrentProbesStage([guard, router], clock)(make_ctx(clock), downstream)


async def test_no_probes_passes_through(clock: FakeClock) -> None:
    ctx = make_ctx(clock)
    await ConcurrentProbesStage([], clock)(ctx, downstream)
    assert ctx.response_headers["downstream"] == "yes"


async def test_outer_cancellation_cancels_probes(clock: FakeClock) -> None:
    router = FakeProbe("router", 90, Annotate(), delay=5)
    stage = ConcurrentProbesStage([router], clock)
    task = asyncio.create_task(stage(make_ctx(clock), downstream))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert router.cancelled


class InstantProbe:
    def __init__(self, name: str, precedence: int) -> None:
        self.name = name
        self.precedence = precedence

    async def __call__(self, ctx: RequestContext, /) -> ProbeOutcome:
        return Annotate()


async def test_probes_that_never_suspend_cost_no_loop_round_trip(clock: FakeClock) -> None:
    stage = ConcurrentProbesStage([InstantProbe("router", 90), InstantProbe("cache", 50)], clock)
    ctx = make_ctx(clock)
    step = stage(ctx, downstream)
    # the whole stage finishes on its first step: no task scheduling, wait or gather hops
    with pytest.raises(StopIteration) as done:
        step.send(None)
    assert isinstance(done.value.value, PipelineResult)
    assert ctx.response_headers["downstream"] == "yes"
