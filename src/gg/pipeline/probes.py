import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import structlog

from gg.core.clock import Clock
from gg.core.context import RequestContext
from gg.core.errors import GGError
from gg.pipeline.stage import Next, PipelineResult

log = structlog.get_logger("gg.pipeline.probes")


@dataclass(frozen=True, slots=True)
class Annotate:
    apply: Callable[[RequestContext], None] | None = None


@dataclass(frozen=True, slots=True)
class ShortCircuit:
    result_factory: Callable[[RequestContext], PipelineResult]


@dataclass(frozen=True, slots=True)
class Reject:
    error: GGError


type ProbeOutcome = Annotate | ShortCircuit | Reject


class Probe(Protocol):
    name: str
    # lower is more authoritative: guards 10, semantic cache 50, router 90
    precedence: int

    async def __call__(self, ctx: RequestContext, /) -> ProbeOutcome: ...


class _Undecided:
    pass


_UNDECIDED = _Undecided()


class ConcurrentProbesStage:
    """runs read-only probes in parallel and lets the most authoritative decisive outcome win.

    a Reject or ShortCircuit only counts once every more authoritative probe has returned Annotate, so a
    semantic hit never bypasses a guard block; probes that can no longer change the result are cancelled.
    """

    name = "probes"

    def __init__(self, probes: Sequence[Probe], clock: Clock) -> None:
        self._probes = tuple(sorted(probes, key=lambda p: p.precedence))
        self._clock = clock

    @property
    def probe_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self._probes)

    async def __call__(self, ctx: RequestContext, call_next: Next, /) -> PipelineResult:
        outcomes = await self._run(ctx)
        for outcome in outcomes:
            if isinstance(outcome, Annotate):
                if outcome.apply is not None:
                    outcome.apply(ctx)
            elif isinstance(outcome, Reject):
                raise outcome.error
            else:
                return outcome.result_factory(ctx)
        return await call_next(ctx)

    async def _run(self, ctx: RequestContext) -> list[ProbeOutcome]:
        loop = asyncio.get_running_loop()
        # eager start: probes that finish without suspending cost no event-loop round trips under load
        tasks = [
            asyncio.Task(self._timed(p, ctx), loop=loop, name=f"probe:{p.name}", eager_start=True)
            for p in self._probes
        ]
        try:
            while isinstance(decided := _decide(tasks), _Undecided):
                pending = {t for t in tasks if not t.done()}
                await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            return decided
        finally:
            running = [t for t in tasks if not t.done()]
            for task in running:
                task.cancel()
            # retrieves every exception so losers never log "exception was never retrieved"
            for task in tasks:
                if task.done() and not task.cancelled():
                    task.exception()
            if running:
                await asyncio.gather(*running, return_exceptions=True)

    async def _timed(self, probe: Probe, ctx: RequestContext) -> ProbeOutcome:
        start = self._clock.monotonic()
        ctx.timings.start(f"probes.{probe.name}", start)
        try:
            return await probe(ctx)
        except Exception:
            log.exception("probe.failed", probe=probe.name)
            raise
        finally:
            ctx.timings.record(f"probes.{probe.name}", self._clock.monotonic() - start)


def _decide(tasks: Sequence[asyncio.Task[ProbeOutcome]]) -> list[ProbeOutcome] | _Undecided:
    """walks probes in precedence order; returns the outcomes up to the decisive one, or all annotations"""
    outcomes: list[ProbeOutcome] = []
    for task in tasks:
        if not task.done():
            return _UNDECIDED
        outcome = task.result()
        outcomes.append(outcome)
        if not isinstance(outcome, Annotate):
            return outcomes
    return outcomes
