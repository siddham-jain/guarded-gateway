import asyncio
import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

import structlog

from gg.core.aio import CpuSaturatedError, TaskSupervisor
from gg.core.clock import Clock
from gg.core.guard_types import Verdict
from gg.guardrails.base import (
    ON_ERROR_VERDICT,
    ErrorKind,
    GuardBackendError,
    GuardContext,
    GuardFinding,
    GuardMetrics,
    Guardrail,
    Mode,
    NullGuardMetrics,
    OnError,
    Redaction,
    RequestPredicate,
    Segment,
    SkipReason,
)

log = structlog.get_logger("gg.guardrails")

INPUT_PRE = range(0, 2)
OUTPUT_DETECT = range(0, 9)


@dataclass(frozen=True, slots=True)
class GuardSettings:
    name: str
    mode: Mode
    on_error: OnError
    timeout_s: float
    sample_rate: float = 1.0
    when: RequestPredicate | None = None


@dataclass(frozen=True, slots=True)
class BoundGuard:
    guard: Guardrail
    settings: GuardSettings

    @property
    def name(self) -> str:
        return self.settings.name

    @property
    def tier(self) -> int:
        return self.guard.tier


class GuardChain:
    """guards of one stage in policy (yaml) order; mode=off guards are dropped when the chain is built"""

    def __init__(self, guards: Iterable[BoundGuard] = ()) -> None:
        self.guards = tuple(guards)

    def tier(self, n: int) -> tuple[BoundGuard, ...]:
        return tuple(g for g in self.guards if g.tier == n)

    def tiers(self) -> list[int]:
        return sorted({g.tier for g in self.guards})

    def select(self, keep: Callable[[BoundGuard], bool]) -> "GuardChain":
        return GuardChain(g for g in self.guards if keep(g))

    def get(self, name: str) -> BoundGuard | None:
        return next((g for g in self.guards if g.name == name), None)

    def __bool__(self) -> bool:
        return bool(self.guards)

    def __len__(self) -> int:
        return len(self.guards)


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    would_verdict: Verdict
    findings: tuple[GuardFinding, ...]
    segments: tuple[Segment, ...]

    @property
    def blocked(self) -> bool:
        return self.verdict is Verdict.BLOCK

    @property
    def error_closed(self) -> bool:
        # a block caused only by a guard that could not decide (on_error=block) -> 503, not 400
        blocks = [f for f in self.findings if f.verdict is Verdict.BLOCK]
        return bool(blocks) and all(f.error_closed for f in blocks)

    def redactions(self, *, enforced_only: bool) -> list[Redaction]:
        return [
            r
            for f in self.findings
            if f.error is None
            and (f.verdict is Verdict.REDACT or (not enforced_only and f.would_verdict is Verdict.REDACT))
            for r in f.redactions
        ]

    @classmethod
    def combine(cls, findings: Sequence[GuardFinding], segments: tuple[Segment, ...]) -> "Decision":
        return cls(
            verdict=max((f.verdict for f in findings), default=Verdict.ALLOW),
            would_verdict=max((f.would_verdict for f in findings), default=Verdict.ALLOW),
            # transformed text stays in the segments; findings are logged and must never carry it
            findings=tuple(replace(f, replacements=()) if f.replacements else f for f in findings),
            segments=segments,
        )


def sampled(request_id: str, guard: str, rate: float) -> bool:
    # deterministic per request so a rerun samples identically
    digest = hashlib.sha256(f"{request_id}:{guard}".encode()).digest()
    return int.from_bytes(digest[:4]) / 2**32 < rate


class GuardrailEngine:
    """runs tiers in order and the guards of one tier concurrently; an enforce block ends the run"""

    def __init__(
        self,
        *,
        clock: Clock,
        metrics: GuardMetrics | None = None,
        supervisor: TaskSupervisor | None = None,
    ) -> None:
        self._clock = clock
        self.metrics: GuardMetrics = metrics or NullGuardMetrics()
        self._supervisor = supervisor

    async def run(
        self,
        chain: GuardChain,
        gctx: GuardContext,
        *,
        tiers: range | None = None,
        deadline_s: float | None = None,
    ) -> Decision:
        phase_end = None if deadline_s is None else self._clock.monotonic() + deadline_s
        findings: list[GuardFinding] = []
        segments = gctx.segments
        selected = [t for t in chain.tiers() if tiers is None or t in tiers]
        for i, tier in enumerate(selected):
            ctx = replace(gctx, segments=segments, prior=tuple(findings))
            tier_findings = await self._run_tier(chain.tier(tier), ctx, phase_end)
            findings.extend(tier_findings)
            if any(f.verdict is Verdict.BLOCK for f in tier_findings):
                self._shadow_followups(chain, selected[i + 1 :], ctx)
                break
            segments = _apply_replacements(segments, tier_findings)
        return Decision.combine(findings, segments)

    async def _run_tier(
        self, guards: Sequence[BoundGuard], ctx: GuardContext, phase_end: float | None
    ) -> list[GuardFinding]:
        if len(guards) == 1:
            return [await self._run_one(guards[0], ctx, phase_end)]
        loop = asyncio.get_running_loop()
        # eager start: cpu-only guards finish before their first await and skip a scheduler round trip
        tasks = {
            asyncio.Task(
                self._run_one(g, ctx, phase_end), loop=loop, name=f"guard-{g.name}", eager_start=True
            ): g
            for g in guards
        }
        done_findings: list[GuardFinding] = []
        pending: set[asyncio.Task[GuardFinding]] = set(tasks)
        try:
            while pending:
                done = {t for t in pending if t.done()}
                if not done:
                    done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                pending -= done
                done_findings.extend(t.result() for t in done)
                if any(f.verdict is Verdict.BLOCK for f in done_findings):
                    break
        finally:
            cancelled = [t for t in pending if not (tasks[t].settings.mode is Mode.SHADOW and self._adopt(t))]
            for task in cancelled:
                task.cancel()
            if cancelled:
                await asyncio.gather(*cancelled, return_exceptions=True)
        # keep policy order so transforms and reports are deterministic
        order = {g.name: i for i, g in enumerate(guards)}
        return sorted(done_findings, key=lambda f: order.get(f.guard, 0))

    def _adopt(self, task: asyncio.Task[GuardFinding]) -> bool:
        if self._supervisor is None:
            return False

        async def finish() -> None:
            await asyncio.wait((task,))

        return self._supervisor.spawn(finish(), name=f"{task.get_name()}-offpath")

    def _shadow_followups(self, chain: GuardChain, tiers: Sequence[int], ctx: GuardContext) -> None:
        # shadow guards in tiers skipped by a block still run, off the critical path, so they are measured
        shadow = [g for t in tiers for g in chain.tier(t) if g.settings.mode is Mode.SHADOW]
        if not shadow or self._supervisor is None:
            return

        async def followup() -> None:
            await asyncio.gather(*(self._run_one(g, ctx, None) for g in shadow))

        self._supervisor.spawn(followup(), name="guard-shadow-followups")

    async def _run_one(self, bound: BoundGuard, ctx: GuardContext, phase_end: float | None) -> GuardFinding:
        s = bound.settings
        guard = bound.guard
        if s.when is not None and not s.when(ctx.request):
            return self._skip(bound, ctx, "not_applicable")
        if s.sample_rate < 1.0 and not sampled(ctx.request_id, s.name, s.sample_rate):
            return self._skip(bound, ctx, "sampled_out")
        timeout = s.timeout_s
        if phase_end is not None:
            timeout = max(0.001, min(timeout, phase_end - self._clock.monotonic()))
        start = self._clock.monotonic()
        error: ErrorKind | None = None
        raw: GuardFinding | None = None
        try:
            async with asyncio.timeout(timeout):
                raw = await guard.check(ctx)
        except TimeoutError:
            error = "timeout"
        except CpuSaturatedError:
            error = "overload"
        except GuardBackendError as exc:
            log.warning("guardrail.backend_failed", guard=s.name, stage=ctx.stage, error=str(exc))
            error = "exception"
        except Exception:
            log.exception("guardrail.failed", guard=s.name, stage=ctx.stage)
            error = "exception"
        elapsed = self._clock.monotonic() - start
        if raw is None or error is not None:
            error = error or "exception"
            self.metrics.error(ctx.stage, s.name, error)
            raw = GuardFinding(
                guard=s.name, stage=ctx.stage, verdict=ON_ERROR_VERDICT[s.on_error], score=None, reason=error
            )
        detected = raw.verdict
        # a shadow guard never changes the request, even when it errors with on_error=block
        effective = detected if s.mode is Mode.ENFORCE else Verdict.ALLOW
        self.metrics.decision(ctx.stage, s.name, detected.name.lower(), s.mode.value)
        self.metrics.duration(ctx.stage, s.name, elapsed)
        return replace(
            raw,
            guard=s.name,
            stage=ctx.stage,
            verdict=effective,
            would_verdict=detected,
            mode="enforce" if s.mode is Mode.ENFORCE else "shadow",
            tier=guard.tier,
            latency_ms=elapsed * 1000,
            error=error,
        )

    def _skip(self, bound: BoundGuard, ctx: GuardContext, reason: SkipReason) -> GuardFinding:
        s = bound.settings
        self.metrics.decision(ctx.stage, s.name, "skip", s.mode.value)
        return GuardFinding(
            guard=s.name,
            stage=ctx.stage,
            verdict=Verdict.ALLOW,
            score=None,
            reason=reason,
            mode="enforce" if s.mode is Mode.ENFORCE else "shadow",
            tier=bound.tier,
            skipped=reason,
        )


def _apply_replacements(
    segments: tuple[Segment, ...], findings: Iterable[GuardFinding]
) -> tuple[Segment, ...]:
    # transforms (normalize) apply in policy order regardless of mode; mode only gates their verdict
    by_index = {seg.index: seg for f in findings if f.error is None for seg in f.replacements}
    if not by_index:
        return segments
    return tuple(by_index.get(seg.index, seg) for seg in segments)
