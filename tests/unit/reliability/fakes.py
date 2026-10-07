import asyncio
import random
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.deployment import Capabilities, Deployment, Timeouts
from gg.core.errors import ProviderError, ProviderErrorKind
from gg.core.routing_types import PlanEntry, RouteDecision, RoutePlan
from gg.core.schema import ChatChunk, ChatRequest, ChunkChoice, Delta, Usage
from gg.reliability.breaker import BreakerListener, BreakerRegistry, BreakerStateName
from gg.reliability.executor import Executor
from gg.reliability.policy import RetryConfig, RetryPolicy
from tests.conftest import make_ctx, make_request

USAGE = Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)


@dataclass
class Step:
    """one scripted upstream call: optional stall, content chunks, then an error or a finish + usage chunk"""

    chunks: tuple[str, ...] = ("hello", " world")
    error: ProviderError | None = None
    stall_before_s: float = 0.0
    stall_after_s: float = 0.0
    finish: bool = True
    usage: Usage | None = USAGE


def ok(*chunks: str) -> Step:
    return Step(chunks=chunks or ("hello", " world"))


def err(
    kind: ProviderErrorKind,
    *,
    status: int = 500,
    code: str | None = None,
    retry_after_s: float | None = None,
    quota_reset_at: datetime | None = None,
    scope: Any = "deployment",
    provider: str = "fake",
) -> ProviderError:
    return ProviderError(
        kind,
        provider=provider,
        status=status,
        code=code,
        retry_after_s=retry_after_s,
        quota_reset_at=quota_reset_at,
        scope=scope,
        message="scripted failure",
    )


def fail(kind: ProviderErrorKind, *, after: tuple[str, ...] = (), **kwargs: Any) -> Step:
    return Step(chunks=after, error=err(kind, **kwargs))


def overloaded() -> Step:
    return fail("fallback", status=529, code="overloaded")


def stall(seconds: float = 5.0) -> Step:
    return Step(stall_before_s=seconds)


def _chunk(dep: Deployment, delta: Delta, finish: Any = None, usage: Usage | None = None) -> ChatChunk:
    choices = () if usage is not None else (ChunkChoice(index=0, delta=delta, finish_reason=finish),)
    return ChatChunk(
        id=f"chatcmpl-{dep.id}", created=1, model=dep.upstream_model, choices=choices, usage=usage
    )


class FakeAdapter:
    """scripted ProviderAdapter: pops one Step per stream() call, then falls back to next_step()"""

    def __init__(self, name: str, steps: Iterable[Step] = (), next_step: Callable[[], Step] = ok) -> None:
        self.name = name
        self.steps = deque(steps)
        self.next_step = next_step
        self.calls = 0
        self.closed = 0
        self.cancelled = 0
        self.requests: list[ChatRequest] = []

    def stream(
        self, request: ChatRequest, deployment: Deployment, ctx: RequestContext, /
    ) -> AsyncIterator[ChatChunk]:
        self.calls += 1
        self.requests.append(request)
        step = self.steps.popleft() if self.steps else self.next_step()
        return self._run(step, deployment)

    async def _run(self, step: Step, dep: Deployment) -> AsyncIterator[ChatChunk]:
        try:
            if step.stall_before_s:
                await asyncio.sleep(step.stall_before_s)
            for i, text in enumerate(step.chunks):
                yield _chunk(dep, Delta(role="assistant" if i == 0 else None, content=text))
            if step.stall_after_s:
                await asyncio.sleep(step.stall_after_s)
            if step.error is not None:
                step.error.committed = bool(step.chunks)
                raise step.error
            if step.finish:
                yield _chunk(dep, Delta(), finish="stop")
            if step.usage is not None:
                yield _chunk(dep, Delta(), usage=step.usage)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.closed += 1

    async def aclose(self) -> None:
        return None


class FakeSleep:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)
        await asyncio.sleep(0)


class TransitionLog(BreakerListener):
    def __init__(self) -> None:
        self.events: list[tuple[str, BreakerStateName, BreakerStateName, str]] = []

    def breaker_transition(
        self, deployment_id: str, old: BreakerStateName, new: BreakerStateName, reason: str, /
    ) -> None:
        self.events.append((deployment_id, old, new, reason))


def deployment(
    dep_id: str,
    *,
    provider: str | None = None,
    context: int = 128_000,
    ttft_s: float = 1.0,
    inter_chunk_s: float = 1.0,
) -> Deployment:
    return Deployment(
        id=dep_id,
        provider=provider or dep_id.split("/")[0],
        upstream_model=dep_id.split("/")[-1],
        capabilities=Capabilities(context=context),
        timeouts=Timeouts(ttft_s=ttft_s, inter_chunk_s=inter_chunk_s),
    )


def context_for(
    clock: FakeClock,
    deployments: Iterable[Deployment],
    *,
    stream: bool = False,
    allow_fallback: bool = True,
    tier_change: frozenset[str] = frozenset(),
    **request: Any,
) -> RequestContext:
    ctx = make_ctx(clock, make_request(stream=stream, **request))
    entries = tuple(PlanEntry(deployment=d, tier_change=d.id in tier_change) for d in deployments)
    plan = RoutePlan(alias="gg/test", entries=entries, allow_fallback=allow_fallback)
    ctx.route = RouteDecision(alias="gg/test", tier=None, reason="test", plan=plan)
    return ctx


@dataclass
class Harness:
    clock: FakeClock
    adapters: dict[str, FakeAdapter]
    config: RetryConfig = field(default_factory=RetryConfig)
    transitions: TransitionLog = field(default_factory=TransitionLog)
    seed: int = 7

    def __post_init__(self) -> None:
        self.sleep = FakeSleep(self.clock)
        self.breakers = BreakerRegistry(self.clock, listener=self.transitions)
        self.policy = RetryPolicy(self.clock, self.config, rng=random.Random(self.seed))  # noqa: S311
        self.executor = Executor(self.adapters, self.breakers, self.policy, self.clock, sleep=self.sleep)


async def drain(stream: AsyncIterator[ChatChunk]) -> str:
    text: list[str] = []
    async for chunk in stream:
        text.extend(c.delta.content or "" for c in chunk.choices)
    return "".join(text)
