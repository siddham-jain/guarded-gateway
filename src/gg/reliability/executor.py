import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Literal, Protocol

from gg.core.aio import Deadline
from gg.core.clock import Clock
from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.errors import (
    GGError,
    InternalError,
    ProviderError,
    UpstreamError,
    UpstreamTimeoutError,
)
from gg.core.routing_types import PlanEntry, RoutePlan
from gg.core.schema import ChatChunk, ChatRequest
from gg.core.usage import AttemptRecord, UsageRecord, usage_record_from_chunk
from gg.pipeline.stage import PipelineResult
from gg.pipeline.streams import ChunkStream, collect_stream
from gg.reliability.breaker import BreakerRegistry, CircuitBreaker
from gg.reliability.errors import Failure, SkipReason, client_error, error_from_attempts
from gg.reliability.hooks import NullHooks, ReliabilityHooks
from gg.reliability.policy import Action, RetryPolicy, counts_against_breaker

type Sleep = Callable[[float], Awaitable[None]]
type AttemptOutcome = Literal["ok", "retry", "fallback", "fail"]

_OUTCOME: dict[Action, AttemptOutcome] = {
    "retry": "retry",
    "fallback": "fallback",
    "fallback_larger_context": "fallback",
    "fail": "fail",
}


class StreamingAdapter(Protocol):
    """the slice of gg.providers.base.ProviderAdapter the executor needs; adapters match it structurally"""

    def stream(
        self, request: ChatRequest, deployment: Deployment, ctx: RequestContext, /
    ) -> AsyncIterator[ChatChunk]: ...


class Executor:
    """terminal pipeline handler: walks ctx.route.plan with retries, breakers and fallback.

    upstream is always streamed; the first content chunk is awaited under the ttft timeout so anything that
    fails before it can still be retried. after commit, failures surface inside the stream as a GGError.
    """

    def __init__(
        self,
        adapters: Mapping[str, StreamingAdapter],
        breakers: BreakerRegistry,
        policy: RetryPolicy,
        clock: Clock,
        *,
        hooks: ReliabilityHooks | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._adapters = adapters
        self._breakers = breakers
        self._policy = policy
        self._clock = clock
        self._hooks: ReliabilityHooks = hooks or NullHooks()
        self._sleep = sleep

    async def __call__(self, ctx: RequestContext) -> PipelineResult:
        if ctx.route is None:
            raise InternalError("executor reached without a route plan")
        plan = ctx.route.plan
        fallback = plan.allow_fallback and (ctx.request.gg is None or ctx.request.gg.fallback)
        entries = plan.entries if fallback else plan.entries[:1]
        budget_s = max((e.deployment.timeouts.total_s for e in entries), default=0.0)
        try:
            return await self._run(ctx, plan, entries, ctx.deadline.child(budget_s))
        except GGError as e:
            ctx.response_headers["x-gg-attempts"] = str(len(ctx.attempts))
            self._hooks.request_failed(ctx, e)
            raise

    async def _run(
        self, ctx: RequestContext, plan: RoutePlan, entries: Sequence[PlanEntry], deadline: Deadline
    ) -> PipelineResult:
        cfg = self._policy.config
        failures: list[Failure] = []
        need_context = 0
        tried = 0
        for entry in entries:
            dep = entry.deployment
            breaker = self._breakers.get(dep)
            skip = self._skip_reason(entry, plan, breaker, need_context)
            if skip is not None:
                self._skip(ctx, failures, dep, skip, breaker)
                continue
            if tried > cfg.max_hops:
                break
            tried += 1
            adapter = self._adapters[dep.provider]
            retries = 0
            while True:
                if deadline.remaining() < cfg.min_attempt_s:
                    raise error_from_attempts(failures, deadline_hit=True)
                permit = breaker.try_acquire()
                if permit is None:
                    self._skip(ctx, failures, dep, "circuit_open", breaker)
                    break
                started = self._clock.monotonic()
                ctx.timings.mark("upstream_start", started)
                self._policy.on_attempt(dep.id)
                try:
                    result = await self._attempt(ctx, entry, adapter, deadline, started)
                except ProviderError as e:
                    if counts_against_breaker(e):
                        permit.failure()
                    else:
                        permit.neutral()
                    decision = self._policy.decide(e, dep.id, retries, deadline.remaining())
                    if decision.cooldown is not None:
                        self._breakers.force_open(dep, decision.cooldown, scope=e.scope)
                    self._record(ctx, dep, started, _OUTCOME[decision.action], error=e)
                    failures.append(Failure(dep, error=e))
                    if decision.action == "fail":
                        raise client_error(e) from e
                    if decision.action == "retry":
                        retries += 1
                        await self._sleep(decision.delay_s)
                        continue
                    if decision.action == "fallback_larger_context":
                        need_context = max(need_context, dep.capabilities.context + 1)
                    break
                except BaseException:
                    permit.neutral()
                    raise
                permit.success()
                return result
        raise error_from_attempts(failures)

    def _skip_reason(
        self, entry: PlanEntry, plan: RoutePlan, breaker: CircuitBreaker, need_context: int
    ) -> SkipReason | None:
        dep = entry.deployment
        if entry.tier_change and not plan.allow_tier_change:
            return "tier_change"
        if dep.capabilities.context < need_context:
            return "context_window"
        if dep.provider not in self._adapters:
            return "no_adapter"
        if not breaker.available():
            return "circuit_open"
        return None

    def _skip(
        self,
        ctx: RequestContext,
        failures: list[Failure],
        dep: Deployment,
        reason: SkipReason,
        breaker: CircuitBreaker,
    ) -> None:
        retry_in = breaker.retry_in() if reason == "circuit_open" else None
        failures.append(Failure(dep, skip=reason, retry_in_s=retry_in))
        self._hooks.deployment_skipped(ctx, dep, reason)

    async def _attempt(
        self,
        ctx: RequestContext,
        entry: PlanEntry,
        adapter: StreamingAdapter,
        deadline: Deadline,
        started: float,
    ) -> PipelineResult:
        dep = entry.deployment
        # per-entry overrides from the routing policy (e.g. reasoning_effort for the chosen tier)
        request = ctx.request.model_copy(update=dict(entry.overrides)) if entry.overrides else ctx.request
        upstream = ChunkStream(adapter.stream(request, dep, ctx))
        try:
            ttft_s = min(dep.timeouts.ttft_s, deadline.remaining())
            first = await _next_chunk(upstream, ttft_s, dep, code="ttft_timeout")
            if first is None:
                raise ProviderError(
                    "retryable",
                    provider=dep.provider,
                    status=200,
                    code="empty_stream",
                    message="stream ended before any content",
                    deployment_id=dep.id,
                )
        except BaseException:
            await upstream.aclose()
            raise
        first_at = self._clock.monotonic()

        if ctx.request.stream:

            def set_usage(usage: UsageRecord) -> None:
                ctx.usage = usage

            index = self._commit(ctx, dep, started, first_at, first)
            relay = _relay(first, upstream, dep, deadline, set_usage)
            client = self._client_stream(ctx, dep, relay, index)
            return PipelineResult(source="upstream", stream=ChunkStream(client, inner=upstream))

        usages: list[UsageRecord] = []
        relay = _relay(first, upstream, dep, deadline, usages.append)
        response = await collect_stream(ChunkStream(relay, inner=upstream))
        self._commit(ctx, dep, started, first_at, first)
        ctx.usage = usages[-1] if usages else None
        ctx.timings.mark("upstream_end")
        return PipelineResult(source="upstream", response=response)

    def _commit(
        self, ctx: RequestContext, dep: Deployment, started: float, first_at: float, first: ChatChunk
    ) -> int:
        ctx.served_by = dep
        ctx.timings.mark("upstream_first_token", first_at)
        index = len(ctx.attempts)
        request_id = getattr(first.gg_meta, "upstream_request_id", None)
        self._record(
            ctx,
            dep,
            started,
            "ok",
            ttft_s=first_at - started,
            upstream_request_id=request_id if isinstance(request_id, str) else None,
        )
        ctx.response_headers.update(
            {"x-gg-provider": dep.provider, "x-gg-model": dep.id, "x-gg-attempts": str(len(ctx.attempts))}
        )
        return index

    async def _client_stream(
        self, ctx: RequestContext, dep: Deployment, relay: AsyncGenerator[ChatChunk], index: int
    ) -> AsyncIterator[ChatChunk]:
        try:
            async for chunk in relay:
                yield chunk
        except ProviderError as e:
            # committed: the client already has content, so no retry; the sse writer renders an error event
            self._breakers.get(dep).record_late_failure()
            self._hooks.stream_interrupted(ctx, dep, e)
            raise _interrupted(dep, e) from e
        finally:
            await relay.aclose()
            record = ctx.attempts[index]
            ctx.attempts[index] = replace(record, duration_s=self._clock.monotonic() - record.started_at)
            ctx.timings.mark("upstream_end")

    def _record(
        self,
        ctx: RequestContext,
        dep: Deployment,
        started: float,
        outcome: AttemptOutcome,
        *,
        error: ProviderError | None = None,
        ttft_s: float | None = None,
        upstream_request_id: str | None = None,
    ) -> None:
        record = AttemptRecord(
            deployment_id=dep.id,
            provider=dep.provider,
            started_at=started,
            duration_s=self._clock.monotonic() - started,
            outcome=outcome,
            error_kind=error.kind if error else None,
            status=error.status if error else None,
            upstream_request_id=error.upstream_request_id if error else upstream_request_id,
            ttft_s=ttft_s,
        )
        ctx.attempts.append(record)
        self._hooks.attempt_finished(ctx, dep, record, error)


async def _next_chunk(
    stream: ChunkStream, timeout_s: float, dep: Deployment, *, code: str
) -> ChatChunk | None:
    try:
        async with asyncio.timeout(timeout_s):
            return await anext(stream)
    except StopAsyncIteration:
        return None
    except TimeoutError as e:
        raise ProviderError(
            "retryable",
            provider=dep.provider,
            status=504,
            code=code,
            message=f"no chunk within {timeout_s:.2f}s",
            deployment_id=dep.id,
        ) from e


async def _relay(
    first: ChatChunk,
    upstream: ChunkStream,
    dep: Deployment,
    deadline: Deadline,
    on_usage: Callable[[UsageRecord], None],
) -> AsyncGenerator[ChatChunk]:
    try:
        chunk: ChatChunk | None = first
        while chunk is not None:
            if (usage := usage_record_from_chunk(chunk, dep)) is not None:
                on_usage(usage)
            yield chunk
            timeout_s = min(dep.timeouts.inter_chunk_s, deadline.remaining())
            chunk = await _next_chunk(upstream, timeout_s, dep, code="stall_timeout")
    finally:
        await upstream.aclose()


def _interrupted(dep: Deployment, error: ProviderError) -> GGError:
    if error.code == "stall_timeout":
        return UpstreamTimeoutError(
            f"{dep.provider} stream stalled after partial output; output is incomplete"
        )
    return UpstreamError(
        f"{dep.provider} stream interrupted after partial output ({error.code or error.kind}); "
        "output is incomplete",
        code="upstream_stream_error",
    )
