"""OutputGuardStage wraps everything inner (cache stages, probes, routing, executor)"""

from collections.abc import AsyncIterator
from dataclasses import replace

from gg.core.aio import TaskSupervisor
from gg.core.clock import Clock
from gg.core.context import RequestContext
from gg.core.schema import ChatChunk
from gg.guardrails.base import GuardContext, Mode, Segment
from gg.guardrails.engine import GuardrailEngine
from gg.guardrails.output.common import Finalizer, request_vault
from gg.guardrails.output.runner import OutputGuardRunner, restore_response
from gg.guardrails.output.stream_guard import StreamGuard
from gg.guardrails.policy.effective import EffectivePolicy
from gg.guardrails.stages import EFFECTIVE_POLICY
from gg.pipeline.stage import Next, PipelineResult
from gg.pipeline.streams import StreamAssembler, synthesize_chunks


class OutputGuardStage:
    """windowed checks on streams, full checks on non-stream responses, restore-only for cache hits"""

    name = "guard_out"

    def __init__(
        self, engine: GuardrailEngine, *, clock: Clock, supervisor: TaskSupervisor | None = None
    ) -> None:
        self._engine = engine
        self._clock = clock
        self._supervisor = supervisor
        self._runner = OutputGuardRunner(engine)

    async def __call__(self, ctx: RequestContext, call_next: Next, /) -> PipelineResult:
        result = await call_next(ctx)
        policy = ctx.get(EFFECTIVE_POLICY)
        if policy is None:
            return result
        if result.source != "upstream":
            # cached output passed these guards when written; it only needs this request's own values back
            return self._restore_only(result, ctx, policy)
        if result.stream is not None:
            if self._buffered(ctx, policy):
                return result.map_stream(lambda s: self._buffer(s, ctx, policy))
            guard = StreamGuard(self._engine, policy, ctx, clock=self._clock)
            return result.map_stream(guard.guard)
        if result.response is None:
            return result
        checked = await self._runner.check(result.response, ctx, policy)
        ctx.reply_text = checked.raw_text
        self._schedule_posthoc(ctx, policy, checked.raw_text)
        return replace(result, response=checked.response)

    def _restore_only(
        self, result: PipelineResult, ctx: RequestContext, policy: EffectivePolicy
    ) -> PipelineResult:
        if result.stream is not None:
            guard = StreamGuard(self._engine, policy, ctx, clock=self._clock, detect=False)
            return result.map_stream(guard.guard)
        finalize = Finalizer(policy, ctx.request, request_vault(ctx))
        return result.map_response(lambda r: restore_response(r, finalize))

    @staticmethod
    def _buffered(ctx: RequestContext, policy: EffectivePolicy) -> bool:
        # buffer guards (json_schema) need the whole reply; an applicable one switches to buffer mode
        if policy.doc.output.streaming.mode == "buffer":
            return True
        return any(
            g.guard.streaming == "buffer" and (g.settings.when is None or g.settings.when(ctx.request))
            for g in policy.output_chain.guards
            if g.settings.mode is Mode.ENFORCE
        )

    async def _buffer(
        self, upstream: AsyncIterator[ChatChunk], ctx: RequestContext, policy: EffectivePolicy
    ) -> AsyncIterator[ChatChunk]:
        assembler = StreamAssembler()
        usage: list[ChatChunk] = []
        async for chunk in upstream:
            if chunk.choices:
                assembler.feed(chunk)
            else:
                usage.append(chunk)
        checked = await self._runner.check(assembler.result(), ctx, policy)
        ctx.reply_text = checked.raw_text
        self._schedule_posthoc(ctx, policy, checked.raw_text)
        for chunk in synthesize_chunks(checked.response, include_usage=False):
            yield chunk
        for chunk in usage:
            yield chunk

    def _schedule_posthoc(self, ctx: RequestContext, policy: EffectivePolicy, text: str) -> None:
        # post-hoc guards (moderation, grounding) see placeholder-space text and never change the reply
        chain = policy.posthoc()
        if not chain or self._supervisor is None:
            return
        gctx = GuardContext(
            stage="output",
            request_id=ctx.request_id,
            segments=(Segment(index=0, role="assistant", kind="content", msg=-1, text=text),),
            request=ctx.request,
            vault=request_vault(ctx),
            key=ctx.key,
        )

        async def run() -> None:
            await self._engine.run(chain, gctx)

        self._supervisor.spawn(run(), name=f"guard-posthoc-{ctx.request_id}")
