"""tier 2-3 input guards as a read-only probe, concurrent with the router and the semantic cache"""

from gg.core.context import RequestContext
from gg.guardrails import errors
from gg.guardrails.base import GuardContext
from gg.guardrails.engine import Decision, GuardrailEngine
from gg.guardrails.stages import EFFECTIVE_POLICY, INPUT_DECISION
from gg.guardrails.vault import GuardVault
from gg.pipeline.probes import Annotate, ProbeOutcome, Reject

INPUT_PARALLEL = range(2, 4)


class GuardProbe:
    """runs on the normalised, un-redacted segments the pre stage left behind; a block outranks a cache hit"""

    name = "guardrails"
    precedence = 10

    def __init__(self, engine: GuardrailEngine) -> None:
        self._engine = engine

    async def decide(self, ctx: RequestContext, /) -> Decision | None:
        """the tier 2-3 decision, or None when there is nothing to run"""
        policy = ctx.get(EFFECTIVE_POLICY)
        pre = ctx.get(INPUT_DECISION)
        if policy is None or pre is None:
            return None
        chain = policy.input_chain.select(lambda g: g.tier in INPUT_PARALLEL)
        if not chain:
            return None
        vault = GuardVault.adopt(ctx.vault)
        gctx = GuardContext(
            stage="input",
            request_id=ctx.request_id,
            segments=pre.segments,
            request=ctx.request,
            vault=vault,
            key=ctx.key,
            prior=pre.findings,
        )
        deadline_s = policy.doc.input.phase_deadline_ms.parallel / 1000
        return await self._engine.run(chain, gctx, tiers=INPUT_PARALLEL, deadline_s=deadline_s)

    async def __call__(self, ctx: RequestContext, /) -> ProbeOutcome:
        policy = ctx.get(EFFECTIVE_POLICY)
        decision = await self.decide(ctx)
        if policy is None or decision is None:
            return Annotate()
        if decision.blocked:
            messages = policy.doc.response.messages
            if decision.error_closed:
                return Reject(errors.unavailable(ctx.request_id, policy.ref, messages, stage="input"))
            return Reject(errors.blocked(ctx.request_id, policy.ref, messages, stage="input"))
        findings = decision.findings
        if not findings:
            return Annotate()
        return Annotate(apply=lambda c: c.guard_findings.extend(findings))
