"""pipeline integration of the input pre phase (tiers 0/1)"""

from dataclasses import dataclass

from gg.config.hashing import combined_hash
from gg.core.cache_types import CacheState
from gg.core.context import ContextKey, RequestContext
from gg.core.keypolicy import KeyPolicy
from gg.core.schema import ChatRequest
from gg.guardrails import errors
from gg.guardrails.base import POLICY_REF, GuardContext
from gg.guardrails.engine import INPUT_PRE, Decision, GuardrailEngine
from gg.guardrails.policy.effective import EffectivePolicy
from gg.guardrails.policy.loader import PolicySet
from gg.guardrails.redact import redact_views
from gg.guardrails.segments import extract_segments, with_placeholder_hint, write_back
from gg.guardrails.vault import GuardVault
from gg.pipeline.stage import Next, PipelineResult

EFFECTIVE_POLICY = ContextKey[EffectivePolicy]("guardrails.effective_policy")
INPUT_DECISION = ContextKey[Decision]("guardrails.input_decision")

CACHE_BYPASS_REASON = "unredacted_upstream"


@dataclass(frozen=True, slots=True)
class InputResult:
    decision: Decision
    # what goes upstream: enforce redactions only
    upstream: ChatRequest
    # every redaction, shadow ones too: the only form the router and the caches may see
    scrubbed: ChatRequest
    redactions: int
    bypass_cache: bool


async def run_input(
    engine: GuardrailEngine,
    policy: EffectivePolicy,
    request: ChatRequest,
    vault: GuardVault,
    *,
    request_id: str,
    key: KeyPolicy | None = None,
) -> InputResult:
    segments = extract_segments(request)
    vault.reserve(s.text for s in segments)
    gctx = GuardContext(
        stage="input", request_id=request_id, segments=segments, request=request, vault=vault, key=key
    )
    deadline_s = policy.doc.input.phase_deadline_ms.pre / 1000
    decision = await engine.run(policy.input_chain, gctx, tiers=INPUT_PRE, deadline_s=deadline_s)
    if decision.blocked:
        return InputResult(decision, request, request, 0, bypass_cache=False)
    enforced = decision.redactions(enforced_only=True)
    every = decision.redactions(enforced_only=False)
    upstream_view, scrubbed_view = redact_views(decision.segments, enforced, every, vault)
    upstream = write_back(request, segments, upstream_view)
    scrubbed = write_back(request, segments, scrubbed_view)
    if policy.doc.input.placeholder_hint:
        upstream = with_placeholder_hint(upstream) if enforced else upstream
        scrubbed = with_placeholder_hint(scrubbed) if every else scrubbed
    engine.metrics.placeholders("redacted", "none", len(vault))
    return InputResult(
        decision, upstream, scrubbed, len(enforced), bypass_cache=upstream_view != scrubbed_view
    )


class InputGuardPreStage:
    """tier 0/1 before the exact cache and the router: blocks, or forwards the redacted request"""

    name = "guard_in_pre"

    def __init__(self, policies: PolicySet, engine: GuardrailEngine) -> None:
        self._policies = policies
        self._engine = engine

    async def __call__(self, ctx: RequestContext, call_next: Next, /) -> PipelineResult:
        policy = self._policies.effective(ctx.key, ctx.request)
        ctx.set(EFFECTIVE_POLICY, policy)
        ctx.set(POLICY_REF, policy.ref)
        ctx.config_hash = combined_hash({"config": ctx.config_hash, "policy": policy.hash})
        ctx.response_headers["x-gg-policy"] = policy.ref.header()
        vault = GuardVault.adopt(ctx.vault)
        ctx.vault = vault

        result = await run_input(
            self._engine, policy, ctx.request, vault, request_id=ctx.request_id, key=ctx.key
        )
        decision = result.decision
        ctx.set(INPUT_DECISION, decision)
        ctx.guard_findings.extend(decision.findings)
        if decision.blocked:
            messages = policy.doc.response.messages
            if decision.error_closed:
                raise errors.unavailable(ctx.request_id, policy.ref, messages, stage="input")
            raise errors.blocked(ctx.request_id, policy.ref, messages, stage="input")

        ctx.request = result.upstream
        ctx.scrubbed = result.scrubbed
        if result.bypass_cache:
            # shadow-mode pii went upstream raw: neither cache may key or store this request
            ctx.cache_status = "bypass"
            if ctx.cache is None:
                ctx.cache = CacheState()
            ctx.cache.bypass_reason = CACHE_BYPASS_REASON
            ctx.cache.store = False
        ctx.response_headers["x-gg-guardrails"] = "redacted" if result.redactions else "pass"
        if result.redactions:
            ctx.response_headers["x-gg-redactions"] = str(result.redactions)
        return await call_next(ctx)
