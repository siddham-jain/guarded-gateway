import math
from collections.abc import Callable
from datetime import UTC, datetime

import structlog

from gg.core.context import FinalizerOrder, RequestContext
from gg.core.errors import GGError, RateLimitedError
from gg.core.schema import ChatRequest
from gg.core.usage import TokenEstimates
from gg.limits.base import (
    LIMITS_STATE,
    BudgetExceededError,
    BudgetLedger,
    BudgetUnavailableError,
    Hold,
    LimitReason,
    LimitResult,
    LimitsHooks,
    LimitsState,
    Micros,
    NullLimitsHooks,
    RateLimiter,
    RejectionLimit,
    TokensExceedLimitError,
)
from gg.limits.config import LimitsConfig, budget_caps, usd_to_micros
from gg.limits.cost import CostCalculator
from gg.limits.headers import budget_headers, cost_headers, rate_limit_headers
from gg.pipeline.stage import Next, PipelineResult
from gg.providers.base import ModelCatalog

log = structlog.get_logger("gg.limits")

type PromptEstimator = Callable[[ChatRequest], int]

CACHE_HITS = frozenset({"exact_hit", "semantic_hit"})

_REJECTIONS: dict[LimitReason, RejectionLimit] = {
    "rpm": "rpm",
    "tpm": "tpm",
    "tokens_exceed_limit": "tokens_exceed_limit",
    "concurrency": "concurrency",
}


class LimitsStage:
    """admission right after auth: estimate tokens, take the rate limit, reserve the worst-case budget.

    defers the LIMITS (tpm reconcile + lease release) and BUDGET (settle at the served attempt's cost)
    finalizers; cache hits settle at zero and get their tokens back.
    """

    name = "limits"

    def __init__(
        self,
        config: LimitsConfig,
        *,
        limiter: RateLimiter,
        ledger: BudgetLedger,
        catalog: ModelCatalog,
        costs: CostCalculator,
        estimate_prompt: PromptEstimator,
        hooks: LimitsHooks | None = None,
    ) -> None:
        self._cfg = config
        self._limiter = limiter
        self._ledger = ledger
        self._catalog = catalog
        self._costs = costs
        self._estimate_prompt = estimate_prompt
        self._hooks = hooks or NullLimitsHooks()

    async def __call__(self, ctx: RequestContext, call_next: Next, /) -> PipelineResult:
        estimates = self.estimate(ctx)
        ctx.estimates = estimates
        result = await self._admit(ctx, estimates)
        state = LimitsState(result)
        ctx.set(LIMITS_STATE, state)
        await self._preauthorize(ctx, estimates, state)
        outcome = await call_next(ctx)
        if outcome.response is not None and (cost := self.actual_cost(ctx)) is not None:
            source = (
                ctx.usage.usage_source
                if ctx.usage is not None and ctx.cache_status not in CACHE_HITS
                else None
            )
            ctx.response_headers.update(cost_headers(cost, source))
        return outcome

    def estimate(self, ctx: RequestContext) -> TokenEstimates:
        request = ctx.request
        per_choice = (
            request.max_completion_tokens
            or ctx.key.limits.max_completion_tokens
            or self._cfg.rate_limits.default_output_estimate
        )
        return TokenEstimates(
            prompt_tokens=self._estimate_prompt(request), max_completion_tokens=per_choice * request.n
        )

    async def _admit(self, ctx: RequestContext, estimates: TokenEstimates) -> LimitResult:
        cfg = self._cfg.rate_limits
        need = estimates.prompt_tokens + estimates.max_completion_tokens
        result = await self._limiter.acquire(
            ctx.key.id,
            need,
            limits=cfg.bucket_limits(ctx.key.rate_limits),
            lease_ttl_s=ctx.deadline.remaining() + cfg.lease_grace_s,
        )
        ctx.response_headers.update(rate_limit_headers(result))
        if not result.allowed:
            self._hooks.rejected(_REJECTIONS[result.reason])
            log.info("limits.rejected", key_id=ctx.key.id, reason=result.reason, tokens=need)
            raise self._rate_error(ctx, result, need)
        lease = result.lease
        if lease is not None:

            async def finish() -> None:
                await self._limiter.finish(lease, self.actual_tokens(ctx))

            ctx.finalizers.defer("limits", finish, FinalizerOrder.LIMITS)
        return result

    async def _preauthorize(self, ctx: RequestContext, estimates: TokenEstimates, state: LimitsState) -> None:
        policy = ctx.key.budget
        caps = budget_caps(policy)
        if not caps.enabled:
            return
        amount = self.worst_case(ctx, estimates)
        try:
            hold, status = await self._ledger.preauthorize(
                ctx.key.id,
                amount,
                caps=caps,
                hold_ttl_s=ctx.deadline.remaining() + self._cfg.budgets.hold_grace_s,
            )
        except BudgetExceededError as exc:
            self._hooks.budget_event("hard_limit")
            self._hooks.rejected("budget")
            ctx.response_headers.update(budget_headers(exc.budget, soft_limit_pct=policy.soft_limit_pct))
            raise
        except BudgetUnavailableError:
            if policy.fail_mode == "open":
                self._hooks.budget_event("fail_open")
                log.warning("budget.fail_open", key_id=ctx.key.id)
                return
            self._hooks.budget_event("fail_closed")
            self._hooks.rejected("budget_unavailable")
            raise
        state.hold, state.budget = hold, status
        headers = budget_headers(status, soft_limit_pct=policy.soft_limit_pct)
        if "x-gg-budget-warning" in headers:
            self._hooks.budget_event("soft_limit")
            log.info(
                "budget.soft_limit", key_id=ctx.key.id, period=status.period, ratio=round(status.ratio, 4)
            )
        ctx.response_headers.update(headers)

        async def settle() -> None:
            await self._settle(ctx, hold)

        ctx.finalizers.defer("budget", settle, FinalizerOrder.BUDGET)

    async def _settle(self, ctx: RequestContext, hold: Hold) -> None:
        cost = self.actual_cost(ctx)
        # served but usage never arrived (cut stream): charge the hold, which never under-counts
        await self._ledger.settle(hold, hold.amount if cost is None else cost)

    def worst_case(self, ctx: RequestContext, estimates: TokenEstimates) -> Micros:
        """max over the model's reachable deployments, capped by the key and request per-request caps"""
        deployments = self._catalog.deployments_for(ctx.request.model)
        allowed = [d for d in deployments if ctx.key.allows_deployment(d)] or deployments
        amount = self._costs.estimate_max(
            estimates.prompt_tokens, estimates.max_completion_tokens, allowed, self._received(ctx)
        )
        caps = [ctx.key.budget.max_request_usd]
        if ctx.request.gg is not None:
            caps.append(ctx.request.gg.max_cost_usd)
        return min([amount, *(usd_to_micros(c) for c in caps if c is not None)])

    def actual_cost(self, ctx: RequestContext) -> Micros | None:
        """what the key pays: the served attempt only; None when it was served but usage is unknown"""
        served = ctx.served_by
        if ctx.cache_status in CACHE_HITS or served is None:
            return 0
        if ctx.usage is None:
            return None
        breakdown = self._costs.cost(ctx.usage, served, self._received(ctx))
        return None if breakdown is None else breakdown.total

    @staticmethod
    def actual_tokens(ctx: RequestContext) -> int | None:
        if ctx.cache_status in CACHE_HITS or ctx.served_by is None:
            return 0
        if ctx.usage is None:
            return None
        return ctx.usage.input_tokens + ctx.usage.output_tokens

    @staticmethod
    def _received(ctx: RequestContext) -> datetime:
        return datetime.fromtimestamp(ctx.received_unix, UTC)

    @staticmethod
    def _rate_error(ctx: RequestContext, result: LimitResult, need: int) -> GGError:
        key, limits = ctx.key.id, result.limits
        wait = math.ceil(result.retry_after_s or 0)
        if result.reason == "tokens_exceed_limit":
            return TokensExceedLimitError(
                f"Request too large for key '{key}': {need} tokens requested but the limit is "
                f"{limits.tpm} tokens per min (TPM). Lower max_completion_tokens or shorten the prompt.",
                param="max_completion_tokens",
            )
        if result.reason == "concurrency":
            return RateLimitedError(
                f"Too many concurrent requests for key '{key}': limit {limits.max_concurrent}.",
                code="concurrency_limit_exceeded",
                retry_after_s=max(1, wait),
            )
        if result.reason == "rpm":
            what = f"requests per min (RPM): limit {limits.rpm}"
        else:
            what = f"tokens per min (TPM): limit {limits.tpm}, requested {need}"
        return RateLimitedError(
            f"Rate limit reached for key '{key}' on {what}. "
            f"Please try again in {result.retry_after_s or 0:.1f}s.",
            retry_after_s=max(1, wait),
        )
