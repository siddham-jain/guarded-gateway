import structlog

from gg.core.context import RequestContext
from gg.core.errors import GGError
from gg.core.routing_types import RouteDecision
from gg.pipeline.probes import Annotate, ProbeOutcome, Reject
from gg.providers.base import AliasRef, ModelCatalog
from gg.routing.base import (
    ROUTING_SCORE,
    NullRoutingHooks,
    RoutingHooks,
    RoutingPolicy,
    RoutingRequest,
    RoutingScore,
    RoutingScorer,
)
from gg.routing.config import HeadersConfig
from gg.routing.scorers.null import NullScorer

log = structlog.get_logger("gg.routing")


class RouterProbe:
    """scores router-alias requests in parallel with the other probes and annotates ctx.route"""

    name = "router"
    precedence = 90

    def __init__(
        self,
        catalog: ModelCatalog,
        scorer: RoutingScorer,
        policy: RoutingPolicy,
        headers: HeadersConfig,
        *,
        hooks: RoutingHooks | None = None,
    ) -> None:
        self._catalog = catalog
        self._scorer = scorer
        self._opted_out = NullScorer()
        self._policy = policy
        self._headers = headers
        self._hooks: RoutingHooks = hooks or NullRoutingHooks()

    async def __call__(self, ctx: RequestContext, /) -> ProbeOutcome:
        resolution = self._catalog.resolve(ctx.request.model)
        if not isinstance(resolution, AliasRef) or resolution.kind != "router":
            return Annotate()
        score: RoutingScore | None = None
        try:
            decision = self._policy.presolve(ctx)
            if decision is None:
                scorer = self._opted_out if ctx.key.routing.jev_opt_out else self._scorer
                score = await scorer.score(RoutingRequest.from_context(ctx))
                decision = self._policy.decide(score, ctx)
        except GGError as exc:
            return Reject(exc)
        self._hooks.decided(decision, score)
        log.info(
            "route_decision",
            request_id=ctx.request_id,
            alias=decision.alias,
            tier=decision.tier,
            reason=decision.reason,
            score=decision.score,
            threshold=decision.threshold,
            scorer_version=score.scorer_version if score is not None else None,
            scorer_latency_ms=score.latency_ms if score is not None else None,
            cached=score.cached if score is not None else None,
            policy_version=decision.policy_version,
        )
        return Annotate(lambda c: self._apply(c, decision, score))

    def _apply(self, ctx: RequestContext, decision: RouteDecision, score: RoutingScore | None) -> None:
        ctx.route = decision
        if score is not None:
            ctx.set(ROUTING_SCORE, score)
        digits = self._headers.decimals
        if decision.threshold is not None:
            ctx.response_headers["x-gg-route-threshold"] = f"{decision.threshold:.{digits}f}"
        # only the final scalar is exposed, never jev's answers (no jev passthrough, typesafe mca 2.3(a))
        if decision.score is not None and self._headers.expose_score and ctx.key.routing.expose_score_header:
            ctx.response_headers["x-gg-route-score"] = f"{decision.score:.{digits}f}"
