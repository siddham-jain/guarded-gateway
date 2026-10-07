from collections.abc import Sequence
from typing import Literal

from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.errors import NotFoundError, PermissionDeniedError
from gg.core.routing_types import PlanEntry, RouteDecision, RoutePlan
from gg.pipeline.stage import Next, PipelineResult
from gg.providers.base import AliasRef, DeploymentRef, ModelCatalog


class RoutingStage:
    """resolves the requested model or alias into a RoutePlan.

    router aliases (gg/auto) use the scorer decision a probe stored on ctx.route when present; without a
    scorer they take the key's fallback route so the gateway still serves traffic.
    """

    name = "routing"

    def __init__(self, catalog: ModelCatalog, *, default_route: Literal["strong", "weak"] = "strong") -> None:
        self._catalog = catalog
        self._default_route = default_route

    async def __call__(self, ctx: RequestContext, call_next: Next, /) -> PipelineResult:
        if ctx.route is None:
            ctx.route = self.decide(ctx)
        ctx.response_headers["x-gg-route"] = ctx.route.tier or ctx.route.alias or ""
        ctx.response_headers["x-gg-route-reason"] = ctx.route.reason
        return await call_next(ctx)

    def decide(self, ctx: RequestContext) -> RouteDecision:
        model = ctx.request.model
        resolution = self._catalog.resolve(model)
        if resolution is None:
            raise NotFoundError(f"The model '{model}' does not exist or is not available.", param="model")
        if isinstance(resolution, DeploymentRef):
            plan = self._plan(model, [resolution.deployment], ctx, tier=None)
            return RouteDecision(alias=None, tier=None, reason="direct", plan=plan)
        return self._decide_alias(resolution, ctx)

    def _decide_alias(self, alias: AliasRef, ctx: RequestContext) -> RouteDecision:
        if alias.kind == "group":
            group = next(iter(self._catalog.groups_of(alias.name)), alias.name)
            chain = self._catalog.chain(group, ctx.request)
            plan = self._plan(alias.name, chain, ctx, tier=group)
            return RouteDecision(alias=alias.name, tier=group, reason="alias", plan=plan)
        tier = ctx.key.routing.fallback_route or self._default_route
        chain = self._catalog.chain(tier, ctx.request)
        plan = self._plan(alias.name, chain, ctx, tier=tier)
        return RouteDecision(alias=alias.name, tier=tier, reason="no_scorer", plan=plan)

    @staticmethod
    def _plan(alias: str, chain: Sequence[Deployment], ctx: RequestContext, *, tier: str | None) -> RoutePlan:
        allowed = ctx.key.allowed_providers
        entries = tuple(
            PlanEntry(deployment=d, tier=tier) for d in chain if allowed is None or d.provider in allowed
        )
        if not entries:
            raise PermissionDeniedError(
                f"No deployment of '{alias}' is available to this key.",
                param="model",
                code="model_not_allowed",
            )
        allow_fallback = ctx.request.gg.fallback if ctx.request.gg is not None else True
        return RoutePlan(alias=alias, entries=entries, allow_fallback=allow_fallback)
