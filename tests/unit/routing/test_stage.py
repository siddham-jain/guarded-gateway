from collections.abc import Callable, Sequence

import pytest

from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.errors import NotFoundError, PermissionDeniedError
from gg.core.schema import AssistantMessage, ChatRequest, ChatResponse, Choice
from gg.pipeline.stage import PipelineResult
from gg.providers.base import AliasRef, DeploymentRef, PublicModel, Resolution
from gg.routing.stage import RoutingStage
from tests.conftest import make_ctx, make_key, make_request

WEAK = Deployment(id="openai/gpt-6-luna", provider="openai", upstream_model="gpt-6-luna")
STRONG_A = Deployment(id="openai/gpt-6.1-sol", provider="openai", upstream_model="gpt-6.1-sol")
STRONG_B = Deployment(id="deepseek/deepseek-v4", provider="deepseek", upstream_model="deepseek-v4")


class FakeCatalog:
    hash = "test"

    def resolve(self, model: str, /) -> Resolution | None:
        if model == "gg/auto":
            return AliasRef("gg/auto", "router")
        if model in ("weak", "strong"):
            return AliasRef(model, "group")
        if model == WEAK.id:
            return DeploymentRef(WEAK)
        return None

    def get(self, deployment_id: str, /) -> Deployment:
        raise KeyError(deployment_id)

    def chain(self, group: str, request: ChatRequest, /) -> list[Deployment]:
        return [WEAK] if group == "weak" else [STRONG_A, STRONG_B]

    def groups_of(self, alias: str, /) -> tuple[str, ...]:
        return (alias,) if alias in ("weak", "strong") else ()

    def deployments_for(self, model: str, /) -> list[Deployment]:
        return []

    def list_public(self, allowed: Callable[[str], bool], /) -> Sequence[PublicModel]:
        return []


async def terminal(ctx: RequestContext) -> PipelineResult:
    msg = AssistantMessage(content="ok")
    resp = ChatResponse(
        id="x", created=1, model="m", choices=(Choice(index=0, message=msg, finish_reason="stop"),)
    )
    return PipelineResult(source="upstream", response=resp)


async def test_direct_deployment(clock: FakeClock) -> None:
    ctx = make_ctx(clock, make_request(model=WEAK.id))
    await RoutingStage(FakeCatalog())(ctx, terminal)
    assert ctx.route is not None
    assert ctx.route.reason == "direct"
    assert [e.deployment for e in ctx.route.plan.entries] == [WEAK]


async def test_router_alias_without_scorer_uses_fallback_route(clock: FakeClock) -> None:
    ctx = make_ctx(clock)
    await RoutingStage(FakeCatalog())(ctx, terminal)
    assert ctx.route is not None
    assert ctx.route.tier == "strong"
    assert ctx.route.reason == "no_scorer"
    assert ctx.response_headers["x-gg-route"] == "strong"


async def test_key_fallback_route_and_provider_allow_list(clock: FakeClock) -> None:
    ctx = make_ctx(clock)
    ctx.key = make_key(routing={"fallback_route": "strong"}, allowed_providers=["deepseek"])
    await RoutingStage(FakeCatalog())(ctx, terminal)
    assert ctx.route is not None
    assert [e.deployment for e in ctx.route.plan.entries] == [STRONG_B]


async def test_fallback_flag_is_carried_into_the_plan(clock: FakeClock) -> None:
    ctx = make_ctx(clock, make_request(model="weak", gg={"fallback": False}))
    await RoutingStage(FakeCatalog())(ctx, terminal)
    assert ctx.route is not None
    assert not ctx.route.plan.allow_fallback


async def test_unknown_model_and_fully_filtered_plan(clock: FakeClock) -> None:
    with pytest.raises(NotFoundError):
        await RoutingStage(FakeCatalog())(make_ctx(clock, make_request(model="nope")), terminal)
    ctx = make_ctx(clock, make_request(model="weak"))
    ctx.key = make_key(allowed_providers=["anthropic"])
    with pytest.raises(PermissionDeniedError):
        await RoutingStage(FakeCatalog())(ctx, terminal)
