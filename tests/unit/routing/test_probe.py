from typing import Any

from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.errors import PermissionDeniedError
from gg.core.schema import AssistantMessage, ChatResponse, Choice
from gg.pipeline.probes import Annotate, ConcurrentProbesStage, Reject
from gg.pipeline.stage import PipelineResult
from gg.routing.base import ROUTING_SCORE, RoutingRequest, RoutingScore
from gg.routing.config import HeadersConfig, RoutingConfig
from gg.routing.policy import ThresholdPolicy
from gg.routing.probe import RouterProbe
from gg.routing.session import SessionStore
from gg.routing.stage import RoutingStage
from tests.conftest import make_ctx, make_key, make_request
from tests.unit.routing.support import WEAK, FakeCatalog


class CountingScorer:
    name = "static"
    version = "static:test"

    def __init__(self, value: float) -> None:
        self.value = value
        self.requests: list[RoutingRequest] = []

    async def score(self, req: RoutingRequest, /) -> RoutingScore:
        self.requests.append(req)
        return RoutingScore(
            score=self.value, raw_score=self.value, scorer=self.name, scorer_version=self.version
        )


def make_probe(clock: FakeClock, value: float = 0.8, **headers: Any) -> tuple[RouterProbe, CountingScorer]:
    scorer = CountingScorer(value)
    catalog = FakeCatalog()
    policy = ThresholdPolicy(RoutingConfig(), catalog, SessionStore(10, 60, clock))
    return RouterProbe(catalog, scorer, policy, HeadersConfig(**headers)), scorer


async def run(probe: RouterProbe, ctx: RequestContext) -> None:
    outcome = await probe(ctx)
    assert isinstance(outcome, Annotate)
    if outcome.apply is not None:
        outcome.apply(ctx)


async def terminal(ctx: RequestContext) -> PipelineResult:
    msg = AssistantMessage(content="ok")
    resp = ChatResponse(
        id="x", created=1, model="m", choices=(Choice(index=0, message=msg, finish_reason="stop"),)
    )
    return PipelineResult(source="upstream", response=resp)


async def test_non_router_models_are_left_alone(clock: FakeClock) -> None:
    probe, scorer = make_probe(clock)
    for model in (WEAK.id, "gg/weak", "unknown/model"):
        ctx = make_ctx(clock, make_request(model=model))
        outcome = await probe(ctx)
        assert outcome == Annotate()
    assert scorer.requests == []


async def test_router_alias_is_scored_and_annotated(clock: FakeClock) -> None:
    probe, scorer = make_probe(clock, 0.734)
    ctx = make_ctx(clock)
    await run(probe, ctx)
    assert ctx.route is not None
    assert (ctx.route.tier, ctx.route.reason) == ("strong", "scored")
    assert ctx.response_headers["x-gg-route-score"] == "0.73"
    assert ctx.response_headers["x-gg-route-threshold"] == "0.50"
    score = ctx.get(ROUTING_SCORE)
    assert score is not None
    assert score.score == 0.734
    assert len(scorer.requests) == 1


async def test_scorer_sees_the_scrubbed_request(clock: FakeClock) -> None:
    probe, scorer = make_probe(clock)
    ctx = make_ctx(clock, make_request(messages=[{"role": "user", "content": "mail bob@example.com"}]))
    ctx.scrubbed = make_request(messages=[{"role": "user", "content": "mail [EMAIL_1]"}])
    await run(probe, ctx)
    assert scorer.requests[0].messages[0].content == "mail [EMAIL_1]"


async def test_score_header_respects_key_and_config(clock: FakeClock) -> None:
    probe, _ = make_probe(clock)
    ctx = make_ctx(clock)
    ctx.key = make_key(routing={"expose_score_header": False})
    await run(probe, ctx)
    assert "x-gg-route-score" not in ctx.response_headers
    assert "x-gg-route-threshold" in ctx.response_headers

    hidden, _ = make_probe(clock, expose_score=False)
    ctx = make_ctx(clock)
    await run(hidden, ctx)
    assert "x-gg-route-score" not in ctx.response_headers


async def test_opted_out_key_never_reaches_the_scorer(clock: FakeClock) -> None:
    probe, scorer = make_probe(clock)
    ctx = make_ctx(clock)
    ctx.key = make_key(routing={"jev_opt_out": True})
    await run(probe, ctx)
    assert scorer.requests == []
    assert ctx.route is not None
    assert (ctx.route.tier, ctx.route.reason) == ("strong", "fallback:disabled")
    assert "x-gg-route-score" not in ctx.response_headers


async def test_tool_continuation_skips_the_scorer(clock: FakeClock) -> None:
    probe, scorer = make_probe(clock, 0.1)
    await run(probe, make_ctx(clock, make_request(gg={"session_id": "agent-1"})))
    follow = make_request(
        gg={"session_id": "agent-1"},
        messages=[
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
        ],
    )
    ctx = make_ctx(clock, follow)
    await run(probe, ctx)
    assert len(scorer.requests) == 1
    assert ctx.route is not None
    assert (ctx.route.tier, ctx.route.reason) == ("weak", "session")


async def test_probe_and_routing_stage_together(clock: FakeClock) -> None:
    probe, _ = make_probe(clock, 0.2)
    catalog = FakeCatalog()
    probes = ConcurrentProbesStage([probe], clock)
    routing = RoutingStage(catalog)

    async def after_probes(ctx: RequestContext) -> PipelineResult:
        return await routing(ctx, terminal)

    ctx = make_ctx(clock)
    await probes(ctx, after_probes)
    assert ctx.response_headers["x-gg-route"] == "weak"
    assert ctx.response_headers["x-gg-route-reason"] == "scored"
    assert ctx.route is not None
    assert ctx.route.plan.entries[0].deployment == WEAK
    assert "probes.router" in ctx.timings.durations

    direct = make_ctx(clock, make_request(model=WEAK.id))
    await probes(direct, after_probes)
    assert direct.response_headers["x-gg-route-reason"] == "direct"


async def test_no_deployment_for_the_key_is_a_reject(clock: FakeClock) -> None:
    probe, _ = make_probe(clock)
    ctx = make_ctx(clock)
    ctx.key = make_key(allowed_providers=["nobody"])
    outcome = await probe(ctx)
    assert isinstance(outcome, Reject)
    assert isinstance(outcome.error, PermissionDeniedError)
