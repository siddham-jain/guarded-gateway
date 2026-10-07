import asyncio
from collections.abc import Mapping
from typing import Any

import httpx2

from gg.core.clock import FakeClock
from gg.core.routing_types import RouteDecision
from gg.routing.base import RoutingRequest, RoutingScore
from gg.routing.decorators import CachingScorer, DeadlineScorer, MetricsScorer
from gg.routing.scorers.null import NullScorer, StaticScorer
from gg.routing.ttl_cache import TTLCache
from tests.unit.routing.support import fixture, json_response, make_client, make_scorer, routing_request, user

REQ = routing_request(user("Prove there are infinitely many primes of the form 4k+3."))


class Slow:
    name = "slow"
    version = "slow:1"

    async def score(self, req: RoutingRequest, /) -> RoutingScore:
        await asyncio.sleep(5)
        raise AssertionError("unreachable")


class Broken:
    name = "broken"
    version = "broken:1"

    async def score(self, req: RoutingRequest, /) -> RoutingScore:
        raise KeyError("bug")


class Hooks:
    def __init__(self) -> None:
        self.scores: list[RoutingScore] = []

    def scored(self, score: RoutingScore, duration_s: float, /) -> None:
        self.scores.append(score)

    def decided(self, decision: RouteDecision, score: RoutingScore | None, /) -> None:
        return None


async def test_deadline_turns_a_slow_scorer_into_a_timeout_fallback(clock: FakeClock) -> None:
    async with asyncio.timeout(1):
        score = await DeadlineScorer(Slow(), 0.02, clock).score(REQ)
    assert score.fallback
    assert score.fallback_reason == "timeout"
    assert score.scorer_version == "slow:1"


async def test_deadline_never_raises_on_scorer_bugs(clock: FakeClock) -> None:
    score = await DeadlineScorer(Broken(), 0.5, clock).score(REQ)
    assert score.fallback_reason == "parse_error"


async def test_cache_stores_raw_answers_and_rederives(clock: FakeClock) -> None:
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return json_response(fixture("p18")["response"])

    store: TTLCache[Mapping[str, Any]] = TTLCache(10, 60, clock)
    scorer = CachingScorer(make_scorer(make_client(handler, clock)), store)
    first = await scorer.score(REQ)
    second = await scorer.score(
        routing_request(user("Prove there are infinitely  many primes of the form 4k+3."))
    )
    assert len(calls) == 1
    assert not first.cached
    assert second.cached
    assert second.score == first.score
    assert len(store) == 1
    cached_raw = next(iter(store._items.values()))[1]
    assert set(cached_raw) == {"model", "answers", "usage"}

    clock.advance(61)
    await scorer.score(REQ)
    assert len(calls) == 2


async def test_cache_skips_fallbacks(clock: FakeClock) -> None:
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(500)

    store: TTLCache[Mapping[str, Any]] = TTLCache(10, 60, clock)
    scorer = CachingScorer(make_scorer(make_client(handler, clock)), store)
    first = await scorer.score(REQ)
    calls_after_first = len(calls)
    await scorer.score(REQ)
    assert first.fallback_reason == "http_5xx"
    assert len(store) == 0
    assert len(calls) > calls_after_first


async def test_metrics_scorer_reports_final_outcome(clock: FakeClock) -> None:
    hooks = Hooks()
    scorer = MetricsScorer(DeadlineScorer(Slow(), 0.01, clock), hooks, clock)
    await scorer.score(REQ)
    assert [s.fallback_reason for s in hooks.scores] == ["timeout"]
    assert scorer.name == "slow"
    assert scorer.version == "slow:1"


async def test_null_and_static_scorers(clock: FakeClock) -> None:
    null = await NullScorer().score(REQ)
    assert null.fallback_reason == "disabled"
    static = await StaticScorer(0.3).score(REQ)
    assert not static.fallback
    assert static.score == 0.3
