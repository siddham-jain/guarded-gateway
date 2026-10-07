from gg.routing.base import RoutingRequest, RoutingScore


class NullScorer:
    """routing disabled, no jev key, or the key opted out: no prompt leaves the gateway"""

    name = "null"
    version = "null"

    async def score(self, req: RoutingRequest, /) -> RoutingScore:
        return RoutingScore.fallback_for(self.name, self.version, "disabled")


class StaticScorer:
    """a fixed score; for dev and tests"""

    name = "static"

    def __init__(self, score: float) -> None:
        self._score = score
        self.version = f"static:{score}"

    async def score(self, req: RoutingRequest, /) -> RoutingScore:
        return RoutingScore(
            score=self._score, raw_score=self._score, scorer=self.name, scorer_version=self.version
        )
