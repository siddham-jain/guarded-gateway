from gg.core.routing_types import RouteDecision
from gg.observability.metrics import Metrics
from gg.routing.base import RoutingScore


class RoutingMetrics:
    """RoutingHooks implementation feeding the gg_routing_* and gg_router_* series"""

    def __init__(self, metrics: Metrics) -> None:
        self._metrics = metrics

    def scored(self, score: RoutingScore, duration_s: float, /) -> None:
        m = self._metrics
        outcome = f"fallback_{score.fallback_reason}" if score.fallback else "ok"
        m.router_score_duration.labels(score.scorer, outcome).observe(duration_s)
        m.routing_decision_cache.labels("lru", "hit" if score.cached else "miss").inc()
        if not score.fallback:
            m.router_score.labels(score.scorer).observe(score.score)

    def decided(self, decision: RouteDecision, score: RoutingScore | None, /) -> None:
        m = self._metrics
        alias = m.labels("alias", decision.alias or "")
        tier = m.labels("tier", decision.tier or "")
        reason = decision.reason.split(":", 1)[0]
        scorer_tier = (score.tier if score is not None and not score.fallback else None) or "none"
        m.routing_decisions.labels(alias, tier, reason, scorer_tier).inc()
