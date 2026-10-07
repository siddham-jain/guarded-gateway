from gg.cache.base import DistanceResult, Layer, LookupResult, SingleFlightOutcome, StoreResult
from gg.observability.metrics import Metrics


class CacheMetrics:
    """CacheHooks implementation feeding the gg_cache_*, gg_cost_saved_* and gg_tokens_saved_* series"""

    def __init__(self, metrics: Metrics) -> None:
        self._metrics = metrics

    def lookup(self, layer: Layer, result: LookupResult, /) -> None:
        self._metrics.cache_lookups.labels(layer, result).inc()

    def lookup_duration(self, layer: Layer, seconds: float, /) -> None:
        self._metrics.cache_lookup_duration.labels(layer).observe(seconds)

    def semantic_distance(self, result: DistanceResult, distance: float, /) -> None:
        self._metrics.semantic_cache_distance.labels(result).observe(distance)

    def store(self, layer: Layer, result: StoreResult, reason: str, /) -> None:
        m = self._metrics
        m.cache_stores.labels(layer, result, m.labels("cache_reason", reason)).inc()

    def bypass(self, reason: str, /) -> None:
        m = self._metrics
        m.cache_bypass.labels(m.labels("cache_reason", reason)).inc()

    def singleflight(self, outcome: SingleFlightOutcome, /) -> None:
        self._metrics.cache_singleflight.labels(outcome).inc()

    def cost_saved(self, layer: Layer, usd: float, /) -> None:
        if usd > 0:
            self._metrics.cost_saved.labels(layer).inc(usd)

    def tokens_saved(self, layer: Layer, kind: str, tokens: int, /) -> None:
        m = self._metrics
        if tokens > 0:
            m.tokens_saved.labels(layer, m.labels("token_type", kind)).inc(tokens)
