import asyncio
from collections.abc import Mapping
from typing import Any

import structlog

from gg.core.clock import Clock
from gg.routing.base import CacheableScorer, RoutingHooks, RoutingRequest, RoutingScore, RoutingScorer
from gg.routing.ttl_cache import TTLCache

log = structlog.get_logger("gg.routing")


class DeadlineScorer:
    """enforces the total scoring deadline; the never-raises guarantee of the scorer contract lives here"""

    def __init__(self, inner: RoutingScorer, deadline_s: float, clock: Clock) -> None:
        self._inner = inner
        self._deadline_s = deadline_s
        self._clock = clock
        self.name = inner.name

    @property
    def version(self) -> str:
        return self._inner.version

    async def score(self, req: RoutingRequest, /) -> RoutingScore:
        start = self._clock.monotonic()
        try:
            async with asyncio.timeout(self._deadline_s):
                return await self._inner.score(req)
        except TimeoutError:
            return RoutingScore.fallback_for(self.name, self.version, "timeout", self._ms_since(start))
        except Exception:
            log.exception("routing.scorer_bug", scorer=self.name)
            return RoutingScore.fallback_for(self.name, self.version, "parse_error", self._ms_since(start))

    def _ms_since(self, start: float) -> float:
        return round((self._clock.monotonic() - start) * 1000, 1)


class CachingScorer:
    """caches the scorer's raw answers, not decisions, so threshold or policy changes never invalidate it"""

    def __init__(self, inner: CacheableScorer, store: TTLCache[Mapping[str, Any]]) -> None:
        self._inner = inner
        self._store = store
        self.name = inner.name

    @property
    def version(self) -> str:
        return self._inner.version

    async def score(self, req: RoutingRequest, /) -> RoutingScore:
        key = self._inner.cache_key(req)
        if key is None:
            return await self._inner.score(req)
        raw = self._store.get(key)
        if raw is not None:
            return self._inner.from_raw(raw, req)
        result = await self._inner.score(req)
        if not result.fallback:
            self._store.put(key, result.raw)
        return result


class MetricsScorer:
    """reports every final score, including deadline fallbacks, to the routing hooks"""

    def __init__(self, inner: RoutingScorer, hooks: RoutingHooks, clock: Clock) -> None:
        self._inner = inner
        self._hooks = hooks
        self._clock = clock
        self.name = inner.name

    @property
    def version(self) -> str:
        return self._inner.version

    async def score(self, req: RoutingRequest, /) -> RoutingScore:
        start = self._clock.monotonic()
        result = await self._inner.score(req)
        self._hooks.scored(result, self._clock.monotonic() - start)
        return result
