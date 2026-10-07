"""pipeline integration: the exact stage (inside guard_out, before probes) and the semantic probe"""

import asyncio

import structlog

from gg.cache.base import CachedResponse, CacheHooks, Embedder, Layer, LookupResult, Pricer, SemanticIndex
from gg.cache.config import CacheConfig
from gg.cache.failopen import FailOpenCache
from gg.cache.keys import CacheKeyBuilder, semantic_text
from gg.cache.policy import decide
from gg.cache.recorder import StreamRecorder
from gg.cache.replay import to_result
from gg.cache.writer import SEMANTIC_VECTOR, CachePlan, CacheWriter, PendingWrite
from gg.core.cache_types import CacheState
from gg.core.clock import Clock
from gg.core.context import ContextKey, FinalizerOrder, RequestContext
from gg.pipeline.probes import Annotate, ProbeOutcome, ShortCircuit
from gg.pipeline.stage import Next, PipelineResult

log = structlog.get_logger("gg.cache")

HEADER = "x-gg-cache"
CACHE_PLAN = ContextKey[CachePlan]("cache.plan")


class HitResponder:
    """turns an entry into this request's reply; pii restore happens in guard_out with this request's vault"""

    def __init__(self, clock: Clock, hooks: CacheHooks, pricer: Pricer | None) -> None:
        self._clock = clock
        self._hooks = hooks
        self._pricer = pricer

    @staticmethod
    def resolvable(ctx: RequestContext, entry: CachedResponse) -> bool:
        return all(ctx.vault.resolve(p) is not None for p in entry.placeholders)

    def _saved_usd(self, entry: CachedResponse) -> float | None:
        if self._pricer is not None and entry.upstream is not None:
            usd = self._pricer(entry.upstream.record())
            if usd is not None:
                return usd
        return entry.cost_usd

    def serve(
        self, ctx: RequestContext, entry: CachedResponse, layer: Layer, distance: float | None = None
    ) -> PipelineResult:
        semantic = layer == "semantic"
        ctx.cache_status = "semantic_hit" if semantic else "exact_hit"
        headers = ctx.response_headers
        headers[HEADER] = "SEMANTIC_HIT" if semantic else "HIT"
        headers["x-gg-cache-age"] = str(max(0, int(self._clock.time() - entry.created_at)))
        if distance is not None:
            headers["x-gg-cache-distance"] = f"{distance:.4f}"
            if ctx.cache is not None:
                ctx.cache.semantic_distance = distance
        if entry.provider is not None and entry.served_by is not None:
            headers["x-gg-provider"] = entry.provider
            headers["x-gg-model"] = entry.served_by
        headers["x-gg-cost-usd"] = "0"
        saved = self._saved_usd(entry)
        if saved is not None:
            headers["x-gg-cost-saved-usd"] = f"{saved:.6f}"
            self._hooks.cost_saved(layer, saved)
        if entry.upstream is not None:
            self._hooks.tokens_saved(layer, "input", entry.upstream.input_tokens)
            self._hooks.tokens_saved(layer, "output", entry.upstream.output_tokens)
        source = "semantic_cache" if semantic else "exact_cache"
        return to_result(entry, ctx, source, int(self._clock.time()))


class ExactCacheStage:
    name = "cache_exact"

    def __init__(
        self,
        cfg: CacheConfig,
        cache: FailOpenCache,
        *,
        keys: CacheKeyBuilder,
        responder: HitResponder,
        writer: CacheWriter,
        hooks: CacheHooks,
        clock: Clock,
        index: SemanticIndex | None = None,
    ) -> None:
        self._cfg = cfg
        self._cache = cache
        self._keys = keys
        self._responder = responder
        self._writer = writer
        self._hooks = hooks
        self._clock = clock
        self._index = index

    async def __call__(self, ctx: RequestContext, call_next: Next, /) -> PipelineResult:
        state = ctx.cache if ctx.cache is not None else CacheState()
        ctx.cache = state
        index_up = self._index is not None and self._index.available
        decision = decide(ctx, self._cfg, backend_up=self._cache.available, index_up=index_up)
        if decision.bypass:
            ctx.cache_status = "bypass"
            state.bypass_reason = state.bypass_reason or decision.reason
            state.store = False
            ctx.response_headers[HEADER] = "BYPASS"
            self._hooks.bypass(decision.reason or "disabled")
            self._hooks.lookup("exact", "bypass")
            return await call_next(ctx)

        plan = CachePlan(decision, self._keys.build(ctx))
        state.exact_key = plan.key.redis_key
        state.store = decision.store
        ctx.set(CACHE_PLAN, plan)
        pending = PendingWrite(plan)
        if decision.lookup:
            hit = await self._lookup(ctx, pending)
            if hit is not None:
                return hit
        else:
            self._hooks.bypass(decision.reason or "client_refresh")
            self._hooks.lookup("exact", "bypass")
        ctx.cache_status = "miss"
        ctx.response_headers[HEADER] = "MISS"
        if decision.store or pending.lease is not None:
            ctx.finalizers.defer(
                "cache_write", lambda: self._writer.finish(ctx, pending), FinalizerOrder.CACHE_WRITE
            )
        result = await call_next(ctx)
        pending.result = result
        if result.stream is not None and result.source == "upstream" and decision.store:
            recorder = StreamRecorder(self._cfg.exact.max_value_bytes)
            pending.recorder = recorder
            result = result.map_stream(recorder.tee)
        return result

    async def _lookup(self, ctx: RequestContext, pending: PendingWrite) -> PipelineResult | None:
        key = pending.plan.key
        start = self._clock.monotonic()
        entry, result = await self._cache.lookup(key.redis_key)
        if entry is not None and not self._responder.resolvable(ctx, entry):
            entry, result = None, "miss"
        if entry is None and result == "miss" and self._cfg.exact.singleflight and not ctx.request.stream:
            entry, result = await self._single_flight(ctx, pending)
        self._hooks.lookup_duration("exact", self._clock.monotonic() - start)
        self._hooks.lookup("exact", result)
        if entry is None:
            return None
        return self._responder.serve(ctx, entry, "exact")

    async def _single_flight(
        self, ctx: RequestContext, pending: PendingWrite
    ) -> tuple[CachedResponse | None, LookupResult]:
        """the leader takes the lock and goes upstream; followers poll for its value, then proceed"""
        key = pending.plan.key
        exact = self._cfg.exact
        pending.lease = await self._cache.acquire(key.lock_key, exact.lock_ttl_ms)
        if pending.lease is not None:
            self._hooks.singleflight("leader")
            return None, "miss"
        for _ in range(max(1, round(exact.singleflight_wait_s / exact.singleflight_poll_s))):
            await asyncio.sleep(exact.singleflight_poll_s)
            entry = await self._cache.get(key.redis_key)
            if entry is None and not await self._cache.held(key.lock_key):
                # the leader released; its value, if any, was written just before
                entry = await self._cache.get(key.redis_key)
                if entry is None:
                    self._hooks.singleflight("waited_miss")
                    return None, "miss"
            if entry is not None:
                if not self._responder.resolvable(ctx, entry):
                    self._hooks.singleflight("waited_miss")
                    return None, "miss"
                self._hooks.singleflight("waited_hit")
                return entry, "hit"
        self._hooks.singleflight("waited_timeout")
        return None, "miss"


class SemanticCacheProbe:
    """runs beside tier-2 guards and the router; a guard block outranks it (precedence 50)"""

    name = "semantic_cache"
    precedence = 50

    def __init__(
        self,
        cfg: CacheConfig,
        cache: FailOpenCache,
        *,
        index: SemanticIndex,
        embedder: Embedder,
        responder: HitResponder,
        hooks: CacheHooks,
        clock: Clock,
    ) -> None:
        self._cfg = cfg
        self._cache = cache
        self._index = index
        self._embedder = embedder
        self._responder = responder
        self._hooks = hooks
        self._clock = clock

    async def __call__(self, ctx: RequestContext, /) -> ProbeOutcome:
        plan = ctx.get(CACHE_PLAN)
        if plan is None or not (plan.decision.lookup and plan.decision.semantic):
            return Annotate()
        start = self._clock.monotonic()
        try:
            async with asyncio.timeout(self._cfg.semantic.deadline_s):
                vector = (await self._embedder.embed([semantic_text(ctx.scrubbed or ctx.request)]))[0]
                match = await self._index.search(vector, plan.key.tags)
                within = match is not None and match.distance <= plan.decision.threshold
                entry = await self._cache.get(match.exact_key) if match is not None and within else None
        except TimeoutError:
            self._hooks.lookup("semantic", "timeout")
            return Annotate()
        except Exception as exc:
            log.warning("cache.semantic_lookup_failed", error=type(exc).__name__)
            self._hooks.lookup("semantic", "error")
            return Annotate()
        finally:
            self._hooks.lookup_duration("semantic", self._clock.monotonic() - start)

        if match is not None:
            self._hooks.semantic_distance("hit" if within else "miss", match.distance)
        if match is None or entry is None or not self._responder.resolvable(ctx, entry):
            self._hooks.lookup("semantic", "miss")
            return Annotate(apply=lambda c: c.set(SEMANTIC_VECTOR, vector))
        self._hooks.lookup("semantic", "hit")
        distance = match.distance
        return ShortCircuit(lambda c: self._responder.serve(c, entry, "semantic", distance))
