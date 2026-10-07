"""cacheability (C8 §4.3) and semantic eligibility (§4.4); pure, every refusal has a bounded reason"""

from dataclasses import dataclass

from gg.cache.config import CacheConfig
from gg.cache.keys import semantic_text
from gg.core.context import RequestContext
from gg.core.schema import ChatRequest, FilePart, InputAudioPart, TextPart

# input guards set ctx.cache.bypass_reason when shadow-mode pii went upstream unredacted (L8)
PII_UNREDACTED = "pii_unredacted"


@dataclass(frozen=True, slots=True)
class CacheDecision:
    lookup: bool
    store: bool
    semantic: bool
    ttl_s: int
    threshold: float
    reason: str | None = None
    semantic_reason: str | None = None

    @property
    def bypass(self) -> bool:
        return not (self.lookup or self.store)


def _source(ctx: RequestContext) -> ChatRequest:
    return ctx.scrubbed or ctx.request


def _bypass_reason(ctx: RequestContext, cfg: CacheConfig, backend_up: bool) -> str | None:
    request = _source(ctx)
    ext = ctx.original.gg
    if not cfg.exact.enabled or ctx.key.cache.scope == "off":
        return "disabled"
    if ctx.cache is not None and ctx.cache.bypass_reason is not None:
        return PII_UNREDACTED
    if ext is not None and ext.cache == "off":
        return "client_off"
    if not (request.temperature == 0 or request.seed is not None or ctx.key.cache.allow_sampled):
        return "sampled"
    if request.n > 1:
        return "n_gt_1"
    if request.has_tools() or request.tool_choice is not None:
        return "tools"
    for message in request.messages:
        if isinstance(message.content, tuple) and any(
            isinstance(p, InputAudioPart | FilePart) for p in message.content
        ):
            return "unsupported_content"
    if not backend_up:
        return "backend_down"
    return None


def _semantic_reason(ctx: RequestContext, cfg: CacheConfig, index_up: bool) -> str | None:
    request = _source(ctx)
    ext = ctx.original.gg
    if not (cfg.semantic.enabled and ctx.key.cache.semantic and (ext is None or ext.semantic_cache)):
        return "sem_disabled"
    users = [m for m in request.messages if m.role == "user"]
    if len(users) != 1 or any(m.role not in ("system", "developer", "user") for m in request.messages):
        return "multi_turn"
    content = users[0].content
    if isinstance(content, tuple) and not all(isinstance(p, TextPart) for p in content):
        return "non_text"
    if len(ctx.vault):
        # a paraphrase may carry other placeholders than the cached reply, so redacted prompts stay exact-only
        return "has_placeholders"
    text = semantic_text(request)
    if len(text) < cfg.semantic.min_user_chars:
        return "too_short"
    if len(text) > cfg.semantic.max_user_chars:
        return "too_long"
    if not index_up:
        return "sem_unavailable"
    return None


def ttl_for(ctx: RequestContext, cfg: CacheConfig) -> int:
    ext = ctx.original.gg
    wanted = (
        ext.cache_ttl_s if ext is not None and ext.cache_ttl_s is not None else ctx.key.cache.default_ttl_s
    )
    ceiling = min(cfg.exact.max_ttl_s, ctx.key.cache.max_ttl_s)
    return max(cfg.exact.min_ttl_s, min(wanted, ceiling))


def threshold_for(ctx: RequestContext, cfg: CacheConfig) -> float:
    ext = ctx.original.gg
    if ext is not None and ext.cache_threshold is not None:
        return min(cfg.semantic.distance_threshold, ext.cache_threshold)
    return cfg.semantic.distance_threshold


def decide(ctx: RequestContext, cfg: CacheConfig, *, backend_up: bool, index_up: bool) -> CacheDecision:
    ttl_s = ttl_for(ctx, cfg)
    threshold = threshold_for(ctx, cfg)
    reason = _bypass_reason(ctx, cfg, backend_up)
    if reason is not None:
        return CacheDecision(False, False, False, ttl_s, threshold, reason=reason, semantic_reason=reason)
    mode = ctx.original.gg.cache if ctx.original.gg is not None else "default"
    lookup = mode != "refresh"
    store = mode != "no_store"
    semantic_reason = _semantic_reason(ctx, cfg, index_up)
    return CacheDecision(
        lookup=lookup,
        store=store,
        semantic=semantic_reason is None,
        ttl_s=ttl_s,
        threshold=threshold,
        reason=None if lookup else "client_refresh",
        semantic_reason=semantic_reason,
    )
