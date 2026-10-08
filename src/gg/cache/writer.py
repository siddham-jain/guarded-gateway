"""the CACHE_WRITE finalizer: stores only clean, fully guarded replies, in placeholder form"""

import asyncio
from dataclasses import dataclass
from typing import Literal

import structlog

from gg.cache.base import CachedResponse, CacheHooks, CacheKey, Embedder, Pricer, SavedUsage, SemanticIndex
from gg.cache.config import CacheConfig
from gg.cache.failopen import FailOpenCache
from gg.cache.keys import semantic_text
from gg.cache.policy import CacheDecision
from gg.cache.recorder import StreamRecorder
from gg.core.aio import TaskSupervisor
from gg.core.clock import Clock
from gg.core.context import ContextKey, RequestContext
from gg.core.guard_types import Verdict
from gg.core.schema import ChatResponse
from gg.guardrails.base import POLICY_REF
from gg.pipeline.stage import PipelineResult

log = structlog.get_logger("gg.cache.writer")

type VerdictState = Literal["allow", "blocked", "pending"]

_VERDICT_POLL_S = 0.1

# the probe's embedding of this request, reused by the semantic write
SEMANTIC_VECTOR = ContextKey[list[float]]("cache.semantic_vector")


@dataclass(frozen=True, slots=True)
class CachePlan:
    decision: CacheDecision
    key: CacheKey


@dataclass(slots=True)
class PendingWrite:
    plan: CachePlan
    lease: str | None = None
    result: PipelineResult | None = None
    recorder: StreamRecorder | None = None


def verdict_state(ctx: RequestContext) -> VerdictState:
    verdict = ctx.output_verdict
    if verdict is None:
        # no guardrails wired means nothing to wait for; wired guards that never settled mean unknown
        return "allow" if ctx.get(POLICY_REF) is None else "pending"
    if verdict.verdict is not Verdict.ALLOW:
        return "blocked"
    return "pending" if verdict.post_hoc_pending else "allow"


def build_entry(
    response: ChatResponse, ctx: RequestContext, *, now: float, pricer: Pricer | None
) -> CachedResponse | str:
    """the entry to store, or the bounded reason it must not be stored"""
    if len(response.choices) != 1:
        return "n_gt_1"
    choice = response.choices[0]
    message = choice.message
    if message.tool_calls:
        return "tool_calls"
    if choice.finish_reason not in ("stop", "length"):
        return "finish_reason"
    if not message.content and not message.refusal:
        return "empty"
    text = f"{message.content or ''}\n{message.refusal or ''}"
    vault = ctx.vault.placeholders()
    if any(value and value in text for value in vault.values()):
        return "vault_value_in_output"
    upstream = SavedUsage.of(ctx.usage) if ctx.usage is not None else None
    served = ctx.served_by
    return CachedResponse(
        content=message.content,
        refusal=message.refusal,
        finish_reason=choice.finish_reason,
        usage=response.usage,
        upstream=upstream,
        provider=served.provider if served is not None else None,
        served_by=served.id if served is not None else None,
        response_model=response.model,
        system_fingerprint=response.system_fingerprint,
        cost_usd=pricer(ctx.usage) if pricer is not None and ctx.usage is not None else None,
        created_at=now,
        request_id=ctx.request_id,
        placeholders=tuple(p for p in vault if p in text),
    )


class CacheWriter:
    def __init__(
        self,
        cfg: CacheConfig,
        cache: FailOpenCache,
        *,
        clock: Clock,
        supervisor: TaskSupervisor,
        hooks: CacheHooks,
        index: SemanticIndex | None = None,
        embedder: Embedder | None = None,
        pricer: Pricer | None = None,
    ) -> None:
        self._cfg = cfg
        self._cache = cache
        self._clock = clock
        self._supervisor = supervisor
        self._hooks = hooks
        self._index = index
        self._embedder = embedder
        self._pricer = pricer

    async def finish(self, ctx: RequestContext, pending: PendingWrite) -> None:
        try:
            if pending.plan.decision.store:
                await self._write(ctx, pending)
        finally:
            # released once the response is out, never held across a post-hoc wait
            if pending.lease is not None:
                await self._cache.release(pending.plan.key.lock_key, pending.lease)

    def _skip(self, reason: str) -> None:
        self._hooks.store("exact", "skipped", reason)

    def _final_response(self, ctx: RequestContext, pending: PendingWrite) -> ChatResponse | str | None:
        result = pending.result
        if result is None:
            return "error"
        if result.source != "upstream":
            return None
        if ctx.outcome != "completed":
            return "disconnect" if ctx.outcome == "client_disconnected" else "error"
        if result.stream is None:
            return result.response or "error"
        recorder = pending.recorder
        if recorder is None or not recorder.complete:
            return "error"
        if recorder.overflow:
            return "too_large"
        return recorder.response() or "error"

    async def _write(self, ctx: RequestContext, pending: PendingWrite) -> None:
        response = self._final_response(ctx, pending)
        if response is None:
            return
        if isinstance(response, str):
            self._skip(response)
            return
        entry = build_entry(response, ctx, now=self._clock.time(), pricer=self._pricer)
        if isinstance(entry, str):
            if entry == "vault_value_in_output":
                log.error("cache.vault_value_in_output", request_id=ctx.request_id)
                self._hooks.store("exact", "rejected", entry)
            else:
                self._skip(entry)
            return
        state = verdict_state(ctx)
        if state == "blocked":
            self._skip("guard_blocked")
        elif state == "allow":
            await self._store(ctx, pending.plan, entry)
        elif ctx.output_verdict is not None and ctx.output_verdict.post_hoc_pending:
            self._supervisor.spawn(
                self._after_post_hoc(ctx, pending.plan, entry), name="cache-write-post-hoc"
            )
        else:
            self._skip("guard_pending")

    async def _after_post_hoc(self, ctx: RequestContext, plan: CachePlan, entry: CachedResponse) -> None:
        state = verdict_state(ctx)
        for _ in range(max(1, round(self._cfg.exact.post_hoc_wait_s / _VERDICT_POLL_S))):
            if state != "pending":
                break
            await asyncio.sleep(_VERDICT_POLL_S)
            state = verdict_state(ctx)
        if state == "allow":
            await self._store(ctx, plan, entry)
        else:
            self._skip("guard_blocked" if state == "blocked" else "guard_pending")

    async def _store(self, ctx: RequestContext, plan: CachePlan, entry: CachedResponse) -> None:
        decision = plan.decision
        # a refresh skipped the lookup, so it overwrites instead of SET NX
        outcome = await self._cache.put(
            plan.key.redis_key, entry, decision.ttl_s, replace=not decision.lookup
        )
        if outcome == "stored":
            self._hooks.store("exact", "stored", "ok")
        elif outcome == "error":
            self._hooks.store("exact", "error", "backend")
        else:
            self._skip(outcome)
        if outcome == "stored" and decision.semantic:
            await self._store_vector(ctx, plan)

    async def _store_vector(self, ctx: RequestContext, plan: CachePlan) -> None:
        if self._index is None or self._embedder is None:
            return
        try:
            text = semantic_text(ctx.scrubbed or ctx.request)
            vector = ctx.get(SEMANTIC_VECTOR)
            if vector is None:
                vector = (await self._embedder.embed([text]))[0]
            await self._index.add(vector, plan.key.tags, plan.key.redis_key, plan.decision.ttl_s, text)
        except Exception as exc:
            log.warning("cache.semantic_store_failed", error=type(exc).__name__)
            self._hooks.store("semantic", "error", "backend")
            return
        self._hooks.store("semantic", "stored", "ok")
