from collections.abc import Mapping

import orjson
import structlog

from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.errors import ProviderError
from gg.core.schema import ChatRequest
from gg.providers.adapter_base import BaseHTTPAdapter, Prepared
from gg.providers.anthropic.errors import REQUEST_ID_HEADER, classify, rate_limit_snapshot
from gg.providers.anthropic.quirks import AnthropicQuirks
from gg.providers.anthropic.request import (
    MESSAGES_PATH,
    RequestInfo,
    build_body,
    history_tool_call_ids,
    uses_thinking,
)
from gg.providers.anthropic.stream import AnthropicStreamTranslator
from gg.providers.http import UpstreamRequest
from gg.providers.meta import ResponseMeta
from gg.providers.openai_compat.quirks import deep_merge
from gg.providers.runtime import AdapterDeps, ProviderRuntime
from gg.providers.state.thinking import ThinkingBlocks, ThinkingStore
from gg.providers.stream_base import StreamTranslator

log = structlog.get_logger("gg.providers.anthropic")


class AnthropicAdapter(BaseHTTPAdapter):
    """native messages api; always streams upstream and round-trips thinking blocks through the store"""

    def __init__(self, runtime: ProviderRuntime, deps: AdapterDeps) -> None:
        super().__init__(runtime, deps)
        self.quirks = AnthropicQuirks.model_validate(runtime.quirks)
        self._per_deployment: dict[str, AnthropicQuirks] = {}
        self._thinking = ThinkingStore(deps.state)

    def quirks_for(self, dep: Deployment) -> AnthropicQuirks:
        if not dep.quirks:
            return self.quirks
        cached = self._per_deployment.get(dep.id)
        if cached is None:
            cached = AnthropicQuirks.model_validate(deep_merge(self.runtime.quirks, dep.quirks))
            self._per_deployment[dep.id] = cached
        return cached

    def prepare(self, request: ChatRequest, dep: Deployment, ctx: RequestContext) -> Prepared:
        prep = super().prepare(request, dep, ctx)
        if request.reasoning_effort == "none" and dep.capabilities.thinking_mode == "adaptive":
            # the checker clamps none to the lowest effort; adaptive models can still skip up-front thinking
            prep.request = prep.request.model_copy(update={"reasoning_effort": "none"})
        return prep

    async def build_upstream(self, prep: Prepared) -> UpstreamRequest:
        dep = prep.deployment
        quirks = self.quirks_for(dep)
        thinking = await self._lookup_thinking(prep) if uses_thinking(dep) else {}
        translated = build_body(
            prep.request, dep, quirks, RequestInfo(key_id=prep.ctx.key.id), thinking=thinking
        )
        flags = ("thinking_reinjected",) if translated.thinking_injected else ()
        prep.note(ignored=translated.ignored, adjustments=translated.adjustments)
        prep.meta = prep.meta.but(flags=(*prep.meta.flags, *flags))
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream",
            "anthropic-version": quirks.api_version,
            **self.runtime.auth.headers(self.runtime.api_key),
        }
        return UpstreamRequest("POST", MESSAGES_PATH, headers, orjson.dumps(translated.body))

    async def _lookup_thinking(self, prep: Prepared) -> dict[str, ThinkingBlocks]:
        ids = history_tool_call_ids(prep.request)
        if not ids:
            return {}
        try:
            return await self._thinking.lookup(prep.ctx.key.id, prep.deployment.upstream_model, ids)
        except Exception:
            # fail open: without the blocks anthropic answers with less reasoning continuity, not an error
            log.warning("thinking_store.lookup_failed", deployment=prep.deployment.id, exc_info=True)
            return {}

    def stream_translator(self, prep: Prepared) -> StreamTranslator:
        return AnthropicStreamTranslator(
            chunk_id=prep.chunk_id,
            created=prep.ctx.received_unix,
            model=prep.deployment.id,
            provider=self.name,
            expose_reasoning=self.quirks_for(prep.deployment).expose_reasoning,
        )

    def classify_error(self, status: int, body: bytes, headers: Mapping[str, str]) -> ProviderError:
        return classify(status, body, headers, provider=self.name, now=self.deps.clock.now())

    def response_meta(self, prep: Prepared, headers: Mapping[str, str]) -> ResponseMeta:
        lowered = {k.lower(): v for k, v in headers.items()}
        return prep.meta.but(
            upstream_request_id=lowered.get(REQUEST_ID_HEADER),
            ratelimit=rate_limit_snapshot(headers, self.deps.clock.now()),
        )

    async def persist_state(self, prep: Prepared, translator: StreamTranslator) -> None:
        if isinstance(translator, AnthropicStreamTranslator) and translator.thinking_writes:
            await self._thinking.save(
                prep.ctx.key.id, prep.deployment.upstream_model, translator.thinking_writes
            )

    def _meta(self, base: ResponseMeta, tr: StreamTranslator) -> ResponseMeta:
        meta = super()._meta(base, tr)
        if isinstance(tr, AnthropicStreamTranslator) and tr.refusal_category is not None:
            return meta.but(refusal_category=tr.refusal_category)
        return meta
