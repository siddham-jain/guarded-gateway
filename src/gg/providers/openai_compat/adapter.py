from collections.abc import Mapping

import orjson

from gg.core.deployment import Deployment
from gg.core.errors import ProviderError
from gg.providers.adapter_base import BaseHTTPAdapter, Prepared
from gg.providers.http import UpstreamRequest
from gg.providers.meta import ResponseMeta, rate_limit_snapshot
from gg.providers.openai_compat.errors import classify
from gg.providers.openai_compat.quirks import QuirkProfile, deep_merge
from gg.providers.openai_compat.request import RequestInfo, build_body, history_key, history_keys
from gg.providers.openai_compat.stream import CompatStreamTranslator
from gg.providers.runtime import AdapterDeps, ProviderRuntime
from gg.providers.state.reasoning import ReasoningStore
from gg.providers.stream_base import StreamTranslator

CHAT_PATH = "/chat/completions"
_HISTORY_MODES = ("required", "required_with_tools")


class OpenAICompatibleAdapter(BaseHTTPAdapter):
    """one adapter for every chat-completions host; differences come from the provider's quirk profile"""

    def __init__(self, runtime: ProviderRuntime, deps: AdapterDeps) -> None:
        super().__init__(runtime, deps)
        self.quirks = QuirkProfile.model_validate(runtime.quirks)
        self._per_deployment: dict[str, QuirkProfile] = {}
        self._reasoning = ReasoningStore(deps.state)

    def quirks_for(self, dep: Deployment) -> QuirkProfile:
        if not dep.quirks:
            return self.quirks
        cached = self._per_deployment.get(dep.id)
        if cached is None:
            cached = QuirkProfile.model_validate(deep_merge(self.runtime.quirks, dep.quirks))
            self._per_deployment[dep.id] = cached
        return cached

    def _info(self, prep: Prepared) -> RequestInfo:
        flags = prep.ctx.key.flags
        return RequestInfo(
            request_id=prep.ctx.request_id,
            key_id=prep.ctx.key.id,
            received_unix=prep.ctx.received_unix,
            allow_store=flags.allow_store,
            allow_service_tier=flags.allow_service_tier,
        )

    async def build_upstream(self, prep: Prepared) -> UpstreamRequest:
        quirks = self.quirks_for(prep.deployment)
        history: dict[str, str] = {}
        if quirks.reasoning.history in _HISTORY_MODES:
            history = await self._reasoning.lookup(prep.ctx.key.id, self.name, history_keys(prep.request))
        translated = build_body(prep.request, prep.deployment, quirks, self._info(prep), history=history)
        prep.note(ignored=translated.ignored, adjustments=translated.adjustments)
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream",
            **quirks.extra_headers,
            **self.runtime.auth.headers(self.runtime.api_key),
        }
        if quirks.client_request_id_header:
            headers[quirks.client_request_id_header] = prep.ctx.request_id
        return UpstreamRequest("POST", CHAT_PATH, headers, orjson.dumps(translated.body))

    def stream_translator(self, prep: Prepared) -> StreamTranslator:
        return CompatStreamTranslator(
            chunk_id=prep.chunk_id,
            created=prep.ctx.received_unix,
            model=prep.deployment.id,
            quirks=self.quirks_for(prep.deployment),
            provider=self.name,
            request_id=prep.ctx.request_id,
        )

    def classify_error(self, status: int, body: bytes, headers: Mapping[str, str]) -> ProviderError:
        return classify(
            status, body, headers, provider=self.name, quirks=self.quirks, now=self.deps.clock.now()
        )

    def response_meta(self, prep: Prepared, headers: Mapping[str, str]) -> ResponseMeta:
        lowered = {k.lower(): v for k, v in headers.items()}
        request_id = next((lowered[h] for h in self.quirks.request_id_headers if h in lowered), None)
        return prep.meta.but(
            upstream_request_id=request_id,
            ratelimit=rate_limit_snapshot(headers, self.quirks.rate_limit_headers),
        )

    def transport_kind(self) -> str:
        return self.quirks.errors.transport_kind

    async def persist_state(self, prep: Prepared, translator: StreamTranslator) -> None:
        quirks = self.quirks_for(prep.deployment)
        reasoning = "".join(translator.reasoning_parts)
        if quirks.reasoning.history not in _HISTORY_MODES or not reasoning:
            return
        turn: dict[str, object] = {"content": translator.output_text()}
        if translator.tool_call_ids:
            turn["tool_calls"] = [{"id": translator.tool_call_ids[0]}]
        key = history_key(turn)
        if key is not None:
            await self._reasoning.save(prep.ctx.key.id, self.name, {key: reasoning})
