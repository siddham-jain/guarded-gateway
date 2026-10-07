from collections.abc import Mapping
from urllib.parse import quote

import orjson

from gg.core.errors import ProviderError
from gg.providers.adapter_base import BaseHTTPAdapter, Prepared
from gg.providers.gemini.errors import classify
from gg.providers.gemini.quirks import GeminiQuirks
from gg.providers.gemini.request import build_body, history_call_ids
from gg.providers.gemini.stream import GeminiStreamTranslator
from gg.providers.http import UpstreamRequest
from gg.providers.runtime import AdapterDeps, ProviderRuntime
from gg.providers.state.signatures import SignatureStore
from gg.providers.stream_base import StreamTranslator

API_KEY_HEADER = "x-goog-api-key"


class GeminiAdapter(BaseHTTPAdapter):
    """native generateContent; upstream calls always stream, the key only ever travels in a header"""

    def __init__(self, runtime: ProviderRuntime, deps: AdapterDeps) -> None:
        super().__init__(runtime, deps)
        self.quirks = GeminiQuirks.model_validate(runtime.quirks)
        self._signatures = SignatureStore(deps.state)

    async def build_upstream(self, prep: Prepared) -> UpstreamRequest:
        key_id = prep.ctx.key.id
        stored = await self._signatures.lookup(key_id, history_call_ids(prep.request))
        translated = build_body(prep.request, prep.deployment, self.quirks, key_id=key_id, signatures=stored)
        prep.note(ignored=translated.ignored, adjustments=translated.adjustments)
        if translated.dummy_signatures:
            prep.meta = prep.meta.but(flags=(*prep.meta.flags, "dummy_signature"))
        headers = {"content-type": "application/json", "accept": "text/event-stream"}
        if self.runtime.api_key is not None:
            headers[API_KEY_HEADER] = self.runtime.api_key.get_secret_value()
        path = f"/models/{quote(prep.deployment.upstream_model, safe='-._')}:streamGenerateContent?alt=sse"
        return UpstreamRequest("POST", path, headers, orjson.dumps(translated.body))

    def stream_translator(self, prep: Prepared) -> StreamTranslator:
        return GeminiStreamTranslator(
            chunk_id=prep.chunk_id,
            created=prep.ctx.received_unix,
            model=prep.deployment.id,
            provider=self.name,
            request_id=prep.ctx.request_id,
            now=self.deps.clock.now(),
            emit_signatures=self.quirks.emit_thought_signatures,
        )

    def classify_error(self, status: int, body: bytes, headers: Mapping[str, str]) -> ProviderError:
        return classify(status, body, headers, provider=self.name, now=self.deps.clock.now())

    async def persist_state(self, prep: Prepared, translator: StreamTranslator) -> None:
        if isinstance(translator, GeminiStreamTranslator) and translator.signatures:
            await self._signatures.save(prep.ctx.key.id, translator.signatures)
