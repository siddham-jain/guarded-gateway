from typing import Any

import httpx2
import orjson
from pydantic import SecretStr

from gg.core.clock import FakeClock
from gg.core.deployment import Deployment
from gg.providers.gemini.adapter import GeminiAdapter
from gg.providers.http import HttpClientFactory
from gg.providers.runtime import AdapterDeps, ProviderRuntime
from gg.providers.state.memory import InMemoryStateStore
from tests.unit.providers.support import SECRET, Handler, Recorder, make_dep

BASE_URL = "https://generativelanguage.test/v1beta"
GEMINI_3: dict[str, Any] = {
    "effort_levels": frozenset({"low", "medium", "high"}),
    "effort_clamp": {"none": "low", "minimal": "low"},
    "thinking_mode": "levels",
    "sampling_params": "advisory",
    "parallel_tool_calls": False,
    "prefill": False,
    "stop_max": 5,
    "vision": True,
    "pdf": True,
    "n": False,
    "max_output": 65_536,
}


def gemini_dep(model: str = "gemini-3.8-flash", **caps: Any) -> Deployment:
    return make_dep("gemini", model, **{**GEMINI_3, **caps})


def make_gemini_adapter(
    handler: Handler,
    *,
    provider: str = "gemini",
    quirks: dict[str, Any] | None = None,
    state: InMemoryStateStore | None = None,
) -> tuple[GeminiAdapter, Recorder]:
    recorder = Recorder(handler)
    runtime = ProviderRuntime(
        name=provider, type="gemini", base_url=BASE_URL, api_key=SecretStr(SECRET), quirks=quirks or {}
    )
    deps = AdapterDeps(
        http=HttpClientFactory(transports={provider: httpx2.MockTransport(recorder)}),
        clock=FakeClock(),
        state=state or InMemoryStateStore(clock=FakeClock()),
    )
    return GeminiAdapter(runtime, deps), recorder


def gemini_sse(*events: Any) -> bytes:
    return b"".join(b"data: " + orjson.dumps(e) + b"\n\n" for e in events)


def candidate_event(
    parts: list[dict[str, Any]] | None, finish: str | None = None, usage: dict[str, Any] | None = None
) -> dict[str, Any]:
    cand: dict[str, Any] = {"index": 0}
    if parts is not None:
        cand["content"] = {"role": "model", "parts": parts}
    if finish is not None:
        cand["finishReason"] = finish
    event: dict[str, Any] = {"candidates": [cand], "modelVersion": "gemini-test", "responseId": "resp-1"}
    if usage is not None:
        event["usageMetadata"] = usage
    return event


CONFORMANCE_FINISH = {
    "stop": "STOP",
    "tool_calls": "STOP",
    "length": "MAX_TOKENS",
    "content_filter": "SAFETY",
}
CONFORMANCE_USAGE = {
    "promptTokenCount": 20,
    "cachedContentTokenCount": 5,
    "candidatesTokenCount": 4,
    "thoughtsTokenCount": 2,
    "totalTokenCount": 26,
}


class GeminiUpstream:
    """conformance upstream: replays openai-shaped scenario deltas as streamGenerateContent events"""

    request_id = "resp-conf"
    auth_header = "x-goog-api-key"
    json_schema_supported = True

    def _event(
        self, parts: list[dict[str, Any]], finish: str | None = None, usage: bool = False
    ) -> dict[str, Any]:
        cand: dict[str, Any] = {"index": 0, "content": {"role": "model", "parts": parts}}
        if finish is not None:
            cand["finishReason"] = finish
        event: dict[str, Any] = {
            "candidates": [cand],
            "modelVersion": "conf",
            "responseId": self.request_id,
            "noise": {"unknown": True},
        }
        if usage:
            event["usageMetadata"] = CONFORMANCE_USAGE
        return event

    def body(
        self, deltas: list[dict[str, Any]], finish: str, *, done: bool = True, usage: bool = True
    ) -> bytes:
        # gemini has no [DONE]: `done` is accepted for parity, the finish event ends the stream
        events = [self._event([{"text": "thinking", "thought": True}])]
        calls: dict[int, dict[str, str]] = {}
        for delta in deltas:
            if delta.get("content"):
                events.append(self._event([{"text": delta["content"]}]))
            for fragment in delta.get("tool_calls", ()):
                call = calls.setdefault(fragment["index"], {"id": "", "name": "", "arguments": ""})
                call["id"] = fragment.get("id") or call["id"]
                fn = fragment.get("function", {})
                call["name"] = fn.get("name") or call["name"]
                call["arguments"] += fn.get("arguments", "")
        if calls:
            parts: list[dict[str, Any]] = [
                {"functionCall": {"id": c["id"], "name": c["name"], "args": orjson.loads(c["arguments"])}}
                for _, c in sorted(calls.items())
            ]
            parts[0]["thoughtSignature"] = "c2ln"
            events.append(self._event(parts))
        events.append(self._event([], CONFORMANCE_FINISH[finish], usage=usage))
        return gemini_sse(*events)

    def text(self, **kw: Any) -> bytes:
        return self.body([{"content": "Hello"}, {"content": " world"}], "stop", **kw)

    def assert_tool_loop(self, body: dict[str, Any]) -> None:
        contents = body["contents"]
        assert [c["role"] for c in contents] == ["user", "model", "user"]
        call = contents[1]["parts"][0]
        result = contents[2]["parts"][0]["functionResponse"]
        assert result["id"] == call["functionCall"]["id"] == "call_a"
        assert result["name"] == call["functionCall"]["name"] == "get_weather"
        assert call["thoughtSignature"]

    def assert_json_schema(self, body: dict[str, Any]) -> None:
        text = body["generationConfig"]["responseFormat"]["text"]
        assert text == {"mimeType": "application/json", "schema": {"type": "object"}}
