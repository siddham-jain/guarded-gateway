from typing import Any, ClassVar

import httpx2
import orjson
from pydantic import SecretStr

from gg.core.clock import FakeClock
from gg.core.deployment import Deployment
from gg.core.errors import ProviderError
from gg.core.schema import ChatChunk
from gg.providers.anthropic.adapter import AnthropicAdapter
from gg.providers.anthropic.stream import AnthropicStreamTranslator
from gg.providers.http import HttpClientFactory
from gg.providers.runtime import AdapterDeps, AuthSpec, ProviderRuntime
from gg.providers.sse import SSEParser
from gg.providers.state.memory import InMemoryStateStore
from tests.unit.providers.support import SECRET, Handler, Recorder, make_dep

BASE_URL = "https://api.anthropic.test"
CLAUDE_5_5: dict[str, Any] = {
    "forced_tool_choice": False,
    "prefill": False,
    "sampling_params": "none",
    "temperature_max": 1.0,
    "effort_levels": frozenset({"low", "medium", "high", "xhigh", "max"}),
    "effort_clamp": {"none": "low", "minimal": "low"},
    "thinking_mode": "adaptive",
    "vision": True,
    "pdf": True,
    "n": False,
    "logprobs": False,
    "max_output": 128_000,
    "min_max_tokens": 1024,
    "cache_min_tokens": 512,
}
HAIKU: dict[str, Any] = {
    **CLAUDE_5_5,
    "forced_tool_choice": True,
    "prefill": True,
    "sampling_params": "temp_or_top_p",
    "effort_levels": frozenset({"none", "low", "medium", "high"}),
    "effort_clamp": {"minimal": "none", "xhigh": "high", "max": "high"},
    "thinking_mode": "manual",
    "mid_conversation_system": False,
    "max_output": 64_000,
    "min_max_tokens": 1,
    "cache_min_tokens": 4096,
}


def sonnet(**caps: Any) -> Deployment:
    defaults = caps.pop("defaults", {"reasoning_effort": "medium", "max_completion_tokens": 16000})
    return make_dep("anthropic", "claude-sonnet-5-5", defaults=defaults, **{**CLAUDE_5_5, **caps})


def opus(**caps: Any) -> Deployment:
    defaults = {"reasoning_effort": "medium", "max_completion_tokens": 16000}
    return make_dep(
        "anthropic",
        "claude-opus-5-5",
        defaults=defaults,
        **{**CLAUDE_5_5, "thinking_mode": "adaptive_always", **caps},
    )


def haiku(**caps: Any) -> Deployment:
    defaults = caps.pop("defaults", {"reasoning_effort": "none", "max_completion_tokens": 4096})
    return make_dep("anthropic", "claude-haiku-4-5", defaults=defaults, **{**HAIKU, **caps})


def make_anthropic_adapter(
    handler: Handler,
    *,
    provider: str = "anthropic",
    quirks: dict[str, Any] | None = None,
    state: Any = None,
) -> tuple[AnthropicAdapter, Recorder]:
    recorder = Recorder(handler)
    runtime = ProviderRuntime(
        name=provider,
        type="anthropic",
        base_url=BASE_URL,
        api_key=SecretStr(SECRET),
        auth=AuthSpec(header="x-api-key", scheme=None),
        quirks=quirks or {},
    )
    deps = AdapterDeps(
        http=HttpClientFactory(transports={provider: httpx2.MockTransport(recorder)}),
        clock=FakeClock(),
        state=state if state is not None else InMemoryStateStore(clock=FakeClock()),
    )
    return AnthropicAdapter(runtime, deps), recorder


def event(obj: dict[str, Any]) -> bytes:
    return b"event: " + str(obj["type"]).encode() + b"\ndata: " + orjson.dumps(obj) + b"\n\n"


def anthropic_sse(*events: dict[str, Any]) -> bytes:
    return b"".join(event(e) for e in events)


def message_start(usage: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    message = {
        "id": "msg_conf",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 15, "cache_read_input_tokens": 5, "output_tokens": 1}
        if usage is None
        else usage,
    }
    return {"type": "message_start", "message": message, **extra}


def text_block(index: int, *texts: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}
    ]
    events.extend(
        {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": t}}
        for t in texts
    )
    events.append({"type": "content_block_stop", "index": index})
    return events


def message_end(reason: str, usage: dict[str, Any] | None = None, **delta: Any) -> list[dict[str, Any]]:
    return [
        {
            "type": "message_delta",
            "delta": {"stop_reason": reason, "stop_sequence": None, **delta},
            "usage": {"output_tokens": 6} if usage is None else usage,
        },
        {"type": "message_stop"},
    ]


def run_translator(
    raw: bytes, tr: AnthropicStreamTranslator | None = None
) -> tuple[AnthropicStreamTranslator, list[ChatChunk]]:
    tr = tr or AnthropicStreamTranslator(
        chunk_id="chatcmpl-x", created=1, model="anthropic/m", provider="anthropic"
    )
    parser = SSEParser()
    out: list[ChatChunk] = []
    for ev in [*parser.feed(raw), *parser.flush()]:
        out.extend(tr.feed(ev))
    return tr, out


def run_until_error(raw: bytes) -> tuple[list[ChatChunk], ProviderError]:
    tr = AnthropicStreamTranslator(
        chunk_id="chatcmpl-x", created=1, model="anthropic/m", provider="anthropic"
    )
    parser = SSEParser()
    out: list[ChatChunk] = []
    try:
        for ev in [*parser.feed(raw), *parser.flush()]:
            out.extend(tr.feed(ev))
    except ProviderError as err:
        return out, err
    raise AssertionError("translator did not raise")


class AnthropicUpstream:
    """conformance upstream: replays openai-shaped scenario deltas as messages api sse events"""

    request_id = "msg_conf"
    auth_header = "x-api-key"
    json_schema_supported = True
    FINISH: ClassVar[dict[str, str]] = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "refusal",
    }

    def _noisy(self, obj: dict[str, Any]) -> dict[str, Any]:
        return {**obj, "noise": {"unknown": True}}

    def body(
        self, deltas: list[dict[str, Any]], finish: str, *, done: bool = True, usage: bool = True
    ) -> bytes:
        # anthropic has no [DONE]: message_stop ends the stream, so `done=False` drops it
        events: list[dict[str, Any]] = [message_start(usage=None if usage else {})]
        events.extend(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "hm"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "c2"},
                },
                {"type": "content_block_stop", "index": 0},
            ]
        )
        index = 1
        text_open = False
        blocks: dict[int, int] = {}
        for delta in deltas:
            if delta.get("content"):
                if not text_open:
                    events.append(
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {"type": "text", "text": ""},
                        }
                    )
                    text_open = True
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": delta["content"]},
                    }
                )
            for fragment in delta.get("tool_calls", ()):
                fn = fragment.get("function", {})
                if fragment["index"] not in blocks:
                    if text_open or blocks:
                        events.append({"type": "content_block_stop", "index": index})
                        index += 1
                        text_open = False
                    blocks[fragment["index"]] = index
                    block = {"type": "tool_use", "id": fragment["id"], "name": fn["name"], "input": {}}
                    events.append({"type": "content_block_start", "index": index, "content_block": block})
                partial = {"type": "input_json_delta", "partial_json": fn.get("arguments", "")}
                events.append(
                    {"type": "content_block_delta", "index": blocks[fragment["index"]], "delta": partial}
                )
        events.append({"type": "content_block_stop", "index": index})
        details = {"type": "refusal", "category": "cyber"} if finish == "content_filter" else None
        if done:
            end = message_end(self.FINISH[finish], {"output_tokens": 6} if usage else {})
            if details:
                end[0]["delta"]["stop_details"] = details
            events.extend(end)
        return anthropic_sse(*(self._noisy(e) for e in events))

    def text(self, **kw: Any) -> bytes:
        return self.body([{"content": "Hello"}, {"content": " world"}], "stop", **kw)

    def assert_tool_loop(self, body: dict[str, Any]) -> None:
        messages = body["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        call = messages[1]["content"][0]
        result = messages[2]["content"][0]
        assert call == {"type": "tool_use", "id": "call_a", "name": "get_weather", "input": {}}
        assert result == {"type": "tool_result", "tool_use_id": "call_a", "content": "sunny"}

    def assert_json_schema(self, body: dict[str, Any]) -> None:
        assert body["output_config"]["format"] == {"type": "json_schema", "schema": {"type": "object"}}
