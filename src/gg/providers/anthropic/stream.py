from collections.abc import Mapping
from typing import Any, cast

import orjson

from gg.core.errors import ProviderError
from gg.core.schema import ChatChunk, FinishReason, FunctionCallDelta, ToolCallDelta
from gg.providers.anthropic.errors import classify_stream_error
from gg.providers.anthropic.usage import counts_from_usage
from gg.providers.sse import SSEEvent
from gg.providers.state.thinking import ThinkingBlocks
from gg.providers.stream_base import StreamTranslator

FINISH_MAP: Mapping[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}
_THINKING_TYPES = ("thinking", "redacted_thinking")


def _dict(value: Any) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


class AnthropicStreamTranslator(StreamTranslator):
    """messages api sse -> canonical chunks; also collects the thinking each tool_use needs next turn"""

    def __init__(
        self, *, chunk_id: str, created: int, model: str, provider: str, expose_reasoning: bool = False
    ) -> None:
        super().__init__(chunk_id=chunk_id, created=created, model=model)
        self.provider = provider
        self.expose_reasoning = expose_reasoning
        self.refusal_category: str | None = None
        self.thinking_writes: dict[str, ThinkingBlocks] = {}
        self._usage: dict[str, Any] = {}
        self._open: dict[int, dict[str, Any]] = {}
        self._tool_ordinal: dict[int, int] = {}
        self._segment: ThinkingBlocks = []
        self._stop_reason: str | None = None
        self._stop_details: dict[str, Any] = {}

    def feed(self, event: SSEEvent) -> list[ChatChunk]:
        data = event.data.strip()
        if not data:
            return []
        try:
            obj: Any = orjson.loads(data)
        except orjson.JSONDecodeError as exc:
            raise ProviderError(
                "retryable",
                provider=self.provider,
                status=200,
                code="bad_upstream_response",
                message=str(exc),
            ) from exc
        payload = _dict(obj)
        kind = payload.get("type") or ("error" if "error" in payload else event.event)
        if kind == "message_start":
            self._message_start(_dict(payload.get("message")))
        elif kind == "content_block_start":
            return self._block_start(payload)
        elif kind == "content_block_delta":
            return self._block_delta(payload)
        elif kind == "content_block_stop":
            self._block_stop(payload)
        elif kind == "message_delta":
            self._message_delta(payload)
        elif kind == "message_stop":
            return self._message_stop()
        elif kind == "error":
            raise classify_stream_error(payload, provider=self.provider)
        # ping and event types added later are ignored
        return []

    def _message_start(self, message: dict[str, Any]) -> None:
        self.upstream_id = self.upstream_id or (_str(message.get("id")) or None)
        self.served_model = _str(message.get("model")) or self.served_model
        self._take_usage(_dict(message.get("usage")))

    def _take_usage(self, usage: dict[str, Any]) -> None:
        if not usage:
            return
        # message_delta usage is cumulative and may repeat or refine the counts from message_start
        self._usage.update({k: v for k, v in usage.items() if v is not None})
        counts = counts_from_usage(self._usage)
        if counts is not None:
            self.counts = counts
            self.raw_usage = dict(self._usage)

    def _index(self, payload: dict[str, Any]) -> int:
        index = payload.get("index")
        return index if isinstance(index, int) else 0

    def _block_start(self, payload: dict[str, Any]) -> list[ChatChunk]:
        index = self._index(payload)
        block = _dict(payload.get("content_block"))
        kind = block.get("type")
        if kind == "text":
            self._open[index] = {"type": "text"}
            return self._text(_str(block.get("text")))
        if kind == "tool_use":
            ordinal = len(self._tool_ordinal)
            self._tool_ordinal[index] = ordinal
            call_id = _str(block.get("id"))
            self._open[index] = {"type": "tool_use", "id": call_id}
            self.tool_call_ids.append(call_id)
            delta = ToolCallDelta.model_construct(
                index=ordinal,
                id=call_id,
                type="function",
                function=FunctionCallDelta.model_construct(name=_str(block.get("name")), arguments=""),
            )
            return [self.chunk((self.choice(0, tool_calls=(delta,)),))]
        if kind == "thinking":
            self._open[index] = {
                "type": "thinking",
                "thinking": _str(block.get("thinking")),
                "signature": _str(block.get("signature")),
            }
        elif kind == "redacted_thinking":
            self._open[index] = dict(block)
        return []

    def _block_delta(self, payload: dict[str, Any]) -> list[ChatChunk]:
        index = self._index(payload)
        delta = _dict(payload.get("delta"))
        kind = delta.get("type")
        block = self._open.get(index)
        if kind == "text_delta":
            return self._text(_str(delta.get("text")))
        if kind == "input_json_delta":
            fragment = _str(delta.get("partial_json"))
            ordinal = self._tool_ordinal.get(index)
            if not fragment or ordinal is None:
                return []
            call = ToolCallDelta.model_construct(
                index=ordinal, function=FunctionCallDelta.model_construct(name=None, arguments=fragment)
            )
            return [self.chunk((self.choice(0, tool_calls=(call,)),))]
        if kind == "thinking_delta":
            text = _str(delta.get("thinking"))
            if block is not None and block.get("type") == "thinking":
                block["thinking"] = block["thinking"] + text
            if not text:
                return []
            self.reasoning_parts.append(text)
            if self.expose_reasoning:
                return [self.chunk((self.choice(0, reasoning_content=text),))]
        elif kind == "signature_delta" and block is not None and block.get("type") == "thinking":
            block["signature"] = block["signature"] + _str(delta.get("signature"))
        return []

    def _block_stop(self, payload: dict[str, Any]) -> None:
        block = self._open.pop(self._index(payload), None)
        if block is None:
            return
        if block.get("type") in _THINKING_TYPES:
            self._segment.append(block)
        elif block.get("type") == "tool_use":
            # the thinking that led to this call must go back verbatim with it on the next turn
            if self._segment and block["id"]:
                self.thinking_writes[block["id"]] = self._segment
            self._segment = []

    def _message_delta(self, payload: dict[str, Any]) -> None:
        delta = _dict(payload.get("delta"))
        reason = delta.get("stop_reason")
        if isinstance(reason, str) and reason:
            self._stop_reason = reason
        details = _dict(delta.get("stop_details"))
        if details:
            self._stop_details = details
        self._take_usage(_dict(payload.get("usage")))

    def _message_stop(self) -> list[ChatChunk]:
        self.finished = True
        reason = self._stop_reason or "end_turn"
        self.provider_finish_reason = reason
        finish = FINISH_MAP.get(reason)
        if finish is None:
            self.flag("unknown_finish_reason")
            finish = "stop"
        if reason == "pause_turn":
            self.flag("pause_turn")
        fields: dict[str, Any] = {}
        if reason == "refusal":
            category = self._stop_details.get("category")
            self.refusal_category = category if isinstance(category, str) else "unspecified"
            explanation = _str(self._stop_details.get("explanation"))
            if explanation:
                fields["refusal"] = explanation
        return [self.chunk((self.choice(0, finish_reason=finish, **fields),))]

    def _text(self, text: str) -> list[ChatChunk]:
        if not text:
            return []
        self.output_parts.append(text)
        return [self.chunk((self.choice(0, content=text),))]
