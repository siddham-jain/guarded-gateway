from typing import Any, Literal

from gg.api.anthropic_ingress.response import error_payload, message_id, stop_reason, usage_block
from gg.core.errors import GGError
from gg.core.jsonutil import dumps
from gg.core.schema import ChatChunk, ToolCallDelta, Usage

COMMENT_KEEPALIVE = b": keep-alive\n\n"


def event_frame(payload: dict[str, Any]) -> bytes:
    return b"event: " + str(payload["type"]).encode() + b"\ndata: " + dumps(payload) + b"\n\n"


class AnthropicStreamEncoder:
    """canonical chunks -> anthropic messages sse events (message_start ... message_stop)"""

    def __init__(self, *, request_id: str, model: str) -> None:
        self._id = message_id(request_id)
        self._model = model
        self._started = False
        self._index = -1
        self._open: Literal["text", "tool_use"] | None = None
        self._tool: int | None = None
        self._finish: str | None = None
        self._usage: Usage | None = None

    def chunk(self, chunk: ChatChunk) -> list[bytes]:
        frames = self._start(chunk.model)
        if chunk.usage is not None:
            self._usage = chunk.usage
        for choice in chunk.choices:
            if choice.index != 0:
                continue
            delta = choice.delta
            text = (delta.content or "") + (delta.refusal or "")
            if text:
                frames += self._text(text)
            for call in delta.tool_calls or ():
                frames += self._tool_call(call)
            if choice.finish_reason is not None:
                self._finish = choice.finish_reason
        return frames

    def error(self, error: GGError) -> list[bytes]:
        return [event_frame(error_payload(error))]

    def end(self) -> list[bytes]:
        frames = self._start(self._model) + self._close()
        frames.append(
            event_frame(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason(self._finish), "stop_sequence": None},
                    "usage": usage_block(self._usage),
                }
            )
        )
        frames.append(event_frame({"type": "message_stop"}))
        return frames

    def keepalive(self) -> bytes:
        # the sdk rejects any event before message_start, so an early keepalive stays a comment
        return event_frame({"type": "ping"}) if self._started else COMMENT_KEEPALIVE

    def _start(self, model: str) -> list[bytes]:
        if self._started:
            return []
        self._started = True
        message = {
            "id": self._id,
            "type": "message",
            "role": "assistant",
            "model": model or self._model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return [event_frame({"type": "message_start", "message": message})]

    def _open_block(self, block: dict[str, Any]) -> list[bytes]:
        frames = self._close()
        self._index += 1
        self._open = block["type"]
        return [
            *frames,
            event_frame({"type": "content_block_start", "index": self._index, "content_block": block}),
        ]

    def _close(self) -> list[bytes]:
        if self._open is None:
            return []
        self._open = None
        self._tool = None
        return [event_frame({"type": "content_block_stop", "index": self._index})]

    def _delta(self, delta: dict[str, Any]) -> bytes:
        return event_frame({"type": "content_block_delta", "index": self._index, "delta": delta})

    def _text(self, text: str) -> list[bytes]:
        frames = [] if self._open == "text" else self._open_block({"type": "text", "text": ""})
        frames.append(self._delta({"type": "text_delta", "text": text}))
        return frames

    def _tool_call(self, call: ToolCallDelta) -> list[bytes]:
        frames: list[bytes] = []
        if self._open != "tool_use" or self._tool != call.index:
            name = call.function.name if call.function is not None else None
            block = {
                "type": "tool_use",
                "id": call.id or f"toolu_{self._id[4:]}_{call.index}",
                "name": name or "",
                "input": {},
            }
            frames = self._open_block(block)
            self._tool = call.index
        arguments = call.function.arguments if call.function is not None else None
        if arguments:
            frames.append(self._delta({"type": "input_json_delta", "partial_json": arguments}))
        return frames
