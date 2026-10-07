import hashlib
from collections.abc import Mapping
from typing import Any, cast

import orjson

from gg.core.errors import ProviderError
from gg.core.schema import (
    FINISH_REASONS,
    ChatChunk,
    ChunkChoice,
    FinishReason,
    FunctionCallDelta,
    ToolCallDelta,
)
from gg.providers.openai_compat.errors import classify_stream_error
from gg.providers.openai_compat.quirks import QuirkProfile
from gg.providers.sse import SSEEvent
from gg.providers.stream_base import StreamTranslator
from gg.providers.usage import counts_from_usage, get_path

DEFAULT_FINISH_MAP: Mapping[str, str] = {
    "function_call": "tool_calls",
    "tool_call": "tool_calls",
    "tool_use": "tool_calls",
    "eos": "stop",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_length": "length",
    "model_context_window_exceeded": "length",
    "sensitive": "content_filter",
}
_REASONING_KEYS = ("reasoning_content", "reasoning")


class ThinkSplitter:
    """splits <think>...</think> out of streamed content, tolerating tags split across chunks"""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self.inside = False
        self._pending = ""

    def feed(self, text: str) -> tuple[str, str]:
        buf = self._pending + text
        self._pending = ""
        content: list[str] = []
        reasoning: list[str] = []
        while buf:
            tag = self.CLOSE if self.inside else self.OPEN
            side = reasoning if self.inside else content
            idx = buf.find(tag)
            if idx >= 0:
                side.append(buf[:idx])
                buf = buf[idx + len(tag) :]
                self.inside = not self.inside
                continue
            keep = next((n for n in range(min(len(tag) - 1, len(buf)), 0, -1) if tag.startswith(buf[-n:])), 0)
            side.append(buf[: len(buf) - keep])
            self._pending = buf[len(buf) - keep :]
            break
        return "".join(content), "".join(reasoning)

    def flush(self) -> tuple[str, str]:
        rest, self._pending = self._pending, ""
        return ("", rest) if self.inside else (rest, "")


def _text_of(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_text_of(v) for v in cast("list[Any]", value))
    if isinstance(value, dict):
        item = cast("dict[str, Any]", value)
        return _text_of(item.get("text", ""))
    return ""


def split_content_array(parts: list[Any]) -> tuple[str, str]:
    """mistral-style content arrays: [{type: thinking, thinking: [...]}, {type: text, text}]"""
    content: list[str] = []
    reasoning: list[str] = []
    for part in parts:
        if isinstance(part, str):
            content.append(part)
        elif isinstance(part, dict):
            item = cast("dict[str, Any]", part)
            if item.get("type") == "thinking":
                reasoning.append(_text_of(item.get("thinking", "")))
            else:
                content.append(_text_of(item.get("text", "")))
    return "".join(content), "".join(reasoning)


class CompatStreamTranslator(StreamTranslator):
    def __init__(
        self, *, chunk_id: str, created: int, model: str, quirks: QuirkProfile, provider: str, request_id: str
    ) -> None:
        super().__init__(chunk_id=chunk_id, created=created, model=model)
        self.q = quirks
        self.provider = provider
        self.request_id = request_id
        self.done = False
        self._finish_map = {**DEFAULT_FINISH_MAP, **quirks.response.finish_reason_map}
        self._tools: dict[int, dict[int, tuple[int, str]]] = {}
        self._next_tool: dict[int, int] = {}
        self._last_tool: dict[int, int] = {}
        self._tool_seen: set[int] = set()
        self._think: dict[int, ThinkSplitter] = {}

    def feed(self, event: SSEEvent) -> list[ChatChunk]:
        data = event.data.strip()
        if not data:
            return []
        if data == "[DONE]":
            self.done = True
            self.finished = True
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
        if not isinstance(obj, dict):
            return []
        payload = cast("dict[str, Any]", obj)
        if payload.get("error"):
            raise classify_stream_error(payload, provider=self.provider, quirks=self.q)
        self._check_body_status(payload)
        self._capture(payload)
        choices = payload.get("choices")
        out: list[ChunkChoice] = []
        if isinstance(choices, list):
            for raw in cast("list[Any]", choices):
                if isinstance(raw, dict):
                    built = self._choice(cast("dict[str, Any]", raw))
                    if built is not None:
                        out.append(built)
        extras = {k: payload[k] for k in self.q.response.passthrough_fields if k in payload}
        if self.all_started_finished() and not self.done:
            self.finished = True
        if not out and not extras:
            return []
        return [self.chunk(tuple(out), **extras)]

    def _check_body_status(self, payload: dict[str, Any]) -> None:
        check = self.q.response.check_body_status
        if check is None:
            return
        status = get_path(payload, check.path)
        if status is None or status == check.ok:
            return
        message = get_path(payload, check.message_path) if check.message_path else None
        raise classify_stream_error(
            {"error": {"code": str(status), "message": str(message or "")}},
            provider=self.provider,
            quirks=self.q,
        )

    def _capture(self, payload: dict[str, Any]) -> None:
        if self.upstream_id is None and isinstance(payload.get("id"), str):
            self.upstream_id = payload["id"]
        if isinstance(payload.get("model"), str):
            self.served_model = payload["model"]
        if isinstance(payload.get("provider"), str):
            self.upstream_provider = payload["provider"]
        for root in self.q.response.usage_roots:
            usage = get_path(payload, root)
            if isinstance(usage, dict):
                raw = cast("dict[str, Any]", usage)
                counts = counts_from_usage(raw, self.q.response.usage_map)
                if counts is not None:
                    self.counts = counts
                    self.raw_usage = raw
        if self.q.response.cost_path is not None:
            cost = get_path(payload, self.q.response.cost_path)
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                self.cost_usd = float(cost) * self.q.response.cost_scale

    def _choice(self, raw: dict[str, Any]) -> ChunkChoice | None:
        index = raw.get("index", 0)
        index = index if isinstance(index, int) else 0
        delta_raw = raw.get("delta") or raw.get("message") or {}
        delta = cast("dict[str, Any]", delta_raw) if isinstance(delta_raw, dict) else {}
        fields: dict[str, Any] = {}

        content, reasoning = self._content(index, delta)
        if content:
            fields["content"] = content
            if index == 0:
                self.output_parts.append(content)
        if reasoning:
            if index == 0:
                self.reasoning_parts.append(reasoning)
            if self.q.reasoning.expose:
                fields["reasoning_content"] = reasoning
        if isinstance(delta.get("refusal"), str) and delta["refusal"]:
            fields["refusal"] = delta["refusal"]
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            fields["tool_calls"] = tuple(self._tool_deltas(index, cast("list[Any]", tool_calls)))

        finish = self._finish(index, raw)
        first = index not in self._started
        if not fields and finish is None and not first:
            return None
        if not fields and finish is None and first and not delta:
            return None
        return self.choice(index, finish_reason=finish, **fields)

    def _content(self, index: int, delta: dict[str, Any]) -> tuple[str, str]:
        raw = delta.get("content")
        reasoning = ""
        for key in _REASONING_KEYS:
            value = delta.get(key)
            if isinstance(value, str) and value:
                reasoning = value
                break
        if isinstance(raw, list):
            content, extra = split_content_array(cast("list[Any]", raw))
            return content, reasoning + extra
        content = raw if isinstance(raw, str) else ""
        if self.q.reasoning.field == "think_tags" and content:
            splitter = self._think.setdefault(index, ThinkSplitter())
            content, extra = splitter.feed(content)
            reasoning += extra
        return content, reasoning

    def _tool_deltas(self, choice: int, fragments: list[Any]) -> list[ToolCallDelta]:
        state = self._tools.setdefault(choice, {})
        out: list[ToolCallDelta] = []
        self._tool_seen.add(choice)
        for position, fragment in enumerate(fragments):
            if not isinstance(fragment, dict):
                continue
            frag = cast("dict[str, Any]", fragment)
            fn_raw = frag.get("function")
            fn = cast("dict[str, Any]", fn_raw) if isinstance(fn_raw, dict) else {}
            up_index = frag.get("index")
            fid = frag.get("id") if isinstance(frag.get("id"), str) and frag.get("id") else None
            name = fn.get("name") if isinstance(fn.get("name"), str) else None
            if not isinstance(up_index, int):
                # hosts that omit the index: a new id or name starts a call, otherwise continue the last one
                if fid or name:
                    up_index = max(state) + 1 if state else position
                else:
                    up_index = self._last_tool.get(choice, 0)
            known = state.get(up_index)
            arguments = fn.get("arguments")
            args = (
                arguments
                if isinstance(arguments, str)
                else (orjson.dumps(arguments).decode() if arguments else None)
            )
            if known is None or (fid is not None and known[1] != fid):
                gg_index = self._next_tool.get(choice, 0)
                self._next_tool[choice] = gg_index + 1
                call_id = (
                    fid
                    or "call_gg_"
                    + hashlib.sha1(  # noqa: S324
                        f"{self.request_id}:{choice}:{gg_index}".encode()
                    ).hexdigest()[:24]
                )
                state[up_index] = (gg_index, call_id)
                if choice == 0:
                    self.tool_call_ids.append(call_id)
                delta = ToolCallDelta.model_construct(
                    index=gg_index,
                    id=call_id,
                    type="function",
                    function=FunctionCallDelta.model_construct(name=name or "", arguments=args or ""),
                )
            else:
                delta = ToolCallDelta.model_construct(
                    index=known[0],
                    function=FunctionCallDelta.model_construct(name=None, arguments=args or ""),
                )
            self._last_tool[choice] = up_index
            out.append(delta)
        return out

    def _finish(self, index: int, raw: dict[str, Any]) -> FinishReason | None:
        reason = raw.get("finish_reason")
        if not isinstance(reason, str) or not reason:
            return None
        native = raw.get("native_finish_reason")
        self.provider_finish_reason = native if isinstance(native, str) else reason
        if reason in self.q.response.finish_reason_errors:
            raise ProviderError(
                "retryable",
                provider=self.provider,
                status=200,
                code="upstream_error",
                message=f"upstream finished with {reason}",
            )
        mapped = reason if reason in FINISH_REASONS else self._finish_map.get(reason)
        if mapped is None or mapped not in FINISH_REASONS:
            self.flag("unknown_finish_reason")
            mapped = "stop"
        if mapped == "stop" and index in self._tool_seen:
            mapped = "tool_calls"
        splitter = self._think.get(index)
        if splitter is not None:
            rest, extra = splitter.flush()
            if rest and index == 0:
                self.output_parts.append(rest)
            if extra and index == 0:
                self.reasoning_parts.append(extra)
        return cast("FinishReason", mapped)
