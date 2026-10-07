import hashlib
from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

import orjson

from gg.core.errors import ProviderError
from gg.core.schema import ChatChunk, ChunkChoice, FinishReason, FunctionCallDelta, ToolCallDelta
from gg.providers.gemini.errors import classify_stream_error
from gg.providers.gemini.request import SYNTHETIC_ID_PREFIX
from gg.providers.sse import SSEEvent
from gg.providers.stream_base import StreamTranslator
from gg.providers.usage import TokenCounts

_CONTENT_FILTER = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "LANGUAGE",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "ESCALATION",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
    }
)
FINISH_MAP: Mapping[str, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "NO_IMAGE": "stop",
    **dict.fromkeys(_CONTENT_FILTER, "content_filter"),
}
# a stop with a flag; the malformed pair is retried instead while nothing has been emitted
_FLAGGED = frozenset(
    {
        "MALFORMED_FUNCTION_CALL",
        "MALFORMED_RESPONSE",
        "UNEXPECTED_TOOL_CALL",
        "TOO_MANY_TOOL_CALLS",
        "OTHER",
        "FINISH_REASON_UNSPECIFIED",
    }
)
_RETRY_BEFORE_CONTENT = frozenset({"MALFORMED_FUNCTION_CALL", "MALFORMED_RESPONSE"})


def _count(raw: Mapping[str, Any], name: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def usage_counts(raw: Mapping[str, Any]) -> TokenCounts | None:
    """usageMetadata -> counts; promptTokenCount includes cached tokens, candidates exclude thoughts"""
    if "promptTokenCount" not in raw and "candidatesTokenCount" not in raw:
        return None
    thoughts = _count(raw, "thoughtsTokenCount")
    return TokenCounts(
        input=_count(raw, "promptTokenCount") + _count(raw, "toolUsePromptTokenCount"),
        output=_count(raw, "candidatesTokenCount") + thoughts,
        cached=_count(raw, "cachedContentTokenCount"),
        reasoning=thoughts,
    ).clamped()


class GeminiStreamTranslator(StreamTranslator):
    """streamGenerateContent sse -> canonical chunks; gemini has no [DONE], so a finishReason ends it"""

    def __init__(
        self,
        *,
        chunk_id: str,
        created: int,
        model: str,
        provider: str,
        request_id: str,
        now: datetime,
        emit_signatures: bool = True,
    ) -> None:
        super().__init__(chunk_id=chunk_id, created=created, model=model)
        self.provider = provider
        self.request_id = request_id
        self.now = now
        self.emit_signatures = emit_signatures
        self.signatures: dict[str, str] = {}
        self._tool_counts: dict[int, int] = {}
        self._emitted = False

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
        if not isinstance(obj, dict):
            return []
        payload = cast("dict[str, Any]", obj)
        if payload.get("error"):
            raise classify_stream_error(payload, provider=self.provider, now=self.now)
        self._capture(payload)
        candidates: Any = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            self._check_prompt_block(payload)
            return []
        out: list[ChunkChoice] = []
        for raw in cast("list[Any]", candidates):
            if isinstance(raw, dict):
                built = self._candidate(cast("dict[str, Any]", raw))
                if built is not None:
                    out.append(built)
        if self.all_started_finished():
            self.finished = True
        return [self.chunk(tuple(out))] if out else []

    def _check_prompt_block(self, payload: dict[str, Any]) -> None:
        feedback: Any = payload.get("promptFeedback")
        reason: Any = feedback.get("blockReason") if isinstance(feedback, dict) else None
        if isinstance(reason, str) and reason:
            self.provider_finish_reason = reason
            raise ProviderError(
                "content_filter",
                provider=self.provider,
                status=200,
                code="content_filter",
                message=f"prompt blocked by upstream safety filter ({reason})",
            )

    def _capture(self, payload: dict[str, Any]) -> None:
        if self.upstream_id is None and isinstance(payload.get("responseId"), str):
            self.upstream_id = payload["responseId"]
        if isinstance(payload.get("modelVersion"), str):
            self.served_model = payload["modelVersion"]
        usage: Any = payload.get("usageMetadata")
        if isinstance(usage, dict):
            raw = cast("dict[str, Any]", usage)
            counts = usage_counts(raw)
            if counts is not None:
                self.counts = counts
                self.raw_usage = raw

    def _candidate(self, raw: dict[str, Any]) -> ChunkChoice | None:
        index = raw.get("index", 0)
        index = index if isinstance(index, int) and not isinstance(index, bool) else 0
        content: Any = raw.get("content")
        parts: Any = content.get("parts") if isinstance(content, dict) else None
        texts: list[str] = []
        calls: list[ToolCallDelta] = []
        for part in cast("list[Any]", parts) if isinstance(parts, list) else []:
            if not isinstance(part, dict):
                continue
            item = cast("dict[str, Any]", part)
            call: Any = item.get("functionCall")
            if isinstance(call, dict):
                calls.append(
                    self._tool_call(index, cast("dict[str, Any]", call), item.get("thoughtSignature"))
                )
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text:
                continue
            if item.get("thought") is True:
                if index == 0:
                    self.reasoning_parts.append(text)
                continue
            texts.append(text)
        fields: dict[str, Any] = {}
        if texts:
            fields["content"] = "".join(texts)
            if index == 0:
                self.output_parts.append(fields["content"])
        if calls:
            fields["tool_calls"] = tuple(calls)
        if fields:
            self._emitted = True
        finish = self._finish(index, raw.get("finishReason"))
        if not fields and finish is None:
            return None
        return self.choice(index, finish_reason=finish, **fields)

    def _tool_call(self, index: int, call: dict[str, Any], signature: Any) -> ToolCallDelta:
        ordinal = self._tool_counts.get(index, 0)
        self._tool_counts[index] = ordinal + 1
        upstream_id = call.get("id")
        call_id = (
            upstream_id
            if isinstance(upstream_id, str) and upstream_id
            else SYNTHETIC_ID_PREFIX
            + hashlib.sha1(f"{self.request_id}:{index}:{ordinal}".encode()).hexdigest()[:24]  # noqa: S324
        )
        if index == 0:
            self.tool_call_ids.append(call_id)
        name = call.get("name")
        args = call.get("args")
        extra: dict[str, Any] = {}
        if isinstance(signature, str) and signature:
            self.signatures[call_id] = signature
            if self.emit_signatures:
                extra["extra_content"] = {"google": {"thought_signature": signature}}
        return ToolCallDelta.model_construct(
            index=ordinal,
            id=call_id,
            type="function",
            function=FunctionCallDelta.model_construct(
                name=name if isinstance(name, str) else "",
                arguments=orjson.dumps(args if args is not None else {}).decode(),
            ),
            **extra,
        )

    def _finish(self, index: int, reason: Any) -> FinishReason | None:
        if not isinstance(reason, str) or not reason:
            return None
        self.provider_finish_reason = reason
        if reason == "MISSING_THOUGHT_SIGNATURE":
            raise ProviderError(
                "fallback",
                provider=self.provider,
                status=200,
                code="missing_thought_signature",
                message="gemini rejected a replayed function call without its thought signature",
            )
        if reason in _RETRY_BEFORE_CONTENT and not self._emitted:
            raise ProviderError(
                "retryable",
                provider=self.provider,
                status=200,
                code=reason.lower(),
                message=f"upstream finished with {reason}",
            )
        mapped = FINISH_MAP.get(reason)
        if mapped is None:
            self.flag(reason.lower() if reason in _FLAGGED else "unknown_finish_reason")
            mapped = "stop"
        if mapped == "stop" and index in self._tool_counts:
            return "tool_calls"
        return mapped
