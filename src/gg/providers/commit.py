from typing import Any

from gg.core.schema import ChatChunk, ChunkChoice, Delta, FinishReason, ToolCallDelta

DEFAULT_MAX_BUFFER_BYTES = 65_536


def _commits(chunk: ChatChunk) -> bool:
    return chunk.has_content() or any(c.finish_reason is not None for c in chunk.choices)


def _size(chunk: ChatChunk) -> int:
    total = 0
    for choice in chunk.choices:
        d = choice.delta
        total += len(d.content or "") + len(d.refusal or "")
        reasoning = (d.model_extra or {}).get("reasoning_content")
        if isinstance(reasoning, str):
            total += len(reasoning)
    return total


def _join(parts: list[str]) -> str | None:
    return "".join(parts) if parts else None


def merge_chunks(chunks: list[ChatChunk]) -> ChatChunk:
    """folds role, reasoning, content and tool fragments of several chunks into one, per choice"""
    if len(chunks) == 1:
        return chunks[0]
    order: list[int] = []
    roles: dict[int, Any] = {}
    content: dict[int, list[str]] = {}
    refusal: dict[int, list[str]] = {}
    reasoning: dict[int, list[str]] = {}
    tools: dict[int, list[ToolCallDelta]] = {}
    finish: dict[int, FinishReason | None] = {}
    for chunk in chunks:
        for choice in chunk.choices:
            i = choice.index
            if i not in finish:
                order.append(i)
                finish[i] = None
            d = choice.delta
            if roles.get(i) is None:
                roles[i] = d.role
            if d.content:
                content.setdefault(i, []).append(d.content)
            if d.refusal:
                refusal.setdefault(i, []).append(d.refusal)
            extra = (d.model_extra or {}).get("reasoning_content")
            if isinstance(extra, str) and extra:
                reasoning.setdefault(i, []).append(extra)
            if d.tool_calls:
                tools.setdefault(i, []).extend(d.tool_calls)
            if choice.finish_reason is not None:
                finish[i] = choice.finish_reason
    choices: list[ChunkChoice] = []
    for i in order:
        fields: dict[str, Any] = {
            "role": roles.get(i),
            "content": _join(content.get(i, [])),
            "refusal": _join(refusal.get(i, [])),
            "tool_calls": tuple(tools[i]) if i in tools else None,
        }
        if i in reasoning:
            fields["reasoning_content"] = "".join(reasoning[i])
        choices.append(
            ChunkChoice.model_construct(
                index=i, delta=Delta.model_construct(**fields), finish_reason=finish[i]
            )
        )
    first = chunks[0]
    return ChatChunk.model_construct(
        id=first.id, created=first.created, model=first.model, choices=tuple(choices)
    )


class CommitGate:
    """holds chunks until the first content-bearing chunk, then flushes them merged into one.

    role-only, empty, reasoning and usage chunks never commit; a finish chunk does.
    """

    def __init__(self, max_bytes: int = DEFAULT_MAX_BUFFER_BYTES) -> None:
        self.committed = False
        self.forced = False
        self._max_bytes = max_bytes
        self._buffer: list[ChatChunk] = []
        self._buffered_bytes = 0

    def push_many(self, chunks: list[ChatChunk]) -> list[ChatChunk]:
        if self.committed:
            return chunks
        out: list[ChatChunk] = []
        for i, chunk in enumerate(chunks):
            if not chunk.choices:
                self._buffer.append(chunk)
                continue
            self._buffer.append(chunk)
            self._buffered_bytes += _size(chunk)
            if _commits(chunk) or self._buffered_bytes > self._max_bytes:
                self.forced = not _commits(chunk)
                self.committed = True
                out.extend(self._drain())
                out.extend(chunks[i + 1 :])
                return out
        return out

    def close(self, tail: list[ChatChunk]) -> list[ChatChunk]:
        if self.committed:
            return tail
        self._buffer.extend(tail)
        self.committed = True
        return self._drain()

    def _drain(self) -> list[ChatChunk]:
        with_choices = [c for c in self._buffer if c.choices]
        usage_only = [c for c in self._buffer if not c.choices]
        self._buffer = []
        merged = [merge_chunks(with_choices)] if with_choices else []
        return merged + usage_only
