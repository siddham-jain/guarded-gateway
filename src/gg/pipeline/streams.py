from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from gg.core.schema import (
    AssistantMessage,
    ChatChunk,
    ChatResponse,
    Choice,
    ChunkChoice,
    Delta,
    FinishReason,
    FunctionCall,
    FunctionCallDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
)


class ChunkStream:
    """async iterator of chunks whose aclose() reaches the adapter, closing the upstream connection.

    iterate from a single task only: the adapter's http stream must exit in the task that entered it.
    """

    def __init__(self, source: AsyncIterator[ChatChunk], *, inner: "ChunkStream | None" = None) -> None:
        self._source = source
        self._inner = inner
        self._closed = False

    def __aiter__(self) -> "ChunkStream":
        return self

    async def __anext__(self) -> ChatChunk:
        return await anext(self._source)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            aclose = getattr(self._source, "aclose", None)
            if aclose is not None:
                await aclose()
        finally:
            if self._inner is not None:
                await self._inner.aclose()

    def map(self, fn: Callable[[AsyncIterator[ChatChunk]], AsyncIterator[ChatChunk]]) -> "ChunkStream":
        return ChunkStream(fn(self), inner=self)


@dataclass(slots=True)
class _ToolCallParts:
    id: str | None = None
    name: str | None = None
    arguments: list[str] = field(default_factory=lambda: [])


@dataclass(slots=True)
class _ChoiceParts:
    content: list[str] = field(default_factory=lambda: [])
    refusal: list[str] = field(default_factory=lambda: [])
    tool_calls: dict[int, _ToolCallParts] = field(default_factory=lambda: {})
    finish_reason: FinishReason | None = None


class StreamAssembler:
    """accumulates chunks into a ChatResponse; tool-call fragments are merged by index"""

    def __init__(self) -> None:
        self._choices: dict[int, _ChoiceParts] = {}
        self._id: str | None = None
        self._created = 0
        self._model = ""
        self.usage: Usage | None = None
        self.chunks = 0

    def feed(self, chunk: ChatChunk) -> None:
        self.chunks += 1
        self._id = self._id or chunk.id
        self._created = self._created or chunk.created
        self._model = self._model or chunk.model
        if chunk.usage is not None:
            self.usage = chunk.usage
        for choice in chunk.choices:
            parts = self._choices.setdefault(choice.index, _ChoiceParts())
            delta = choice.delta
            if delta.content:
                parts.content.append(delta.content)
            if delta.refusal:
                parts.refusal.append(delta.refusal)
            for tc in delta.tool_calls or ():
                call = parts.tool_calls.setdefault(tc.index, _ToolCallParts())
                if tc.id:
                    call.id = tc.id
                if tc.function is not None:
                    if tc.function.name:
                        call.name = tc.function.name
                    if tc.function.arguments:
                        call.arguments.append(tc.function.arguments)
            if choice.finish_reason is not None:
                parts.finish_reason = choice.finish_reason

    def text_so_far(self, choice: int = 0) -> str:
        parts = self._choices.get(choice)
        return "".join(parts.content) if parts else ""

    def result(self) -> ChatResponse:
        choices: list[Choice] = []
        for index in sorted(self._choices):
            parts = self._choices[index]
            tool_calls = tuple(
                ToolCall(
                    id=call.id or f"call_{index}_{i}",
                    function=FunctionCall(name=call.name or "", arguments="".join(call.arguments)),
                )
                for i, call in sorted(parts.tool_calls.items())
            )
            message = AssistantMessage(
                content="".join(parts.content) if parts.content else None,
                refusal="".join(parts.refusal) if parts.refusal else None,
                tool_calls=tool_calls or None,
            )
            choices.append(Choice(index=index, message=message, finish_reason=parts.finish_reason))
        return ChatResponse(
            id=self._id or "",
            created=self._created,
            model=self._model,
            choices=tuple(choices),
            usage=self.usage,
        )


def synthesize_chunks(
    response: ChatResponse, *, include_usage: bool, chunk_chars: int = 32
) -> list[ChatChunk]:
    def chunk(choices: tuple[ChunkChoice, ...], usage: Usage | None = None) -> ChatChunk:
        return ChatChunk(
            id=response.id, created=response.created, model=response.model, choices=choices, usage=usage
        )

    out: list[ChatChunk] = []
    for choice in response.choices:
        msg = choice.message
        out.append(chunk((ChunkChoice(index=choice.index, delta=Delta(role="assistant")),)))
        text = msg.content or ""
        for start in range(0, len(text), chunk_chars):
            delta = Delta(content=text[start : start + chunk_chars])
            out.append(chunk((ChunkChoice(index=choice.index, delta=delta),)))
        if msg.refusal:
            out.append(chunk((ChunkChoice(index=choice.index, delta=Delta(refusal=msg.refusal)),)))
        for i, call in enumerate(msg.tool_calls or ()):
            delta = Delta(
                tool_calls=(
                    ToolCallDelta(
                        index=i,
                        id=call.id,
                        type="function",
                        function=FunctionCallDelta(
                            name=call.function.name, arguments=call.function.arguments
                        ),
                    ),
                )
            )
            out.append(chunk((ChunkChoice(index=choice.index, delta=delta),)))
        out.append(
            chunk((ChunkChoice(index=choice.index, delta=Delta(), finish_reason=choice.finish_reason),))
        )
    if include_usage and response.usage is not None:
        out.append(chunk((), usage=response.usage))
    return out


def synthesize_stream(response: ChatResponse, *, include_usage: bool, chunk_chars: int = 32) -> ChunkStream:
    chunks = synthesize_chunks(response, include_usage=include_usage, chunk_chars=chunk_chars)

    async def gen() -> AsyncIterator[ChatChunk]:
        for c in chunks:
            yield c

    return ChunkStream(gen())


async def collect_stream(stream: ChunkStream) -> ChatResponse:
    assembler = StreamAssembler()
    try:
        async for chunk in stream:
            assembler.feed(chunk)
    finally:
        await stream.aclose()
    return assembler.result()
