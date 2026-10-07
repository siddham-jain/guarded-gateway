from typing import Any

from gg.core.schema import ChatChunk, ChunkChoice, Delta, FinishReason, Usage
from gg.providers.sse import SSEEvent
from gg.providers.usage import TokenCounts


class StreamTranslator:
    """per-request state machine from upstream events to canonical chunks; subclasses stay pure (no i/o)

    feed() returns chunks for one event and may raise ProviderError; tail() returns the closing chunks
    (synthesised finish if needed, then the usage chunk) once the upstream stream ended cleanly.
    """

    def __init__(self, *, chunk_id: str, created: int, model: str) -> None:
        self.chunk_id = chunk_id
        self.created = created
        self.model = model
        self.finished = False
        self.served_model: str | None = None
        self.upstream_id: str | None = None
        self.upstream_provider: str | None = None
        self.provider_finish_reason: str | None = None
        self.counts: TokenCounts | None = None
        self.raw_usage: dict[str, Any] | None = None
        self.cost_usd: float | None = None
        self.flags: list[str] = []
        self.output_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.tool_call_ids: list[str] = []
        self._started: list[int] = []
        self._finished_choices: set[int] = set()

    def feed(self, event: SSEEvent) -> list[ChatChunk]:
        raise NotImplementedError

    def flag(self, name: str) -> None:
        if name not in self.flags:
            self.flags.append(name)

    def chunk(self, choices: tuple[ChunkChoice, ...], usage: Usage | None = None, **extra: Any) -> ChatChunk:
        return ChatChunk.model_construct(
            id=self.chunk_id, created=self.created, model=self.model, choices=choices, usage=usage, **extra
        )

    def choice(self, index: int, *, finish_reason: FinishReason | None = None, **delta: Any) -> ChunkChoice:
        if index not in self._started:
            self._started.append(index)
            delta.setdefault("role", "assistant")
        if finish_reason is not None:
            self._finished_choices.add(index)
        return ChunkChoice.model_construct(
            index=index, delta=Delta.model_construct(**delta), finish_reason=finish_reason
        )

    def all_started_finished(self) -> bool:
        return bool(self._started) and all(i in self._finished_choices for i in self._started)

    def tail(self, usage: Usage) -> list[ChatChunk]:
        pending = [i for i in self._started if i not in self._finished_choices] or (
            [] if self._started else [0]
        )
        out: list[ChatChunk] = []
        if pending:
            self.flag("synthesized_finish")
            out.append(self.chunk(tuple(self.choice(i, finish_reason="stop") for i in pending)))
        out.append(self.chunk((), usage=usage))
        return out

    def output_text(self) -> str:
        return "".join(self.output_parts)
