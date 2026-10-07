from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from gg.core.context import RequestContext
from gg.core.schema import ChatChunk, ChatResponse
from gg.pipeline.streams import ChunkStream

type ResultSource = Literal["upstream", "exact_cache", "semantic_cache", "synthetic"]
type StageOutcome = Literal["continued", "short_circuit", "rejected", "error"]


@dataclass(frozen=True, slots=True)
class PipelineResult:
    source: ResultSource
    response: ChatResponse | None = None
    stream: ChunkStream | None = None

    def map_response(self, fn: Callable[[ChatResponse], ChatResponse]) -> "PipelineResult":
        if self.response is None:
            return self
        return replace(self, response=fn(self.response))

    def map_stream(
        self, fn: Callable[[AsyncIterator[ChatChunk]], AsyncIterator[ChatChunk]]
    ) -> "PipelineResult":
        if self.stream is None:
            return self
        return replace(self, stream=self.stream.map(fn))


type Next = Callable[[RequestContext], Awaitable[PipelineResult]]
type Handler = Callable[[RequestContext], Awaitable[PipelineResult]]


class Stage(Protocol):
    name: str

    async def __call__(self, ctx: RequestContext, call_next: Next, /) -> PipelineResult: ...


class PipelineObserver(Protocol):
    def stage_finished(
        self, ctx: RequestContext, stage: str, exclusive_s: float, outcome: StageOutcome, /
    ) -> None: ...


class NullObserver:
    def stage_finished(
        self, ctx: RequestContext, stage: str, exclusive_s: float, outcome: StageOutcome, /
    ) -> None:
        return None
