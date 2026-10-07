from collections.abc import AsyncIterator

from gg.core.schema import ChatChunk, ChatResponse
from gg.pipeline.streams import StreamAssembler


class StreamRecorder:
    """tees the committed upstream stream; only a stream that ran to its end yields a response"""

    def __init__(self, max_chars: int) -> None:
        self._assembler = StreamAssembler()
        self._max_chars = max_chars
        self._chars = 0
        self.complete = False
        self.overflow = False

    def _feed(self, chunk: ChatChunk) -> None:
        for choice in chunk.choices:
            delta = choice.delta
            self._chars += len(delta.content or "") + len(delta.refusal or "")
            for call in delta.tool_calls or ():
                self._chars += len(call.function.arguments or "") if call.function else 0
        if self._chars > self._max_chars:
            self.overflow = True
            return
        self._assembler.feed(chunk)

    async def tee(self, source: AsyncIterator[ChatChunk]) -> AsyncIterator[ChatChunk]:
        async for chunk in source:
            if not self.overflow:
                self._feed(chunk)
            yield chunk
        # not reached on an error, a guard abort that raises, or a client disconnect (aclose)
        self.complete = True

    def response(self) -> ChatResponse | None:
        if not self.complete or self.overflow:
            return None
        return self._assembler.result()
