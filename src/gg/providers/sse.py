import codecs
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

_LINE_END = re.compile(r"\r\n|\r|\n")


@dataclass(frozen=True, slots=True)
class SSEEvent:
    data: str
    event: str = "message"
    id: str | None = None


class SSEParser:
    """incremental whatwg event-stream parser; safe for utf-8 sequences and line ends split across reads"""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer = ""
        self._pending_cr = False
        self._data: list[str] = []
        self._event = ""
        self._id: str | None = None
        self._last_id: str | None = None
        self._first = True

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        return self._consume(self._decoder.decode(chunk))

    def flush(self) -> list[SSEEvent]:
        # a trailing event without the final blank line is still dispatched at eof
        events = self._consume(self._decoder.decode(b"", final=True))
        if self._buffer:
            self._line(self._buffer, events)
            self._buffer = ""
        self._dispatch(events)
        return events

    def _consume(self, text: str) -> list[SSEEvent]:
        events: list[SSEEvent] = []
        if not text:
            return events
        if self._first:
            text = text.removeprefix("\ufeff")
            self._first = False
        if self._pending_cr and text.startswith("\n"):
            text = text[1:]
        self._pending_cr = False
        buf = self._buffer + text
        start = 0
        for match in _LINE_END.finditer(buf):
            self._line(buf[start : match.start()], events)
            start = match.end()
        self._buffer = buf[start:]
        # a trailing cr may be the first half of a crlf split across reads
        self._pending_cr = buf.endswith("\r")
        return events

    def _line(self, line: str, events: list[SSEEvent]) -> None:
        if not line:
            self._dispatch(events)
            return
        if line.startswith(":"):
            return
        name, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._data.append(value)
        elif name == "event":
            self._event = value
        elif name == "id" and "\x00" not in value:
            self._id = value

    def _dispatch(self, events: list[SSEEvent]) -> None:
        if self._id is not None:
            self._last_id = self._id
            self._id = None
        if self._data:
            events.append(SSEEvent("\n".join(self._data), self._event or "message", self._last_id))
        self._data = []
        self._event = ""


async def aiter_sse(chunks: AsyncIterator[bytes]) -> AsyncIterator[SSEEvent]:
    parser = SSEParser()
    async for chunk in chunks:
        for event in parser.feed(chunk):
            yield event
    for event in parser.flush():
        yield event
