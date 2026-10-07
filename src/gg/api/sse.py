import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import anyio
import structlog
from starlette.responses import Response
from starlette.types import Receive

from gg.api.deps import ApiServices
from gg.api.disconnect import wait_for_disconnect
from gg.api.errors import to_gg_error
from gg.api.headers import build_response_headers
from gg.api.responses import FinalizingStreamResponse, error_outcome, finalize_request
from gg.core.context import RequestContext
from gg.core.errors import GGError, InternalError, ProviderError, UpstreamError
from gg.core.jsonutil import dumps
from gg.core.schema import ChatChunk, to_wire

log = structlog.get_logger("gg.api.sse")

DONE_FRAME = b"data: [DONE]\n\n"
KEEPALIVE_FRAME = b": keep-alive\n\n"
DISCONNECTED_STATUS = 499
_PRODUCER_STOP_S = 2.0


def data_frame(payload: Any) -> bytes:
    return b"data: " + dumps(payload) + b"\n\n"


class _Signal(Enum):
    END = "end"
    KEEPALIVE = "keepalive"
    COMMIT_TIMEOUT = "commit_timeout"


@dataclass(frozen=True, slots=True)
class _Chunk:
    chunk: ChatChunk


@dataclass(frozen=True, slots=True)
class _Failed:
    error: GGError


type _Msg = _Chunk | _Failed | _Signal


class _StalledError(Exception):
    pass


class _DisconnectedError(Exception):
    pass


def visible_chunk(chunk: ChatChunk, *, include_usage: bool) -> ChatChunk | None:
    """adapters always report usage; the client only sees it when it asked via stream_options"""
    if include_usage or chunk.usage is None:
        return chunk
    if not chunk.choices:
        return None
    return chunk.model_copy(update={"usage": None})


class StreamEncoder(Protocol):
    """turns canonical chunks and stream end states into wire frames for one request"""

    def chunk(self, chunk: ChatChunk) -> list[bytes]: ...

    def error(self, error: GGError) -> list[bytes]: ...

    def end(self) -> list[bytes]: ...

    def keepalive(self) -> bytes: ...


class OpenAIStreamEncoder:
    def __init__(self, *, include_usage: bool) -> None:
        self._include_usage = include_usage

    def chunk(self, chunk: ChatChunk) -> list[bytes]:
        shown = visible_chunk(chunk, include_usage=self._include_usage)
        return [] if shown is None else [data_frame(to_wire(shown))]

    def error(self, error: GGError) -> list[bytes]:
        return [data_frame(error.to_body()), DONE_FRAME]

    def end(self) -> list[bytes]:
        return [DONE_FRAME]

    def keepalive(self) -> bytes:
        return KEEPALIVE_FRAME


class SSEWriter:
    """runs the pipeline in one producer task and relays its chunks as sse frames.

    headers are committed on the first message or after sse_header_commit_s; errors before that are real
    http statuses, errors after it are an in-stream error event followed by [DONE].
    """

    def __init__(
        self,
        services: ApiServices,
        ctx: RequestContext,
        receive: Receive,
        encoder: StreamEncoder | None = None,
    ) -> None:
        self._services = services
        self._ctx = ctx
        self._receive = receive
        self._encoder = encoder or OpenAIStreamEncoder(include_usage=ctx.original.wants_usage())
        self._queue: asyncio.Queue[_Msg] = asyncio.Queue(maxsize=1)
        self._producer: asyncio.Task[None] | None = None
        self._body: AsyncGenerator[bytes] | None = None
        self._chunks_put = 0

    async def open(self) -> Response:
        ctx = self._ctx
        self._producer = asyncio.create_task(self._produce(), name=f"gg-sse-{ctx.request_id}")
        try:
            first = await self._first_message()
        except _DisconnectedError:
            ctx.outcome = "client_disconnected"
            await self.close()
            await finalize_request(ctx, DISCONNECTED_STATUS, self._services)
            return Response(status_code=DISCONNECTED_STATUS)
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self.close()
            raise
        if isinstance(first, _Failed):
            ctx.outcome = error_outcome(first.error)
            await self.close()
            raise first.error
        early = first is _Signal.COMMIT_TIMEOUT
        if early:
            log.info("sse.early_commit", waited_s=self._services.settings.server.sse_header_commit_s)
        headers = build_response_headers(ctx, streaming=True, early_commit=early)
        self._body = self._frames(first)
        return FinalizingStreamResponse(
            self._body, headers=headers, ctx=ctx, services=self._services, cleanup=self._finish
        )

    async def close(self) -> None:
        """closes the relay, cancels the producer (which closes the upstream) and waits for it"""
        if self._body is not None:
            await self._body.aclose()
        producer = self._producer
        if producer is not None:
            producer.cancel()
            await asyncio.wait((producer,), timeout=_PRODUCER_STOP_S)

    async def _finish(self) -> None:
        await self.close()
        # the relay sets the outcome when it ends the stream itself; otherwise the client went away
        if self._ctx.outcome is None:
            self._ctx.outcome = "client_disconnected"

    async def _first_message(self) -> _Msg:
        get = asyncio.create_task(self._queue.get())
        watch = asyncio.create_task(wait_for_disconnect(self._receive))
        try:
            done, _ = await asyncio.wait(
                (get, watch),
                timeout=self._services.settings.server.sse_header_commit_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            get.cancel()
            watch.cancel()
            await asyncio.gather(get, watch, return_exceptions=True)
        if not get.cancelled():
            return get.result()
        if watch in done:
            raise _DisconnectedError
        return _Signal.COMMIT_TIMEOUT

    async def _next_message(self) -> _Msg:
        try:
            async with asyncio.timeout(self._services.settings.server.sse_keepalive_s):
                return await self._queue.get()
        except TimeoutError:
            return _Signal.KEEPALIVE

    async def _frames(self, first: _Msg) -> AsyncGenerator[bytes]:
        ctx = self._ctx
        encoder = self._encoder
        msg = first
        while True:
            match msg:
                case _Chunk(chunk=chunk):
                    frames = encoder.chunk(chunk)
                    if frames:
                        ctx.timings.mark("client_first_byte")
                    for frame in frames:
                        yield frame
                case _Failed(error=error):
                    ctx.outcome = error_outcome(error)
                    for frame in encoder.error(error):
                        yield frame
                    return
                case _Signal.END:
                    ctx.outcome = ctx.outcome or "completed"
                    for frame in encoder.end():
                        yield frame
                    return
                case _Signal.KEEPALIVE | _Signal.COMMIT_TIMEOUT:
                    yield encoder.keepalive()
            msg = await self._next_message()

    async def _produce(self) -> None:
        final: _Msg = _Signal.END
        try:
            await self._pump()
        except asyncio.CancelledError:
            raise
        except _StalledError:
            log.warning("sse.producer_stalled")
            return
        except Exception as exc:
            final = _Failed(self._stream_error(exc))
        try:
            await self._put(final)
        except _StalledError:
            log.warning("sse.producer_stalled")

    async def _pump(self) -> None:
        result = await self._services.run_pipeline(self._ctx)
        stream = result.stream
        if stream is None:
            raise InternalError("Streaming request produced no stream.")
        try:
            async for chunk in stream:
                await self._put(_Chunk(chunk))
                self._chunks_put += 1
        finally:
            await stream.aclose()

    async def _put(self, msg: _Msg) -> None:
        # stall guard: a vanished consumer must not pin the producer past the request deadline
        try:
            async with asyncio.timeout(max(self._ctx.deadline.remaining(), 1.0)):
                await self._queue.put(msg)
        except TimeoutError:
            raise _StalledError from None

    def _stream_error(self, exc: Exception) -> GGError:
        if isinstance(exc, ProviderError) and self._chunks_put:
            return UpstreamError(
                "The upstream provider failed while streaming.", code="upstream_stream_error"
            )
        return to_gg_error(exc)
