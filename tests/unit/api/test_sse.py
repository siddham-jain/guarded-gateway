import asyncio
from typing import Any

import pytest
from starlette.responses import Response
from starlette.types import Message, Receive

from gg.api.responses import FinalizingStreamResponse
from gg.api.sse import DONE_FRAME, KEEPALIVE_FRAME, SSEWriter, data_frame, visible_chunk
from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.errors import InternalError, UpstreamError
from gg.core.schema import ChatChunk, Usage
from gg.pipeline.stage import PipelineResult
from tests.conftest import make_ctx, make_request
from tests.unit.api.fakes import FakePipeline, chunk, make_services, make_settings, response_for

SCOPE: dict[str, Any] = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}}


class Client:
    """fake asgi peer: records what was sent and can disconnect after n data frames"""

    def __init__(self, disconnect_after: int | None = None) -> None:
        self.messages: list[Message] = []
        self.disconnected = asyncio.Event()
        self._disconnect_after = disconnect_after

    @property
    def body(self) -> bytes:
        return b"".join(m.get("body", b"") for m in self.messages if m["type"] == "http.response.body")

    async def receive(self) -> Message:
        await self.disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(self, message: Message) -> None:
        self.messages.append(message)
        if self._disconnect_after is not None and self.body.count(b"data: {") >= self._disconnect_after:
            self.disconnected.set()


def _ctx(clock: FakeClock, **request: Any) -> RequestContext:
    return make_ctx(clock, make_request(model="mock/echo", stream=True, **request))


def _writer(pipeline: Any, ctx: RequestContext, receive: Receive, **server: Any) -> SSEWriter:
    settings = make_settings(**{"sse_header_commit_s": 1.0, "sse_keepalive_s": 1.0, **server})
    return SSEWriter(make_services(pipeline, settings=settings), ctx, receive)


async def _drive(response: Response, client: Client) -> None:
    await response(SCOPE, client.receive, client.send)


async def test_happy_path(clock: FakeClock) -> None:
    pipeline = FakePipeline()
    ctx = _ctx(clock)
    client = Client()
    response = await _writer(pipeline, ctx, client.receive).open()
    assert isinstance(response, FinalizingStreamResponse)
    assert response.headers["x-gg-provider"] == "mock"
    await _drive(response, client)
    frames = client.body.split(b"\n\n")
    assert frames[-2] == b"data: [DONE]"
    assert all(f.startswith(b"data: {") for f in frames[:-2])
    assert ctx.outcome == "completed"
    assert pipeline.finalizer_runs == [("completed", "req_test")]
    assert pipeline.probe.closed
    assert "client_first_byte" in ctx.timings.marks


async def test_failure_before_commit_raises(clock: FakeClock) -> None:
    pipeline = FakePipeline(error=UpstreamError("down"))
    ctx = _ctx(clock)
    with pytest.raises(UpstreamError):
        await _writer(pipeline, ctx, Client().receive).open()
    assert ctx.outcome == "upstream_error"
    assert pipeline.finalizer_runs == []


async def test_failure_on_first_chunk_is_http_error(clock: FakeClock) -> None:
    pipeline = FakePipeline(fail_after=0, fail_with=RuntimeError("x"))
    with pytest.raises(InternalError):
        await _writer(pipeline, _ctx(clock), Client().receive).open()
    assert pipeline.probe.closed


async def test_missing_stream_is_internal_error(clock: FakeClock) -> None:
    async def run(ctx: RequestContext) -> PipelineResult:
        return PipelineResult(source="upstream", response=response_for(ctx))

    with pytest.raises(InternalError):
        await _writer(run, _ctx(clock), Client().receive).open()


async def test_early_commit_then_content(clock: FakeClock) -> None:
    pipeline = FakePipeline(prime_delay_s=0.1)
    ctx = _ctx(clock)
    client = Client()
    response = await _writer(
        pipeline, ctx, client.receive, sse_header_commit_s=0.02, sse_keepalive_s=0.03
    ).open()
    assert "x-gg-provider" not in response.headers
    await _drive(response, client)
    assert client.body.startswith(KEEPALIVE_FRAME)
    assert client.body.endswith(DONE_FRAME)
    assert ctx.outcome == "completed"


async def test_failure_after_commit_is_error_event(clock: FakeClock) -> None:
    pipeline = FakePipeline(fail_after=2, fail_with=UpstreamError("gone", code="upstream_stream_error"))
    ctx = _ctx(clock)
    client = Client()
    await _drive(await _writer(pipeline, ctx, client.receive).open(), client)
    frames = [f for f in client.body.split(b"\n\n") if f]
    assert frames[-1] == b"data: [DONE]"
    assert frames[-2] == data_frame(UpstreamError("gone", code="upstream_stream_error").to_body()).strip()
    assert ctx.outcome == "upstream_error"
    assert pipeline.finalizer_runs == [("upstream_error", "req_test")]


async def test_disconnect_before_commit(clock: FakeClock) -> None:
    pipeline = FakePipeline(prime_delay_s=5)
    ctx = _ctx(clock)
    client = Client()
    asyncio.get_running_loop().call_later(0.02, client.disconnected.set)
    response = await _writer(pipeline, ctx, client.receive, sse_header_commit_s=10).open()
    assert response.status_code == 499
    assert ctx.outcome == "client_disconnected"
    assert pipeline.finalizer_runs == [("client_disconnected", "req_test")]


async def test_disconnect_mid_stream_cancels_producer(clock: FakeClock) -> None:
    pipeline = FakePipeline(chunks=[chunk(str(i)) for i in range(100)], chunk_delay_s=0.005)
    ctx = _ctx(clock)
    client = Client(disconnect_after=3)
    await _drive(await _writer(pipeline, ctx, client.receive).open(), client)
    assert pipeline.probe.closed
    assert pipeline.probe.cancelled
    assert pipeline.probe.yielded < 100
    assert DONE_FRAME not in client.body
    assert ctx.outcome == "client_disconnected"
    assert pipeline.finalizer_runs == [("client_disconnected", "req_test")]


async def test_keepalive_cadence(clock: FakeClock) -> None:
    pipeline = FakePipeline(chunks=[chunk("a"), chunk("b")], chunk_delay_s=0.15)
    client = Client()
    response = await _writer(
        pipeline, _ctx(clock), client.receive, sse_header_commit_s=1, sse_keepalive_s=0.04
    ).open()
    await _drive(response, client)
    between = client.body.split(b"data: {")[1]
    assert between.count(KEEPALIVE_FRAME) >= 2


async def test_cancelled_response_still_finalizes(clock: FakeClock) -> None:
    pipeline = FakePipeline(chunks=[chunk(str(i)) for i in range(100)], chunk_delay_s=0.01)
    ctx = _ctx(clock)
    client = Client()
    response = await _writer(pipeline, ctx, client.receive).open()
    task = asyncio.create_task(_drive(response, client))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pipeline.probe.closed
    assert pipeline.finalizer_runs == [("client_disconnected", "req_test")]


def _usage_chunk(choices: bool) -> ChatChunk:
    base = chunk("x")
    usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return base.model_copy(update={"usage": usage, "choices": base.choices if choices else ()})


def test_visible_chunk() -> None:
    plain = chunk("x")
    assert visible_chunk(plain, include_usage=False) is plain
    assert visible_chunk(_usage_chunk(False), include_usage=False) is None
    stripped = visible_chunk(_usage_chunk(True), include_usage=False)
    assert stripped is not None
    assert stripped.usage is None
    assert stripped.choices
    kept = _usage_chunk(False)
    assert visible_chunk(kept, include_usage=True) is kept


def test_data_frame() -> None:
    assert data_frame({"a": 1}) == b'data: {"a":1}\n\n'
