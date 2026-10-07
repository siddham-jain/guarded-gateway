import asyncio
import socket
from collections.abc import AsyncIterator
from typing import Any

import httpx2
import pytest
import uvicorn
from tests.unit.api.fakes import TOKENS, FakePipeline, build_app, chunk, make_services, make_settings

from gg.api.deps import ApiServices

HEADERS = {"authorization": f"Bearer {TOKENS['demo']}"}
BODY: dict[str, Any] = {"model": "mock/echo", "messages": [{"role": "user", "content": "hi"}]}


async def _serve(services: ApiServices, http: str = "h11") -> AsyncIterator[str]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(build_app(services), log_level="warning", lifespan="off", http=http)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:  # noqa: ASYNC110
        await asyncio.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task
        sock.close()


async def _wait_for(predicate: Any, timeout_s: float = 2.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not predicate():  # noqa: ASYNC110
            await asyncio.sleep(0.01)


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline(chunks=[chunk(str(i)) for i in range(50)], chunk_delay_s=0.02)


@pytest.fixture(params=["h11", "httptools"])
async def base_url(request: pytest.FixtureRequest, pipeline: FakePipeline) -> AsyncIterator[str]:
    services = make_services(pipeline, settings=make_settings(sse_header_commit_s=0.05, sse_keepalive_s=0.05))
    async for url in _serve(services, http=request.param):
        yield url


async def test_disconnect_mid_stream_cancels_upstream(base_url: str, pipeline: FakePipeline) -> None:
    async with (
        httpx2.AsyncClient(base_url=base_url) as client,
        client.stream(
            "POST", "/v1/chat/completions", headers=HEADERS, json={**BODY, "stream": True}
        ) as response,
    ):
        assert response.status_code == 200
        frames = 0
        async for line in response.aiter_lines():
            if line.startswith("data: {"):
                frames += 1
            if frames == 3:
                break
    await _wait_for(lambda: pipeline.probe.closed and pipeline.finalizer_runs)
    assert pipeline.probe.cancelled
    assert pipeline.probe.yielded < 50
    assert pipeline.finalizer_runs[0][0] == "client_disconnected"


async def test_disconnect_while_priming(base_url: str, pipeline: FakePipeline) -> None:
    pipeline.prime_delay_s = 5
    async with (
        httpx2.AsyncClient(base_url=base_url) as client,
        client.stream(
            "POST", "/v1/chat/completions", headers=HEADERS, json={**BODY, "stream": True}
        ) as response,
    ):
        assert response.status_code == 200
        async for line in response.aiter_lines():
            assert line.startswith(": keep-alive") or line == ""
            break
    await _wait_for(lambda: bool(pipeline.finalizer_runs))
    assert pipeline.finalizer_runs[0][0] == "client_disconnected"


async def test_disconnect_before_header_commit() -> None:
    pipeline = FakePipeline(prime_delay_s=5)
    services = make_services(pipeline, settings=make_settings(sse_header_commit_s=10))
    async for url in _serve(services):
        async with httpx2.AsyncClient(base_url=url) as client:
            request = asyncio.create_task(
                client.post("/v1/chat/completions", headers=HEADERS, json={**BODY, "stream": True})
            )
            await _wait_for(pipeline.started.is_set)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        await _wait_for(lambda: bool(pipeline.finalizer_runs))
    assert pipeline.finalizer_runs[0][0] == "client_disconnected"


async def test_disconnect_during_non_stream_call() -> None:
    pipeline = FakePipeline(prime_delay_s=5)
    services = make_services(pipeline)
    async for url in _serve(services):
        async with httpx2.AsyncClient(base_url=url) as client:
            request = asyncio.create_task(client.post("/v1/chat/completions", headers=HEADERS, json=BODY))
            await _wait_for(pipeline.started.is_set)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        await _wait_for(lambda: bool(pipeline.finalizer_runs))
    assert pipeline.finalizer_runs[0][0] == "client_disconnected"
