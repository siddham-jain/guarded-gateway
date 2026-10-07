import time
from collections.abc import AsyncIterator

import httpx2
import orjson
import pytest
from mock_upstream.app import app as mock_app

BODY = {"model": "gpt-mock", "messages": [{"role": "user", "content": "a b c"}]}


@pytest.fixture
async def client() -> AsyncIterator[httpx2.AsyncClient]:
    transport = httpx2.ASGITransport(app=mock_app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://mock") as client:
        await client.post("/_stats")
        yield client


async def test_knobs_shape_latency_and_tokens(client: httpx2.AsyncClient) -> None:
    started = time.perf_counter()
    headers = {"x-mock-ttft-ms": "10", "x-mock-itl-ms": "4", "x-mock-output-tokens": "5"}
    resp = await client.post("/v1/chat/completions", json=BODY, headers=headers)
    elapsed = time.perf_counter() - started
    assert resp.status_code == 200
    assert resp.json()["usage"]["completion_tokens"] == 5
    assert elapsed >= 0.03


async def test_env_knobs_apply_when_no_header(
    client: httpx2.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOCK_ERROR_STATUS", "503")
    resp = await client.post("/v1/chat/completions", json=BODY)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "server_is_overloaded"


async def test_stream_emits_tokens_then_done(client: httpx2.AsyncClient) -> None:
    resp = await client.post(
        "/v1/chat/completions",
        json={**BODY, "stream": True, "stream_options": {"include_usage": True}},
        headers={"x-mock-output-tokens": "4"},
    )
    events = [line[6:] for line in resp.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [orjson.loads(e) for e in events[:-1]]
    assert sum(1 for c in chunks if c["choices"] and c["choices"][0]["delta"].get("content")) == 4
    assert chunks[-1]["usage"]["completion_tokens"] == 4


async def test_stats_count_requests_and_return_to_idle(client: httpx2.AsyncClient) -> None:
    await client.post("/v1/chat/completions", json=BODY)
    await client.post("/v1/chat/completions", json={**BODY, "stream": True})
    await client.post("/v1/chat/completions", json=BODY, headers={"x-mock-error-status": "429"})
    stats = (await client.get("/_stats")).json()
    assert stats["requests"] == 3
    assert stats["in_flight"] == 0
    assert stats["peak_in_flight"] >= 1
    reset = (await client.post("/_stats")).json()
    assert reset == {"requests": 0, "in_flight": 0, "peak_in_flight": 0, "cancelled": 0}
