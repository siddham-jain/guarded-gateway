from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2
import openai
import pytest
from fastapi import FastAPI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tests.unit.api.fakes import TOKENS, FakePipeline, build_app, chunk, make_services, make_settings

from gg.api.deps import ApiServices
from gg.core.errors import (
    GuardrailBlockedError,
    ProviderError,
    RateLimitedError,
    ServiceUnavailableError,
    UpstreamError,
    UpstreamTimeoutError,
)

BASE_URL = "http://gg.test/v1"
MESSAGES: Any = [{"role": "user", "content": "hi"}]

type ClientFactory = Callable[..., openai.AsyncOpenAI]


def make_client(app: FastAPI, api_key: str = TOKENS["demo"]) -> openai.AsyncOpenAI:
    transport = httpx2.ASGITransport(app=app, raise_app_exceptions=False)
    return openai.AsyncOpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        http_client=httpx2.AsyncClient(transport=transport),
        max_retries=0,
    )


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def services(pipeline: FakePipeline) -> ApiServices:
    return make_services(pipeline)


@pytest.fixture
def app(services: ApiServices) -> FastAPI:
    return build_app(services)


@pytest.fixture
async def oai(app: FastAPI) -> AsyncIterator[openai.AsyncOpenAI]:
    async with make_client(app) as client:
        yield client


async def test_non_stream_roundtrip(oai: openai.AsyncOpenAI, pipeline: FakePipeline) -> None:
    raw = await oai.chat.completions.with_raw_response.create(model="mock/echo", messages=MESSAGES)
    completion = raw.parse()
    assert isinstance(completion, ChatCompletion)
    assert completion.choices[0].message.content == "hello there"
    assert completion.usage is not None
    assert completion.usage.total_tokens == 5
    headers = raw.headers
    assert headers["x-request-id"].startswith("req_")
    assert headers["x-gg-provider"] == "mock"
    assert headers["x-gg-model"] == "mock/echo"
    assert headers["x-gg-config"] == "0123456789ab"
    assert headers["x-gg-route"] == "direct"
    assert headers["cache-control"] == "no-store"
    assert "auth;dur=" in headers["server-timing"]
    assert "gw;dur=" in headers["server-timing"]
    assert headers["x-content-type-options"] == "nosniff"
    assert pipeline.finalizer_runs == [("completed", pipeline.calls[0].request_id)]


async def test_inbound_credentials_never_reach_context(
    oai: openai.AsyncOpenAI, pipeline: FakePipeline
) -> None:
    await oai.chat.completions.create(model="mock/echo", messages=MESSAGES)
    ctx = pipeline.calls[0]
    assert TOKENS["demo"] not in repr(vars(ctx.request)) + repr(ctx.request.model_extra)
    assert ctx.key.id == "demo"


async def test_stream_roundtrip_with_usage(oai: openai.AsyncOpenAI, pipeline: FakePipeline) -> None:
    stream = await oai.chat.completions.create(
        model="mock/echo", messages=MESSAGES, stream=True, stream_options={"include_usage": True}
    )
    chunks = [c async for c in stream]
    assert all(isinstance(c, ChatCompletionChunk) for c in chunks)
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "hello there"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 5
    assert pipeline.finalizer_runs[0][0] == "completed"
    assert pipeline.probe.closed


async def test_stream_strips_usage_when_not_requested(oai: openai.AsyncOpenAI) -> None:
    stream = await oai.chat.completions.create(model="mock/echo", messages=MESSAGES, stream=True)
    chunks = [c async for c in stream]
    assert chunks
    assert all(c.usage is None for c in chunks)
    assert all(c.choices for c in chunks)


async def test_stream_raw_framing(app: FastAPI) -> None:
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://gg.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {TOKENS['demo']}"},
            json={"model": "mock/echo", "messages": MESSAGES, "stream": True},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert "content-length" not in response.headers
    events = response.text.split("\n\n")
    assert events[-1] == ""
    assert events[-2] == "data: [DONE]"
    assert all(e.startswith("data: {") for e in events[:-2])


@pytest.mark.parametrize(
    ("error", "status", "code", "should_retry", "exc_type"),
    [
        (
            RateLimitedError("slow down", retry_after_s=3),
            429,
            "rate_limit_exceeded",
            "true",
            openai.RateLimitError,
        ),
        (UpstreamError("all failed"), 502, "upstream_error", "false", openai.InternalServerError),
        (
            ServiceUnavailableError("none healthy"),
            503,
            "no_healthy_deployment",
            "false",
            openai.InternalServerError,
        ),
        (UpstreamTimeoutError("too slow"), 504, "upstream_timeout", "false", openai.InternalServerError),
        (GuardrailBlockedError("blocked"), 400, "guardrail_blocked", "false", openai.BadRequestError),
        (RuntimeError("secret detail sk-abc"), 500, "internal_error", "false", openai.InternalServerError),
        (
            ProviderError("fallback", provider="mock", status=500),
            502,
            "upstream_error",
            "false",
            openai.InternalServerError,
        ),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_pipeline_errors_before_commit_are_http_errors(
    pipeline: FakePipeline,
    oai: openai.AsyncOpenAI,
    error: BaseException,
    status: int,
    code: str,
    should_retry: str,
    exc_type: type[openai.APIStatusError],
    stream: bool,
) -> None:
    pipeline.error = error
    with pytest.raises(exc_type) as info:
        await oai.chat.completions.create(model="mock/echo", messages=MESSAGES, stream=stream)
    assert info.value.status_code == status
    assert info.value.code == code
    assert info.value.response.headers["x-should-retry"] == should_retry
    assert info.value.response.headers["x-request-id"].startswith("req_")
    assert "secret detail" not in info.value.response.text
    if status == 429:
        assert info.value.response.headers["retry-after"] == "3"
    expected = "internal_error" if status == 500 else ("rejected" if status < 500 else "upstream_error")
    assert pipeline.finalizer_runs == [(expected, pipeline.calls[0].request_id)]


async def _collect(stream: AsyncIterator[ChatCompletionChunk], into: list[ChatCompletionChunk]) -> None:
    async for c in stream:
        into.append(c)  # noqa: PERF401 - keeps the chunks seen before an error


async def test_mid_stream_error_becomes_error_event(oai: openai.AsyncOpenAI, pipeline: FakePipeline) -> None:
    pipeline.fail_after = 2
    pipeline.fail_with = ProviderError("retryable", provider="mock", status=500, committed=True)
    stream = await oai.chat.completions.create(model="mock/echo", messages=MESSAGES, stream=True)
    seen: list[ChatCompletionChunk] = []
    with pytest.raises(openai.APIError) as info:
        await _collect(stream, seen)
    assert len(seen) == 2
    assert "upstream provider failed while streaming" in info.value.message
    assert pipeline.finalizer_runs[0][0] == "upstream_error"
    assert pipeline.probe.closed


async def test_mid_stream_raw_error_event(app: FastAPI, pipeline: FakePipeline) -> None:
    pipeline.fail_after = 1
    pipeline.fail_with = RuntimeError("boom")
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://gg.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {TOKENS['demo']}"},
            json={"model": "mock/echo", "messages": MESSAGES, "stream": True},
        )
    events = [e for e in response.text.split("\n\n") if e]
    assert events[-1] == "data: [DONE]"
    assert events[-2].startswith('data: {"error":')
    assert '"code":"internal_error"' in events[-2]
    assert "finish_reason" not in events[-2]
    assert "boom" not in response.text


async def test_early_header_commit_sends_keepalives() -> None:
    pipeline = FakePipeline(prime_delay_s=0.2)
    services = make_services(pipeline, settings=make_settings(sse_header_commit_s=0.05, sse_keepalive_s=0.05))
    app = build_app(services)
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://gg.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {TOKENS['demo']}"},
            json={"model": "mock/echo", "messages": MESSAGES, "stream": True},
        )
    assert response.status_code == 200
    assert response.text.startswith(": keep-alive\n\n")
    assert "x-gg-provider" not in response.headers
    assert response.text.endswith("data: [DONE]\n\n")
    async with make_client(app) as oai:
        stream = await oai.chat.completions.create(model="mock/echo", messages=MESSAGES, stream=True)
        text = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])
    assert text == "hello there"


async def test_failure_after_early_commit_is_error_event() -> None:
    pipeline = FakePipeline(prime_delay_s=0.15, error=UpstreamError("all failed"))
    services = make_services(pipeline, settings=make_settings(sse_header_commit_s=0.05, sse_keepalive_s=0.05))
    async with make_client(build_app(services)) as oai:
        stream = await oai.chat.completions.create(model="mock/echo", messages=MESSAGES, stream=True)
        with pytest.raises(openai.APIError):
            await _collect(stream, [])
    assert pipeline.finalizer_runs[0][0] == "upstream_error"


async def test_keepalive_between_slow_chunks() -> None:
    pipeline = FakePipeline(chunks=[chunk("a"), chunk("b")], chunk_delay_s=0.12)
    services = make_services(pipeline, settings=make_settings(sse_header_commit_s=1, sse_keepalive_s=0.05))
    app = build_app(services)
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://gg.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {TOKENS['demo']}"},
            json={"model": "mock/echo", "messages": MESSAGES, "stream": True},
        )
    assert ": keep-alive" in response.text
    assert response.text.count("data: {") == 2


async def test_ignored_and_stripped_params_header(oai: openai.AsyncOpenAI, pipeline: FakePipeline) -> None:
    raw = await oai.chat.completions.with_raw_response.create(
        model="mock/echo",
        messages=MESSAGES,
        service_tier="priority",
        store=True,
        extra_body={"gg": {"cache": "off", "fallback": False, "session_id": "s1"}},
    )
    assert raw.headers["x-gg-ignored-params"] == "gg.cache,gg.session_id,service_tier,store"
    working = pipeline.calls[0].request
    assert "service_tier" not in (working.model_extra or {})
    assert "store" not in (working.model_extra or {})
    assert working.gg is not None
    assert working.gg.fallback is False
