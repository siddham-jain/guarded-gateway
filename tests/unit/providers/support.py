from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import aclosing
from functools import cache
from pathlib import Path
from typing import Any

import httpx2
import orjson
from pydantic import SecretStr

from gg.config.loader import load_file
from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.deployment import Capabilities, Deployment
from gg.core.errors import ProviderError
from gg.core.schema import ChatChunk, ChatRequest
from gg.providers.catalog.loader import resolve_quirks
from gg.providers.catalog.schema import ModelsConfig
from gg.providers.http import HttpClientFactory
from gg.providers.openai_compat.adapter import OpenAICompatibleAdapter
from gg.providers.runtime import AdapterDeps, ProviderRuntime
from gg.providers.state.memory import InMemoryStateStore
from tests.conftest import make_ctx, make_request

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures"
MODELS_YAML = ROOT / "config" / "models.yaml"
SECRET = "sk-test-secret-123"

type Handler = Callable[[httpx2.Request], httpx2.Response]


def fixture(provider: str, name: str) -> bytes:
    return (FIXTURES / provider / name).read_bytes()


def sse(*events: Any, done: bool = True) -> bytes:
    parts = [b"data: " + orjson.dumps(e) + b"\n\n" for e in events]
    if done:
        parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


def text_chunk(
    content: str | None = None, *, finish: str | None = None, role: bool = False, **extra: Any
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = "assistant"
    if content is not None:
        delta["content"] = content
    return {
        "id": "up-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        **extra,
    }


def usage_chunk(prompt: int = 10, completion: int = 5) -> dict[str, Any]:
    return {
        "id": "up-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


class Parts(httpx2.AsyncByteStream):
    """response body delivered in several reads, to exercise split frames"""

    def __init__(self, parts: Iterable[bytes]) -> None:
        self.parts = list(parts)
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in self.parts:
            yield part

    async def aclose(self) -> None:
        self.closed = True


def sse_response(
    body: bytes, *, status: int = 200, headers: dict[str, str] | None = None, split: int | None = None
) -> httpx2.Response:
    parts = [body[i : i + split] for i in range(0, len(body), split)] if split else [body]
    return httpx2.Response(
        status, headers={"content-type": "text/event-stream", **(headers or {})}, stream=Parts(parts)
    )


def json_response(status: int, body: Any, headers: dict[str, str] | None = None) -> httpx2.Response:
    return httpx2.Response(
        status, content=orjson.dumps(body), headers={"content-type": "application/json", **(headers or {})}
    )


@cache
def models_config() -> ModelsConfig:
    return load_file(MODELS_YAML, ModelsConfig)


def profile_quirks(provider: str) -> dict[str, Any]:
    config = models_config()
    return resolve_quirks(config, config.providers[provider].quirks)


def compat_providers() -> list[str]:
    return sorted(n for n, p in models_config().providers.items() if p.type == "openai_compat")


def make_dep(provider: str = "test", model: str = "test-model", **caps: Any) -> Deployment:
    defaults = caps.pop("defaults", {})
    quirks = caps.pop("quirks", {})
    capabilities = Capabilities(**{"max_output": 1024, "logprobs": True, "n": True, **caps})
    return Deployment(
        id=f"{provider}/{model}",
        provider=provider,
        upstream_model=model,
        capabilities=capabilities,
        defaults=defaults,
        quirks=quirks,
    )


class Recorder:
    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.handler(request)

    def body(self, i: int = -1) -> dict[str, Any]:
        return orjson.loads(self.requests[i].content)


def make_adapter(
    handler: Handler,
    *,
    provider: str = "test",
    quirks: dict[str, Any] | None = None,
    base_url: str = "https://upstream.test/v1",
    max_in_flight: int | None = None,
    state: InMemoryStateStore | None = None,
) -> tuple[OpenAICompatibleAdapter, Recorder]:
    recorder = Recorder(handler)
    runtime = ProviderRuntime(
        name=provider,
        type="openai_compat",
        base_url=base_url,
        api_key=SecretStr(SECRET),
        quirks=quirks if quirks is not None else {},
        max_in_flight=max_in_flight,
    )
    deps = AdapterDeps(
        http=HttpClientFactory(transports={provider: httpx2.MockTransport(recorder)}),
        clock=FakeClock(),
        state=state or InMemoryStateStore(clock=FakeClock()),
    )
    return OpenAICompatibleAdapter(runtime, deps), recorder


def ctx_for(request: ChatRequest | None = None, request_id: str = "req_test") -> RequestContext:
    ctx = make_ctx(FakeClock(), request or make_request())
    ctx.request_id = request_id
    return ctx


async def collect(
    adapter: Any, request: ChatRequest, dep: Deployment, ctx: RequestContext | None = None
) -> list[ChatChunk]:
    async with aclosing(adapter.stream(request, dep, ctx or ctx_for(request))) as chunks:
        return [chunk async for chunk in chunks]


async def collect_until_error(
    adapter: Any, request: ChatRequest, dep: Deployment, ctx: RequestContext | None = None
) -> tuple[list[ChatChunk], ProviderError]:
    """chunks yielded before the stream raised, plus the error; fails if the stream ends cleanly"""
    seen: list[ChatChunk] = []
    try:
        async with aclosing(adapter.stream(request, dep, ctx or ctx_for(request))) as chunks:
            async for chunk in chunks:
                seen.append(chunk)  # noqa: PERF401 - partial output must survive the exception
    except ProviderError as err:
        return seen, err
    raise AssertionError("stream ended without a ProviderError")


def text_of(chunks: list[ChatChunk]) -> str:
    return "".join(c.delta.content or "" for ch in chunks for c in ch.choices)
