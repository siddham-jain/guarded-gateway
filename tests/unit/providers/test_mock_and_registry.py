from typing import Any

import httpx2
import orjson
import pytest
from mock_upstream.app import app as mock_app
from pydantic import SecretStr

from gg.config.settings import Settings
from gg.core.clock import FakeClock
from gg.core.errors import ProviderError
from gg.providers.catalog.loader import build_catalog
from gg.providers.http import HttpClientFactory
from gg.providers.meta import ResponseMeta
from gg.providers.mock.adapter import MockAdapter
from gg.providers.openai_compat.adapter import OpenAICompatibleAdapter
from gg.providers.registry import adapter_registry, build_adapters
from gg.providers.runtime import AdapterDeps, ProviderRuntime
from tests.conftest import make_request
from tests.unit.providers.support import (
    collect,
    collect_until_error,
    ctx_for,
    make_dep,
    models_config,
    text_of,
)

TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]


def mock_adapter(env: str = "dev") -> MockAdapter:
    runtime = ProviderRuntime(name="mock", type="mock", base_url="mock://local")
    return MockAdapter(runtime, AdapterDeps(http=HttpClientFactory(), clock=FakeClock(), env=env))


def mock_dep(**scenario: Any):
    return make_dep("mock", "echo", defaults={"mock": {"text": "echo", **scenario}})


async def test_mock_echoes_last_user_message_in_chunks() -> None:
    request = make_request(messages=[{"role": "user", "content": "one two three four five six"}])
    chunks = await collect(mock_adapter(), request, mock_dep(chunk_tokens=2))
    assert text_of(chunks) == "one two three four five six"
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[0].choices[0].delta.content
    assert chunks[-2].choices[0].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].choices == ()
    meta = chunks[-1].gg_meta
    assert isinstance(meta, ResponseMeta)
    assert meta.usage is not None
    assert meta.usage.output_tokens > 0


async def test_mock_tool_call_and_fixed_text() -> None:
    chunks = await collect(
        mock_adapter(), make_request(tools=TOOLS), mock_dep(tool_call={"arguments": {"q": 1}})
    )
    call = chunks[0].choices[0].delta.tool_calls
    assert call
    assert call[0].function
    assert call[0].function.name == "lookup"
    assert orjson.loads(call[0].function.arguments or "") == {"q": 1}
    assert chunks[-2].choices[0].finish_reason == "tool_calls"
    fixed = await collect(mock_adapter(), make_request(), mock_dep(text="fixed reply"))
    assert text_of(fixed) == "fixed reply"


async def test_mock_fault_first_n_attempts_goes_through_real_classifier() -> None:
    adapter = mock_adapter()
    dep = mock_dep(fail={"status": 529, "first_n_attempts": 1})
    ctx = ctx_for(request_id="req_flaky")
    with pytest.raises(ProviderError) as exc:
        await collect(adapter, make_request(), dep, ctx)
    assert (exc.value.kind, exc.value.code, exc.value.committed) == ("retryable", "overloaded", False)
    assert text_of(await collect(adapter, make_request(), dep, ctx)) == "hello"


async def test_mock_midstream_fault_is_committed() -> None:
    adapter = mock_adapter()
    dep = mock_dep(chunk_tokens=1, fail={"status": 500, "after_chunks": 1})
    request = make_request(messages=[{"role": "user", "content": "a b c"}])
    seen, err = await collect_until_error(adapter, request, dep)
    assert len(seen) == 1
    assert err.committed


async def test_request_override_ignored_in_prod() -> None:
    request = make_request(mock={"text": "override"})
    assert text_of(await collect(mock_adapter(), request, mock_dep())) == "override"
    assert text_of(await collect(mock_adapter("prod"), request, mock_dep())) == "hello"


async def test_complete_via_mock() -> None:
    response = await mock_adapter().complete(make_request(), mock_dep(), ctx_for())
    assert response.choices[0].message.content == "hello"
    assert response.usage is not None


async def test_build_adapters_only_for_enabled_providers() -> None:
    settings = Settings(_env_file=None, providers={"groq": {"api_key": "k"}})  # pyright: ignore[reportCallIssue]
    cat = build_catalog(models_config(), settings)
    http = HttpClientFactory()
    adapters = build_adapters(cat, AdapterDeps(http=http))
    assert set(adapters) == {"mock", "groq"}
    assert isinstance(adapters["groq"], OpenAICompatibleAdapter)
    assert isinstance(adapters["mock"], MockAdapter)
    assert {"anthropic", "gemini", "mock", "openai_compat"} <= set(adapter_registry().names())
    await http.aclose()


async def test_openai_compat_adapter_against_mock_upstream_app() -> None:
    transport = httpx2.ASGITransport(app=mock_app)
    runtime = ProviderRuntime(
        name="mock_http", type="openai_compat", base_url="http://mock/v1", api_key=SecretStr("unused")
    )
    adapter = OpenAICompatibleAdapter(
        runtime, AdapterDeps(http=HttpClientFactory(default_transport=transport))
    )
    dep = make_dep("mock_http", "gpt-mock")
    request = make_request(messages=[{"role": "user", "content": "ping pong"}])
    chunks = await collect(adapter, request, dep)
    assert text_of(chunks) == "ping pong"
    meta = chunks[-1].gg_meta
    assert isinstance(meta, ResponseMeta)
    assert meta.usage is not None
    assert meta.usage.usage_source == "reported"


@pytest.mark.parametrize(
    ("headers", "kind", "code", "committed"),
    [
        ({"x-mock-error-status": "429"}, "quota_minute", "rate_limited", False),
        ({"x-mock-error-status": "529"}, "retryable", "overloaded", False),
        ({"x-mock-fail-after-chunks": "1"}, "retryable", "truncated", True),
        ({"x-mock-error-after-chunks": "0"}, "retryable", "upstream_error", False),
    ],
)
async def test_mock_upstream_fault_injection(
    headers: dict[str, str], kind: str, code: str, committed: bool
) -> None:
    transport = httpx2.ASGITransport(app=mock_app)
    runtime = ProviderRuntime(
        name="mock_http", type="openai_compat", base_url="http://mock/v1", quirks={"extra_headers": headers}
    )
    adapter = OpenAICompatibleAdapter(
        runtime, AdapterDeps(http=HttpClientFactory(default_transport=transport))
    )
    request = make_request(messages=[{"role": "user", "content": "a b c"}])
    with pytest.raises(ProviderError) as exc:
        await collect(adapter, request, make_dep("mock_http", "gpt-mock"))
    assert (exc.value.kind, exc.value.code, exc.value.committed) == (kind, code, committed)
