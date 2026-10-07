"""the real composition root, end to end, driven by the stock openai sdk"""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest
import yaml
from fastapi import FastAPI
from mock_upstream.app import app as mock_upstream_app
from openai.types.chat import ChatCompletion

from gg.app.factory import Overrides, build_app
from gg.config.settings import Settings
from gg.core.clock import SystemClock
from gg.limits.spend_guard import InMemorySpendGuard, SpendGuardSettings

ROOT = Path(__file__).resolve().parents[2]
TOKENS: dict[str, str] = yaml.safe_load((ROOT / "tests/fixtures/keys/tokens.yaml").read_text())
MESSAGES: Any = [{"role": "user", "content": "ping from the walking skeleton"}]


def settings(profile: str, **extra: Any) -> Settings:
    return Settings(  # pyright: ignore[reportCallIssue]
        _env_file=None,
        env="test",
        log_format="console",
        log_level="warning",
        config_dir=ROOT / "config",
        keys_file=ROOT / "tests/fixtures/keys/keys.yaml",
        model_profile=profile,
        **extra,
    )


def spend_guard() -> InMemorySpendGuard:
    caps = SpendGuardSettings(  # pyright: ignore[reportCallIssue]
        _env_file=None, cap_daily_usd="1.00", cap_total_usd="1.00", cap_run_usd="1.00"
    )
    return InMemorySpendGuard(caps, clock=SystemClock())


@pytest.fixture
async def ci_app() -> AsyncIterator[FastAPI]:
    app = build_app(settings("ci"), overrides=Overrides(spend_guard=spend_guard()))
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def http_app() -> AsyncIterator[FastAPI]:
    # mock_http is a real openai_compat adapter speaking http to mock_upstream, in process
    overrides = Overrides(
        transports={"mock_http": httpx2.ASGITransport(app=mock_upstream_app)}, spend_guard=spend_guard()
    )
    app = build_app(
        settings("loadtest", providers={"mock_http": {"base_url": "http://mock-upstream/v1"}}),
        overrides=overrides,
    )
    async with app.router.lifespan_context(app):
        yield app


def client(app: FastAPI) -> openai.AsyncOpenAI:
    transport = httpx2.ASGITransport(app=app)
    return openai.AsyncOpenAI(
        base_url="http://gg.test/v1",
        api_key=TOKENS["demo"],
        http_client=httpx2.AsyncClient(transport=transport),
        max_retries=0,
    )


async def test_non_stream_through_router_alias(ci_app: FastAPI) -> None:
    raw = await client(ci_app).chat.completions.with_raw_response.create(model="gg/auto", messages=MESSAGES)
    completion: ChatCompletion = raw.parse()
    assert completion.choices[0].message.content == "ping from the walking skeleton"
    assert completion.usage is not None
    assert raw.headers["x-gg-provider"] == "mock"
    assert raw.headers["x-gg-route"] == "strong"
    assert raw.headers["x-request-id"]


async def test_stream_with_usage(ci_app: FastAPI) -> None:
    stream = await client(ci_app).chat.completions.create(
        model="gg/auto", messages=MESSAGES, stream=True, stream_options={"include_usage": True}
    )
    text, usage = "", None
    async for chunk in stream:
        if chunk.choices:
            text += chunk.choices[0].delta.content or ""
        if chunk.usage is not None:
            usage = chunk.usage
    assert text == "ping from the walking skeleton"
    assert usage is not None


async def test_fallback_hides_a_failing_first_deployment(ci_app: FastAPI) -> None:
    raw = await client(ci_app).chat.completions.with_raw_response.create(
        model="gg/resilient", messages=MESSAGES
    )
    assert raw.parse().choices[0].message.content == "ping from the walking skeleton"
    assert raw.headers["x-gg-attempts"] == "2"


async def test_models_list_and_metrics(ci_app: FastAPI) -> None:
    models = await client(ci_app).models.list()
    ids = {m.id async for m in models}
    assert {"gg/auto", "gg/weak", "gg/strong"} <= ids
    await client(ci_app).chat.completions.create(model="gg/weak", messages=MESSAGES)
    await client(ci_app).chat.completions.create(model="gg/auto", messages=MESSAGES)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=ci_app)) as http:
        body = (await http.get("http://gg.test/metrics")).text
    assert "gg_requests_total" in body
    assert "gg_gateway_overhead_seconds" in body
    assert 'gg_routing_decisions_total{alias="gg/auto"' in body


async def test_bad_key_is_rejected(ci_app: FastAPI) -> None:
    bad = openai.AsyncOpenAI(
        base_url="http://gg.test/v1",
        api_key="sk-not-a-gg-key",
        http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=ci_app)),
        max_retries=0,
    )
    with pytest.raises(openai.AuthenticationError):
        await bad.chat.completions.create(model="gg/auto", messages=MESSAGES)


async def test_openai_compatible_adapter_over_http(http_app: FastAPI) -> None:
    gg = client(http_app)
    completion = await gg.chat.completions.create(model="gg/weak", messages=MESSAGES)
    assert completion.choices[0].message.content
    stream = await gg.chat.completions.create(model="gg/strong", messages=MESSAGES, stream=True)
    chunks = [c async for c in stream]
    assert any(c.choices and c.choices[0].delta.content for c in chunks)


async def test_rate_limit_headers_and_exact_cache_hit(ci_app: FastAPI) -> None:
    gg = client(ci_app)
    first = await gg.chat.completions.with_raw_response.create(
        model="gg/weak", messages=MESSAGES, temperature=0
    )
    second = await gg.chat.completions.with_raw_response.create(
        model="gg/weak", messages=MESSAGES, temperature=0
    )
    assert first.headers["x-gg-cache"] == "MISS"
    assert second.headers["x-gg-cache"] == "HIT"
    assert second.parse().choices[0].message.content == first.parse().choices[0].message.content
    assert "x-ratelimit-remaining-requests" in first.headers


async def test_component_metrics_are_exported(ci_app: FastAPI) -> None:
    await client(ci_app).chat.completions.create(model="gg/weak", messages=MESSAGES, temperature=0)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=ci_app)) as http:
        body = (await http.get("http://gg.test/metrics")).text
    for name in ("gg_cache_lookups_total", "gg_guardrail_decisions_total", "gg_circuit_state"):
        assert name in body, name


async def test_demo_chaos_falls_over_to_the_next_deployment(ci_app: FastAPI) -> None:
    raw = await client(ci_app).chat.completions.with_raw_response.create(
        model="gg/demo-chaos", messages=MESSAGES
    )
    assert raw.headers["x-gg-model"] == "mock/echo"
    assert raw.headers["x-gg-fallback"] == "true"
