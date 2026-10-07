import asyncio
from collections.abc import AsyncIterator

import httpx2
import openai
import pytest
from fastapi import FastAPI
from openai.types import Model
from pydantic import SecretStr
from tests.unit.api.fakes import TOKENS, build_app, make_services

from gg.api.deps import ApiServices
from gg.config.settings import MetricsSettings, ServerSettings, Settings
from gg.core.lifecycle import HealthStatus


def _client(app: FastAPI, key_id: str = "demo") -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url="http://gg.test/v1",
        api_key=TOKENS[key_id],
        http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
        max_retries=0,
    )


@pytest.fixture
def services() -> ApiServices:
    return make_services()


@pytest.fixture
def app(services: ApiServices) -> FastAPI:
    return build_app(services)


@pytest.fixture
async def raw(app: FastAPI) -> AsyncIterator[httpx2.AsyncClient]:
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://gg.test"
    ) as client:
        yield client


async def test_models_list_filtered_by_key(app: FastAPI) -> None:
    async with _client(app) as oai:
        page = await oai.models.list()
        ids = [m.id for m in page.data]
        assert all(isinstance(m, Model) for m in page.data)
        # demo allows gg/* and mock/*; premium needs a flag
        assert ids == ["gg/auto", "gg/weak", "mock/echo", "mock/free"]
        echo = next(m for m in page.data if m.id == "mock/echo")
        assert echo.owned_by == "mock"
        assert echo.model_extra is not None
        assert echo.model_extra["gg"] == {"context_window": 128_000}
    async with _client(app, "restricted") as oai:
        assert [m.id for m in (await oai.models.list()).data] == ["gg/auto", "gg/weak"]


async def test_models_retrieve(app: FastAPI) -> None:
    async with _client(app) as oai:
        model = await oai.models.retrieve("gg/auto")
        assert model.id == "gg/auto"
        assert (await oai.models.retrieve("mock/echo")).owned_by == "mock"
        with pytest.raises(openai.NotFoundError) as info:
            await oai.models.retrieve("other/big")
    assert info.value.code == "model_not_found"


async def test_models_cache_control(raw: httpx2.AsyncClient) -> None:
    response = await raw.get("/v1/models", headers={"authorization": f"Bearer {TOKENS['demo']}"})
    assert response.headers["cache-control"] == "private, max-age=60"


async def test_models_require_auth(raw: httpx2.AsyncClient) -> None:
    response = await raw.get("/v1/models")
    assert response.status_code == 401


async def test_healthz(raw: httpx2.AsyncClient) -> None:
    response = await raw.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"].startswith("req_")
    assert response.headers["referrer-policy"] == "no-referrer"
    assert (await raw.head("/healthz")).status_code == 200


class _Check:
    def __init__(self, name: str, status: str = "ok", *, gating: bool = True, delay_s: float = 0) -> None:
        self.name = name
        self._status = status
        self._gating = gating
        self._delay_s = delay_s

    async def check(self) -> HealthStatus:
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        return HealthStatus(name=self.name, status=self._status, gating=self._gating)  # pyright: ignore[reportArgumentType]


async def test_readyz_states() -> None:
    services = make_services(health_checks=(_Check("config"), _Check("breakers", "down", gating=False)))
    app = build_app(services)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://gg.test") as raw:
        response = await raw.get("/readyz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["config_hash"] == "0123456789abcdef"
        assert body["checks"]["breakers"]["status"] == "down"
        services.state.begin_drain()
        response = await raw.get("/readyz")
        assert response.status_code == 503
        assert response.json()["status"] == "draining"


@pytest.mark.parametrize("check", [_Check("redis", "down"), _Check("slow", delay_s=1)])
async def test_readyz_gating_failure(check: _Check) -> None:
    app = build_app(make_services(health_checks=(check,)))
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://gg.test") as raw:
        response = await raw.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


async def test_readyz_before_start() -> None:
    services = make_services()
    services.state.ready = False
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=build_app(services)), base_url="http://gg.test"
    ) as raw:
        assert (await raw.get("/readyz")).status_code == 503


async def test_version(raw: httpx2.AsyncClient) -> None:
    body = (await raw.get("/version")).json()
    assert set(body) == {"version", "git_sha", "image_digest", "built_at", "config_hash"}
    assert body["config_hash"] == "0123456789abcdef"


def _metrics_app(token: str | None, *, renderer: bool = True) -> FastAPI:
    settings = Settings(
        env="test",
        server=ServerSettings(),
        metrics=MetricsSettings(bearer_token=SecretStr(token) if token else None),
    )
    services = make_services(
        settings=settings,
        metrics_renderer=(lambda: (b"gg_up 1\n", "text/plain; version=0.0.4")) if renderer else None,
    )
    return build_app(services)


async def test_metrics_open() -> None:
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=_metrics_app(None)), base_url="http://gg.test"
    ) as raw:
        response = await raw.get("/metrics")
    assert response.status_code == 200
    assert response.text == "gg_up 1\n"
    assert response.headers["content-type"].startswith("text/plain")


async def test_metrics_bearer() -> None:
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=_metrics_app("s3cret")), base_url="http://gg.test"
    ) as raw:
        assert (await raw.get("/metrics")).status_code == 401
        assert (await raw.get("/metrics", headers={"authorization": "Bearer wrong"})).status_code == 401
        assert (await raw.get("/metrics", headers={"authorization": "Bearer s3cret"})).status_code == 200


async def test_metrics_disabled() -> None:
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=_metrics_app(None, renderer=False)), base_url="http://gg.test"
    ) as raw:
        assert (await raw.get("/metrics")).status_code == 404


async def test_cors_preflight() -> None:
    settings = Settings(env="test", server=ServerSettings(cors_origins=("https://app.example",)))
    app = build_app(make_services(settings=settings))
    headers = {
        "origin": "https://app.example",
        "access-control-request-method": "POST",
        "access-control-request-headers": "authorization, x-stainless-os",
    }
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://gg.test") as raw:
        allowed = await raw.options("/v1/chat/completions", headers=headers)
        denied = await raw.options(
            "/v1/chat/completions", headers={**headers, "origin": "https://evil.example"}
        )
        simple = await raw.get("/healthz", headers={"origin": "https://app.example"})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://app.example"
    assert "access-control-allow-origin" not in denied.headers
    assert "x-request-id" in simple.headers["access-control-expose-headers"]
