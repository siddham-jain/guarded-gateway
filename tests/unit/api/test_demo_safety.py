from collections.abc import AsyncIterator
from dataclasses import replace

import httpx2
import pytest
from fastapi import FastAPI

from gg.api.deps import ApiServices
from gg.auth.config import KeysConfig
from gg.auth.keys import generate_key
from gg.auth.resolver import CachingKeyResolver
from gg.auth.store import YamlKeyStore
from gg.config.settings import AuthFailureSettings, Settings
from gg.core.clock import SystemClock
from tests.unit.api.fakes import TOKENS, build_app, make_services

BAD_TOKEN = "gg-test-" + "x" * 49
BODY = {"model": "gg/weak", "messages": [{"role": "user", "content": "hi"}]}


async def _client(app: FastAPI) -> AsyncIterator[httpx2.AsyncClient]:
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://gg.test"
    ) as client:
        yield client


async def _chat(client: httpx2.AsyncClient, token: str) -> httpx2.Response:
    return await client.post("/v1/chat/completions", json=BODY, headers={"authorization": f"Bearer {token}"})


@pytest.fixture
async def limited() -> AsyncIterator[httpx2.AsyncClient]:
    settings = Settings(env="test", auth_failures=AuthFailureSettings(max_failures=3, window_s=60))
    async for client in _client(build_app(make_services(settings=settings))):
        yield client


async def test_failed_auth_limiter_blocks_the_ip(limited: httpx2.AsyncClient) -> None:
    for _ in range(3):
        assert (await _chat(limited, BAD_TOKEN)).status_code == 401
    blocked = await _chat(limited, TOKENS["demo"])
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "too_many_auth_failures"
    assert int(blocked.headers["retry-after"]) > 0
    assert "x-request-id" in blocked.headers
    assert (await limited.get("/healthz")).status_code == 200


async def test_successful_requests_do_not_count(limited: httpx2.AsyncClient) -> None:
    for _ in range(5):
        assert (await _chat(limited, TOKENS["demo"])).status_code == 200
    assert (await _chat(limited, BAD_TOKEN)).status_code == 401


async def test_limiter_can_be_disabled() -> None:
    settings = Settings(env="test", auth_failures=AuthFailureSettings(enabled=False, max_failures=1))
    async for client in _client(build_app(make_services(settings=settings))):
        for _ in range(3):
            assert (await _chat(client, BAD_TOKEN)).status_code == 401


def _with_admin(services: ApiServices) -> tuple[ApiServices, str]:
    admin = generate_key("test")
    keys = KeysConfig.model_validate(
        {
            "keys": [
                {
                    "id": "ops",
                    "name": "Ops",
                    "hash": admin.hash,
                    "prefix": admin.prefix,
                    "created_at": "2026-10-07",
                    "tags": ["admin"],
                }
            ]
        }
    )
    resolver = CachingKeyResolver(YamlKeyStore(keys), SystemClock(), allow_test_keys=True)
    return replace(services, keys=resolver), admin.token


async def test_maintenance_returns_503_except_for_admin_keys() -> None:
    services, admin_token = _with_admin(make_services(settings=Settings(env="test", maintenance=True)))
    async for client in _client(build_app(services)):
        down = await _chat(client, BAD_TOKEN)
        assert down.status_code == 503
        assert down.json()["error"]["code"] == "maintenance"
        assert "retry-after" in down.headers
        assert (await client.get("/v1/models", headers={"authorization": "Bearer nope"})).status_code == 503
        assert (await _chat(client, admin_token)).status_code == 200
        assert (await client.get("/healthz")).status_code == 200
        assert (await client.get("/readyz")).status_code == 200


async def test_maintenance_off_by_default() -> None:
    async for client in _client(build_app(make_services())):
        assert (await _chat(client, TOKENS["demo"])).status_code == 200


async def test_middleware_errors_use_the_anthropic_shape_on_messages() -> None:
    async for client in _client(build_app(make_services(settings=Settings(env="test", maintenance=True)))):
        down = await client.post("/v1/messages", json={}, headers={"x-api-key": "nope"})
        assert down.status_code == 503
        assert down.json()["type"] == "error"
        assert down.json()["error"]["type"] == "overloaded_error"
        assert down.json()["error"]["details"]["code"] == "maintenance"
