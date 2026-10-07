import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx2
import openai
import pytest
from fastapi import FastAPI
from tests.unit.api.fakes import TOKENS, FakePipeline, build_app, make_services, make_settings

from gg.auth.keys import generate_key
from gg.config.settings import ServerSettings, Settings

MESSAGES: Any = [{"role": "user", "content": "the secret prompt text"}]


def _client(app: FastAPI, api_key: str = TOKENS["demo"]) -> openai.AsyncOpenAI:
    transport = httpx2.ASGITransport(app=app, raise_app_exceptions=False)
    return openai.AsyncOpenAI(
        base_url="http://gg.test/v1",
        api_key=api_key,
        http_client=httpx2.AsyncClient(transport=transport),
        max_retries=0,
    )


def _raw(app: FastAPI) -> httpx2.AsyncClient:
    transport = httpx2.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx2.AsyncClient(transport=transport, base_url="http://gg.test")


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def app(pipeline: FakePipeline) -> FastAPI:
    return build_app(make_services(pipeline))


@pytest.fixture
async def raw(app: FastAPI) -> AsyncIterator[httpx2.AsyncClient]:
    async with _raw(app) as client:
        yield client


def _assert_error(response: httpx2.Response, status: int, code: str | None, param: str | None = None) -> None:
    assert response.status_code == status
    error = response.json()["error"]
    assert error["code"] == code
    assert error["param"] == param
    assert response.headers["x-should-retry"] == "false"
    assert response.headers["x-request-id"].startswith("req_")
    assert "secret prompt text" not in response.text


@pytest.mark.parametrize(
    ("key", "code"),
    [
        ("gg-live-" + "a" * 49, "invalid_api_key"),
        ("sk-proj-abcdefghijklmnop", "invalid_api_key"),
        (generate_key("test").token, "invalid_api_key"),
        (TOKENS["disabled"], "api_key_disabled"),
        (TOKENS["expired"], "api_key_expired"),
    ],
)
async def test_auth_failures(app: FastAPI, pipeline: FakePipeline, key: str, code: str) -> None:
    async with _client(app, key) as oai:
        with pytest.raises(openai.AuthenticationError) as info:
            await oai.chat.completions.create(model="mock/echo", messages=MESSAGES)
    assert info.value.status_code == 401
    assert info.value.code == code
    assert info.value.response.headers["www-authenticate"] == 'Bearer realm="gg"'
    assert info.value.response.headers["x-should-retry"] == "false"
    assert pipeline.calls == []


async def test_missing_key(raw: httpx2.AsyncClient) -> None:
    response = await raw.post("/v1/chat/completions", json={"model": "mock/echo", "messages": MESSAGES})
    _assert_error(response, 401, "missing_api_key")


async def test_test_key_rejected_in_prod() -> None:
    settings = Settings(env="prod", log_format="json", server=ServerSettings())
    app = build_app(make_services(settings=settings))
    async with _client(app) as oai:
        with pytest.raises(openai.AuthenticationError) as info:
            await oai.chat.completions.create(model="mock/echo", messages=MESSAGES)
    assert info.value.code == "invalid_api_key"


@pytest.mark.parametrize(
    ("token", "model", "exc_type", "status", "code"),
    [
        ("demo", "other/big", openai.PermissionDeniedError, 403, "model_not_allowed"),
        ("restricted", "mock/echo", openai.PermissionDeniedError, 403, "model_not_allowed"),
        ("demo", "mock/missing", openai.NotFoundError, 404, "model_not_found"),
        ("demo", "mock/premium", openai.PermissionDeniedError, 403, "model_not_allowed"),
    ],
)
async def test_model_authorization(
    app: FastAPI, token: str, model: str, exc_type: type[openai.APIStatusError], status: int, code: str
) -> None:
    async with _client(app, TOKENS[token]) as oai:
        with pytest.raises(exc_type) as info:
            await oai.chat.completions.create(model=model, messages=MESSAGES)
    assert info.value.status_code == status
    assert info.value.code == code
    assert info.value.param == "model"  # pyright: ignore[reportAttributeAccessIssue]


async def test_cap_exceeded(app: FastAPI) -> None:
    async with _client(app, TOKENS["restricted"]) as oai:
        with pytest.raises(openai.BadRequestError) as info:
            await oai.chat.completions.create(model="gg/auto", messages=MESSAGES, max_tokens=500)
    assert info.value.code == "invalid_value"
    assert info.value.param == "max_completion_tokens"  # pyright: ignore[reportAttributeAccessIssue]


async def test_unknown_gg_field(app: FastAPI) -> None:
    async with _client(app) as oai:
        with pytest.raises(openai.BadRequestError) as info:
            await oai.chat.completions.create(
                model="mock/echo", messages=MESSAGES, extra_body={"gg": {"nope": 1}}
            )
    assert info.value.code == "unknown_parameter"
    assert info.value.param == "gg.nope"  # pyright: ignore[reportAttributeAccessIssue]


async def test_gg_override_needs_permission(app: FastAPI) -> None:
    async with _client(app) as oai:
        with pytest.raises(openai.PermissionDeniedError) as info:
            await oai.chat.completions.create(
                model="gg/auto", messages=MESSAGES, extra_body={"gg": {"route_threshold": 0.3}}
            )
    assert info.value.code == "gg_override_not_allowed"


@pytest.mark.parametrize(
    ("content", "code", "param"),
    [
        (b"{not json", "invalid_json", None),
        (b"[1, 2]", "invalid_json", None),
        (
            b'{"messages": [{"role": "user", "content": "the secret prompt text"}]}',
            "missing_required_parameter",
            "model",
        ),
        (
            b'{"model": "mock/echo", "messages": [{"role": "user", "content": 5}]}',
            "invalid_type",
            "messages[0].content",
        ),
        (
            b'{"model": "mock/echo", "messages": [{"role": "user", "content": "x"}], "temperature": 9}',
            "invalid_value",
            "temperature",
        ),
        (
            b'{"model": "mock/echo", "messages": [{"role": "user", "content": "x"}], '
            b'"max_tokens": 5, "max_completion_tokens": 6}',
            "invalid_value",
            "max_tokens",
        ),
    ],
)
async def test_bad_bodies(raw: httpx2.AsyncClient, content: bytes, code: str, param: str | None) -> None:
    response = await raw.post(
        "/v1/chat/completions",
        content=content,
        headers={"authorization": f"Bearer {TOKENS['demo']}", "content-type": "application/json"},
    )
    _assert_error(response, 400, code, param)


@pytest.mark.parametrize(
    ("headers", "code"),
    [
        ({"content-type": "text/plain"}, "unsupported_media_type"),
        ({"content-type": "application/json", "content-encoding": "gzip"}, "unsupported_content_encoding"),
    ],
)
async def test_unsupported_media(raw: httpx2.AsyncClient, headers: dict[str, str], code: str) -> None:
    response = await raw.post(
        "/v1/chat/completions",
        content=b"{}",
        headers={"authorization": f"Bearer {TOKENS['demo']}", **headers},
    )
    _assert_error(response, 415, code)


async def test_body_too_large() -> None:
    app = build_app(make_services(settings=make_settings(max_body_bytes=200)))
    async with _client(app) as oai:
        with pytest.raises(openai.APIStatusError) as info:
            await oai.chat.completions.create(
                model="mock/echo", messages=[{"role": "user", "content": "x" * 500}]
            )
    assert info.value.status_code == 413
    assert info.value.code == "request_too_large"
    assert info.value.response.headers["connection"] == "close"


async def test_chunked_body_too_large() -> None:
    app = build_app(make_services(settings=make_settings(max_body_bytes=200)))

    async def body() -> AsyncIterator[bytes]:
        for _ in range(10):
            yield b"x" * 64

    async with _raw(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=body(),
            headers={"authorization": f"Bearer {TOKENS['demo']}", "content-type": "application/json"},
        )
    _assert_error(response, 413, "request_too_large")


async def test_slow_body_times_out() -> None:
    app = build_app(make_services(settings=make_settings(body_read_timeout_s=0.05)))

    async def body() -> AsyncIterator[bytes]:
        yield b'{"model": '
        await asyncio.sleep(0.3)
        yield b'"mock/echo"}'

    async with _raw(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=body(),
            headers={"authorization": f"Bearer {TOKENS['demo']}", "content-type": "application/json"},
        )
    assert response.status_code == 408
    assert response.json()["error"]["code"] == "request_timeout"


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [
        ("POST", "/v1/foo", 404),
        ("POST", "/v1/chat/completions/", 404),
        ("POST", "/chat/completions", 404),
        ("GET", "/v1/chat/completions", 405),
    ],
)
async def test_unknown_routes_are_openai_shaped(
    raw: httpx2.AsyncClient, method: str, path: str, status: int
) -> None:
    response = await raw.request(method, path, headers={"authorization": f"Bearer {TOKENS['demo']}"})
    _assert_error(response, status, None)
    assert response.json()["error"]["message"] == f"Invalid URL ({method} {path})"


async def test_draining_rejects_new_requests(app: FastAPI) -> None:
    app.state.services.state.begin_drain()
    async with _client(app) as oai:
        with pytest.raises(openai.InternalServerError) as info:
            await oai.chat.completions.create(model="mock/echo", messages=MESSAGES)
    assert info.value.status_code == 503
    assert info.value.code == "not_ready"
    assert info.value.response.headers["x-should-retry"] == "true"
    assert info.value.response.headers["retry-after"] == "1"


async def test_unhandled_exception_outside_chat_is_500(app: FastAPI) -> None:
    def explode(allowed: Any) -> Any:
        raise RuntimeError("catalog exploded with sk-secret")

    app.state.services.catalog.list_public = explode
    async with _raw(app) as client:
        response = await client.get("/v1/models", headers={"authorization": f"Bearer {TOKENS['demo']}"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert response.headers["x-request-id"].startswith("req_")
    assert response.headers["x-frame-options"] == "DENY"
    assert "sk-secret" not in response.text
