from datetime import UTC, datetime
from typing import Any

import httpx2
import pytest

from gg.core.errors import ProviderError
from gg.providers.errors import (
    after_commit,
    parse_duration,
    parse_error_body,
    parse_retry_after,
    transport_error,
)
from gg.providers.openai_compat.errors import classify, classify_stream_error
from gg.providers.openai_compat.quirks import QuirkProfile
from tests.unit.providers.support import fixture, profile_quirks

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("6m0s", 360.0),
        ("1s", 1.0),
        ("1.5s", 1.5),
        ("200ms", 0.2),
        ("1h2m3s", 3723.0),
        ("39s", 39.0),
        ("12", 12.0),
        ("0.5", 0.5),
        ("", None),
        ("soon", None),
        ("5x", None),
        ("s5", None),
    ],
)
def test_parse_duration(raw: str, expected: float | None) -> None:
    assert parse_duration(raw) == expected


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "7"}, 7.0),
        ({"Retry-After": "1.5"}, 1.5),
        ({"retry-after-ms": "250"}, 0.25),
        ({"retry-after-ms": "250", "retry-after": "9"}, 0.25),
        ({"retry-after": "Mon, 05 Oct 2026 12:00:30 GMT"}, 30.0),
        ({"retry-after": "1000"}, 120.0),
        ({"retry-after": "-3"}, None),
        ({"retry-after": "garbage"}, None),
        ({"retry-after-ms": "x", "retry-after": "2"}, 2.0),
        ({}, None),
    ],
)
def test_parse_retry_after(headers: dict[str, str], expected: float | None) -> None:
    assert parse_retry_after(headers, NOW) == expected


@pytest.mark.parametrize(
    ("data", "message", "code", "type_"),
    [
        ({"error": {"message": "m", "type": "t", "code": "c"}}, "m", "c", "t"),
        ({"error": "Invalid username or password."}, "Invalid username or password.", None, None),
        ({"code": "not-found", "error": "no model"}, "no model", "not-found", None),
        ({"detail": "Not authenticated"}, "Not authenticated", None, None),
        ({"code": 403, "reason": "NOT_ENOUGH_BALANCE", "message": "nb"}, "nb", "403", "NOT_ENOUGH_BALANCE"),
        ({"success": False, "errors": [{"code": 3036, "message": "daily"}]}, "daily", "3036", None),
        ({"error": {"code": 402, "message": "credits", "metadata": {"a": 1}}}, "credits", "402", None),
        ("plain text", "plain text", None, None),
        (None, "", None, None),
    ],
)
def test_parse_error_body_envelopes(data: Any, message: str, code: str | None, type_: str | None) -> None:
    body = parse_error_body(data)
    assert (body.message, body.code, body.type) == (message, code, type_)


def run(
    status: int,
    body: bytes | dict[str, Any] | str,
    headers: dict[str, str] | None = None,
    provider: str = "openai",
) -> ProviderError:
    raw = (
        body
        if isinstance(body, bytes)
        else (body.encode() if isinstance(body, str) else httpx2.Response(200, json=body).content)
    )
    quirks = (
        QuirkProfile.model_validate(profile_quirks(provider)) if provider != "generic" else QuirkProfile()
    )
    return classify(status, raw, headers or {}, provider=provider, quirks=quirks, now=NOW)


@pytest.mark.parametrize(
    ("status", "name", "kind", "code"),
    [
        (429, "err_429_rate_limit.json", "quota_minute", "rate_limited"),
        (429, "err_429_insufficient_quota.json", "auth", "billing"),
        (400, "err_400_context_length.json", "fallback", "context_length"),
        (400, "err_400_unsupported_param.json", "fallback", "capability_drift"),
        (503, "err_503_overloaded.json", "retryable", "overloaded"),
        (401, "err_401.json", "auth", "auth"),
    ],
)
def test_openai_fixture_errors(status: int, name: str, kind: str, code: str) -> None:
    err = run(status, fixture("openai", name), {"retry-after": "3", "x-request-id": "req_up"})
    assert (err.kind, err.code) == (kind, code)
    assert err.upstream_request_id == "req_up"
    assert err.scope == ("provider" if kind == "auth" else "deployment")
    assert err.retry_after_s == (3.0 if kind in ("quota_minute", "retryable") else None)


@pytest.mark.parametrize(
    ("status", "body", "kind", "code"),
    [
        (402, {"error": {"message": "pay"}}, "auth", "billing"),
        (403, {"error": {"message": "region"}}, "auth", "permission"),
        (404, {"error": {"message": "no such model"}}, "fallback", "model_not_found"),
        (408, "", "retryable", "timeout"),
        (409, "", "retryable", "upstream_error"),
        (413, {"error": {"message": "too big"}}, "client", "invalid_request"),
        (422, {"detail": [{"msg": "extra inputs are not permitted"}]}, "fallback", "capability_drift"),
        (429, {"error": {"message": "Limit tokens per day reached"}}, "quota_day", "per_day"),
        (429, {"error": {"type": "engine_overloaded_error", "message": "busy"}}, "retryable", "overloaded"),
        (498, {"error": {"message": "flex"}}, "fallback", "flex_capacity"),
        (499, "", "client", "cancelled"),
        (500, "<html>oops</html>", "retryable", "upstream_error"),
        (502, "", "retryable", "upstream_error"),
        (529, {"error": {"type": "overloaded_error"}}, "retryable", "overloaded"),
        (
            400,
            {
                "error": {
                    "message": "Input data may contain inappropriate content",
                    "code": "data_inspection_failed",
                }
            },
            "content_filter",
            "content_filter",
        ),
        (400, {"error": {"message": "bad role"}}, "client", "invalid_request"),
    ],
)
def test_generic_classification(status: int, body: Any, kind: str, code: str) -> None:
    err = run(status, body, provider="generic")
    assert (err.kind, err.code, err.status) == (kind, code, status)


def test_should_retry_false_turns_retryable_into_fallback() -> None:
    assert run(500, "", {"x-should-retry": "false"}, provider="generic").kind == "fallback"


def test_quota_day_reset_time_from_retry_after() -> None:
    err = run(429, {"error": {"message": "requests per day"}}, {"retry-after": "60"}, provider="generic")
    assert err.quota_reset_at == datetime(2026, 10, 5, 12, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("provider", "status", "name", "kind", "code"),
    [
        ("zai", 400, "err_1301.json", "content_filter", "content_filter"),
        ("zai", 429, "err_1113.json", "auth", "billing"),
        ("deepseek", 402, "err_402.json", "auth", "billing"),
        ("openrouter", 402, "err_402.json", "auth", "billing"),
        ("openrouter", 403, "err_403_moderation.json", "content_filter", "content_filter"),
        ("cloudflare", 429, "err_3036.json", "quota_day", "per_day"),
        ("novita", 403, "err_balance.json", "auth", "billing"),
        ("nebius", 422, "err_422_detail.json", "fallback", "capability_drift"),
        ("xai", 404, "err_404.json", "fallback", "model_not_found"),
        ("ollama", 404, "err_404_not_pulled.json", "fallback", "model_not_found"),
        ("groq", 498, "err_498_flex.json", "fallback", "flex_capacity"),
    ],
)
def test_provider_profile_errors(provider: str, status: int, name: str, kind: str, code: str) -> None:
    err = run(status, fixture(provider, name), provider=provider)
    assert (err.kind, err.code) == (kind, code)


def test_status_map_overrides_generic_rule() -> None:
    assert run(503, {"error": {"message": "no provider"}}, provider="openrouter").code == "no_provider"
    assert run(503, "", provider="ollama").kind == "fallback"
    assert run(498, "", provider="cohere").kind == "auth"


def test_stream_error_classification() -> None:
    quirks = QuirkProfile()
    server = classify_stream_error(
        {"error": {"type": "server_error", "message": "x"}}, provider="p", quirks=quirks
    )
    assert (server.kind, server.status) == ("retryable", 200)
    invalid = classify_stream_error({"error": {"type": "invalid_request_error"}}, provider="p", quirks=quirks)
    assert invalid.kind == "client"
    upstream = classify_stream_error(
        {"error": {"code": 502, "message": "Provider disconnected"}}, provider="p", quirks=quirks
    )
    assert upstream.kind == "retryable"


def test_after_commit_copies_and_transport_errors() -> None:
    err = ProviderError("retryable", provider="p", status=500, code="x")
    committed = after_commit(err)
    assert committed.committed
    assert not err.committed
    assert committed.code == "x"
    assert after_commit(committed) is committed
    assert transport_error(httpx2.ConnectError("refused"), provider="p").code == "connect_error"
    timeout = transport_error(httpx2.ReadTimeout("slow"), provider="p")
    assert (timeout.kind, timeout.status, timeout.code) == ("retryable", 0, "timeout")
