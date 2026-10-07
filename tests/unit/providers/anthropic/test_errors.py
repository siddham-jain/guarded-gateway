from datetime import UTC, datetime

import pytest

from gg.providers.anthropic.errors import classify, classify_stream_error, rate_limit_snapshot
from tests.unit.providers.support import fixture

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("status", "name", "kind", "code"),
    [
        (400, "err_400_invalid.json", "client", "invalid_request"),
        (400, "err_400_spend_limit.json", "auth", "spend_limit"),
        (400, "err_400_prompt_too_long.json", "fallback", "context_length"),
        (400, "err_400_prefill.json", "fallback", "capability_drift"),
        (401, "err_401.json", "auth", "auth"),
        (402, "err_402.json", "auth", "billing"),
        (403, "err_403.json", "auth", "permission"),
        (404, "err_404.json", "fallback", "model_not_found"),
        (413, "err_413.json", "client", "request_too_large"),
        (429, "err_429_rate_limit.json", "quota_minute", "rate_limited"),
        (429, "err_429_spend_cap.json", "auth", "billing"),
        (500, "err_500.json", "retryable", "upstream_error"),
        (504, "err_504.json", "retryable", "timeout"),
        (529, "err_529.json", "fallback", "overloaded"),
    ],
)
def test_fixture_errors(status: int, name: str, kind: str, code: str) -> None:
    err = classify(status, fixture("anthropic", name), {}, provider="anthropic", now=NOW)
    assert (err.kind, err.code, err.status) == (kind, code, status)
    assert err.scope == ("provider" if kind == "auth" else "deployment")
    assert err.upstream_request_id == "req_018EeWyXxfu5pfWkrYcMdjWG"
    assert err.message


def test_retry_after_and_request_id_header() -> None:
    headers = {"retry-after": "7", "request-id": "req_hdr"}
    err = classify(
        429, fixture("anthropic", "err_429_rate_limit.json"), headers, provider="anthropic", now=NOW
    )
    assert err.retry_after_s == 7.0
    assert err.upstream_request_id == "req_hdr"
    overloaded = classify(
        529, fixture("anthropic", "err_529.json"), {"retry-after": "3"}, provider="anthropic"
    )
    assert overloaded.retry_after_s == 3.0
    spend = classify(
        429, fixture("anthropic", "err_429_spend_cap.json"), {"retry-after": "3"}, provider="anthropic"
    )
    assert spend.retry_after_s is None


@pytest.mark.parametrize(
    ("status", "body", "kind", "code"),
    [
        (409, b"", "retryable", "upstream_error"),
        (503, b"<html>bad gateway</html>", "retryable", "overloaded"),
        (408, b"", "retryable", "timeout"),
        (422, b"{}", "client", "invalid_request"),
        (502, b"not json", "retryable", "upstream_error"),
    ],
)
def test_status_fallbacks_tolerate_odd_bodies(status: int, body: bytes, kind: str, code: str) -> None:
    err = classify(status, body, {}, provider="anthropic")
    assert (err.kind, err.code) == (kind, code)


@pytest.mark.parametrize(
    ("error_type", "kind", "code"),
    [
        ("overloaded_error", "fallback", "overloaded"),
        ("api_error", "retryable", "upstream_error"),
        ("rate_limit_error", "quota_minute", "rate_limited"),
        ("invalid_request_error", "client", "invalid_request"),
        ("billing_error", "auth", "billing"),
        ("something_new", "retryable", "upstream_error"),
    ],
)
def test_stream_error_events(error_type: str, kind: str, code: str) -> None:
    payload = {"type": "error", "error": {"type": error_type, "message": "m"}}
    err = classify_stream_error(payload, provider="anthropic")
    assert (err.kind, err.code, err.status) == (kind, code, 200)


def test_ratelimit_headers_parse_rfc3339_resets() -> None:
    headers = {
        "anthropic-ratelimit-requests-limit": "1000",
        "anthropic-ratelimit-requests-remaining": "999",
        "anthropic-ratelimit-requests-reset": "2026-10-05T12:00:30Z",
        "anthropic-ratelimit-tokens-limit": "2000000",
        "anthropic-ratelimit-tokens-remaining": "1990000",
        "anthropic-ratelimit-tokens-reset": "not a date",
    }
    snap = rate_limit_snapshot(headers, NOW)
    assert snap is not None
    assert (snap.limit_requests, snap.remaining_requests, snap.reset_requests_s) == (1000, 999, 30.0)
    assert (snap.limit_tokens, snap.remaining_tokens, snap.reset_tokens_s) == (2_000_000, 1_990_000, None)
    assert rate_limit_snapshot({"x-ratelimit-remaining-requests": "1"}, NOW) is None
