from datetime import UTC, datetime

import orjson
import pytest

from gg.providers.gemini.errors import classify, classify_stream_error, next_pacific_midnight, parse_status
from tests.unit.providers.support import fixture

NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)


def err(name: str, status: int, headers: dict[str, str] | None = None, now: datetime = NOW):
    return classify(status, fixture("gemini", name), headers or {}, provider="gemini", now=now)


@pytest.mark.parametrize(
    ("name", "status", "kind", "code"),
    [
        ("err_400_api_key_invalid.json", 400, "auth", "auth"),
        ("err_400_failed_precondition.json", 400, "auth", "billing_or_region"),
        ("err_400_missing_signature.json", 400, "fallback", "missing_thought_signature"),
        ("err_400_thinking_level.json", 400, "fallback", "capability_drift"),
        ("err_400_invalid_argument.json", 400, "client", "invalid_request"),
        ("err_402.json", 402, "auth", "billing"),
        ("err_403.json", 403, "auth", "permission"),
        ("err_404.json", 404, "fallback", "model_not_found"),
        ("err_429_per_minute.json", 429, "quota_minute", "rate_limited"),
        ("err_429_per_day.json", 429, "quota_day", "per_day"),
        ("err_500_context_too_long.json", 500, "fallback", "context_length"),
        ("err_503.json", 503, "retryable", "overloaded"),
    ],
)
def test_fixture_classification(name: str, status: int, kind: str, code: str) -> None:
    error = err(name, status)
    assert (error.kind, error.code, error.status) == (kind, code, status)
    assert error.scope == ("provider" if kind == "auth" else "deployment")
    assert error.message


def test_per_minute_uses_retry_info_over_headers() -> None:
    error = err("err_429_per_minute.json", 429, {"retry-after": "5"})
    assert error.retry_after_s == 39.0
    assert error.quota_reset_at is None


def test_per_day_resets_at_next_pacific_midnight_and_ignores_retry_delay() -> None:
    error = err("err_429_per_day.json", 429)
    assert error.retry_after_s is None
    # 2026-10-05 12:00 utc is 05:00 pdt; the next pacific midnight is 2026-10-06 07:00 utc
    assert error.quota_reset_at == datetime(2026, 10, 6, 7, tzinfo=UTC)


@pytest.mark.parametrize(
    ("now", "reset"),
    [
        # 23:59 pdt on oct 31: the reset is nov 1 00:00, still pdt (utc-7)
        (datetime(2026, 11, 1, 6, 59, tzinfo=UTC), datetime(2026, 11, 1, 7, tzinfo=UTC)),
        # dst ends 2026-11-01 02:00 local; the next midnight is pst (utc-8)
        (datetime(2026, 11, 1, 7, 0, tzinfo=UTC), datetime(2026, 11, 2, 8, tzinfo=UTC)),
        # dst starts 2027-03-14 02:00 local; the next midnight is pdt (utc-7)
        (datetime(2027, 3, 14, 9, tzinfo=UTC), datetime(2027, 3, 15, 7, tzinfo=UTC)),
    ],
)
def test_pacific_midnight_is_dst_correct(now: datetime, reset: datetime) -> None:
    assert next_pacific_midnight(now) == reset


def test_unknown_429_falls_back_to_retry_after_header() -> None:
    body = orjson.dumps({"error": {"code": 429, "message": "slow down", "status": "RESOURCE_EXHAUSTED"}})
    error = classify(429, body, {"Retry-After": "7"}, provider="gemini", now=NOW)
    assert (error.kind, error.retry_after_s) == ("quota_minute", 7.0)


def test_retry_delay_is_capped() -> None:
    body = orjson.dumps(
        {
            "error": {
                "code": 429,
                "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3600s"}],
            }
        }
    )
    assert classify(429, body, {}, provider="gemini", now=NOW).retry_after_s == 120.0


@pytest.mark.parametrize(
    ("status", "kind", "code"),
    [
        (401, "auth", "auth"),
        (408, "retryable", "timeout"),
        (409, "client", "invalid_request"),
        (499, "client", "cancelled"),
        (500, "retryable", "upstream_error"),
        (504, "retryable", "timeout"),
    ],
)
def test_status_only_classification(status: int, kind: str, code: str) -> None:
    error = classify(status, b"", {}, provider="gemini", now=NOW)
    assert (error.kind, error.code) == (kind, code)
    assert error.message == f"upstream returned {status}"


def test_non_json_and_array_bodies() -> None:
    assert classify(502, b"<html>bad gateway</html>", {}, provider="gemini", now=NOW).message.startswith(
        "<html>"
    )
    wrapped = b"[" + fixture("gemini", "err_400_api_key_invalid.json") + b"]"
    assert classify(400, wrapped, {}, provider="gemini", now=NOW).kind == "auth"


def test_parse_status_reads_rpc_details() -> None:
    rpc = parse_status(orjson.loads(fixture("gemini", "err_429_per_day.json")))
    assert rpc.code == 429
    assert rpc.status == "RESOURCE_EXHAUSTED"
    assert rpc.quota_ids == ("GenerateRequestsPerDayPerProjectPerModel-FreeTier",)
    assert rpc.retry_delay_s == 1.0


def test_stream_error_keeps_status_200_and_kind_from_code() -> None:
    error = classify_stream_error(
        {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}},
        provider="gemini",
        now=NOW,
    )
    assert (error.kind, error.status) == ("quota_minute", 200)
    bare = classify_stream_error({"error": {"message": "boom"}}, provider="gemini", now=NOW)
    assert (bare.kind, bare.code, bare.status) == ("retryable", "upstream_error", 200)
