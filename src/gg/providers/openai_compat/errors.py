from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import orjson

from gg.core.errors import ProviderError, ProviderErrorKind
from gg.providers.errors import ErrorBody, parse_error_body, parse_retry_after
from gg.providers.openai_compat.quirks import ErrorRule, QuirkProfile

_CONTEXT = (
    "context_length",
    "context length",
    "context window",
    "maximum context",
    "prompt is too long",
    "input is too long",
    "too many tokens",
    "reduce the length",
    "prompt too long",
    "exceeds the model",
)
_UNSUPPORTED = (
    "unsupported_parameter",
    "unsupported_value",
    "unsupported parameter",
    "not supported",
    "unknown parameter",
    "unrecognized request argument",
    "extra inputs are not permitted",
    "extra_forbidden",
    "extra fields not permitted",
    "is not allowed",
)
_BILLING = (
    "insufficient_quota",
    "credit",
    "balance",
    "billing",
    "spend_limit",
    "spending limit",
    "usage_limit",
    "payment",
    "arrearage",
    "exceeded your current quota",
    "exceeded_current_quota",
)
_CONTENT = ("content_filter", "data_inspection", "datainspectionfailed", "sensitive", "moderation", "flagged")
_DAILY = ("per day", "per-day", "daily", "requests per day", "tokens per day", "rpd", "tpd")
_OVERLOAD = (
    "overloaded",
    "server_is_overloaded",
    "engine_overloaded",
    "capacity",
    "queue_full",
    "try again later",
)

_KIND_CODES: Mapping[ProviderErrorKind, str] = {
    "retryable": "upstream_error",
    "fallback": "unavailable",
    "client": "invalid_request",
    "auth": "auth",
    "quota_minute": "rate_limited",
    "quota_day": "per_day",
    "content_filter": "content_filter",
}


def _has(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def _header(headers: Mapping[str, str], names: tuple[str, ...]) -> str | None:
    lowered = {k.lower(): v for k, v in headers.items()}
    for name in names:
        if (value := lowered.get(name.lower())) is not None:
            return value
    return None


def _rule(eb: ErrorBody, status: int, quirks: QuirkProfile) -> ErrorRule | None:
    for candidate in (eb.code, eb.type):
        if candidate is not None and candidate in quirks.errors.code_map:
            return quirks.errors.code_map[candidate]
    return quirks.errors.status_map.get(status)


def _generic(status: int, eb: ErrorBody, headers: Mapping[str, str]) -> tuple[ProviderErrorKind, str]:
    text = eb.text()
    if status in (400, 413, 422):
        if _has(text, _CONTEXT):
            return "fallback", "context_length"
        if _has(text, _BILLING):
            return "auth", "billing"
        if _has(text, _CONTENT):
            return "content_filter", "content_filter"
        if _has(text, _UNSUPPORTED):
            return "fallback", "capability_drift"
        return "client", "invalid_request"
    if status in (401, 403):
        if _has(text, _BILLING):
            return "auth", "billing"
        return "auth", "auth" if status == 401 else "permission"
    if status == 402:
        return "auth", "billing"
    if status == 404:
        return "fallback", "model_not_found"
    if status == 410:
        return "fallback", "model_not_found"
    if status == 429:
        if _has(text, _BILLING):
            return "auth", "billing"
        if _has(text, _DAILY):
            return "quota_day", "per_day"
        if _has(text, ("engine_overloaded", "server_is_overloaded", "queue_full")):
            return "retryable", "overloaded"
        return "quota_minute", "rate_limited"
    if status == 498:
        return "fallback", "flex_capacity"
    if status == 499:
        return "client", "cancelled"
    if status in (503, 529) or (status >= 500 and _has(text, _OVERLOAD)):
        return "retryable", "overloaded"
    if status in (408, 504):
        return "retryable", "timeout"
    if status == 409 or status >= 500:
        return "retryable", "upstream_error"
    if 400 <= status < 500:
        return "client", "invalid_request"
    return "retryable", "bad_upstream_response"


def classify_payload(
    status: int,
    data: Any,
    headers: Mapping[str, str],
    *,
    provider: str,
    quirks: QuirkProfile,
    now: datetime | None = None,
) -> ProviderError:
    eb = parse_error_body(data)
    rule = _rule(eb, status, quirks)
    if rule is not None:
        kind, code = rule.kind, rule.code or _KIND_CODES[rule.kind]
    else:
        kind, code = _generic(status, eb, headers)
    if kind == "retryable" and _header(headers, ("x-should-retry",)) == "false":
        kind = "fallback"
    retry_after = (
        parse_retry_after(headers, now) if kind in ("retryable", "quota_minute", "quota_day") else None
    )
    reset_at = None
    if kind == "quota_day" and retry_after is not None:
        reset_at = (now or datetime.now(UTC)) + timedelta(seconds=retry_after)
    return ProviderError(
        kind,
        provider=provider,
        status=status,
        code=code,
        retry_after_s=retry_after,
        message=eb.message or f"upstream returned {status}",
        upstream_request_id=_header(headers, quirks.request_id_headers),
        quota_reset_at=reset_at,
        scope="provider" if kind == "auth" else "deployment",
    )


def classify(
    status: int,
    body: bytes,
    headers: Mapping[str, str],
    *,
    provider: str,
    quirks: QuirkProfile,
    now: datetime | None = None,
) -> ProviderError:
    try:
        data: Any = orjson.loads(body) if body.strip() else ""
    except orjson.JSONDecodeError:
        data = body.decode("utf-8", errors="replace")
    return classify_payload(status, data, headers, provider=provider, quirks=quirks, now=now)


def classify_stream_error(payload: Any, *, provider: str, quirks: QuirkProfile) -> ProviderError:
    """an in-band error event after a 200 (openai {"error":...}, openrouter error chunk, minimax base_resp)"""
    eb = parse_error_body(payload)
    status = int(eb.code) if eb.code is not None and eb.code.isdigit() and 400 <= int(eb.code) < 600 else 0
    if status or _rule(eb, 0, quirks) is not None:
        err = classify_payload(status, payload, {}, provider=provider, quirks=quirks)
        return ProviderError(
            err.kind,
            provider=provider,
            status=200,
            code=err.code,
            message=err.message,
            retry_after_s=err.retry_after_s,
            scope=err.scope,
        )
    text = eb.text()
    if _has(text, ("invalid_request", "invalid request")):
        kind, code = "client", "invalid_request"
    elif _has(text, _CONTEXT):
        kind, code = "fallback", "context_length"
    elif _has(text, _CONTENT):
        kind, code = "content_filter", "content_filter"
    elif _has(text, _OVERLOAD):
        kind, code = "retryable", "overloaded"
    else:
        kind, code = "retryable", "upstream_error"
    return ProviderError(
        kind, provider=provider, status=200, code=code, message=eb.message or "upstream stream error"
    )
