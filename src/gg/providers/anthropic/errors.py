from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

import orjson

from gg.core.errors import ProviderError, ProviderErrorKind
from gg.providers.errors import parse_retry_after
from gg.providers.meta import RateLimitSnapshot

REQUEST_ID_HEADER = "request-id"

_USAGE_LIMIT = ("specified api usage limits", "usage limit", "spend limit")
_CONTEXT = ("prompt is too long", "context window", "context length", "too many tokens")
_DRIFT = ("prefill", "tool_choice", "thinking", "budget_tokens", "temperature", "top_p", "effort")

# in-band sse `error` events (http 200 already sent)
_STREAM_KINDS: Mapping[str, tuple[ProviderErrorKind, str]] = {
    "overloaded_error": ("fallback", "overloaded"),
    "api_error": ("retryable", "upstream_error"),
    "timeout_error": ("retryable", "timeout"),
    "rate_limit_error": ("quota_minute", "rate_limited"),
    "authentication_error": ("auth", "auth"),
    "permission_error": ("auth", "permission"),
    "billing_error": ("auth", "billing"),
    "not_found_error": ("fallback", "model_not_found"),
    "invalid_request_error": ("client", "invalid_request"),
    "request_too_large": ("client", "request_too_large"),
}


class AnthropicErrorBody:
    """view over `{"type": "error", "error": {"type", "message", "details"}, "request_id"}`"""

    __slots__ = ("error_code", "message", "request_id", "type")

    def __init__(self, data: Any) -> None:
        self.type: str | None = None
        self.message = ""
        self.error_code: str | None = None
        self.request_id: str | None = None
        if isinstance(data, str):
            self.message = data[:2000]
            return
        if not isinstance(data, dict):
            return
        body = cast("dict[str, Any]", data)
        request_id = body.get("request_id")
        self.request_id = request_id if isinstance(request_id, str) else None
        error = body.get("error")
        if not isinstance(error, dict):
            return
        err = cast("dict[str, Any]", error)
        self.type = err.get("type") if isinstance(err.get("type"), str) else None
        message = err.get("message")
        self.message = str(message)[:2000] if message is not None else ""
        details = err.get("details")
        if isinstance(details, dict):
            code = cast("dict[str, Any]", details).get("error_code")
            self.error_code = code if isinstance(code, str) else None


def _has(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def _kind(status: int, eb: AnthropicErrorBody) -> tuple[ProviderErrorKind, str]:
    text = eb.message.lower()
    if status == 400:
        if _has(text, _USAGE_LIMIT):
            return "auth", "spend_limit"
        if _has(text, _CONTEXT):
            return "fallback", "context_length"
        if _has(text, _DRIFT):
            return "fallback", "capability_drift"
        return "client", "invalid_request"
    if status == 401:
        return "auth", "auth"
    if status == 402:
        return "auth", "billing"
    if status == 403:
        return "auth", "permission"
    if status == 404:
        return "fallback", "model_not_found"
    if status == 413:
        return "client", "request_too_large"
    if status == 429:
        if eb.error_code == "enforced_spend_limit_reached":
            return "auth", "billing"
        return "quota_minute", "rate_limited"
    if status == 529:
        # overloaded: try another deployment now (c4 §3.4), counts toward the breaker
        return "fallback", "overloaded"
    if status in (408, 504):
        return "retryable", "timeout"
    if status == 503:
        return "retryable", "overloaded"
    if status == 409 or status >= 500:
        return "retryable", "upstream_error"
    if 400 <= status < 500:
        return "client", "invalid_request"
    return "retryable", "bad_upstream_response"


def _load(body: bytes) -> Any:
    try:
        return orjson.loads(body) if body.strip() else ""
    except orjson.JSONDecodeError:
        return body.decode("utf-8", errors="replace")


def classify(
    status: int, body: bytes, headers: Mapping[str, str], *, provider: str, now: datetime | None = None
) -> ProviderError:
    eb = AnthropicErrorBody(_load(body))
    kind, code = _kind(status, eb)
    lowered = {k.lower(): v for k, v in headers.items()}
    retry_after = (
        parse_retry_after(headers, now) if kind in ("retryable", "quota_minute", "fallback") else None
    )
    return ProviderError(
        kind,
        provider=provider,
        status=status,
        code=code,
        retry_after_s=retry_after,
        message=eb.message or f"upstream returned {status}",
        upstream_request_id=lowered.get(REQUEST_ID_HEADER) or eb.request_id,
        scope="provider" if kind == "auth" else "deployment",
    )


def classify_stream_error(payload: Any, *, provider: str) -> ProviderError:
    eb = AnthropicErrorBody(payload)
    kind, code = _STREAM_KINDS.get(eb.type or "", ("retryable", "upstream_error"))
    return ProviderError(
        kind,
        provider=provider,
        status=200,
        code=code,
        message=eb.message or "upstream stream error",
        upstream_request_id=eb.request_id,
        scope="provider" if kind == "auth" else "deployment",
    )


def _reset_s(value: str | None, now: datetime) -> float | None:
    if value is None:
        return None
    try:
        when = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        return None
    return max(0.0, (when - now).total_seconds())


def _int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def rate_limit_snapshot(headers: Mapping[str, str], now: datetime) -> RateLimitSnapshot | None:
    """anthropic-ratelimit-{requests,tokens}-{limit,remaining,reset}; resets are rfc 3339 timestamps"""
    lowered = {k.lower(): v for k, v in headers.items()}

    def get(dimension: str, field: str) -> str | None:
        return lowered.get(f"anthropic-ratelimit-{dimension}-{field}")

    snapshot = RateLimitSnapshot(
        remaining_requests=_int(get("requests", "remaining")),
        remaining_tokens=_int(get("tokens", "remaining")),
        limit_requests=_int(get("requests", "limit")),
        limit_tokens=_int(get("tokens", "limit")),
        reset_requests_s=_reset_s(get("requests", "reset"), now),
        reset_tokens_s=_reset_s(get("tokens", "reset"), now),
    )
    return None if snapshot == RateLimitSnapshot() else snapshot
