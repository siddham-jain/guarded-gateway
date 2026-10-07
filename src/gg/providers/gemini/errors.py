from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta, timezone, tzinfo
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import orjson

from gg.core.errors import ProviderError, ProviderErrorKind
from gg.providers.errors import MAX_RETRY_AFTER_S, parse_duration, parse_retry_after

_CONTEXT = (
    "context is too long",
    "input context",
    "exceeds the maximum number of tokens",
    "too many tokens",
    "prompt is too long",
)
_SIGNATURE = ("thought_signature", "thought signature")
_UNSUPPORTED = ("thinking level", "thinking_level", "thinkinglevel", "not supported", "unsupported")
_RETRYABLE_STATUS = {408: "timeout", 503: "overloaded", 504: "timeout"}


@dataclass(frozen=True, slots=True)
class RpcStatus:
    """the parts of a google.rpc.Status error body gg acts on"""

    code: int | None = None
    status: str | None = None
    message: str = ""
    reasons: tuple[str, ...] = ()
    quota_ids: tuple[str, ...] = ()
    retry_delay_s: float | None = None


def _dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [cast("dict[str, Any]", v) for v in cast("list[Any]", value) if isinstance(v, dict)]


def parse_status(data: Any) -> RpcStatus:
    if isinstance(data, list) and data:
        data = cast("list[Any]", data)[0]
    if isinstance(data, str):
        return RpcStatus(message=data[:2000])
    if not isinstance(data, dict):
        return RpcStatus()
    error: Any = cast("dict[str, Any]", data).get("error")
    if isinstance(error, str):
        return RpcStatus(message=error[:2000])
    if not isinstance(error, dict):
        return RpcStatus()
    err = cast("dict[str, Any]", error)
    reasons: list[str] = []
    quota_ids: list[str] = []
    delay: float | None = None
    for detail in _dicts(err.get("details")):
        kind = str(detail.get("@type", ""))
        if kind.endswith("google.rpc.ErrorInfo") and isinstance(detail.get("reason"), str):
            reasons.append(detail["reason"])
        elif kind.endswith("google.rpc.QuotaFailure"):
            quota_ids.extend(str(v.get("quotaId", "")) for v in _dicts(detail.get("violations")))
        elif kind.endswith("google.rpc.RetryInfo") and isinstance(detail.get("retryDelay"), str):
            delay = parse_duration(detail["retryDelay"])
    code = err.get("code")
    status = err.get("status")
    return RpcStatus(
        code=code if isinstance(code, int) and not isinstance(code, bool) else None,
        status=status if isinstance(status, str) else None,
        message=str(err.get("message") or "")[:2000],
        reasons=tuple(reasons),
        quota_ids=tuple(quota_ids),
        retry_delay_s=delay,
    )


def _pacific() -> tzinfo:
    try:
        return ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError:
        # no tz database in the image: pst is never earlier than the real reset
        return timezone(timedelta(hours=-8))


def next_pacific_midnight(now: datetime) -> datetime:
    """gemini per-day quotas reset at midnight pacific time"""
    tz = _pacific()
    local = now.astimezone(tz)
    return datetime.combine(local.date() + timedelta(days=1), time(), tzinfo=tz).astimezone(UTC)


def _has(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def _kind(status: int, rpc: RpcStatus) -> tuple[ProviderErrorKind, str]:
    text = rpc.message.lower()
    if "API_KEY_INVALID" in rpc.reasons:
        return "auth", "auth"
    if status == 400:
        if rpc.status == "FAILED_PRECONDITION":
            return "auth", "billing_or_region"
        if _has(text, _SIGNATURE):
            return "fallback", "missing_thought_signature"
        if _has(text, _CONTEXT):
            return "fallback", "context_length"
        if _has(text, _UNSUPPORTED):
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
    if status == 429:
        if any("perday" in q.lower() for q in rpc.quota_ids):
            return "quota_day", "per_day"
        return "quota_minute", "rate_limited"
    if status == 499:
        return "client", "cancelled"
    if status in _RETRYABLE_STATUS:
        return "retryable", _RETRYABLE_STATUS[status]
    if status >= 500:
        if _has(text, _CONTEXT):
            return "fallback", "context_length"
        return "retryable", "upstream_error"
    if 400 <= status < 500:
        return "client", "invalid_request"
    return "retryable", "upstream_error"


def classify_payload(
    status: int, data: Any, headers: Mapping[str, str], *, provider: str, now: datetime
) -> ProviderError:
    rpc = parse_status(data)
    kind, code = _kind(status, rpc)
    retry_after: float | None = None
    reset_at: datetime | None = None
    if kind == "quota_day":
        # retryDelay is not trusted here: per-day 429s persist until the pacific reset
        reset_at = next_pacific_midnight(now)
    elif kind in ("retryable", "quota_minute"):
        delay = rpc.retry_delay_s
        retry_after = min(max(delay, 0.0), MAX_RETRY_AFTER_S) if delay is not None else None
        retry_after = retry_after if retry_after is not None else parse_retry_after(headers, now)
    return ProviderError(
        kind,
        provider=provider,
        status=status,
        code=code,
        retry_after_s=retry_after,
        message=rpc.message or f"upstream returned {status}",
        quota_reset_at=reset_at,
        scope="provider" if kind == "auth" else "deployment",
    )


def classify(
    status: int, body: bytes, headers: Mapping[str, str], *, provider: str, now: datetime
) -> ProviderError:
    try:
        data: Any = orjson.loads(body) if body.strip() else None
    except orjson.JSONDecodeError:
        data = body.decode("utf-8", errors="replace")
    return classify_payload(status, data, headers, provider=provider, now=now)


def classify_stream_error(payload: Any, *, provider: str, now: datetime) -> ProviderError:
    """an {"error": {...}} event after the 200; its code is the http status gemini would have sent"""
    rpc = parse_status(payload)
    status = rpc.code if rpc.code is not None and 400 <= rpc.code < 600 else 0
    err = classify_payload(status, payload, {}, provider=provider, now=now)
    return ProviderError(
        err.kind,
        provider=provider,
        status=200,
        code=err.code,
        message=rpc.message or "upstream stream error",
        retry_after_s=err.retry_after_s,
        quota_reset_at=err.quota_reset_at,
        scope=err.scope,
    )
