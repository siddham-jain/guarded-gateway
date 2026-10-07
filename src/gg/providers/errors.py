import re
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from gg.core.errors import ProviderError, ProviderErrorKind

__all__ = [
    "MAX_RETRY_AFTER_S",
    "ErrorBody",
    "after_commit",
    "capability_mismatch",
    "parse_duration",
    "parse_error_body",
    "parse_retry_after",
    "transport_error",
]

MAX_RETRY_AFTER_S = 120.0

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|us|µs|ns|h|m|s)")
_DURATION_UNITS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}


def parse_duration(value: str) -> float | None:
    """go-style ("6m0s", "1.5s", "200ms") and google ("39s") durations; bare numbers are seconds"""
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    pos = 0
    total = 0.0
    for match in _DURATION_PART.finditer(text):
        if match.start() != pos:
            return None
        total += float(match.group(1)) * _DURATION_UNITS[match.group(2)]
        pos = match.end()
    return total if pos == len(text) and pos > 0 else None


def parse_retry_after(headers: Mapping[str, str], now: datetime | None = None) -> float | None:
    """retry-after-ms, then retry-after (seconds or http-date); garbage or negative -> None, capped"""
    lowered = {k.lower(): v for k, v in headers.items()}
    seconds: float | None = None
    if (ms := lowered.get("retry-after-ms")) is not None:
        try:
            seconds = float(ms) / 1000
        except ValueError:
            seconds = None
    if seconds is None and (raw := lowered.get("retry-after")) is not None:
        seconds = _retry_after_value(raw, now)
    if seconds is None or seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_S)


def _retry_after_value(raw: str, now: datetime | None) -> float | None:
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return (when - (now or datetime.now(UTC))).total_seconds()


class ErrorBody:
    """tolerant view over the error envelopes openai-compatible hosts return"""

    __slots__ = ("code", "message", "metadata", "type")

    def __init__(
        self, message: str, type_: str | None, code: str | None, metadata: Mapping[str, Any]
    ) -> None:
        self.message = message
        self.type = type_
        self.code = code
        self.metadata = metadata

    def text(self) -> str:
        return " ".join(p for p in (self.type, self.code, self.message) if p).lower()


def _str_or_none(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    return str(value)


def parse_error_body(data: Any) -> ErrorBody:
    """openai {"error": {...}}, hf {"error": "..."}, nebius/cerebras {"detail": ...},
    novita {"code", "reason"}, cloudflare {"errors": [...]} and bare text"""
    if isinstance(data, str):
        return ErrorBody(data[:2000], None, None, {})
    if not isinstance(data, dict):
        return ErrorBody("", None, None, {})
    payload: Any = data.get("error", data)
    errors = data.get("errors")
    if "error" not in data and isinstance(errors, list) and errors and isinstance(errors[0], dict):
        payload = errors[0]
    if isinstance(payload, str):
        return ErrorBody(payload[:2000], _str_or_none(data.get("type")), _str_or_none(data.get("code")), {})
    if not isinstance(payload, dict):
        return ErrorBody("", None, None, {})
    message = payload.get("message") or payload.get("detail") or payload.get("msg") or ""
    if isinstance(message, (dict, list)):
        message = str(message)
    metadata = payload.get("metadata")
    return ErrorBody(
        str(message)[:2000],
        _str_or_none(payload.get("type") or payload.get("reason")),
        _str_or_none(payload.get("code")),
        metadata if isinstance(metadata, dict) else {},
    )


def transport_error(
    exc: BaseException, *, provider: str, kind: ProviderErrorKind = "retryable"
) -> ProviderError:
    name = type(exc).__name__
    code = "timeout" if "Timeout" in name else "connect_error" if "Connect" in name else "transport_error"
    return ProviderError(kind, provider=provider, status=0, code=code, message=f"{name}: {exc}"[:500])


def capability_mismatch(provider: str, deployment_id: str, violations: tuple[str, ...]) -> ProviderError:
    return ProviderError(
        "fallback",
        provider=provider,
        status=0,
        code="capability_mismatch",
        message="deployment cannot serve this request: " + ", ".join(violations),
        deployment_id=deployment_id,
        violations=violations,
    )


def after_commit(err: ProviderError) -> ProviderError:
    if err.committed:
        return err
    return ProviderError(
        err.kind,
        provider=err.provider,
        status=err.status,
        code=err.code,
        retry_after_s=err.retry_after_s,
        message=err.message,
        upstream_request_id=err.upstream_request_id,
        quota_reset_at=err.quota_reset_at,
        scope=err.scope,
        committed=True,
        deployment_id=err.deployment_id,
        violations=err.violations,
    )
