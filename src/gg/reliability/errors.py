from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from gg.core.deployment import Deployment
from gg.core.errors import (
    ContentFilterError,
    ContextLengthError,
    GGError,
    InvalidRequestError,
    ProviderError,
    RateLimitedError,
    ServiceUnavailableError,
    UpstreamError,
    UpstreamTimeoutError,
)
from gg.core.jsonutil import JSONValue

type SkipReason = Literal["circuit_open", "tier_change", "context_window", "no_adapter"]

_TIMEOUT_CODES = frozenset({"ttft_timeout", "stall_timeout", "timeout", "read_timeout", "deadline_exceeded"})
_CONTEXT_SKIPS = frozenset({"context_window", "tier_change"})
_HEALTH_RETRY_MAX_S = 10.0
_OVERLOAD_RETRY_S = 2.0


@dataclass(frozen=True, slots=True)
class Failure:
    """one failed attempt (error set) or one skipped candidate (skip set)"""

    deployment: Deployment
    error: ProviderError | None = None
    skip: SkipReason | None = None
    retry_in_s: float | None = None


def _should_retry(value: bool) -> dict[str, str]:
    return {"x-should-retry": "true" if value else "false"}


def _is_timeout(error: ProviderError) -> bool:
    return error.code in _TIMEOUT_CODES or error.status in (408, 504)


def _is_overload(error: ProviderError) -> bool:
    return error.code == "overloaded" or error.status in (503, 529)


def client_error(error: ProviderError) -> GGError:
    message = f"{error.provider}: {error.message or 'request rejected upstream'}"
    if error.kind == "content_filter":
        return ContentFilterError(message)
    return InvalidRequestError(message, code=error.code)


def error_from_attempts(failures: Sequence[Failure], *, deadline_hit: bool = False) -> GGError:
    """maps an exhausted plan to one client error; first matching rule wins (C4 §3.9)"""
    errors = [f.error for f in failures if f.error is not None]
    skips = [f for f in failures if f.skip is not None]

    if (
        errors
        and all(e.code == "context_length" for e in errors)
        and all(s.skip in _CONTEXT_SKIPS for s in skips)
    ):
        return ContextLengthError("The prompt is too long for every available model.", param="messages")
    if not errors and not deadline_hit:
        waits = [s.retry_in_s for s in skips if s.retry_in_s is not None]
        retry = max(1.0, min(waits)) if waits else None
        return ServiceUnavailableError(
            "No healthy deployment is available for this model.",
            code="no_healthy_deployment",
            retry_after_s=retry,
            headers=_should_retry(retry is not None and retry <= _HEALTH_RETRY_MAX_S),
        )
    if errors and all(e.kind == "quota_minute" for e in errors):
        waits = [e.retry_after_s for e in errors if e.retry_after_s is not None]
        waits += [s.retry_in_s for s in skips if s.skip == "circuit_open" and s.retry_in_s is not None]
        return RateLimitedError(
            "The upstream provider is rate limiting requests.",
            code="upstream_rate_limited",
            retry_after_s=max(1.0, min(waits)) if waits else None,
        )
    if deadline_hit or all(_is_timeout(e) for e in errors):
        return UpstreamTimeoutError("The upstream provider did not respond in time.")
    if all(e.kind in ("auth", "quota_day") for e in errors):
        return UpstreamError("The gateway's upstream account was rejected.", code="upstream_account_error")
    if any(_is_overload(e) for e in errors) or any(s.skip == "circuit_open" for s in skips):
        return ServiceUnavailableError(
            "The upstream provider is overloaded.",
            code="upstream_overloaded",
            retry_after_s=_OVERLOAD_RETRY_S,
            headers=_should_retry(True),
        )
    attempts: list[JSONValue] = [{"provider": e.provider, "status": e.status, "code": e.code} for e in errors]
    return UpstreamError("All upstream providers failed.", details={"attempts": attempts})
