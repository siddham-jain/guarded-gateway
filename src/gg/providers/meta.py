from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from gg.core.usage import UsageRecord
from gg.providers.errors import parse_duration


@dataclass(frozen=True, slots=True)
class RateLimitSnapshot:
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    limit_requests: int | None = None
    limit_tokens: int | None = None
    reset_requests_s: float | None = None
    reset_tokens_s: float | None = None


DEFAULT_RATE_LIMIT_HEADERS: Mapping[str, str] = {
    "remaining_requests": "x-ratelimit-remaining-requests",
    "remaining_tokens": "x-ratelimit-remaining-tokens",
    "limit_requests": "x-ratelimit-limit-requests",
    "limit_tokens": "x-ratelimit-limit-tokens",
    "reset_requests_s": "x-ratelimit-reset-requests",
    "reset_tokens_s": "x-ratelimit-reset-tokens",
}


def _int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def rate_limit_snapshot(headers: Mapping[str, str], names: Mapping[str, str]) -> RateLimitSnapshot | None:
    lowered = {k.lower(): v for k, v in headers.items()}
    values: dict[str, Any] = {}
    for field_name, header in names.items():
        raw = lowered.get(header.lower())
        if raw is None:
            continue
        values[field_name] = parse_duration(raw) if field_name.startswith("reset_") else _int(raw)
    return RateLimitSnapshot(**values) if values else None


@dataclass(frozen=True, slots=True)
class ResponseMeta:
    """non-serialised metadata attached to chunks/responses with with_meta()"""

    provider: str
    deployment_id: str
    upstream_model: str
    served_model: str | None = None
    upstream_request_id: str | None = None
    upstream_provider: str | None = None
    ignored_params: tuple[str, ...] = ()
    adjustments: tuple[str, ...] = ()
    provider_finish_reason: str | None = None
    refusal_category: str | None = None
    ratelimit: RateLimitSnapshot | None = None
    usage: UsageRecord | None = None
    upstream_ttft_s: float | None = None
    upstream_cost_usd: float | None = None
    flags: tuple[str, ...] = ()

    def but(self, **changes: Any) -> "ResponseMeta":
        return replace(self, **changes)
