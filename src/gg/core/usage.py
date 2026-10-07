from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from gg.core.deployment import Deployment
from gg.core.schema import ChatChunk

type UsageSource = Literal["reported", "partial", "estimated"]


@dataclass(frozen=True, slots=True)
class UsageRecord:
    provider: str
    deployment_id: str
    upstream_model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    reasoning_tokens: int = 0
    usage_source: UsageSource = "reported"
    service_tier: str | None = None
    raw: Mapping[str, Any] = field(default_factory=lambda: {})


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    deployment_id: str
    provider: str
    started_at: float
    duration_s: float
    outcome: Literal["ok", "retry", "fallback", "fail"]
    error_kind: str | None = None
    status: int | None = None
    upstream_request_id: str | None = None
    ttft_s: float | None = None


@dataclass(frozen=True, slots=True)
class TokenEstimates:
    prompt_tokens: int
    max_completion_tokens: int


def usage_record_from_chunk(chunk: ChatChunk, deployment: Deployment) -> UsageRecord | None:
    """prefer the adapter's normalised record in gg_meta; fall back to the wire usage block"""
    reported = getattr(chunk.gg_meta, "usage", None)
    if isinstance(reported, UsageRecord):
        return reported
    usage = chunk.usage
    if usage is None:
        return None
    prompt = usage.prompt_tokens_details
    completion = usage.completion_tokens_details
    return UsageRecord(
        provider=deployment.provider,
        deployment_id=deployment.id,
        upstream_model=deployment.upstream_model,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cached_input_tokens=prompt.cached_tokens if prompt else 0,
        cache_write_tokens=prompt.cache_write_tokens if prompt else 0,
        reasoning_tokens=completion.reasoning_tokens if completion else 0,
        raw=usage.model_dump(mode="json"),
    )
