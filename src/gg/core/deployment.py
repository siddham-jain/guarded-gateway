from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Literal

type SamplingParams = Literal["full", "temp_or_top_p", "none", "when_effort_none", "advisory"]
type ImageUrlMode = Literal["provider_fetch", "gateway_fetch", "unsupported"]
type ThinkingMode = Literal["none", "manual", "adaptive", "adaptive_always", "levels", "effort", "toggle"]

EFFORT_ORDER: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True, slots=True)
class Capabilities:
    tools: bool = True
    forced_tool_choice: bool = True
    parallel_tool_calls: bool = True
    prefill: bool = True
    sampling_params: SamplingParams = "full"
    effort_levels: frozenset[str] = frozenset()
    json_schema: bool = True
    json_object: bool = True
    vision: bool = False
    n: bool = False
    logprobs: bool = False
    max_output: int = 8192
    context: int = 128_000
    tools_require_effort: str | None = None
    effort_clamp: Mapping[str, str] = field(default_factory=lambda: {})
    temperature_max: float = 2.0
    stop_max: int | None = None
    json_schema_strict: bool = True
    image_url: ImageUrlMode = "provider_fetch"
    pdf: bool = False
    audio: bool = False
    thinking_mode: ThinkingMode = "none"
    mid_conversation_system: bool = True
    min_max_tokens: int = 1
    cache_min_tokens: int | None = None
    honors_cancellation: bool = True


@dataclass(frozen=True, slots=True)
class LongContextPricing:
    threshold_input_tokens: int
    input_mult: Decimal
    output_mult: Decimal


@dataclass(frozen=True, slots=True)
class Pricing:
    """usd per 1m tokens, valid from effective_from (inclusive)"""

    effective_from: date
    input: Decimal
    output: Decimal
    cached_input: Decimal | None = None
    cache_write: Decimal | None = None
    cache_write_1h: Decimal | None = None
    long_context: LongContextPricing | None = None


@dataclass(frozen=True, slots=True)
class PriceSchedule:
    periods: tuple[Pricing, ...]
    billed: bool = True

    def at(self, ts: datetime) -> Pricing | None:
        day = ts.astimezone(UTC).date()
        current: Pricing | None = None
        for period in sorted(self.periods, key=lambda p: p.effective_from):
            if period.effective_from <= day:
                current = period
        return current


@dataclass(frozen=True, slots=True)
class Timeouts:
    connect_s: float = 3.0
    ttft_s: float = 10.0
    inter_chunk_s: float = 20.0
    total_s: float = 120.0


@dataclass(frozen=True, slots=True)
class Deployment:
    id: str
    provider: str
    upstream_model: str
    capabilities: Capabilities = field(default_factory=Capabilities)
    pricing: PriceSchedule = field(default_factory=lambda: PriceSchedule(periods=()))
    defaults: Mapping[str, Any] = field(default_factory=lambda: {})
    tier: str | None = None
    timeouts: Timeouts = field(default_factory=Timeouts)
    tags: frozenset[str] = frozenset()
    status: Literal["active", "preview", "deprecated", "disabled"] = "active"
    shutdown_date: date | None = None
    enabled: bool = True
    canonical_model: str | None = None
    quirks: Mapping[str, Any] = field(default_factory=lambda: {})
