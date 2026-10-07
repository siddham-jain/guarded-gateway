from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from gg.core.context import ContextKey, RequestContext
from gg.core.routing_types import RouteDecision
from gg.core.schema import Message

type Tier = Literal["weak", "strong"]
type EffortHint = Literal["none", "low", "medium", "high"]
type FallbackReason = Literal[
    "timeout",
    "network",
    "http_429",
    "http_529",
    "http_5xx",
    "http_4xx",
    "auth",
    "firewall_403",
    "circuit_open",
    "parse_error",
    "disabled",
    "unscorable",
]


@dataclass(frozen=True, slots=True)
class RoutingRequest:
    request_id: str
    key_id: str
    messages: tuple[Message, ...]
    tools_present: bool

    @classmethod
    def from_context(cls, ctx: RequestContext) -> "RoutingRequest":
        # scorers are third-party: only the redacted request may leave the gateway, never ctx.original
        source = ctx.scrubbed or ctx.request
        return cls(
            request_id=ctx.request_id,
            key_id=ctx.key.id,
            messages=source.messages,
            tools_present=source.has_tools(),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutingScore:
    """scorer output; score approximates P(strong beats weak), the policy routes strong iff score >= alpha"""

    score: float
    raw_score: float
    scorer: str
    scorer_version: str
    has_probabilities: bool = False
    tier: str | None = None
    tier_probabilities: Mapping[str, float] = field(default_factory=lambda: {})
    difficulty: float | None = None
    strong_helps: float | None = None
    needs_reasoning: float | None = None
    task_type: str | None = None
    task_type_probabilities: Mapping[str, float] = field(default_factory=lambda: {})
    reasoning_effort_hint: EffortHint | None = None
    guard_signals: Mapping[str, float] = field(default_factory=lambda: {})
    confidence: float | None = None
    latency_ms: float = 0.0
    input_tokens: int | None = None
    cost_usd: float | None = None
    cached: bool = False
    fallback: bool = False
    fallback_reason: FallbackReason | None = None
    request_id: str | None = None
    response_model: str | None = None
    # analysis only; never training data for another router (typesafe mca 2.3(b))
    raw: Mapping[str, Any] = field(default_factory=lambda: {})

    @classmethod
    def fallback_for(
        cls,
        scorer: str,
        version: str,
        reason: FallbackReason,
        latency_ms: float = 0.0,
        *,
        request_id: str | None = None,
    ) -> "RoutingScore":
        # 1.0 fails open toward strong should a consumer ignore the fallback flag
        return cls(
            score=1.0,
            raw_score=1.0,
            scorer=scorer,
            scorer_version=version,
            latency_ms=latency_ms,
            fallback=True,
            fallback_reason=reason,
            request_id=request_id,
        )


class RoutingScorer(Protocol):
    name: str

    @property
    def version(self) -> str: ...

    async def score(self, req: RoutingRequest, /) -> RoutingScore:
        """never raises; failures come back as fallback=True with a fallback_reason"""
        ...


class CacheableScorer(RoutingScorer, Protocol):
    def cache_key(self, req: RoutingRequest, /) -> str | None:
        """None means do not cache, e.g. an unscorable request"""
        ...

    def from_raw(self, raw: Mapping[str, Any], req: RoutingRequest, /) -> RoutingScore: ...


class RoutingPolicy(Protocol):
    def presolve(self, ctx: RequestContext, /) -> RouteDecision | None:
        """a decision that needs no score (session reuse, key policy), so no scorer call is wasted"""
        ...

    def decide(self, score: RoutingScore, ctx: RequestContext, /) -> RouteDecision: ...


class RoutingHooks(Protocol):
    def scored(self, score: RoutingScore, duration_s: float, /) -> None: ...

    def decided(self, decision: RouteDecision, score: RoutingScore | None, /) -> None: ...


class NullRoutingHooks:
    def scored(self, score: RoutingScore, duration_s: float, /) -> None:
        return None

    def decided(self, decision: RouteDecision, score: RoutingScore | None, /) -> None:
        return None


# full scorer output for later stages (c6 advisory guard signals); ctx.route only keeps the scalar
ROUTING_SCORE: ContextKey[RoutingScore] = ContextKey("routing.score")
