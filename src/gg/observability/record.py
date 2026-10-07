from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Literal, Protocol

from gg.core.cache_types import CacheStatus
from gg.core.context import ContextKey, Outcome, RequestContext
from gg.core.deployment import Deployment
from gg.core.usage import AttemptRecord, UsageRecord
from gg.observability.timing import CACHE_HITS, RequestTimings, compute_timings

type StatusClass = Literal["2xx", "4xx", "5xx", "disconnect"]
type DisconnectPhase = Literal["before_commit", "before_first_token", "mid_stream", "non_stream"]


class PricedCost(Protocol):
    @property
    def total(self) -> int: ...
    @property
    def parts(self) -> Mapping[str, int]: ...
    @property
    def effective_from(self) -> date: ...


class Pricer(Protocol):
    """gg.limits.cost.CostCalculator satisfies this; observability never imports limits"""

    def cost(self, usage: UsageRecord, deployment: Deployment, at: datetime, /) -> PricedCost | None: ...


# the deployment a weak-routed request would have used had it been routed strong (the coordinator wires it)
type StrongDeployment = Callable[[RequestContext], Deployment | None]


@dataclass(frozen=True, slots=True)
class ResponseStatus:
    """set by the response writer when it knows the http status and error type"""

    status: int
    error_type: str | None = None


RESPONSE_STATUS = ContextKey[ResponseStatus]("observability.response_status")
RECORD = ContextKey["RequestRecord"]("observability.record")

_OUTCOME_STATUS: Mapping[Outcome | None, ResponseStatus] = {
    "completed": ResponseStatus(200),
    "guard_aborted": ResponseStatus(200),
    "client_disconnected": ResponseStatus(499),
    "rejected": ResponseStatus(400, "invalid_request"),
    "upstream_error": ResponseStatus(502, "upstream_unavailable"),
    "internal_error": ResponseStatus(500, "internal"),
    "shutdown": ResponseStatus(503, "internal"),
    None: ResponseStatus(500, "internal"),
}


def micros_to_usd(micros: int) -> Decimal:
    return Decimal(micros).scaleb(-6)


def _response_status(ctx: RequestContext) -> ResponseStatus:
    explicit = ctx.get(RESPONSE_STATUS)
    if explicit is not None:
        return explicit
    if ctx.outcome == "upstream_error" and "client_first_byte" in ctx.timings.marks:
        return ResponseStatus(200, "upstream_midstream")
    return _OUTCOME_STATUS[ctx.outcome]


def _status_class(outcome: Outcome | None, status: ResponseStatus) -> StatusClass:
    if outcome == "client_disconnected" or status.status == 499:
        return "disconnect"
    if status.error_type == "upstream_midstream" or status.status >= 500:
        return "5xx"
    return "4xx" if status.status >= 400 else "2xx"


def _disconnect_phase(ctx: RequestContext) -> DisconnectPhase | None:
    if ctx.outcome != "client_disconnected":
        return None
    if not ctx.request.stream:
        return "non_stream"
    marks = ctx.timings.marks
    if "client_first_byte" in marks:
        return "mid_stream"
    return "before_first_token" if "upstream_first_token" in marks else "before_commit"


def _attempts(ctx: RequestContext, timings: RequestTimings) -> tuple[AttemptRecord, ...]:
    if ctx.attempts:
        return tuple(ctx.attempts)
    start = ctx.timings.marks.get("upstream_start")
    if ctx.served_by is None or start is None or ctx.cache_status in CACHE_HITS:
        return ()
    # single-attempt executors that only set marks still get attempt metrics
    return (
        AttemptRecord(
            deployment_id=ctx.served_by.id,
            provider=ctx.served_by.provider,
            started_at=start,
            duration_s=timings.upstream or 0.0,
            outcome="fail" if ctx.outcome == "upstream_error" else "ok",
            ttft_s=timings.upstream_ttft,
        ),
    )


def _provider(ctx: RequestContext, attempts: tuple[AttemptRecord, ...]) -> str:
    if ctx.cache_status in CACHE_HITS:
        return "cache"
    if ctx.served_by is not None:
        return ctx.served_by.provider
    if ctx.usage is not None:
        return ctx.usage.provider
    return attempts[-1].provider if attempts else "none"


def _routed_weak(ctx: RequestContext) -> bool:
    return ctx.route is not None and ctx.route.tier == "weak"


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """built once per request at the end; feeds metrics and the request.completed log line"""

    request_id: str
    trace_id: str | None
    key_id: str
    alias: str
    served_by: str | None
    provider: str
    tier: str | None
    stream: bool
    status: int
    status_class: StatusClass
    error_type: str | None
    outcome: Outcome | None
    cache_status: CacheStatus
    disconnect_phase: DisconnectPhase | None
    attempts: tuple[AttemptRecord, ...]
    usage: UsageRecord | None
    cost: PricedCost | None
    timings: RequestTimings
    stages: Mapping[str, float]
    config_hash: str
    strong_equiv_cost: PricedCost | None = None

    @property
    def upstream_usage(self) -> UsageRecord | None:
        return None if self.cache_status in CACHE_HITS else self.usage

    @classmethod
    def from_ctx(
        cls,
        ctx: RequestContext,
        *,
        now: float,
        pricer: Pricer | None = None,
        strong_deployment: StrongDeployment | None = None,
    ) -> "RequestRecord":
        timings = compute_timings(ctx, now=now)
        attempts = _attempts(ctx, timings)
        status = _response_status(ctx)
        served = ctx.served_by
        cost = None
        strong_equiv = None
        usage = ctx.usage
        if pricer and served and usage and ctx.cache_status not in CACHE_HITS:
            at = datetime.fromtimestamp(ctx.received_unix, UTC)
            cost = pricer.cost(usage, served, at)
            strong = strong_deployment(ctx) if strong_deployment and _routed_weak(ctx) else None
            if strong is not None:
                strong_equiv = pricer.cost(usage, strong, at)
        tier = served.tier if served is not None and served.tier else None
        if tier is None and ctx.route is not None:
            tier = ctx.route.tier
        return cls(
            request_id=ctx.request_id,
            trace_id=ctx.trace_id,
            key_id=ctx.key.id,
            alias=ctx.original.model,
            served_by=served.id if served is not None else None,
            provider=_provider(ctx, attempts),
            tier=tier,
            stream=ctx.request.stream,
            status=status.status,
            status_class=_status_class(ctx.outcome, status),
            error_type=status.error_type,
            outcome=ctx.outcome,
            cache_status=ctx.cache_status,
            disconnect_phase=_disconnect_phase(ctx),
            attempts=attempts,
            usage=ctx.usage,
            cost=cost,
            timings=timings,
            stages=dict(ctx.timings.durations),
            config_hash=ctx.config_hash,
            strong_equiv_cost=strong_equiv,
        )

    def to_log(self) -> dict[str, Any]:
        """fields for request.completed; ids, counts and timings only, never prompt or output text"""
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "key_id": self.key_id,
            "alias": self.alias,
            "served_by": self.served_by,
            "provider": self.provider,
            "tier": self.tier,
            "stream": self.stream,
            "status": self.status,
            "status_class": self.status_class,
            "error_type": self.error_type,
            "outcome": self.outcome,
            "cache": self.cache_status,
            "attempts": [_attempt_log(a) for a in self.attempts],
            "usage": _usage_log(self.usage),
            "cost_usd": _cost_log(self.cost),
            "pricing_effective_from": self.cost.effective_from.isoformat() if self.cost else None,
            "timing_ms": _timing_log(self.timings, self.stages),
            "config_hash": self.config_hash,
        }


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def _attempt_log(a: AttemptRecord) -> dict[str, Any]:
    return {
        "deployment": a.deployment_id,
        "provider": a.provider,
        "result": a.outcome,
        "error_kind": a.error_kind,
        "status": a.status,
        "ms": _ms(a.duration_s),
    }


def _usage_log(u: UsageRecord | None) -> dict[str, Any] | None:
    if u is None:
        return None
    return {
        "input": u.input_tokens,
        "cached_input": u.cached_input_tokens,
        "cache_write": u.cache_write_tokens,
        "output": u.output_tokens,
        "reasoning": u.reasoning_tokens,
        "source": u.usage_source,
    }


def _cost_log(cost: PricedCost | None) -> dict[str, str] | None:
    if cost is None:
        return None
    out = {name: f"{micros_to_usd(m):f}" for name, m in cost.parts.items()}
    out["total"] = f"{micros_to_usd(cost.total):f}"
    return out


def _timing_log(t: RequestTimings, stages: Mapping[str, float]) -> dict[str, Any]:
    values = {
        "total": t.total,
        "pre_upstream": t.pre_upstream,
        "failover": t.failover,
        "upstream": t.upstream,
        "upstream_ttft": t.upstream_ttft,
        "ttft": t.ttft,
        "ttft_added": t.ttft_added,
        "post": t.post,
        "stream_tail": t.stream_tail,
        "overhead": t.overhead,
        "tpot": t.tpot,
    }
    out: dict[str, Any] = {name: _ms(v) for name, v in values.items() if v is not None}
    out["stages"] = {name: _ms(v) for name, v in stages.items()}
    return out
