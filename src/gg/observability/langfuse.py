"""langfuse traces, built once per request after the last byte and sent over otlp/json.

tree: root (the request) → one span per pipeline stage (probes nested under `probes`) → one generation per
upstream attempt. timestamps are the gateway's own monotonic marks mapped to wall time, so the trace shows
what the request actually waited on. content is off by default; when on, only placeholder-space text goes
out, masked and truncated — never the raw request.
"""

import base64
import hashlib
import re
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from gg.config.settings import LangfuseSettings
from gg.core.clock import Clock
from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.guard_types import Finding, Verdict
from gg.core.ids import new_span_id
from gg.core.jsonutil import dumps_str
from gg.core.schema import to_wire
from gg.core.usage import AttemptRecord, UsageRecord
from gg.observability.otlp import TRACES_PATH, AttrValue, EncodedSpan, SpanData, encode_span
from gg.observability.record import PricedCost, RequestRecord, micros_to_usd

OTLP_PATH = "/api/public/otel"
TRACE_NAME = "chat.completions"
# stages whose time is already shown by their children or the generations
_HIDDEN_STAGES = frozenset({"terminal"})
_PROBE_PREFIX = "probes."

_MASKS = (
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[email]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}"), "[bearer]"),
    (re.compile(r"\b(?:sk|pk|rk|gg|pg_live|pg_test)[-_][A-Za-z0-9_-]{12,}"), "[key]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[key]"),
)


class TraceSubmitter(Protocol):
    def submit(self, trace: list[EncodedSpan], /) -> bool: ...


def otlp_traces_url(host: str) -> str:
    return host.rstrip("/") + OTLP_PATH + TRACES_PATH


def auth_headers(public_key: str, secret_key: str) -> dict[str, str]:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return {"authorization": f"Basic {token}", "x-langfuse-ingestion-version": "4"}


def sampled(trace_id: str, rate: float) -> bool:
    # deterministic per trace id, so a retry of the export makes the same choice
    return rate >= 1 or int(trace_id[:8], 16) < rate * 0x1_0000_0000


def mask(text: str, limit: int) -> str:
    for pattern, replacement in _MASKS:
        text = pattern.sub(replacement, text)
    return text if len(text) <= limit else text[:limit] + "…[truncated]"


def _masked(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return mask(value, limit)
    if isinstance(value, list):
        return [_masked(v, limit) for v in value]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, dict):
        return {k: _masked(v, limit) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    return value


class WallClock:
    """maps the gateway's monotonic seconds to unix nanoseconds"""

    def __init__(self, clock: Clock) -> None:
        self._offset = clock.time() - clock.monotonic()

    def ns(self, monotonic: float) -> int:
        return int((monotonic + self._offset) * 1_000_000_000)

    def iso(self, monotonic: float) -> str:
        return datetime.fromtimestamp(monotonic + self._offset, UTC).isoformat().replace("+00:00", "Z")


def _usd(micros: int) -> float:
    return float(micros_to_usd(micros))


def _usage_details(usage: UsageRecord) -> dict[str, int]:
    uncached = max(0, usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens)
    details = {
        "input": uncached,
        "input_cached": usage.cached_input_tokens,
        "input_cache_write": usage.cache_write_tokens,
        "output": usage.output_tokens,
        "output_reasoning": usage.reasoning_tokens,
        # explicit, because cached and reasoning tokens are already inside input and output
        "total": usage.input_tokens + usage.output_tokens,
    }
    return {k: v for k, v in details.items() if v or k in ("input", "output", "total")}


def _cost_details(cost: PricedCost) -> dict[str, float]:
    details = {name: _usd(m) for name, m in cost.parts.items() if m}
    details["total"] = _usd(cost.total)
    return details


def _findings(findings: list[Finding], stage: str) -> list[Finding]:
    return [f for f in findings if f.stage == stage]


def _findings_attrs(findings: list[Finding]) -> dict[str, AttrValue]:
    if not findings:
        return {}
    # names, verdicts and scores only; matched text never leaves the gateway
    summary = [
        {
            "guard": f.guard,
            "verdict": f.verdict.name.lower(),
            "mode": f.mode,
            "score": f.score,
            "reason": f.reason,
        }
        for f in findings
        if f.verdict is not Verdict.ALLOW or f.error
    ]
    attrs: dict[str, AttrValue] = {"langfuse.observation.metadata.findings": dumps_str(summary)}
    worst = max((f for f in findings if f.mode == "enforce"), key=lambda f: f.verdict, default=None)
    if worst is not None and worst.verdict is Verdict.BLOCK:
        attrs["langfuse.observation.level"] = "ERROR"
        attrs["langfuse.observation.status_message"] = f"blocked by {worst.guard}: {worst.reason}"
    elif summary:
        attrs["langfuse.observation.level"] = "WARNING"
    return attrs


class LangfuseTraceBuilder:
    def __init__(
        self,
        settings: LangfuseSettings,
        *,
        environment: str,
        release: str,
        deployment: Callable[[str], Deployment | None] | None = None,
    ) -> None:
        self._settings = settings
        self._environment = environment
        self._release = release
        self._deployment = deployment

    def build(self, ctx: RequestContext, record: RequestRecord, wall: WallClock) -> list[SpanData]:
        received = ctx.timings.marks.get("received", ctx.received_at)
        end = received + record.timings.total
        common = self._common(ctx, record)
        root_id = new_span_id()
        root = SpanData(
            name=TRACE_NAME,
            span_id=root_id,
            start_ns=wall.ns(received),
            end_ns=wall.ns(end),
            kind="server",
            attributes={**common, **self._root_attrs(ctx, record)},
            error=(record.error_type or "error") if record.status_class == "5xx" else None,
        )
        spans = [root, *self._stages(ctx, root_id, common, wall)]
        spans += self._generations(ctx, record, root_id, common, wall)
        return spans

    def _common(self, ctx: RequestContext, record: RequestRecord) -> dict[str, AttrValue]:
        # langfuse filters per observation, so trace-level fields go on every span
        tags = [record.alias, record.cache_status, record.outcome or "unknown"]
        if record.tier:
            tags.append(record.tier)
        attrs: dict[str, AttrValue] = {
            "langfuse.user.id": ctx.key.id,
            "langfuse.environment": self._environment,
            "langfuse.release": self._release,
            "langfuse.version": ctx.config_hash[:12],
            "langfuse.trace.tags": tuple(tags),
        }
        if ctx.original.user:
            attrs["langfuse.session.id"] = hashlib.sha256(ctx.original.user.encode()).hexdigest()[:16]
        return attrs

    def _root_attrs(self, ctx: RequestContext, record: RequestRecord) -> dict[str, AttrValue]:
        t = record.timings
        meta: dict[str, AttrValue | None] = {
            "request_id": record.request_id,
            "alias": record.alias,
            "served_by": record.served_by,
            "provider": record.provider,
            "tier": record.tier,
            "status": record.status,
            "outcome": record.outcome,
            "cache": record.cache_status,
            "stream": record.stream,
            "attempts": len(record.attempts),
            "cost_usd": _usd(record.cost.total) if record.cost else None,
            "total_ms": round(t.total * 1000, 3),
            "ttft_ms": None if t.ttft is None else round(t.ttft * 1000, 3),
            "overhead_ms": None if t.overhead is None else round(t.overhead * 1000, 3),
        }
        route = ctx.route
        if route is not None:
            meta |= {
                "route_tier": route.tier,
                "route_reason": route.reason,
                "route_score": route.score,
                "route_threshold": route.threshold,
            }
        attrs: dict[str, AttrValue] = {
            "langfuse.trace.name": TRACE_NAME,
            **{f"langfuse.trace.metadata.{k}": v for k, v in meta.items() if v is not None},
        }
        if record.status_class == "5xx":
            attrs["langfuse.observation.level"] = "ERROR"
        elif record.status_class == "4xx":
            attrs["langfuse.observation.level"] = "WARNING"
        attrs |= self._content(ctx)
        return attrs

    def _content(self, ctx: RequestContext) -> dict[str, AttrValue]:
        if self._settings.capture_content == "off":
            return {}
        limit = self._settings.max_content_chars
        attrs: dict[str, AttrValue] = {}
        if ctx.scrubbed is not None:
            messages = to_wire(ctx.scrubbed)["messages"]
            attrs["langfuse.observation.input"] = dumps_str(_masked(messages, limit))
        if ctx.reply_text is not None:
            attrs["langfuse.observation.output"] = dumps_str(mask(ctx.reply_text, limit))
        return attrs

    def _stages(
        self, ctx: RequestContext, root_id: str, common: Mapping[str, AttrValue], wall: WallClock
    ) -> Iterator[SpanData]:
        timings = ctx.timings
        ids = {name: new_span_id() for name in timings.starts}
        for name, start in timings.starts.items():
            if name in _HIDDEN_STAGES:
                continue
            parent = root_id
            label = name
            if name.startswith(_PROBE_PREFIX):
                parent = ids.get("probes", root_id)
                label = "probe." + name.removeprefix(_PROBE_PREFIX)
            seconds = timings.durations.get(name, 0.0)
            attrs: dict[str, AttrValue] = {
                **common,
                "langfuse.observation.type": "guardrail" if "guard" in name else "span",
                "langfuse.observation.metadata.self_ms": round(seconds * 1000, 3),
            }
            if name == "guard_in_pre":
                attrs |= _findings_attrs(_findings(ctx.guard_findings, "input"))
            elif name == "guard_out":
                attrs |= _findings_attrs(_findings(ctx.guard_findings, "output"))
            yield SpanData(
                name=f"gg.{label}",
                span_id=ids[name],
                parent_id=parent,
                start_ns=wall.ns(start),
                end_ns=wall.ns(start + seconds),
                attributes=attrs,
            )

    def _generations(
        self,
        ctx: RequestContext,
        record: RequestRecord,
        root_id: str,
        common: Mapping[str, AttrValue],
        wall: WallClock,
    ) -> Iterator[SpanData]:
        final = next(
            (i for i in range(len(record.attempts) - 1, -1, -1) if record.attempts[i].outcome == "ok"), None
        )
        for i, attempt in enumerate(record.attempts):
            model = self._model(attempt, ctx)
            attrs: dict[str, AttrValue] = {
                **common,
                "langfuse.observation.type": "generation",
                "langfuse.observation.model.name": model,
                "langfuse.observation.model.parameters": dumps_str(_parameters(ctx)),
                "langfuse.observation.metadata.deployment": attempt.deployment_id,
                "langfuse.observation.metadata.provider": attempt.provider,
                "langfuse.observation.metadata.attempt": i + 1,
                "langfuse.observation.metadata.result": attempt.outcome,
            }
            if attempt.upstream_request_id:
                attrs["langfuse.observation.metadata.upstream_request_id"] = attempt.upstream_request_id
            if attempt.ttft_s is not None:
                attrs["langfuse.observation.completion_start_time"] = wall.iso(
                    attempt.started_at + attempt.ttft_s
                )
            error = None
            if attempt.outcome != "ok":
                error = attempt.error_kind or attempt.outcome
                attrs["langfuse.observation.level"] = "ERROR"
                attrs["langfuse.observation.status_message"] = (
                    f"{error} (status {attempt.status})" if attempt.status else error
                )
            if i == final:
                if record.upstream_usage is not None:
                    attrs["langfuse.observation.usage_details"] = dumps_str(
                        _usage_details(record.upstream_usage)
                    )
                if record.cost is not None:
                    attrs["langfuse.observation.cost_details"] = dumps_str(_cost_details(record.cost))
                attrs |= self._content(ctx)
            yield SpanData(
                name=f"chat {model}",
                span_id=new_span_id(),
                parent_id=root_id,
                start_ns=wall.ns(attempt.started_at),
                end_ns=wall.ns(attempt.started_at + attempt.duration_s),
                kind="client",
                attributes=attrs,
                error=error,
            )

    def _model(self, attempt: AttemptRecord, ctx: RequestContext) -> str:
        served = ctx.served_by
        if served is not None and served.id == attempt.deployment_id:
            return served.upstream_model
        deployment = self._deployment(attempt.deployment_id) if self._deployment else None
        return deployment.upstream_model if deployment is not None else attempt.deployment_id


def _parameters(ctx: RequestContext) -> dict[str, Any]:
    req = ctx.request
    params = {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_completion_tokens": req.max_completion_tokens,
        "reasoning_effort": req.reasoning_effort,
        "stream": req.stream,
        "tools": len(req.tools) if req.tools else None,
    }
    return {k: v for k, v in params.items() if v is not None}


class LangfuseTracer:
    """the TRACE finalizer's sink: sample, build, encode, hand to the exporter; never raises into a request"""

    def __init__(
        self, builder: LangfuseTraceBuilder, exporter: TraceSubmitter, *, clock: Clock, sample_rate: float
    ) -> None:
        self._builder = builder
        self._exporter = exporter
        self._clock = clock
        self._sample_rate = sample_rate

    def submit(self, ctx: RequestContext, record: RequestRecord) -> None:
        trace_id = ctx.trace_id
        if trace_id is None or not sampled(trace_id, self._sample_rate):
            return
        spans = self._builder.build(ctx, record, WallClock(self._clock))
        self._exporter.submit([encode_span(trace_id, s) for s in spans])
