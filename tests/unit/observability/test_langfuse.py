from typing import Any

import pytest

from gg.config.settings import LangfuseSettings
from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.guard_types import Finding, Verdict
from gg.core.jsonutil import loads
from gg.core.routing_types import PlanEntry, RouteDecision, RoutePlan
from gg.core.usage import AttemptRecord
from gg.limits.cost import CostCalculator
from gg.observability.langfuse import (
    LangfuseTraceBuilder,
    LangfuseTracer,
    WallClock,
    auth_headers,
    mask,
    otlp_traces_url,
    sampled,
)
from gg.observability.otlp import EncodedSpan, SpanData
from gg.observability.record import RequestRecord
from tests.conftest import make_ctx, make_request
from tests.unit.observability.support import LUNA, luna_usage

HAIKU = Deployment(id="anthropic/claude-haiku-4-5", provider="anthropic", upstream_model="claude-haiku-4-5")
DEPLOYMENTS = {d.id: d for d in (LUNA, HAIKU)}
WALL_ZERO = 1_790_000_000 - 1_000  # FakeClock wall minus its monotonic start


def wall_ns(monotonic: float) -> int:
    return int((monotonic + WALL_ZERO) * 1_000_000_000)


def request_ctx(clock: FakeClock) -> RequestContext:
    ctx = make_ctx(
        clock, make_request(messages=[{"role": "user", "content": "mail jane@corp.io"}], user="u-1")
    )
    ctx.trace_id = "0af7651916cd43dd8448eb211c80319c"
    ctx.config_hash = "abcdef0123456789"
    timings = ctx.timings
    for name, start, seconds in (
        ("limits", 1000.001, 0.0004),
        ("guard_in_pre", 1000.002, 0.002),
        ("probes", 1000.005, 0.300),
        ("probes.router", 1000.005, 0.290),
        ("terminal", 1000.310, 1.0),
    ):
        timings.start(name, start)
        timings.record(name, seconds)
    ctx.attempts = [
        AttemptRecord(HAIKU.id, "anthropic", 1000.31, 0.08, "fallback", error_kind="overloaded", status=529),
        AttemptRecord(LUNA.id, "openai", 1000.40, 0.90, "ok", ttft_s=0.42, upstream_request_id="up-1"),
    ]
    ctx.served_by = LUNA
    ctx.usage = luna_usage()
    ctx.outcome = "completed"
    ctx.route = RouteDecision(
        alias="gg/auto",
        tier="weak",
        reason="score_below_threshold",
        plan=RoutePlan("gg/auto", (PlanEntry(HAIKU), PlanEntry(LUNA))),
        score=0.31,
        threshold=0.5,
    )
    ctx.guard_findings = [Finding("pii", "input", Verdict.REDACT, 0.9, "EMAIL_ADDRESS")]
    ctx.scrubbed = make_request(messages=[{"role": "user", "content": "mail [EMAIL_1]"}])
    ctx.reply_text = "sent to [EMAIL_1] with key sk-abcdefghijklmnop1234"
    timings.mark("completed", 1001.5)
    return ctx


def build(clock: FakeClock, **settings: Any) -> tuple[RequestContext, list[SpanData]]:
    ctx = request_ctx(clock)
    record = RequestRecord.from_ctx(ctx, now=clock.monotonic(), pricer=CostCalculator())
    builder = LangfuseTraceBuilder(
        LangfuseSettings(**settings), environment="dev", release="0.1.0", deployment=DEPLOYMENTS.get
    )
    return ctx, builder.build(ctx, record, WallClock(clock))


def by_name(spans: list[SpanData]) -> dict[str, SpanData]:
    return {s.name: s for s in spans}


def test_span_tree_and_timestamps(clock: FakeClock) -> None:
    _, spans = build(clock)
    names = [s.name for s in spans]
    assert names == [
        "chat.completions",
        "gg.limits",
        "gg.guard_in_pre",
        "gg.probes",
        "gg.probe.router",
        "chat claude-haiku-4-5",
        "chat gpt-6-luna",
    ]
    tree = by_name(spans)
    root = tree["chat.completions"]
    assert (root.start_ns, root.end_ns, root.kind) == (wall_ns(1000), wall_ns(1001.5), "server")
    assert tree["gg.probe.router"].parent_id == tree["gg.probes"].span_id
    assert tree["gg.limits"].parent_id == root.span_id
    limits = tree["gg.limits"]
    assert (limits.start_ns, limits.end_ns) == (wall_ns(1000.001), wall_ns(1000.0014))


def test_trace_fields_are_on_every_span(clock: FakeClock) -> None:
    _, spans = build(clock)
    for span in spans:
        assert span.attributes["langfuse.user.id"] == "test-key"
        assert span.attributes["langfuse.version"] == "abcdef012345"
        assert span.attributes["langfuse.trace.tags"] == ("gg/auto", "miss", "completed", "weak")
        assert len(str(span.attributes["langfuse.session.id"])) == 16
    root = spans[0].attributes
    assert root["langfuse.trace.metadata.route_score"] == 0.31
    assert root["langfuse.trace.metadata.served_by"] == LUNA.id
    assert root["langfuse.trace.metadata.attempts"] == 2


def test_generations_carry_usage_cost_and_failures(clock: FakeClock) -> None:
    _, spans = build(clock)
    tree = by_name(spans)
    failed = tree["chat claude-haiku-4-5"]
    assert failed.error == "overloaded"
    assert failed.attributes["langfuse.observation.level"] == "ERROR"
    assert failed.attributes["langfuse.observation.status_message"] == "overloaded (status 529)"
    assert "langfuse.observation.usage_details" not in failed.attributes

    ok = tree["chat gpt-6-luna"].attributes
    assert ok["langfuse.observation.type"] == "generation"
    assert ok["langfuse.observation.model.name"] == "gpt-6-luna"
    assert loads(str(ok["langfuse.observation.usage_details"])) == {"input": 500, "output": 200, "total": 700}
    # 500 x $0.10/M + 200 x $0.50/M
    assert loads(str(ok["langfuse.observation.cost_details"])) == {
        "input": 0.00005,
        "output": 0.0001,
        "total": 0.00015,
    }
    assert ok["langfuse.observation.completion_start_time"] == "2026-09-21T14:13:20.820000Z"
    assert ok["langfuse.observation.metadata.upstream_request_id"] == "up-1"


def test_guard_findings_are_summarised_without_matched_text(clock: FakeClock) -> None:
    _, spans = build(clock)
    guard = by_name(spans)["gg.guard_in_pre"].attributes
    assert guard["langfuse.observation.type"] == "guardrail"
    assert guard["langfuse.observation.level"] == "WARNING"
    assert loads(str(guard["langfuse.observation.metadata.findings"])) == [
        {"guard": "pii", "verdict": "redact", "mode": "enforce", "score": 0.9, "reason": "EMAIL_ADDRESS"}
    ]


def test_content_is_off_by_default(clock: FakeClock) -> None:
    _, spans = build(clock)
    for span in spans:
        assert "langfuse.observation.input" not in span.attributes
        assert "langfuse.observation.output" not in span.attributes


def test_captured_content_is_placeholder_space_and_masked(clock: FakeClock) -> None:
    _, spans = build(clock, capture_content="scrubbed")
    root = spans[0].attributes
    assert loads(str(root["langfuse.observation.input"])) == [{"role": "user", "content": "mail [EMAIL_1]"}]
    assert loads(str(root["langfuse.observation.output"])) == "sent to [EMAIL_1] with key [key]"
    assert "jane@corp.io" not in str(spans)


def test_mask_and_truncate() -> None:
    text = "a@b.io Bearer abcdefghijk gg-live-1234567890abcd AKIAABCDEFGHIJKLMNOP"
    assert mask(text, 1000) == "[email] [bearer] [key] [key]"
    assert mask("x" * 20, 10) == "x" * 10 + "…[truncated]"


@pytest.mark.parametrize(
    ("trace_id", "rate", "expected"),
    [
        ("00000000" + "0" * 24, 0.5, True),
        ("ffffffff" + "0" * 24, 0.5, False),
        ("ffffffff" + "0" * 24, 1.0, True),
    ],
)
def test_sampling_is_deterministic(trace_id: str, rate: float, expected: bool) -> None:
    assert sampled(trace_id, rate) is expected


def test_endpoint_and_auth() -> None:
    assert otlp_traces_url("http://localhost:3001/") == "http://localhost:3001/api/public/otel/v1/traces"
    headers = auth_headers("pk-lf-1", "sk-lf-2")
    assert headers["authorization"] == "Basic cGstbGYtMTpzay1sZi0y"
    assert headers["x-langfuse-ingestion-version"] == "4"


class Collector:
    def __init__(self) -> None:
        self.traces: list[list[EncodedSpan]] = []

    def submit(self, trace: list[EncodedSpan], /) -> bool:
        self.traces.append(trace)
        return True


def test_tracer_encodes_under_the_request_trace_id(clock: FakeClock) -> None:
    ctx = request_ctx(clock)
    record = RequestRecord.from_ctx(ctx, now=clock.monotonic())
    collector = Collector()
    builder = LangfuseTraceBuilder(LangfuseSettings(), environment="dev", release="0.1.0")
    LangfuseTracer(builder, collector, clock=clock, sample_rate=1.0).submit(ctx, record)
    (trace,) = collector.traces
    assert {s["traceId"] for s in trace} == {ctx.trace_id}
    assert trace[0]["name"] == "chat.completions"
    # without a catalog lookup a failed attempt is named by its deployment id
    assert any(s["name"] == "chat anthropic/claude-haiku-4-5" for s in trace)

    ctx.trace_id = "ffffffff" + "0" * 24
    LangfuseTracer(builder, collector, clock=clock, sample_rate=0.5).submit(ctx, record)
    assert len(collector.traces) == 1
