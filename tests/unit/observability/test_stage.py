from collections.abc import AsyncIterator, Callable, MutableMapping
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from structlog.testing import capture_logs

from gg.core.clock import FakeClock
from gg.core.context import Outcome, RequestContext
from gg.core.deployment import Deployment, PriceSchedule, Pricing
from gg.core.errors import RateLimitedError
from gg.core.routing_types import RouteDecision, RoutePlan
from gg.core.schema import ChatChunk, ChunkChoice, Delta
from gg.core.usage import AttemptRecord
from gg.limits.cost import CostCalculator
from gg.observability.metrics import Metrics
from gg.observability.record import RECORD, RESPONSE_STATUS, ResponseStatus, StrongDeployment
from gg.observability.stage import ObservabilityStage
from gg.pipeline.runner import Pipeline
from gg.pipeline.stage import Next, PipelineResult
from gg.pipeline.streams import ChunkStream
from tests.conftest import make_ctx, make_request
from tests.unit.observability.support import LUNA, RESPONSE, luna_usage, new_metrics

type Terminal = Callable[[RequestContext], Any]


class Harness:
    def __init__(
        self,
        clock: FakeClock,
        *,
        pricer: CostCalculator | None = None,
        strong_deployment: StrongDeployment | None = None,
    ) -> None:
        self.clock = clock
        self.metrics = new_metrics()
        self.stage = ObservabilityStage(
            self.metrics,
            clock=clock,
            pricer=pricer or CostCalculator(self.metrics),
            strong_deployment=strong_deployment,
        )

    def get(self, name: str, **labels: str) -> float | None:
        return self.metrics.registry.get_sample_value(name, labels)

    async def run(
        self, ctx: RequestContext, terminal: Terminal, *, outcome: Outcome | None
    ) -> list[MutableMapping[str, Any]]:
        pipeline = Pipeline([self.stage], terminal, clock=self.clock)
        try:
            await pipeline.run(ctx)
        except Exception:
            if outcome is None:
                raise
        stream = "true" if ctx.request.stream else "false"
        assert self.get("gg_inflight_requests", stream=stream) == 1
        ctx.outcome = outcome
        self.clock.advance(0.002)
        with capture_logs() as logs:
            await ctx.finalizers.run(timeout_s=1)
        assert self.get("gg_inflight_requests", stream=stream) == 0
        return logs


def upstream(clock: FakeClock, *, served: Deployment = LUNA, stream: bool = False) -> Terminal:
    async def terminal(ctx: RequestContext) -> PipelineResult:
        clock.advance(0.010)
        ctx.timings.mark("upstream_start")
        clock.advance(0.100)
        ctx.timings.mark("upstream_first_token")
        clock.advance(0.100)
        ctx.timings.mark("upstream_end")
        ctx.served_by = served
        ctx.usage = luna_usage()
        if stream:
            ctx.timings.mark("client_first_byte")
            return PipelineResult(source="upstream", stream=ChunkStream(_chunks()))
        return PipelineResult(source="upstream", response=RESPONSE)

    return terminal


async def _chunks() -> AsyncIterator[ChatChunk]:
    yield ChatChunk(id="c", created=1, model="m", choices=(ChunkChoice(index=0, delta=Delta(content="x")),))


async def test_non_stream_success_records_everything(clock: FakeClock) -> None:
    h = Harness(clock)
    ctx = make_ctx(clock, make_request(messages=[{"role": "user", "content": "my password is hunter2"}]))
    logs = await h.run(ctx, upstream(clock), outcome="completed")

    labels = {
        "endpoint": "chat",
        "alias": "gg/auto",
        "status_class": "2xx",
        "cache": "miss",
        "stream": "false",
    }
    assert h.get("gg_requests_total", **labels) == 1
    assert h.get("gg_request_duration_seconds_sum", endpoint="chat", stream="false", cache="miss") == (
        pytest.approx(0.212)
    )
    assert h.get("gg_gateway_overhead_seconds_sum", stream="false", phase="total") == pytest.approx(0.012)
    assert h.get("gg_gateway_overhead_seconds_sum", stream="false", phase="pre_upstream") == pytest.approx(
        0.010
    )
    assert h.get("gg_gateway_overhead_seconds_sum", stream="false", phase="post") == pytest.approx(0.002)
    dep = {"provider": "openai", "deployment": LUNA.id}
    assert h.get("gg_upstream_attempts_total", **dep, result="ok", error_kind="none") == 1
    assert h.get("gg_upstream_duration_seconds_sum", **dep, outcome="ok") == pytest.approx(0.2)
    assert h.get("gg_upstream_ttft_seconds_sum", **dep) == pytest.approx(0.1)
    assert h.get("gg_tokens_total", **dep, type="input", usage_source="reported") == 500
    assert h.get("gg_tokens_total", **dep, type="output", usage_source="reported") == 200
    assert h.get("gg_cost_usd_total", **dep, tier="weak", attributed="key") == pytest.approx(150e-6)
    assert h.get("gg_request_cost_usd_count", alias="gg/auto") == 1
    assert h.get("gg_time_per_output_token_seconds_count", **dep) == 1
    assert h.get("gg_ttft_seconds_count", provider="openai", cache="miss") is None

    [line] = logs
    assert line["event"] == "request.completed"
    assert line["log_level"] == "info"
    assert line["request_id"] == "req_test"
    assert line["served_by"] == LUNA.id
    assert line["status"] == 200
    assert line["usage"]["input"] == 500
    assert line["cost_usd"] == {
        "input": "0.000050",
        "cached_input": "0.000000",
        "cache_write": "0.000000",
        "output": "0.000100",
        "total": "0.000150",
    }
    assert line["pricing_effective_from"] == "2026-01-01"
    assert line["timing_ms"]["upstream"] == pytest.approx(200)
    assert "observability" in line["timing_ms"]["stages"]
    assert "hunter2" not in repr(line)
    assert "secret reply" not in repr(line)
    assert ctx.get(RECORD) is not None


async def test_stream_mid_stream_disconnect(clock: FakeClock) -> None:
    h = Harness(clock)
    ctx = make_ctx(clock, make_request(stream=True))
    logs = await h.run(ctx, upstream(clock, stream=True), outcome="client_disconnected")
    labels = {"endpoint": "chat", "alias": "gg/auto", "status_class": "disconnect", "cache": "miss"}
    assert h.get("gg_requests_total", **labels, stream="true") == 1
    assert h.get("gg_client_disconnects_total", phase="mid_stream") == 1
    dep = {"provider": "openai", "deployment": LUNA.id}
    assert h.get("gg_upstream_duration_seconds_count", **dep, outcome="cancelled") == 1
    assert h.get("gg_ttft_seconds_count", provider="openai", cache="miss") == 1
    assert h.get("gg_gateway_overhead_seconds_count", stream="true", phase="ttft_added") == 1
    assert logs[0]["status"] == 499


async def test_stream_disconnect_before_any_upstream_token(clock: FakeClock) -> None:
    h = Harness(clock)

    async def stalled(ctx: RequestContext) -> PipelineResult:
        return PipelineResult(source="upstream", stream=ChunkStream(_chunks()))

    await h.run(make_ctx(clock, make_request(stream=True)), stalled, outcome="client_disconnected")
    assert h.get("gg_client_disconnects_total", phase="before_commit") == 1


async def test_rejection_still_runs_finalizers(clock: FakeClock) -> None:
    h = Harness(clock)

    async def reject(ctx: RequestContext) -> PipelineResult:
        ctx.set(RESPONSE_STATUS, ResponseStatus(429, "rate_limited"))
        raise RateLimitedError("slow down")

    await h.run(make_ctx(clock), reject, outcome="rejected")
    assert h.get("gg_request_errors_total", endpoint="chat", error_type="rate_limited") == 1
    labels = {"endpoint": "chat", "alias": "gg/auto", "cache": "miss", "stream": "false"}
    assert h.get("gg_requests_total", **labels, status_class="4xx") == 1
    assert h.get("gg_gateway_overhead_seconds_count", stream="false", phase="total") is None


async def test_upstream_failure_after_first_byte_is_midstream_5xx(clock: FakeClock) -> None:
    h = Harness(clock)
    ctx = make_ctx(clock, make_request(stream=True))
    logs = await h.run(ctx, upstream(clock, stream=True), outcome="upstream_error")
    assert h.get("gg_request_errors_total", endpoint="chat", error_type="upstream_midstream") == 1
    assert logs[0]["log_level"] == "error"
    assert logs[0]["status_class"] == "5xx"


async def test_cache_hit_counts_no_upstream_tokens_or_overhead(clock: FakeClock) -> None:
    h = Harness(clock)

    async def hit(ctx: RequestContext) -> PipelineResult:
        ctx.cache_status = "exact_hit"
        ctx.served_by = LUNA
        ctx.usage = luna_usage()
        clock.advance(0.003)
        ctx.timings.mark("client_first_byte")
        return PipelineResult(source="exact_cache", stream=ChunkStream(_chunks()))

    logs = await h.run(make_ctx(clock, make_request(stream=True)), hit, outcome="completed")
    assert h.get("gg_ttft_seconds_sum", provider="cache", cache="exact_hit") == pytest.approx(0.003)
    assert (
        h.get("gg_tokens_total", provider="openai", deployment=LUNA.id, type="input", usage_source="reported")
        is None
    )
    assert h.get("gg_gateway_overhead_seconds_count", stream="true", phase="total") is None
    assert logs[0]["cost_usd"] is None
    assert logs[0]["provider"] == "cache"


async def test_fallback_counts_attempts_and_fallbacks(clock: FakeClock) -> None:
    h = Harness(clock)

    async def with_fallback(ctx: RequestContext) -> PipelineResult:
        start = clock.monotonic()
        ctx.attempts = [
            AttemptRecord(
                "anthropic/claude-haiku-4-5", "anthropic", start, 0.3, "fallback", "retryable", 529
            ),
            AttemptRecord(LUNA.id, "openai", start + 0.4, 0.5, "ok", ttft_s=0.2),
        ]
        clock.advance(0.95)
        ctx.served_by = LUNA
        ctx.usage = luna_usage()
        return PipelineResult(source="upstream", response=RESPONSE)

    ctx = make_ctx(clock, make_request(model="gg/resilient"))
    logs = await h.run(ctx, with_fallback, outcome="completed")
    assert (
        h.get("gg_fallbacks_total", from_provider="anthropic", to_provider="openai", reason="retryable") == 1
    )
    haiku = {"provider": "anthropic", "deployment": "anthropic/claude-haiku-4-5"}
    assert h.get("gg_upstream_attempts_total", **haiku, result="fallback", error_kind="retryable") == 1
    assert h.get("gg_upstream_duration_seconds_count", **haiku, outcome="error") == 1
    assert h.get("gg_gateway_overhead_seconds_sum", stream="false", phase="failover") == pytest.approx(0.4)
    assert [a["result"] for a in logs[0]["attempts"]] == ["fallback", "ok"]


async def test_missing_price_logs_null_cost_and_counts(clock: FakeClock) -> None:
    h = Harness(clock)
    unpriced = Deployment(id="mock/echo", provider="mock", upstream_model="echo")
    h.metrics.labels.allow("deployment", [unpriced.id])
    logs = await h.run(make_ctx(clock), upstream(clock, served=unpriced), outcome="completed")
    assert h.get("gg_pricing_missing_total", provider="mock", deployment="mock/echo") == 1
    assert [e["event"] for e in logs] == ["cost.price_missing", "request.completed"]
    assert logs[1]["cost_usd"] is None
    assert (
        h.get(
            "gg_requests_total",
            endpoint="chat",
            alias="gg/auto",
            status_class="2xx",
            cache="miss",
            stream="false",
        )
        == 1
    )


async def test_unknown_alias_is_bounded(clock: FakeClock) -> None:
    h = Harness(clock)
    await h.run(
        make_ctx(clock, make_request(model="whatever/user-typed")), upstream(clock), outcome="completed"
    )
    labels = {"endpoint": "chat", "status_class": "2xx", "cache": "miss", "stream": "false"}
    assert h.get("gg_requests_total", **labels, alias="other") == 1


async def test_metrics_failure_is_counted_and_inflight_still_released(clock: FakeClock) -> None:
    metrics = new_metrics()

    def boom(_: object) -> None:
        raise RuntimeError("sink broke")

    metrics.record_request = boom  # type: ignore[method-assign]
    stage = ObservabilityStage(metrics, clock=clock)
    ctx = make_ctx(clock)

    async def ok(_: RequestContext) -> PipelineResult:
        return PipelineResult(source="upstream", response=RESPONSE)

    await Pipeline([stage], ok, clock=clock).run(ctx)
    ctx.outcome = "completed"
    with capture_logs() as logs:
        await ctx.finalizers.run(timeout_s=1)
    assert metrics.registry.get_sample_value("gg_inflight_requests", {"stream": "false"}) == 0
    assert metrics.registry.get_sample_value("gg_telemetry_errors_total", {"sink": "metrics"}) == 1
    assert [e["event"] for e in logs] == ["finalizer.failed", "request.completed"]


def test_stage_name_matches_label_allowlist() -> None:
    assert ObservabilityStage.name == "observability"
    assert Metrics(process_collectors=False).labels("stage", ObservabilityStage.name) == "observability"


async def test_stage_passes_result_through(clock: FakeClock) -> None:
    stage = ObservabilityStage(new_metrics(), clock=clock)

    async def call_next(ctx: RequestContext) -> PipelineResult:
        return PipelineResult(source="synthetic", response=RESPONSE)

    next_: Next = call_next
    result = await stage(make_ctx(clock), next_)
    assert result.source == "synthetic"


SOL = Deployment(
    id="openai/gpt-6-sol",
    provider="openai",
    upstream_model="gpt-6-sol",
    pricing=PriceSchedule(
        periods=(Pricing(effective_from=date(2026, 1, 1), input=Decimal("2.00"), output=Decimal("8.00")),)
    ),
)


def routed(tier: str) -> RouteDecision:
    return RouteDecision(
        alias="gg/auto", tier=tier, reason="scored", plan=RoutePlan(alias="gg/auto", entries=())
    )


@pytest.mark.parametrize(("tier", "expected"), [("weak", 2600e-6), ("strong", None)])
async def test_strong_equivalent_cost_only_for_weak_routes(
    clock: FakeClock, tier: str, expected: float | None
) -> None:
    seen: list[RequestContext] = []

    def strong(ctx: RequestContext) -> Deployment:
        seen.append(ctx)
        return SOL

    h = Harness(clock, strong_deployment=strong)
    ctx = make_ctx(clock)
    ctx.route = routed(tier)
    await h.run(ctx, upstream(clock), outcome="completed")
    # 500 input x $2/M + 200 output x $8/M, the same tokens at strong prices
    value = h.get("gg_routing_strong_equiv_cost_usd_total", tier="weak")
    assert value == (pytest.approx(expected) if expected is not None else None)
    assert len(seen) == (1 if tier == "weak" else 0)


async def test_strong_equivalent_skipped_on_cache_hit(clock: FakeClock) -> None:
    h = Harness(clock, strong_deployment=lambda _: SOL)
    ctx = make_ctx(clock)
    ctx.route = routed("weak")
    ctx.cache_status = "exact_hit"
    await h.run(ctx, upstream(clock), outcome="completed")
    assert h.get("gg_routing_strong_equiv_cost_usd_total", tier="weak") is None
