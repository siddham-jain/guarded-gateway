import pytest
from prometheus_client.parser import text_string_to_metric_families

from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.observability.metrics import Metrics
from gg.observability.observer import MetricsObserver
from gg.pipeline.runner import Pipeline
from gg.pipeline.stage import Next, PipelineResult
from tests.conftest import make_ctx
from tests.unit.observability.support import RESPONSE

CATALOGUE: dict[str, tuple[str, set[str]]] = {
    "gg_requests": ("counter", {"endpoint", "alias", "status_class", "cache", "stream"}),
    "gg_request_errors": ("counter", {"endpoint", "error_type"}),
    "gg_inflight_requests": ("gauge", {"stream"}),
    "gg_client_disconnects": ("counter", {"phase"}),
    "gg_request_duration_seconds": ("histogram", {"endpoint", "stream", "cache"}),
    "gg_ttft_seconds": ("histogram", {"provider", "cache"}),
    "gg_upstream_ttft_seconds": ("histogram", {"provider", "deployment"}),
    "gg_upstream_duration_seconds": ("histogram", {"provider", "deployment", "outcome"}),
    "gg_gateway_overhead_seconds": ("histogram", {"stream", "phase"}),
    "gg_stage_duration_seconds": ("histogram", {"stage"}),
    "gg_time_per_output_token_seconds": ("histogram", {"provider", "deployment"}),
    "gg_event_loop_lag_seconds": ("histogram", set()),
    "gg_tokens": ("counter", {"provider", "deployment", "type", "usage_source"}),
    "gg_cost_usd": ("counter", {"provider", "deployment", "tier", "attributed"}),
    "gg_request_cost_usd": ("histogram", {"alias"}),
    "gg_upstream_attempts": ("counter", {"provider", "deployment", "result", "error_kind"}),
    "gg_fallbacks": ("counter", {"from_provider", "to_provider", "reason"}),
    "gg_routing_decisions": ("counter", {"alias", "tier", "reason", "scorer_tier"}),
    "gg_router_score_duration_seconds": ("histogram", {"scorer", "outcome"}),
    "gg_router_score": ("histogram", {"scorer"}),
    "gg_routing_decision_cache": ("counter", {"layer", "result"}),
    "gg_pricing_missing": ("counter", {"provider", "deployment"}),
    "gg_spend_guard_blocks": ("counter", {"provider", "period"}),
    "gg_routing_strong_equiv_cost_usd": ("counter", {"tier"}),
    "gg_cache_lookups": ("counter", {"layer", "result"}),
    "gg_cache_lookup_duration_seconds": ("histogram", {"layer"}),
    "gg_semantic_cache_distance": ("histogram", {"result"}),
    "gg_cache_stores": ("counter", {"layer", "result", "reason"}),
    "gg_cache_bypass": ("counter", {"reason"}),
    "gg_cache_singleflight": ("counter", {"outcome"}),
    "gg_cost_saved_usd": ("counter", {"layer"}),
    "gg_tokens_saved": ("counter", {"layer", "type"}),
    "gg_ratelimit_rejections": ("counter", {"limit"}),
    "gg_budget_events": ("counter", {"event"}),
    "gg_ratelimit_backend_errors": ("counter", {"op"}),
    "gg_ratelimit_degraded": ("gauge", set()),
    "gg_budget_hold_expired": ("counter", set()),
    "gg_circuit_state": ("gauge", {"deployment"}),
    "gg_circuit_transitions": ("counter", {"deployment", "to_state", "reason"}),
    "gg_deployment_skips": ("counter", {"deployment", "reason"}),
    "gg_stream_errors": ("counter", {"phase", "provider"}),
    "gg_exhausted": ("counter", {"status", "code"}),
    "gg_provider_credential_errors": ("counter", {"provider"}),
    "gg_guardrail_decisions": ("counter", {"stage", "guardrail", "action", "mode"}),
    "gg_guardrail_duration_seconds": ("histogram", {"stage", "guardrail"}),
    "gg_guardrail_errors": ("counter", {"stage", "guardrail", "kind"}),
    "gg_pii_placeholders": ("counter", {"direction", "strategy"}),
    "gg_telemetry_errors": ("counter", {"sink"}),
    "gg_telemetry_dropped": ("counter", {"sink"}),
    "gg_build_info": ("gauge", {"version", "git_sha", "config_hash"}),
    "gg_config_info": ("gauge", {"config_hash", "models_hash", "routing_hash"}),
}

FORBIDDEN_LABELS = {"request_id", "key_id", "key", "user", "trace_id", "prompt"}


def test_catalogue_names_types_and_labels() -> None:
    metrics = Metrics(process_collectors=False)
    found = {}
    for collected in metrics.registry.collect():
        labels: set[str] = set(getattr(collected, "_labelnames", ()))
        found[collected.name] = collected.type
        assert not labels & FORBIDDEN_LABELS
    for name, (kind, _) in CATALOGUE.items():
        assert found.get(name) == kind, name
    assert set(found) == set(CATALOGUE)
    for attr in vars(metrics).values():
        name = getattr(attr, "_name", None)
        if name is not None:
            expected = CATALOGUE[name.removesuffix("_total")][1]
            assert set(attr._labelnames) == expected, name


def test_private_registries_do_not_collide() -> None:
    a, b = Metrics(), Metrics()
    a.telemetry_error("log")
    assert a.registry.get_sample_value("gg_telemetry_errors_total", {"sink": "log"}) == 1
    assert b.registry.get_sample_value("gg_telemetry_errors_total", {"sink": "log"}) is None


def test_render_is_prometheus_text() -> None:
    metrics = Metrics()
    metrics.set_build_info(version="0.1.0", git_sha="abc1234", config_hash="9f2c" * 8)
    metrics.set_build_info(version="0.1.1", git_sha="def5678", config_hash="a71e" * 8)
    metrics.set_config_info(config_hash="1" * 64, models_hash="2" * 64)
    body, content_type = metrics.render()
    assert content_type.startswith("text/plain")
    families = {f.name: f for f in text_string_to_metric_families(body.decode())}
    build = families["gg_build_info"].samples
    assert [s.labels for s in build] == [
        {"version": "0.1.1", "git_sha": "def5678", "config_hash": "a71ea71ea71e"}
    ]
    assert families["gg_config_info"].samples[0].labels["routing_hash"] == ""


def test_label_guard_maps_unknown_values_to_other_and_reports_once() -> None:
    metrics = Metrics(deployments=["openai/gpt-6-luna"])
    metrics.pricing_missing("openai", "openai/gpt-6-luna")
    metrics.pricing_missing("together", "together/llama")
    metrics.pricing_missing("together", "together/llama")
    get = metrics.registry.get_sample_value
    assert get("gg_pricing_missing_total", {"provider": "openai", "deployment": "openai/gpt-6-luna"}) == 1
    assert get("gg_pricing_missing_total", {"provider": "other", "deployment": "other"}) == 2
    assert get("gg_telemetry_errors_total", {"sink": "label"}) == 2
    assert metrics.labels("alias", None) == "none"


def test_capped_label_admits_new_values_up_to_the_cap() -> None:
    metrics = Metrics(process_collectors=False)
    admitted = [metrics.labels("cache_reason", f"reason_{i}") for i in range(60)]
    assert admitted[:48] == [f"reason_{i}" for i in range(48)]
    assert set(admitted[48:]) == {"other"}
    assert metrics.labels("cache_reason", "reason_3") == "reason_3"
    assert metrics.labels("cache_reason", "x" * 65) == "other"
    assert metrics.registry.get_sample_value("gg_telemetry_errors_total", {"sink": "label"}) == 13


def test_spend_denied_counter() -> None:
    metrics = Metrics()
    metrics.spend_denied("openai", "day")
    assert (
        metrics.registry.get_sample_value(
            "gg_spend_guard_blocks_total", {"provider": "openai", "period": "day"}
        )
        == 1
    )


class Advance:
    def __init__(self, name: str, seconds: float, clock: FakeClock) -> None:
        self.name = name
        self._seconds = seconds
        self._clock = clock

    async def __call__(self, ctx: RequestContext, call_next: Next, /) -> PipelineResult:
        self._clock.advance(self._seconds)
        return await call_next(ctx)


async def test_observer_records_exclusive_stage_time(clock: FakeClock) -> None:
    metrics = Metrics(stages=["custom"])

    async def terminal(ctx: RequestContext) -> PipelineResult:
        clock.advance(0.5)
        return PipelineResult(source="upstream", response=RESPONSE)

    pipeline = Pipeline(
        [Advance("routing", 0.003, clock), Advance("custom", 0.0002, clock)],
        terminal,
        clock=clock,
        observer=MetricsObserver(metrics),
    )
    await pipeline.run(make_ctx(clock))
    get = metrics.registry.get_sample_value
    assert get("gg_stage_duration_seconds_sum", {"stage": "routing"}) == pytest.approx(0.003)
    assert get("gg_stage_duration_seconds_bucket", {"stage": "custom", "le": "0.00025"}) == 1
    assert get("gg_stage_duration_seconds_count", {"stage": "terminal"}) is None
