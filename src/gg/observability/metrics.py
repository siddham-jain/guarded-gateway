from collections.abc import Iterable

import structlog
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.gc_collector import GCCollector
from prometheus_client.platform_collector import PlatformCollector
from prometheus_client.process_collector import ProcessCollector

from gg.observability.labels import LabelGuard
from gg.observability.record import RequestRecord, micros_to_usd
from gg.observability.timing import final_attempt

log = structlog.get_logger("gg.observability")

# bucket sets from the C10 §4.1 catalogue
LAT = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92)
TTFT = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 20, 30)
OVH = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)
STAGE = (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)
TPOT = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1)
COST = (0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1)
DIST = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
SCORE = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)

ENDPOINT_CHAT = "chat"


def _flag(value: bool) -> str:
    return "true" if value else "false"


class Metrics:
    """the gg_* catalogue on a private registry, so two apps in one process never collide"""

    def __init__(
        self,
        *,
        aliases: Iterable[str] = (),
        deployments: Iterable[str] = (),
        providers: Iterable[str] = (),
        stages: Iterable[str] = (),
        tiers: Iterable[str] = (),
        guardrails: Iterable[str] = (),
        process_collectors: bool = True,
    ) -> None:
        self.registry = r = CollectorRegistry()
        if process_collectors:
            ProcessCollector(registry=r)
            PlatformCollector(registry=r)
            GCCollector(registry=r)
        self.labels = LabelGuard(self._unknown_label)
        self.labels.allow("alias", aliases)
        self.labels.allow("deployment", deployments)
        self.labels.allow("provider", providers)
        self.labels.allow("stage", stages)
        self.labels.allow("tier", tiers)
        self.labels.allow("guardrail", guardrails)

        self.requests = Counter(
            "gg_requests_total",
            "Chat requests by logical outcome.",
            ("endpoint", "alias", "status_class", "cache", "stream"),
            registry=r,
        )
        self.request_errors = Counter(
            "gg_request_errors_total",
            "Failed requests by error type.",
            ("endpoint", "error_type"),
            registry=r,
        )
        self.inflight = Gauge("gg_inflight_requests", "Requests in flight.", ("stream",), registry=r)
        self.client_disconnects = Counter(
            "gg_client_disconnects_total", "Client disconnects by phase.", ("phase",), registry=r
        )
        self.request_duration = Histogram(
            "gg_request_duration_seconds",
            "Server-side request duration, receive to last byte.",
            ("endpoint", "stream", "cache"),
            buckets=LAT,
            registry=r,
        )
        self.ttft = Histogram(
            "gg_ttft_seconds",
            "Time to first content byte written to the client (streams).",
            ("provider", "cache"),
            buckets=TTFT,
            registry=r,
        )
        self.upstream_ttft = Histogram(
            "gg_upstream_ttft_seconds",
            "Upstream attempt start to first content delta.",
            ("provider", "deployment"),
            buckets=TTFT,
            registry=r,
        )
        self.upstream_duration = Histogram(
            "gg_upstream_duration_seconds",
            "Upstream attempt duration.",
            ("provider", "deployment", "outcome"),
            buckets=LAT,
            registry=r,
        )
        self.overhead = Histogram(
            "gg_gateway_overhead_seconds",
            "Time the gateway adds on top of the upstream, by phase.",
            ("stream", "phase"),
            buckets=OVH,
            registry=r,
        )
        self.stage_duration = Histogram(
            "gg_stage_duration_seconds",
            "Exclusive time per pipeline stage.",
            ("stage",),
            buckets=STAGE,
            registry=r,
        )
        self.tpot = Histogram(
            "gg_time_per_output_token_seconds",
            "Upstream time per output token after the first.",
            ("provider", "deployment"),
            buckets=TPOT,
            registry=r,
        )
        self.event_loop_lag = Histogram(
            "gg_event_loop_lag_seconds", "Event loop scheduling lag.", buckets=STAGE, registry=r
        )
        self.tokens = Counter(
            "gg_tokens_total",
            "Upstream tokens; input includes cached and cache-write, output includes reasoning.",
            ("provider", "deployment", "type", "usage_source"),
            registry=r,
        )
        self.cost = Counter(
            "gg_cost_usd_total",
            "Upstream cost in USD.",
            ("provider", "deployment", "tier", "attributed"),
            registry=r,
        )
        self.request_cost = Histogram(
            "gg_request_cost_usd", "Cost per served request in USD.", ("alias",), buckets=COST, registry=r
        )
        self.upstream_attempts = Counter(
            "gg_upstream_attempts_total",
            "Upstream attempts by result.",
            ("provider", "deployment", "result", "error_kind"),
            registry=r,
        )
        self.fallbacks = Counter(
            "gg_fallbacks_total",
            "Cross-deployment fallbacks.",
            ("from_provider", "to_provider", "reason"),
            registry=r,
        )
        self.pricing_missing_total = Counter(
            "gg_pricing_missing_total",
            "Usage that could not be priced because no price period applies.",
            ("provider", "deployment"),
            registry=r,
        )
        self.spend_guard_blocks = Counter(
            "gg_spend_guard_blocks_total",
            "Upstream attempts refused by the provider spend guard.",
            ("provider", "period"),
            registry=r,
        )
        self.routing_decisions = Counter(
            "gg_routing_decisions_total",
            "Routing decisions for router aliases.",
            ("alias", "tier", "reason", "scorer_tier"),
            registry=r,
        )
        self.router_score_duration = Histogram(
            "gg_router_score_duration_seconds",
            "Routing scorer latency.",
            ("scorer", "outcome"),
            buckets=TTFT,
            registry=r,
        )
        self.router_score = Histogram(
            "gg_router_score",
            "Routing score, approx P(strong beats weak).",
            ("scorer",),
            buckets=SCORE,
            registry=r,
        )
        self.routing_decision_cache = Counter(
            "gg_routing_decision_cache_total", "Scorer answer cache lookups.", ("layer", "result"), registry=r
        )
        self.routing_strong_equiv_cost = Counter(
            "gg_routing_strong_equiv_cost_usd_total",
            "Estimate: what weak-routed requests would cost at the strong deployment's prices, same tokens.",
            ("tier",),
            registry=r,
        )
        self.cache_lookups = Counter(
            "gg_cache_lookups_total", "Response cache lookups.", ("layer", "result"), registry=r
        )
        self.cache_lookup_duration = Histogram(
            "gg_cache_lookup_duration_seconds",
            "Response cache lookup latency.",
            ("layer",),
            buckets=STAGE,
            registry=r,
        )
        self.semantic_cache_distance = Histogram(
            "gg_semantic_cache_distance",
            "Cosine distance of the nearest semantic cache entry.",
            ("result",),
            buckets=DIST,
            registry=r,
        )
        self.cache_stores = Counter(
            "gg_cache_stores_total", "Response cache writes.", ("layer", "result", "reason"), registry=r
        )
        self.cache_bypass = Counter(
            "gg_cache_bypass_total", "Requests that skipped the cache.", ("reason",), registry=r
        )
        self.cache_singleflight = Counter(
            "gg_cache_singleflight_total",
            "Single-flight lock outcomes on cache misses.",
            ("outcome",),
            registry=r,
        )
        self.cost_saved = Counter(
            "gg_cost_saved_usd_total", "Upstream cost avoided by cache hits in USD.", ("layer",), registry=r
        )
        self.tokens_saved = Counter(
            "gg_tokens_saved_total", "Upstream tokens avoided by cache hits.", ("layer", "type"), registry=r
        )
        self.ratelimit_rejections = Counter(
            "gg_ratelimit_rejections_total",
            "Requests refused by rate limits and budgets.",
            ("limit",),
            registry=r,
        )
        self.budget_events = Counter("gg_budget_events_total", "Key budget events.", ("event",), registry=r)
        self.ratelimit_backend_errors = Counter(
            "gg_ratelimit_backend_errors_total", "Limits backend errors and timeouts.", ("op",), registry=r
        )
        self.ratelimit_degraded = Gauge(
            "gg_ratelimit_degraded", "1 while the local rate-limit fallback is active.", registry=r
        )
        self.budget_hold_expired = Counter(
            "gg_budget_hold_expired_total", "Budget holds swept after their ttl.", registry=r
        )
        self.circuit_state = Gauge(
            "gg_circuit_state", "Breaker state: 0 closed, 1 half-open, 2 open.", ("deployment",), registry=r
        )
        self.circuit_transitions = Counter(
            "gg_circuit_transitions_total",
            "Breaker state transitions.",
            ("deployment", "to_state", "reason"),
            registry=r,
        )
        self.deployment_skips = Counter(
            "gg_deployment_skips_total",
            "Plan entries the executor skipped without calling upstream.",
            ("deployment", "reason"),
            registry=r,
        )
        self.stream_errors = Counter(
            "gg_stream_errors_total", "Upstream stream failures.", ("phase", "provider"), registry=r
        )
        self.exhausted = Counter(
            "gg_exhausted_total",
            "Requests the executor failed after exhausting its plan.",
            ("status", "code"),
            registry=r,
        )
        self.provider_credential_errors = Counter(
            "gg_provider_credential_errors_total", "Upstream auth failures.", ("provider",), registry=r
        )
        self.guardrail_decisions = Counter(
            "gg_guardrail_decisions_total",
            "Guardrail verdicts; shadow rows carry the would-be action.",
            ("stage", "guardrail", "action", "mode"),
            registry=r,
        )
        self.guardrail_duration = Histogram(
            "gg_guardrail_duration_seconds",
            "Guardrail check latency.",
            ("stage", "guardrail"),
            buckets=STAGE,
            registry=r,
        )
        self.guardrail_errors = Counter(
            "gg_guardrail_errors_total", "Guardrail failures.", ("stage", "guardrail", "kind"), registry=r
        )
        self.pii_placeholders = Counter(
            "gg_pii_placeholders_total",
            "PII placeholders by direction.",
            ("direction", "strategy"),
            registry=r,
        )
        self.telemetry_errors = Counter(
            "gg_telemetry_errors_total", "Telemetry failures and label overflows.", ("sink",), registry=r
        )
        self.telemetry_dropped = Counter(
            "gg_telemetry_dropped_total",
            "Telemetry records dropped (full queue or failed export).",
            ("sink",),
            registry=r,
        )
        self.build_info = Gauge(
            "gg_build_info", "Build metadata (always 1).", ("version", "git_sha", "config_hash"), registry=r
        )
        self.config_info = Gauge(
            "gg_config_info",
            "Loaded configuration hashes (always 1).",
            ("config_hash", "models_hash", "routing_hash"),
            registry=r,
        )

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST

    def set_build_info(self, *, version: str, git_sha: str, config_hash: str) -> None:
        self.build_info.clear()
        self.build_info.labels(version, git_sha, config_hash[:12]).set(1)

    def set_config_info(self, *, config_hash: str, models_hash: str = "", routing_hash: str = "") -> None:
        self.config_info.clear()
        self.config_info.labels(config_hash[:12], models_hash[:12], routing_hash[:12]).set(1)

    def observe_stage(self, stage: str, seconds: float) -> None:
        self.stage_duration.labels(self.labels("stage", stage)).observe(seconds)

    def observe_loop_lag(self, seconds: float) -> None:
        self.event_loop_lag.observe(seconds)

    def telemetry_error(self, sink: str) -> None:
        self.telemetry_errors.labels(sink).inc()

    def dropped(self, sink: str, count: int = 1) -> None:
        if count > 0:
            self.telemetry_dropped.labels(sink).inc(count)

    def pricing_missing(self, provider: str, deployment: str, /) -> None:
        self.pricing_missing_total.labels(
            self.labels("provider", provider), self.labels("deployment", deployment)
        ).inc()

    def spend_denied(self, provider: str, reason: str, /) -> None:
        self.spend_guard_blocks.labels(self.labels("provider", provider), reason).inc()

    def record_request(self, rec: RequestRecord) -> None:
        guard = self.labels
        stream = _flag(rec.stream)
        alias = guard("alias", rec.alias)
        provider = guard("provider", rec.provider)
        self.requests.labels(ENDPOINT_CHAT, alias, rec.status_class, rec.cache_status, stream).inc()
        if rec.status_class in ("4xx", "5xx"):
            self.request_errors.labels(ENDPOINT_CHAT, guard("error_type", rec.error_type or "internal")).inc()
        if rec.disconnect_phase is not None:
            self.client_disconnects.labels(rec.disconnect_phase).inc()

        t = rec.timings
        self.request_duration.labels(ENDPOINT_CHAT, stream, rec.cache_status).observe(t.total)
        if t.ttft is not None:
            self.ttft.labels(provider, rec.cache_status).observe(t.ttft)
        for phase, seconds in t.overhead_phases().items():
            self.overhead.labels(stream, phase).observe(seconds)

        self._record_attempts(rec)
        self._record_usage(rec, alias)

    def _record_attempts(self, rec: RequestRecord) -> None:
        guard = self.labels
        final = final_attempt(rec.attempts)
        for i, attempt in enumerate(rec.attempts):
            provider = guard("provider", attempt.provider)
            deployment = guard("deployment", attempt.deployment_id)
            self.upstream_attempts.labels(
                provider, deployment, attempt.outcome, guard("error_kind", attempt.error_kind)
            ).inc()
            outcome = "ok" if attempt.outcome == "ok" else "error"
            if attempt is final and rec.outcome == "client_disconnected":
                outcome = "cancelled"
            self.upstream_duration.labels(provider, deployment, outcome).observe(attempt.duration_s)
            if attempt.ttft_s is not None:
                self.upstream_ttft.labels(provider, deployment).observe(attempt.ttft_s)
            if attempt.outcome == "fallback" and i + 1 < len(rec.attempts):
                self.fallbacks.labels(
                    provider,
                    guard("provider", rec.attempts[i + 1].provider),
                    guard("error_kind", attempt.error_kind),
                ).inc()

    def _record_usage(self, rec: RequestRecord, alias: str) -> None:
        usage = rec.upstream_usage
        if usage is None:
            return
        provider = self.labels("provider", usage.provider)
        deployment = self.labels("deployment", usage.deployment_id)
        counts = {
            "input": usage.input_tokens,
            "cached_input": usage.cached_input_tokens,
            "cache_write": usage.cache_write_tokens,
            "output": usage.output_tokens,
            "reasoning": usage.reasoning_tokens,
        }
        for kind, count in counts.items():
            self.tokens.labels(provider, deployment, kind, usage.usage_source).inc(count)
        if rec.timings.tpot is not None:
            self.tpot.labels(provider, deployment).observe(rec.timings.tpot)
        if rec.cost is not None:
            usd = float(micros_to_usd(rec.cost.total))
            self.cost.labels(provider, deployment, self.labels("tier", rec.tier), "key").inc(usd)
            self.request_cost.labels(alias).observe(usd)
        if rec.strong_equiv_cost is not None:
            # only weak-routed requests get a strong-equivalent price, so savings = this - cost{tier="weak"}
            self.routing_strong_equiv_cost.labels("weak").inc(
                float(micros_to_usd(rec.strong_equiv_cost.total))
            )

    def _unknown_label(self, label: str, value: str) -> None:
        self.telemetry_error("label")
        log.warning("metrics.label_overflow", label=label, value=value[:64])
