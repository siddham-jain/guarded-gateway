import pytest

from gg.cache.base import CacheHooks
from gg.core.clock import FakeClock
from gg.core.errors import ProviderError, ProviderErrorKind, UpstreamError
from gg.core.usage import AttemptRecord
from gg.guardrails.base import GuardMetrics
from gg.limits.base import LimitsHooks
from gg.observability.cache import CacheMetrics
from gg.observability.guardrails import GuardrailMetrics
from gg.observability.limits import LimitsMetrics
from gg.observability.metrics import Metrics
from gg.observability.reliability import ReliabilityMetrics
from gg.reliability.breaker import BreakerConfig, BreakerRegistry
from gg.reliability.hooks import ReliabilityHooks
from tests.conftest import make_ctx
from tests.unit.observability.support import LUNA, new_metrics


def sample(metrics: Metrics, name: str, **labels: str) -> float | None:
    return metrics.registry.get_sample_value(name, labels)


def test_hooks_satisfy_their_protocols() -> None:
    metrics = new_metrics()
    cache: CacheHooks = CacheMetrics(metrics)
    limits: LimitsHooks = LimitsMetrics(metrics)
    reliability: ReliabilityHooks = ReliabilityMetrics(metrics)
    guards: GuardMetrics = GuardrailMetrics(metrics)
    assert all((cache, limits, reliability, guards))


def test_cache_metrics() -> None:
    m = new_metrics()
    hooks = CacheMetrics(m)
    hooks.lookup("exact", "hit")
    hooks.lookup("semantic", "miss")
    hooks.lookup_duration("exact", 0.0004)
    hooks.semantic_distance("hit", 0.04)
    hooks.store("exact", "stored", "ok")
    hooks.store("exact", "skipped", "too_large")
    hooks.bypass("sampled")
    hooks.singleflight("waited_hit")
    hooks.cost_saved("exact", 0.0025)
    hooks.cost_saved("exact", 0.0)
    hooks.tokens_saved("semantic", "input", 120)
    hooks.tokens_saved("semantic", "bogus", 5)

    assert sample(m, "gg_cache_lookups_total", layer="exact", result="hit") == 1
    assert sample(m, "gg_cache_lookups_total", layer="semantic", result="miss") == 1
    assert sample(m, "gg_cache_lookup_duration_seconds_bucket", layer="exact", le="0.0005") == 1
    assert sample(m, "gg_semantic_cache_distance_bucket", result="hit", le="0.05") == 1
    assert sample(m, "gg_cache_stores_total", layer="exact", result="skipped", reason="too_large") == 1
    assert sample(m, "gg_cache_bypass_total", reason="sampled") == 1
    assert sample(m, "gg_cache_singleflight_total", outcome="waited_hit") == 1
    assert sample(m, "gg_cost_saved_usd_total", layer="exact") == pytest.approx(0.0025)
    assert sample(m, "gg_tokens_saved_total", layer="semantic", type="input") == 120
    assert sample(m, "gg_tokens_saved_total", layer="semantic", type="other") == 5


def test_cache_reason_overflow_maps_to_other() -> None:
    m = new_metrics()
    hooks = CacheMetrics(m)
    for i in range(50):
        hooks.bypass(f"r{i}")
    assert sample(m, "gg_cache_bypass_total", reason="r0") == 1
    assert sample(m, "gg_cache_bypass_total", reason="other") == 2
    assert sample(m, "gg_telemetry_errors_total", sink="label") == 2


def test_limits_metrics() -> None:
    m = new_metrics()
    hooks = LimitsMetrics(m)
    hooks.rejected("rpm")
    hooks.rejected("budget")
    hooks.budget_event("soft_limit")
    hooks.backend_error("acquire")
    hooks.degraded(True)
    assert sample(m, "gg_ratelimit_degraded") == 1
    hooks.degraded(False)
    hooks.holds_expired(3)
    hooks.holds_expired(0)

    assert sample(m, "gg_ratelimit_rejections_total", limit="rpm") == 1
    assert sample(m, "gg_ratelimit_rejections_total", limit="budget") == 1
    assert sample(m, "gg_budget_events_total", event="soft_limit") == 1
    assert sample(m, "gg_ratelimit_backend_errors_total", op="acquire") == 1
    assert sample(m, "gg_ratelimit_degraded") == 0
    assert sample(m, "gg_budget_hold_expired_total") == 3


def provider_error(kind: ProviderErrorKind) -> ProviderError:
    return ProviderError(kind, provider="openai", status=401, deployment_id=LUNA.id)


def test_reliability_metrics(clock: FakeClock) -> None:
    m = new_metrics()
    hooks = ReliabilityMetrics(m)
    ctx = make_ctx(clock)
    record = AttemptRecord(
        deployment_id=LUNA.id, provider="openai", started_at=0.0, duration_s=0.1, outcome="fallback"
    )
    hooks.attempt_finished(ctx, LUNA, record, provider_error("auth"))
    hooks.attempt_finished(ctx, LUNA, record, provider_error("retryable"))
    hooks.attempt_finished(ctx, LUNA, record, None)
    hooks.deployment_skipped(ctx, LUNA, "circuit_open")
    hooks.stream_interrupted(ctx, LUNA, provider_error("retryable"))
    hooks.request_failed(ctx, UpstreamError("all upstreams failed", code="upstream_unavailable"))

    assert sample(m, "gg_provider_credential_errors_total", provider="openai") == 1
    assert sample(m, "gg_deployment_skips_total", deployment=LUNA.id, reason="circuit_open") == 1
    assert sample(m, "gg_stream_errors_total", phase="mid_stream", provider="openai") == 1
    assert sample(m, "gg_exhausted_total", status="502", code="upstream_unavailable") == 1
    # attempts are counted once, from the request record
    assert sample(m, "gg_upstream_attempts_total", provider="openai", deployment=LUNA.id) is None


def test_breaker_transitions_drive_the_state_gauge(clock: FakeClock) -> None:
    m = new_metrics()
    breakers = BreakerRegistry(clock, BreakerConfig(consecutive_failures=1), listener=ReliabilityMetrics(m))
    permit = breakers.get(LUNA).try_acquire()
    assert permit is not None
    permit.failure()
    assert sample(m, "gg_circuit_state", deployment=LUNA.id) == 2
    labels = {"deployment": LUNA.id, "to_state": "open", "reason": "trip"}
    assert sample(m, "gg_circuit_transitions_total", **labels) == 1


def test_breaker_unknown_deployment_and_reason_are_bounded() -> None:
    m = new_metrics()
    hooks = ReliabilityMetrics(m)
    hooks.breaker_transition("together/llama", "closed", "open", "invalid_api_key")
    labels = {"deployment": "other", "to_state": "open", "reason": "invalid_api_key"}
    assert sample(m, "gg_circuit_transitions_total", **labels) == 1
    assert sample(m, "gg_circuit_state", deployment="other") is None
    hooks.breaker_transition(LUNA.id, "open", "half_open", "cooldown_elapsed")
    assert sample(m, "gg_circuit_state", deployment=LUNA.id) == 1


def test_guardrail_metrics() -> None:
    m = Metrics(guardrails=["pii_regex", "secrets"], process_collectors=False)
    hooks = GuardrailMetrics(m)
    hooks.decision("input", "pii_regex", "redact", "enforce")
    hooks.decision("output", "secrets", "block", "shadow")
    hooks.duration("input", "pii_regex", 0.0008)
    hooks.error("input", "secrets", "timeout")
    hooks.placeholders("redacted", "none", 3)
    hooks.placeholders("restored", "exact", 0)
    hooks.decision("input", "pii_regex", "explode", "sideways")

    labels = {"stage": "input", "guardrail": "pii_regex"}
    assert sample(m, "gg_guardrail_decisions_total", **labels, action="redact", mode="enforce") == 1
    shadow = {"stage": "output", "guardrail": "secrets", "action": "block", "mode": "shadow"}
    assert sample(m, "gg_guardrail_decisions_total", **shadow) == 1
    assert sample(m, "gg_guardrail_decisions_total", **labels, action="other", mode="other") == 1
    assert sample(m, "gg_guardrail_duration_seconds_count", **labels) == 1
    assert sample(m, "gg_guardrail_errors_total", stage="input", guardrail="secrets", kind="timeout") == 1
    assert sample(m, "gg_pii_placeholders_total", direction="redacted", strategy="none") == 3
    assert sample(m, "gg_pii_placeholders_total", direction="restored", strategy="exact") is None


def test_guardrail_names_are_capped() -> None:
    m = Metrics(process_collectors=False)
    hooks = GuardrailMetrics(m)
    for i in range(40):
        hooks.duration("input", f"guard_{i}", 0.001)
    assert sample(m, "gg_guardrail_duration_seconds_count", stage="input", guardrail="guard_0") == 1
    assert sample(m, "gg_guardrail_duration_seconds_count", stage="input", guardrail="other") == 8
