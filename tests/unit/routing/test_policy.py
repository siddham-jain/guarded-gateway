from typing import Any

import pytest

from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.errors import PermissionDeniedError
from gg.routing.base import RoutingScore
from gg.routing.config import RoutingConfig
from gg.routing.policy import ThresholdPolicy, resolve_threshold
from gg.routing.session import SessionStore
from tests.conftest import make_ctx, make_key, make_request
from tests.unit.routing.support import HAIKU, SONNET, STRONG, STRONG_TOOLS, WEAK, FakeCatalog

PREMIUM = Deployment(
    id="anthropic/claude-opus-5-5", provider="anthropic", upstream_model="opus", tier="premium"
)
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
TOOL_TURN = [
    {"role": "user", "content": "look it up"},
    {
        "role": "assistant",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "result"},
]


def scored(value: float, *, claim: float = 0.0, hint: Any = None) -> RoutingScore:
    return RoutingScore(
        score=value,
        raw_score=value,
        scorer="jev",
        scorer_version="v",
        guard_signals={"routing_claim_present": claim},
        reasoning_effort_hint=hint,
    )


def fallback(reason: Any = "timeout") -> RoutingScore:
    return RoutingScore.fallback_for("jev", "v", reason)


def make_policy(
    clock: FakeClock, catalog: FakeCatalog | None = None, **policy: Any
) -> tuple[ThresholdPolicy, SessionStore]:
    sessions = SessionStore(100, 3600, clock)
    cfg = RoutingConfig.model_validate({"policy": policy})
    return ThresholdPolicy(cfg, catalog or FakeCatalog(), sessions), sessions


def context(clock: FakeClock, *, routing: dict[str, Any] | None = None, **request: Any) -> RequestContext:
    ctx = make_ctx(clock, make_request(**request))
    ctx.key = make_key(routing=routing or {})
    return ctx


ALLOW = {"allow_request_threshold": True}


@pytest.mark.parametrize(
    ("score", "routing", "gg", "tier", "reason", "alpha"),
    [
        (scored(0.7), None, None, "strong", "scored", 0.5),
        (scored(0.3), None, None, "weak", "scored", 0.5),
        (scored(0.5), None, None, "strong", "scored", 0.5),
        (scored(0.6), {"threshold": 0.8}, None, "weak", "scored", 0.8),
        (scored(0.6), {"threshold": 0.8, **ALLOW}, {"route_threshold": 0.4}, "strong", "scored", 0.4),
        (scored(0.6), {"threshold": 0.8}, {"route_threshold": 0.4}, "weak", "scored", 0.8),
        (
            scored(0.3),
            {"threshold_bounds": (0.4, 1.0), **ALLOW},
            {"route_threshold": 0.0},
            "weak",
            "scored",
            0.4,
        ),
        (scored(0.0), {"threshold": 0.0}, None, "strong", "scored", 0.0),
        (scored(0.99), {"threshold": 1.0}, None, "weak", "scored", 1.0),
        (scored(1.0), {"threshold": 1.0}, None, "strong", "scored", 1.0),
        (fallback("timeout"), None, None, "strong", "fallback:timeout", 0.5),
        (fallback("disabled"), {"fallback_route": "weak"}, None, "weak", "fallback:disabled", 0.5),
        (scored(0.2, claim=0.9), None, None, "strong", "override:routing_claim", 0.5),
        (scored(0.2, claim=0.4), None, None, "weak", "scored", 0.5),
        (scored(0.8, claim=0.9), None, None, "strong", "scored", 0.5),
    ],
)
def test_decision_table(
    clock: FakeClock,
    score: RoutingScore,
    routing: dict[str, Any] | None,
    gg: dict[str, Any] | None,
    tier: str,
    reason: str,
    alpha: float,
) -> None:
    policy, _ = make_policy(clock)
    decision = policy.decide(score, context(clock, routing=routing, gg=gg))
    assert (decision.tier, decision.reason, decision.threshold) == (tier, reason, alpha)
    assert decision.alias == "gg/auto"
    assert decision.score == (None if score.fallback else score.score)
    assert all(e.tier == tier for e in decision.plan.entries)


def test_routing_claim_can_be_ignored(clock: FakeClock) -> None:
    policy, _ = make_policy(clock, routing_claim={"action": "ignore"})
    assert policy.decide(scored(0.2, claim=0.9), context(clock)).tier == "weak"


def test_global_fallback_route_and_threshold_from_config(clock: FakeClock) -> None:
    policy, _ = make_policy(clock, threshold=0.9, fallback_route="weak")
    assert policy.decide(scored(0.8), context(clock)).tier == "weak"
    assert policy.decide(fallback(), context(clock)).tier == "weak"


def test_profile_threshold_and_unknown_profile(clock: FakeClock) -> None:
    cfg = RoutingConfig.model_validate(
        {"profiles": {"default": {"weak_group": "weak", "strong_group": "strong", "threshold": 0.2}}}
    )
    ctx = context(clock)
    assert resolve_threshold(ctx, cfg, cfg.profiles["default"]) == 0.2
    policy = ThresholdPolicy(cfg, FakeCatalog(), SessionStore(10, 60, clock))
    assert policy.decide(scored(0.3), context(clock, routing={"profile": "nope"})).tier == "strong"


def test_plain_and_tool_chains_with_fixed_effort(clock: FakeClock) -> None:
    policy, _ = make_policy(clock)
    plain = policy.decide(scored(0.9), context(clock))
    assert [(e.deployment, dict(e.overrides)) for e in plain.plan.entries] == [
        (STRONG, {"reasoning_effort": "low"}),
        (SONNET, {}),
    ]
    tools = policy.decide(scored(0.9), context(clock, tools=TOOLS))
    assert [e.deployment for e in tools.plan.entries] == [STRONG_TOOLS, SONNET]
    weak = policy.decide(scored(0.1), context(clock))
    assert [(e.deployment, dict(e.overrides)) for e in weak.plan.entries] == [
        (WEAK, {"reasoning_effort": "none"}),
        (HAIKU, {}),
    ]


def test_client_effort_wins_and_hinted_mode(clock: FakeClock) -> None:
    policy, _ = make_policy(clock)
    client_set = policy.decide(scored(0.9), context(clock, reasoning_effort="high"))
    assert all(not e.overrides for e in client_set.plan.entries)
    hinted, _ = make_policy(clock, effort={"mode": "hinted"})
    decision = hinted.decide(scored(0.9, hint="high"), context(clock))
    assert decision.plan.entries[0].overrides == {"reasoning_effort": "high"}
    off, _ = make_policy(clock, effort={"mode": "off"})
    assert not off.decide(scored(0.9), context(clock)).plan.entries[0].overrides


def test_key_provider_filter_and_fallback_flag(clock: FakeClock) -> None:
    policy, _ = make_policy(clock)
    ctx = context(clock, gg={"fallback": False})
    ctx.key = make_key(allowed_providers=["anthropic"])
    decision = policy.decide(scored(0.9), ctx)
    assert [e.deployment for e in decision.plan.entries] == [SONNET]
    assert not decision.plan.allow_fallback


def test_tool_continuation_reuses_the_session_tier(clock: FakeClock) -> None:
    policy, _ = make_policy(clock)
    first = context(clock, messages=TOOL_TURN[:1], tools=TOOLS)
    assert policy.presolve(first) is None
    assert policy.decide(scored(0.1), first).tier == "weak"
    follow = context(clock, messages=TOOL_TURN, tools=TOOLS)
    reused = policy.presolve(follow)
    assert reused is not None
    assert (reused.tier, reused.reason, reused.score) == ("weak", "session", None)


def test_explicit_session_id_is_scoped_by_key(clock: FakeClock) -> None:
    policy, _ = make_policy(clock)
    policy.decide(scored(0.9), context(clock, gg={"session_id": "s1"}))
    same = context(clock, messages=TOOL_TURN, gg={"session_id": "s1"})
    reused = policy.presolve(same)
    assert reused is not None
    assert reused.tier == "strong"
    other_key = context(clock, messages=TOOL_TURN, gg={"session_id": "s1"})
    other_key.key = make_key(id="other-key")
    assert policy.presolve(other_key) is None


def test_continuation_without_record_or_reuse_disabled_is_scored(clock: FakeClock) -> None:
    policy, _ = make_policy(clock)
    assert policy.presolve(context(clock, messages=TOOL_TURN)) is None
    disabled, _ = make_policy(clock, session={"reuse_tool_continuations": False})
    disabled.decide(scored(0.1), context(clock, messages=TOOL_TURN[:1]))
    assert disabled.presolve(context(clock, messages=TOOL_TURN)) is None


def test_session_record_expires(clock: FakeClock) -> None:
    policy, _ = make_policy(clock)
    policy.decide(scored(0.1), context(clock, messages=TOOL_TURN[:1]))
    clock.advance(3601)
    assert policy.presolve(context(clock, messages=TOOL_TURN)) is None


@pytest.mark.parametrize(("key_ratchet", "global_ratchet"), [(True, False), (False, True)])
def test_ratchet_never_lowers_the_tier(clock: FakeClock, key_ratchet: bool, global_ratchet: bool) -> None:
    policy, _ = make_policy(clock, session={"ratchet": global_ratchet})
    routing = {"ratchet": key_ratchet}
    policy.decide(scored(0.9), context(clock, routing=routing))
    second = policy.decide(scored(0.1), context(clock, routing=routing))
    assert (second.tier, second.reason) == ("strong", "override:ratchet")


def test_without_ratchet_the_tier_can_drop(clock: FakeClock) -> None:
    policy, _ = make_policy(clock)
    policy.decide(scored(0.9), context(clock))
    assert policy.decide(scored(0.1), context(clock)).tier == "weak"


def test_key_without_strong_access_goes_weak_without_scoring(clock: FakeClock) -> None:
    policy, _ = make_policy(clock, FakeCatalog({"strong": [PREMIUM]}))
    decision = policy.presolve(context(clock))
    assert decision is not None
    assert (decision.tier, decision.reason) == ("weak", "override:key_policy")


def test_empty_tier_falls_over_to_the_other_tier(clock: FakeClock) -> None:
    policy, _ = make_policy(clock, FakeCatalog({"weak": [PREMIUM]}))
    decision = policy.decide(scored(0.1), context(clock))
    assert (decision.tier, decision.reason) == ("strong", "override:capability_fallback")


def test_no_deployment_at_all_is_rejected(clock: FakeClock) -> None:
    policy, _ = make_policy(clock, FakeCatalog({"weak": [PREMIUM], "strong": [PREMIUM]}))
    with pytest.raises(PermissionDeniedError):
        policy.decide(scored(0.1), context(clock))


def test_policy_version_tracks_alpha_and_is_deterministic(clock: FakeClock) -> None:
    policy, _ = make_policy(clock)
    a = policy.decide(scored(0.6), context(clock)).policy_version
    b = policy.decide(scored(0.2), context(clock)).policy_version
    c = policy.decide(scored(0.6), context(clock, routing={"threshold": 0.7})).policy_version
    assert a == b
    assert a != c


@pytest.mark.parametrize("value", [0.0, 0.1, 0.49, 0.5, 0.51, 0.9, 1.0])
def test_raising_alpha_never_adds_strong_decisions(clock: FakeClock, value: float) -> None:
    policy, _ = make_policy(clock)
    tiers = [
        policy.decide(scored(value), context(clock, routing={"threshold": alpha})).tier
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    strong = [t == "strong" for t in tiers]
    assert strong == sorted(strong, reverse=True)
