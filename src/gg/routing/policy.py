from collections.abc import Sequence

import structlog

from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.errors import PermissionDeniedError
from gg.core.jsonutil import canonical_json, sha256_hex
from gg.core.routing_types import PlanEntry, RouteDecision, RoutePlan
from gg.providers.base import ModelCatalog
from gg.routing.base import RoutingScore, Tier
from gg.routing.config import ProfileConfig, RoutingConfig
from gg.routing.session import SessionRecord, SessionStore, is_tool_continuation, session_id

log = structlog.get_logger("gg.routing")

_OTHER: dict[Tier, Tier] = {"weak": "strong", "strong": "weak"}


def resolve_threshold(ctx: RequestContext, cfg: RoutingConfig, profile: ProfileConfig) -> float:
    """request (if the key allows it) > key > profile > global; clamped to the key's bounds"""
    routing = ctx.key.routing
    low, high = routing.threshold_bounds
    ext = ctx.request.gg
    if ext is not None and ext.route_threshold is not None and routing.allow_request_threshold:
        # c2 already rejects out-of-bounds values; the clamp is defensive
        return min(max(ext.route_threshold, low), high)
    if routing.threshold is not None:
        return min(max(routing.threshold, low), high)
    if profile.threshold is not None:
        return min(max(profile.threshold, low), high)
    return min(max(cfg.policy.threshold, low), high)


class ThresholdPolicy:
    """strong iff score >= alpha, then rules that may only raise the tier (routing claim, ratchet)"""

    def __init__(self, cfg: RoutingConfig, catalog: ModelCatalog, sessions: SessionStore) -> None:
        self._cfg = cfg
        self._catalog = catalog
        self._sessions = sessions
        self._base_version = sha256_hex(
            canonical_json(
                {
                    "policy": cfg.policy.model_dump(mode="json"),
                    "profiles": {k: v.model_dump(mode="json") for k, v in cfg.profiles.items()},
                }
            )
        )

    def presolve(self, ctx: RequestContext, /) -> RouteDecision | None:
        profile_name, profile = self._profile(ctx)
        alpha = resolve_threshold(ctx, self._cfg, profile)
        if not self._plan(ctx, profile, "strong") and self._plan(ctx, profile, "weak"):
            return self._decision(ctx, "weak", "override:key_policy", None, alpha, profile_name, profile)
        if self._cfg.policy.session.reuse_tool_continuations and is_tool_continuation(ctx.request):
            record = self._sessions.get(session_id(ctx))
            if record is not None:
                return self._decision(ctx, record.tier, "session", None, alpha, profile_name, profile)
        return None

    def decide(self, score: RoutingScore, ctx: RequestContext, /) -> RouteDecision:
        profile_name, profile = self._profile(ctx)
        alpha = resolve_threshold(ctx, self._cfg, profile)
        tier: Tier
        if score.fallback:
            tier = ctx.key.routing.fallback_route or self._cfg.policy.fallback_route
            reason = f"fallback:{score.fallback_reason}"
        else:
            tier = "strong" if score.score >= alpha else "weak"
            reason = "scored"
            claim = self._cfg.policy.routing_claim
            claimed = score.guard_signals.get("routing_claim_present", 0.0) >= claim.threshold
            # "use the cheap model" text in a prompt must never buy a downgrade
            if tier == "weak" and claim.action == "strong" and claimed:
                tier, reason = "strong", "override:routing_claim"
        sid = session_id(ctx)
        previous = self._sessions.get(sid)
        ratchet = ctx.key.routing.ratchet or self._cfg.policy.session.ratchet
        if ratchet and tier == "weak" and previous is not None and previous.tier == "strong":
            tier, reason = "strong", "override:ratchet"
        decision = self._decision(ctx, tier, reason, score, alpha, profile_name, profile)
        self._sessions.put(sid, SessionRecord(tier, None if score.fallback else score.score))
        return decision

    def _decision(
        self,
        ctx: RequestContext,
        tier: Tier,
        reason: str,
        score: RoutingScore | None,
        alpha: float,
        profile_name: str,
        profile: ProfileConfig,
    ) -> RouteDecision:
        chain = self._plan(ctx, profile, tier)
        if not chain:
            tier, reason = _OTHER[tier], "override:capability_fallback"
            chain = self._plan(ctx, profile, tier)
        alias = ctx.request.model
        if not chain:
            raise PermissionDeniedError(
                f"No deployment of '{alias}' is available to this key.",
                param="model",
                code="model_not_allowed",
            )
        entries = tuple(
            PlanEntry(deployment=d, tier=tier, overrides=self._overrides(ctx, tier, score, d)) for d in chain
        )
        allow_fallback = ctx.request.gg.fallback if ctx.request.gg is not None else True
        return RouteDecision(
            alias=alias,
            tier=tier,
            reason=reason,
            plan=RoutePlan(alias=alias, entries=entries, allow_fallback=allow_fallback),
            score=None if score is None or score.fallback else score.score,
            threshold=alpha,
            policy_version=self._policy_version(profile_name, alpha),
        )

    def _profile(self, ctx: RequestContext) -> tuple[str, ProfileConfig]:
        name = ctx.key.routing.profile
        profile = self._cfg.profiles.get(name)
        if profile is None:
            log.warning("routing.unknown_profile", profile=name, key_id=ctx.key.id)
            name = self._cfg.policy.default_profile
            profile = self._cfg.profiles[name]
        return name, profile

    def _plan(self, ctx: RequestContext, profile: ProfileConfig, tier: Tier) -> Sequence[Deployment]:
        group = profile.strong_group if tier == "strong" else profile.weak_group
        # chain() already swaps in the group's tool-capable chain when the request carries tools
        return [d for d in self._catalog.chain(group, ctx.request) if ctx.key.allows_deployment(d)]

    def _overrides(
        self, ctx: RequestContext, tier: Tier, score: RoutingScore | None, deployment: Deployment
    ) -> dict[str, str]:
        effort = self._cfg.policy.effort
        if effort.mode == "off" or ctx.request.reasoning_effort is not None:
            return {}
        if not deployment.capabilities.effort_levels:
            return {}
        level = effort.strong if tier == "strong" else effort.weak
        if effort.mode == "hinted" and score is not None and score.reasoning_effort_hint is not None:
            level = score.reasoning_effort_hint
        # c3's capability check clamps the level to what the deployment supports
        return {"reasoning_effort": level}

    def _policy_version(self, profile: str, alpha: float) -> str:
        return sha256_hex(f"{self._base_version}:{profile}:{alpha:.4f}")[:16]
