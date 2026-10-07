from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from gg.core.deployment import Deployment


@dataclass(frozen=True, slots=True)
class PlanEntry:
    deployment: Deployment
    tier: str | None = None
    overrides: Mapping[str, Any] = field(default_factory=lambda: {})
    demoted_for: str | None = None
    tier_change: bool = False


@dataclass(frozen=True, slots=True)
class RoutePlan:
    alias: str
    entries: tuple[PlanEntry, ...]
    allow_fallback: bool = True
    allow_tier_change: bool = False


@dataclass(frozen=True, slots=True)
class RouteDecision:
    alias: str | None
    tier: str | None
    reason: str
    plan: RoutePlan
    score: float | None = None
    threshold: float | None = None
    policy_version: str | None = None
