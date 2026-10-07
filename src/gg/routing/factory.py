from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
import structlog

from gg.config.loader import ConfigError, ConfigProblem
from gg.core.clock import Clock
from gg.providers.base import ModelCatalog
from gg.routing.base import NullRoutingHooks, RoutingHooks, RoutingScorer
from gg.routing.config import RoutingConfig
from gg.routing.decorators import CachingScorer, DeadlineScorer, MetricsScorer
from gg.routing.policy import ThresholdPolicy
from gg.routing.probe import RouterProbe
from gg.routing.scorers.jev.client import JevClient
from gg.routing.scorers.jev.questions import QuestionSetError, load_question_set
from gg.routing.scorers.jev.scorer import JevScorer
from gg.routing.scorers.jev.state import StateBuilder
from gg.routing.scorers.null import NullScorer, StaticScorer
from gg.routing.session import SessionStore
from gg.routing.ttl_cache import TTLCache

log = structlog.get_logger("gg.routing")

CONFIG_NAME = "routing.yaml"


@dataclass(frozen=True, slots=True)
class RouterDeps:
    catalog: ModelCatalog
    group_names: Collection[str]
    clock: Clock
    config_dir: Path
    # an http client for a scorer's base url; the composition root owns its lifecycle
    http_client: Callable[[str], httpx2.AsyncClient]
    jev_api_key: str | None = None
    hooks: RoutingHooks = field(default_factory=NullRoutingHooks)


@dataclass(frozen=True, slots=True)
class BuiltScorer:
    scorer: RoutingScorer
    # content hash of files the scorer read (question sets); folded into config_hash
    assets_hash: str = ""


type ScorerBuilder = Callable[[RoutingConfig, RouterDeps], BuiltScorer]

_SCORERS: dict[str, ScorerBuilder] = {}


def register_scorer(name: str, builder: ScorerBuilder) -> None:
    _SCORERS[name] = builder


@dataclass(frozen=True, slots=True)
class Router:
    probe: RouterProbe
    policy: ThresholdPolicy
    scorer: RoutingScorer
    assets_hash: str


def build_router(cfg: RoutingConfig, deps: RouterDeps) -> Router:
    problems = [
        ConfigProblem(CONFIG_NAME, f"profiles.{name}", f"unknown model group '{group}'")
        for name, profile in cfg.profiles.items()
        for group in (profile.weak_group, profile.strong_group)
        if group not in deps.group_names
    ]
    builder = _SCORERS.get(cfg.scorer.type)
    if builder is None:
        known = ", ".join(sorted(_SCORERS))
        problems.append(
            ConfigProblem(CONFIG_NAME, "scorer.type", f"unknown scorer '{cfg.scorer.type}'; known: {known}")
        )
    if problems:
        raise ConfigError(problems)
    assert builder is not None
    built = builder(cfg, deps) if cfg.enabled else BuiltScorer(NullScorer())
    deadline_s = cfg.scorer.deadline_ms / 1000
    scorer = MetricsScorer(DeadlineScorer(built.scorer, deadline_s, deps.clock), deps.hooks, deps.clock)
    sessions = SessionStore(cfg.policy.session.max_entries, cfg.policy.session.ttl_s, deps.clock)
    policy = ThresholdPolicy(cfg, deps.catalog, sessions)
    probe = RouterProbe(deps.catalog, scorer, policy, cfg.headers, hooks=deps.hooks)
    log.info(
        "routing.ready", scorer=scorer.name, scorer_version=scorer.version, threshold=cfg.policy.threshold
    )
    return Router(probe=probe, policy=policy, scorer=scorer, assets_hash=built.assets_hash)


def _build_jev(cfg: RoutingConfig, deps: RouterDeps) -> BuiltScorer:
    jev = cfg.scorer.jev
    try:
        qset = load_question_set(deps.config_dir, jev.question_set, jev.strong_tiers)
    except QuestionSetError as exc:
        raise ConfigError([ConfigProblem(CONFIG_NAME, "scorer.jev.question_set", str(exc))]) from exc
    if not deps.jev_api_key:
        log.warning(
            "routing.jev_disabled", reason="GG_JEV_API_KEY is not set; gg/auto uses the fallback route"
        )
        return BuiltScorer(NullScorer(), qset.content_hash)
    client = JevClient(deps.http_client(_origin(jev.url)), deps.jev_api_key, jev, deps.clock)
    scorer = JevScorer(client, StateBuilder(jev.state), qset, jev, deadline_s=cfg.scorer.deadline_ms / 1000)
    if not cfg.cache.enabled:
        return BuiltScorer(scorer, qset.content_hash)
    store: TTLCache[Mapping[str, Any]] = TTLCache(cfg.cache.lru_entries, cfg.cache.ttl_s, deps.clock)
    return BuiltScorer(CachingScorer(scorer, store), qset.content_hash)


def _origin(url: str) -> str:
    parsed = httpx2.URL(url)
    return f"{parsed.scheme}://{parsed.netloc.decode()}"


register_scorer("jev", _build_jev)
register_scorer("null", lambda cfg, deps: BuiltScorer(NullScorer()))
register_scorer("static", lambda cfg, deps: BuiltScorer(StaticScorer(cfg.scorer.static.score)))
