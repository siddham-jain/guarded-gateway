from pathlib import Path
from typing import Any

import httpx2
import pytest

from gg.config.loader import ConfigError, load_file
from gg.core.clock import FakeClock
from gg.routing.config import RoutingConfig
from gg.routing.factory import RouterDeps, build_router
from tests.unit.routing.support import CONFIG_DIR, FakeCatalog, routing_request, user


def deps(clock: FakeClock, *, key: str | None = "jev-key", config_dir: Path = CONFIG_DIR) -> RouterDeps:
    return RouterDeps(
        catalog=FakeCatalog(),
        group_names=("weak", "strong", "weak-anthropic", "strong-anthropic"),
        clock=clock,
        config_dir=config_dir,
        http_client=lambda base_url: httpx2.AsyncClient(base_url=base_url),
        jev_api_key=key,
    )


def config(**overrides: Any) -> RoutingConfig:
    return RoutingConfig.model_validate(overrides)


def test_shipped_routing_yaml_is_valid() -> None:
    cfg = load_file(CONFIG_DIR / "routing.yaml", RoutingConfig)
    assert cfg.scorer.type == "jev"
    assert cfg.scorer.jev.model == "jev-1.13.0"
    assert cfg.policy.threshold == 0.25
    assert cfg.policy.fallback_route == "strong"
    assert (cfg.scorer.deadline_ms, cfg.scorer.jev.attempt_timeout_ms) == (600, 400)


def test_jev_stack_and_version(clock: FakeClock) -> None:
    router = build_router(config(), deps(clock))
    assert router.scorer.name == "jev"
    assert router.scorer.version == "jev-1.13.0:qset-v1:state-v1:strong_helps"
    assert len(router.assets_hash) == 64


async def test_missing_key_or_disabled_routing_uses_null_scorer(clock: FakeClock) -> None:
    for router in (
        build_router(config(), deps(clock, key=None)),
        build_router(config(enabled=False), deps(clock)),
    ):
        assert router.scorer.name == "null"
        score = await router.scorer.score(routing_request(user("hi")))
        assert score.fallback_reason == "disabled"


async def test_static_scorer_is_registered(clock: FakeClock) -> None:
    router = build_router(config(scorer={"type": "static", "static": {"score": 0.7}}), deps(clock))
    assert (await router.scorer.score(routing_request(user("hi")))).score == 0.7


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"scorer": {"type": "local_classifier"}}, "unknown scorer 'local_classifier'"),
        (
            {"profiles": {"default": {"weak_group": "weak", "strong_group": "nope"}}},
            "unknown model group 'nope'",
        ),
    ],
)
def test_startup_validation(clock: FakeClock, overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        build_router(config(**overrides), deps(clock))


def test_missing_or_mismatched_question_set(clock: FakeClock, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="question set file not found"):
        build_router(config(), deps(clock, config_dir=tmp_path))
    with pytest.raises(ConfigError, match="not 'tier' options"):
        build_router(config(scorer={"jev": {"strong_tiers": ["genius"]}}), deps(clock))


@pytest.mark.parametrize(
    "overrides",
    [
        {"policy": {"threshold": 1.5}},
        {"policy": {"default_profile": "missing"}},
        {"scorer": {"jev": {"model": "jev-latest"}}},
        {"scorer": {"jev": {"max_attempts": 3}}},
        {"scorer": {"jev": {"state": {"request_head_chars": 3500}}}},
        {"typo": True},
    ],
)
def test_invalid_config_is_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="validation error"):
        config(**overrides)
