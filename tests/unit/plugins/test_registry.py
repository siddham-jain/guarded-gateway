import pytest
from pydantic import BaseModel, ConfigDict

from gg.plugins.registry import Registry, RegistryError


class ScorerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold: float = 0.5


def make_scorer(config: ScorerConfig, deps: str) -> str:
    return f"{deps}:{config.threshold}"


def test_create_validates_config_with_plugin_schema() -> None:
    registry: Registry[str, str] = Registry("scorer")
    registry.register("jev", make_scorer, config_model=ScorerConfig)
    assert registry.create("jev", {"threshold": 0.7}, "deps") == "deps:0.7"
    with pytest.raises(RegistryError, match="invalid config"):
        registry.create("jev", {"treshold": 0.7}, "deps")


def test_unknown_and_duplicate_names() -> None:
    registry: Registry[str, str] = Registry("scorer")
    registry.register("jev", make_scorer, config_model=ScorerConfig)
    with pytest.raises(RegistryError, match="already registered"):
        registry.register("jev", make_scorer, config_model=ScorerConfig)
    with pytest.raises(RegistryError, match="available: jev"):
        registry.create("jevv", {}, "deps")
    assert "jev" in registry
    assert registry.names() == ("jev",)
