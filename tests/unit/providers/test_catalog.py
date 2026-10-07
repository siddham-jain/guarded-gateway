from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest

from gg.config.loader import ConfigBundle, ConfigError
from gg.config.settings import Settings
from gg.providers.base import AliasRef, DeploymentRef, ModelCatalog
from gg.providers.catalog.catalog import Catalog
from gg.providers.catalog.loader import build_catalog, check_models_config, validate_models
from gg.providers.catalog.schema import ModelsConfig
from tests.conftest import make_request
from tests.unit.providers.support import models_config

TOOLS = [{"type": "function", "function": {"name": "f"}}]
TODAY = date(2026, 10, 5)


def settings(**providers: dict[str, str]) -> Settings:
    return Settings(_env_file=None, providers=providers)  # pyright: ignore[reportCallIssue]


def catalog(profile: str = "prod", **providers: dict[str, str]) -> Catalog:
    return build_catalog(models_config(), settings(**providers), profile=profile, today=TODAY)


def ids(deps: list[Any]) -> list[str]:
    return [d.id for d in deps]


def minimal(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "providers": {"p": {"type": "openai_compat", "base_url": "https://p.test/v1"}},
        "deployments": [
            {"id": "p/m", "pricing": [{"effective_from": "2026-01-01", "input": 1, "output": 2}]}
        ],
        "groups": {"g": {"chain": ["p/m"]}},
        "aliases": {"gg/g": {"group": "g"}},
    }
    data.update(overrides)
    return data


def problems(data: dict[str, Any]) -> list[str]:
    return [f"{p.path}: {p.message}" for p in validate_models(ModelsConfig.model_validate(data), today=TODAY)]


def test_repo_catalog_is_valid_in_every_profile() -> None:
    config = models_config()
    assert validate_models(config, today=TODAY) == []
    for profile in config.profiles:
        assert isinstance(build_catalog(config, settings(), profile=profile, today=TODAY), Catalog)


def test_catalog_implements_protocol_and_hash_is_stable() -> None:
    cat: ModelCatalog = catalog()
    assert cat.hash == catalog().hash
    assert cat.hash != catalog("ci").hash


def test_providers_enabled_only_with_credentials() -> None:
    assert set(catalog().providers) == {"mock"}
    cat = catalog(openai={"api_key": "k"}, ollama={"base_url": "http://localhost:11434"})
    assert set(cat.providers) == {"mock", "openai", "ollama"}
    assert cat.providers["ollama"].base_url == "http://localhost:11434/v1"
    assert cat.providers["openai"].api_key is not None
    assert catalog(anthropic={"api_key": "k"}).providers["anthropic"].auth.header == "x-api-key"


def test_ci_profile_is_mock_only() -> None:
    cat = catalog("ci", openai={"api_key": "k"})
    assert set(cat.providers) == {"mock"}
    assert ids(cat.chain("weak", make_request())) == ["mock/echo"]
    assert ids(cat.chain("strong", make_request(tools=TOOLS))) == ["mock/echo"]


def test_resolution_rules() -> None:
    cat = catalog(openai={"api_key": "k"}, groq={"api_key": "k"})
    assert cat.resolve("gg/auto") == AliasRef("gg/auto", "router")
    assert cat.resolve("gg/weak") == AliasRef("gg/weak", "group")
    assert cat.resolve("openai/gpt-oss-120b") == AliasRef("openai/gpt-oss-120b", "group")
    luna = cat.resolve("openai/gpt-6-luna")
    assert isinstance(luna, DeploymentRef)
    assert luna.deployment.upstream_model == "gpt-6-luna"
    assert cat.resolve("gpt-6-luna") == luna
    assert cat.resolve("anthropic/claude-haiku-4-5") is None
    assert cat.resolve("nope") is None
    assert cat.resolve("openai/gpt-oss-20b") == AliasRef("openai/gpt-oss-20b", "group")
    assert cat.resolve("openai/gpt-oss-120b") is not None
    assert catalog().resolve("openai/gpt-oss-120b") is None


def test_chain_is_tool_aware_and_skips_disabled() -> None:
    cat = catalog(openai={"api_key": "k"}, groq={"api_key": "k"}, deepinfra={"api_key": "k"})
    assert ids(cat.chain("strong", make_request())) == ["openai/gpt-6.1-sol"]
    assert ids(cat.chain("strong", make_request(tools=TOOLS))) == ["openai/gpt-6-sol"]
    assert ids(cat.chain("gg/weak", make_request())) == [
        "openai/gpt-6-luna",
        "deepinfra/openai/gpt-oss-120b",
        "groq/openai/gpt-oss-120b",
    ]
    assert cat.chain("missing", make_request()) == []


def test_canonical_model_hosts_ordered_by_price() -> None:
    cat = catalog(**{p: {"api_key": "k"} for p in ("groq", "deepinfra", "fireworks", "together", "cerebras")})
    hosts = ids(cat.chain("openai/gpt-oss-120b", make_request()))
    assert hosts[0] == "deepinfra/openai/gpt-oss-120b"
    assert hosts[-1] == "cerebras/openai/gpt-oss-120b"
    assert len(hosts) == 5
    groq = cat.get("groq/openai/gpt-oss-120b")
    assert groq.canonical_model == "openai/gpt-oss-120b"
    assert not groq.capabilities.json_schema
    assert not groq.pricing.billed
    assert "quant:fp8" in cat.get("cerebras/openai/gpt-oss-120b").tags
    assert cat.resolve("gpt-oss-120b") is None


def test_deployments_for_alias_covers_every_chain() -> None:
    cat = catalog(openai={"api_key": "k"})
    assert set(ids(cat.deployments_for("gg/auto"))) == {
        "openai/gpt-6-luna",
        "openai/gpt-6.1-sol",
        "openai/gpt-6-sol",
    }
    assert ids(cat.deployments_for("openai/gpt-6-luna")) == ["openai/gpt-6-luna"]
    assert cat.deployments_for("nope") == []


def test_price_periods_switch_at_boundary() -> None:
    cat = build_catalog(models_config(), settings(), today=TODAY)
    before = cat.price_for("gemini/gemini-3.8-flash", datetime(2026, 12, 31, 23, 59, tzinfo=UTC))
    after = cat.price_for("gemini/gemini-3.8-flash", datetime(2027, 1, 1, tzinfo=UTC))
    assert before is not None
    assert after is not None
    assert (before.input, after.input) == (Decimal("0.75"), Decimal("1.5"))
    assert cat.price_for("openai/gpt-6-luna", datetime(2020, 1, 1, tzinfo=UTC)) is None


def test_timeouts_merge_defaults_provider_and_deployment() -> None:
    cat = catalog(openai={"api_key": "k"}, ollama={"base_url": "http://o"})
    sol = cat.get("openai/gpt-6.1-sol").timeouts
    assert (sol.ttft_s, sol.inter_chunk_s, sol.total_s) == (60, 60, 300)
    local = cat.get("ollama/llama3.2:1b").timeouts
    assert (local.connect_s, local.ttft_s, local.inter_chunk_s) == (1, 15, 20)


def test_list_public_rows() -> None:
    cat = catalog(openai={"api_key": "k"})
    rows = {r.id: r for r in cat.list_public(lambda m: m != "gg/premium")}
    assert "gg/auto" in rows
    assert "gg/premium" not in rows
    assert rows["openai/gpt-6-luna"].owned_by == "openai"
    assert rows["openai/gpt-6-luna"].context_window == 1050000
    assert "anthropic/claude-haiku-4-5" not in rows


def test_compatible_prefilter() -> None:
    cat = catalog(openai={"api_key": "k"})
    deps = [cat.get("openai/gpt-6.1-sol"), cat.get("openai/gpt-6-sol")]
    assert ids(cat.compatible(make_request(tools=TOOLS), deps)) == ["openai/gpt-6-sol"]


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"groups": {"g": {"chain": ["p/missing"]}}}, "unknown deployment or model 'p/missing'"),
        ({"aliases": {"gg/x": {"group": "nope"}}}, "unknown group 'nope'"),
        ({"aliases": {"p/m": {"group": "g"}}}, "alias shadows"),
        (
            {
                "deployments": [
                    {"id": "q/m", "pricing": [{"effective_from": "2026-01-01", "input": 1, "output": 1}]}
                ]
            },
            "unknown provider 'q'",
        ),
        ({"deployments": [{"id": "p/m"}]}, "at least one price period"),
        (
            {
                "deployments": [
                    {
                        "id": "p/m",
                        "pricing": [
                            {"effective_from": "2026-02-01", "input": 1, "output": 1},
                            {"effective_from": "2026-01-01", "input": 1, "output": 1},
                        ],
                    }
                ]
            },
            "strictly increase",
        ),
        (
            {
                "deployments": [
                    {
                        "id": "p/m",
                        "capabilities": {"effort_levels": ["low"], "effort_clamp": {"none": "high"}},
                        "pricing": [{"effective_from": "2026-01-01", "input": 1, "output": 1}],
                    }
                ]
            },
            "'high' not in effort_levels",
        ),
        (
            {
                "deployments": [
                    {
                        "id": "p/m",
                        "shutdown_date": "2026-01-01",
                        "pricing": [{"effective_from": "2025-01-01", "input": 1, "output": 1}],
                    }
                ]
            },
            "was shut down",
        ),
        (
            {
                "groups": {"g": {"chain": ["p/m"], "chain_with_tools": ["p/n"]}},
                "deployments": [
                    {"id": "p/m", "pricing": [{"effective_from": "2026-01-01", "input": 1, "output": 1}]},
                    {
                        "id": "p/n",
                        "capabilities": {"tools": False},
                        "pricing": [{"effective_from": "2026-01-01", "input": 1, "output": 1}],
                    },
                ],
            },
            "does not support tools",
        ),
        (
            {
                "providers": {
                    "p": {"type": "openai_compat", "quirks": {"request": {"max_tokens_param": "nope"}}}
                }
            },
            "providers.p.quirks.request.max_tokens_param",
        ),
        (
            {"providers": {"p": {"type": "openai_compat", "quirks": {"extends": "ghost"}}}},
            "unknown quirk profile",
        ),
        (
            {
                "quirk_profiles": {"a": {"extends": "b"}, "b": {"extends": "a"}},
                "providers": {"p": {"type": "openai_compat", "quirks": {"extends": "a"}}},
            },
            "cycle",
        ),
        ({"profiles": {"x": {"groups": {"g": {"chain": ["p/zzz"]}}}}}, "profiles.x.groups.g.chain.0"),
        ({"profiles": {"x": {"only_providers": ["ghost"]}}}, "unknown provider 'ghost'"),
        (
            {
                "models": {
                    "p/m": {
                        "hosts": [
                            {
                                "provider": "p",
                                "upstream_model": "m",
                                "pricing": [{"effective_from": "2026-01-01", "input": 1, "output": 1}],
                            }
                        ]
                    }
                }
            },
            "collides",
        ),
    ],
)
def test_invalid_configs_report_dotted_paths(overrides: dict[str, Any], needle: str) -> None:
    found = problems(minimal(**overrides))
    assert any(needle in p for p in found), found


def test_minimal_config_is_valid_and_bundle_validator() -> None:
    config = ModelsConfig.model_validate(minimal())
    assert validate_models(config, today=TODAY) == []
    assert list(check_models_config(ConfigBundle({ModelsConfig: config}, {}))) == []
    assert list(check_models_config(ConfigBundle({}, {}))) == []


def test_build_raises_for_unknown_profile_and_forbidden_url() -> None:
    with pytest.raises(ConfigError, match="unknown profile"):
        build_catalog(models_config(), settings(), profile="nope", today=TODAY)
    with pytest.raises(ConfigError, match="forbidden"):
        build_catalog(
            models_config(),
            settings(zai={"api_key": "k", "base_url": "https://api.z.ai/api/coding/paas/v4"}),
            today=TODAY,
        )
