from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from itertools import pairwise
from typing import Any

from pydantic import BaseModel, ValidationError

from gg.config.hashing import section_hash
from gg.config.loader import ConfigBundle, ConfigError, ConfigProblem
from gg.config.settings import Settings
from gg.core.deployment import EFFORT_ORDER, Deployment, PriceSchedule, Timeouts
from gg.core.jsonutil import sha256_hex
from gg.providers.anthropic.quirks import AnthropicQuirks
from gg.providers.catalog.capabilities import CapabilityChecker
from gg.providers.catalog.catalog import Catalog, Group
from gg.providers.catalog.schema import (
    AliasSpec,
    CapabilitiesSpec,
    DeploymentSpec,
    GroupSpec,
    HostSpec,
    ModelsConfig,
)
from gg.providers.gemini.quirks import GeminiQuirks
from gg.providers.openai_compat.quirks import QuirkProfile, deep_merge
from gg.providers.runtime import ProviderRuntime
from gg.providers.usage import estimate_prompt_tokens

FILE = "models.yaml"
DEFAULT_TYPES: frozenset[str] = frozenset({"openai_compat", "mock", "gemini", "anthropic"})
DEFAULT_QUIRK_MODELS: Mapping[str, type[BaseModel]] = {
    "openai_compat": QuirkProfile,
    "gemini": GeminiQuirks,
    "anthropic": AnthropicQuirks,
}
MOCK_BASE_URL = "mock://local"


def resolve_quirks(config: ModelsConfig, raw: Mapping[str, Any]) -> dict[str, Any]:
    """expands `extends: <quirk profile>` chains (deep merge, child wins)"""
    seen: list[str] = []
    layers: list[Mapping[str, Any]] = [raw]
    parent = raw.get("extends")
    while isinstance(parent, str):
        if parent in seen:
            raise ValueError(f"quirk profile cycle: {' -> '.join([*seen, parent])}")
        if parent not in config.quirk_profiles:
            raise ValueError(f"unknown quirk profile '{parent}'")
        seen.append(parent)
        profile = config.quirk_profiles[parent]
        layers.append(profile)
        parent = profile.get("extends")
    merged: dict[str, Any] = {}
    for layer in reversed(layers):
        merged = deep_merge(merged, {k: v for k, v in layer.items() if k != "extends"})
    return merged


def host_deployment_id(canonical: str, host: HostSpec) -> str:
    return host.id or f"{host.provider}/{canonical}"


def _blended_price(spec: DeploymentSpec, today: date) -> Decimal:
    current = [p for p in spec.pricing if p.effective_from <= today]
    period = max(current, key=lambda p: p.effective_from) if current else None
    return period.input + period.output if period else Decimal("Infinity")


@dataclass(frozen=True, slots=True)
class Expanded:
    deployments: list[DeploymentSpec]
    canonical: dict[str, list[str]]
    quantization: dict[str, str]


def expand(config: ModelsConfig, today: date, problems: list[ConfigProblem] | None = None) -> Expanded:
    """turns canonical models x hosts into ordinary deployments; host order follows the model's selection"""
    out = list(config.deployments)
    canonical: dict[str, list[str]] = {}
    quant: dict[str, str] = {}
    for model_id, model in config.models.items():
        members: list[DeploymentSpec] = []
        for i, host in enumerate(model.hosts):
            if (
                model.allow_quantizations is not None
                and (host.quantization or "unknown") not in model.allow_quantizations
            ):
                continue
            try:
                caps = CapabilitiesSpec.model_validate(
                    {**model.capabilities.model_dump(), **host.capabilities}
                )
            except ValidationError as exc:
                if problems is not None:
                    problems.extend(
                        ConfigProblem(FILE, f"models.{model_id}.hosts.{i}.capabilities", e["msg"])
                        for e in exc.errors(include_input=False)
                    )
                continue
            dep_id = host_deployment_id(model_id, host)
            members.append(
                DeploymentSpec(
                    id=dep_id,
                    provider=host.provider,
                    upstream_model=host.upstream_model,
                    capabilities=caps,
                    defaults={**model.defaults, **host.defaults},
                    pricing=host.pricing,
                    status=host.status,
                    billing=host.billing,
                    reliability=host.reliability,
                    quirks=host.quirks,
                    data_policy=host.data_policy,
                    public=model.public,
                )
            )
            if host.quantization:
                quant[dep_id] = host.quantization
        if model.selection == "price":
            members.sort(key=lambda d: _blended_price(d, today))
        canonical[model_id] = [d.id for d in members]
        out.extend(members)
    return Expanded(out, canonical, quant)


def _effort_problems(path: str, spec: DeploymentSpec) -> list[ConfigProblem]:
    caps = spec.capabilities
    levels = set(caps.effort_levels)
    problems: list[ConfigProblem] = []
    for src, dst in caps.effort_clamp.items():
        if dst not in levels:
            problems.append(
                ConfigProblem(
                    FILE, f"{path}.capabilities.effort_clamp.{src}", f"'{dst}' not in effort_levels"
                )
            )
    default = spec.defaults.get("reasoning_effort")
    if default is not None and levels and default not in levels:
        problems.append(
            ConfigProblem(FILE, f"{path}.defaults.reasoning_effort", f"'{default}' not in effort_levels")
        )
    if default is not None and default not in EFFORT_ORDER:
        problems.append(
            ConfigProblem(FILE, f"{path}.defaults.reasoning_effort", f"unknown effort '{default}'")
        )
    return problems


def _chain_problems(
    path: str, group: GroupSpec, specs: Mapping[str, DeploymentSpec], canonical: Mapping[str, list[str]]
) -> list[ConfigProblem]:
    problems: list[ConfigProblem] = []
    for field, members in (("chain", group.chain), ("chain_with_tools", group.chain_with_tools or [])):
        for i, member in enumerate(members):
            if member not in specs and member not in canonical:
                problems.append(
                    ConfigProblem(FILE, f"{path}.{field}.{i}", f"unknown deployment or model '{member}'")
                )
            elif field == "chain_with_tools":
                ids = canonical.get(member, [member])
                if not all(specs[d].capabilities.tools for d in ids if d in specs):
                    problems.append(
                        ConfigProblem(FILE, f"{path}.{field}.{i}", f"'{member}' does not support tools")
                    )
    return problems


def _alias_problems(
    path: str, alias: AliasSpec, groups: Mapping[str, GroupSpec], canonical: Mapping[str, list[str]]
) -> list[ConfigProblem]:
    targets = alias.groups if alias.router else [alias.group or ""]
    return [
        ConfigProblem(FILE, path, f"unknown group '{t}'")
        for t in targets
        if t not in groups and t not in canonical
    ]


def validate_models(
    config: ModelsConfig,
    *,
    quirk_models: Mapping[str, type[BaseModel]] = DEFAULT_QUIRK_MODELS,
    today: date | None = None,
) -> list[ConfigProblem]:
    today = today or datetime.now(UTC).date()
    problems: list[ConfigProblem] = []
    for name, provider in config.providers.items():
        try:
            quirks = resolve_quirks(config, provider.quirks)
        except ValueError as exc:
            problems.append(ConfigProblem(FILE, f"providers.{name}.quirks", str(exc)))
            continue
        model = quirk_models.get(provider.type)
        if model is not None:
            try:
                model.model_validate(quirks)
            except ValidationError as exc:
                problems.extend(
                    ConfigProblem(FILE, f"providers.{name}.quirks." + ".".join(map(str, e["loc"])), e["msg"])
                    for e in exc.errors(include_input=False)
                )
    expanded = expand(config, today, problems)
    specs: dict[str, DeploymentSpec] = {}
    for i, spec in enumerate(expanded.deployments):
        path = f"deployments.{i}" if i < len(config.deployments) else f"models[{spec.id}]"
        if spec.id in specs:
            problems.append(ConfigProblem(FILE, path, f"duplicate deployment id '{spec.id}'"))
        specs[spec.id] = spec
        if spec.provider_name not in config.providers:
            problems.append(ConfigProblem(FILE, path, f"unknown provider '{spec.provider_name}'"))
        if not spec.pricing:
            problems.append(
                ConfigProblem(FILE, f"{path}.pricing", "every deployment needs at least one price period")
            )
        dates = [p.effective_from for p in spec.pricing]
        if any(b <= a for a, b in pairwise(dates)):
            problems.append(
                ConfigProblem(FILE, f"{path}.pricing", "effective_from dates must strictly increase")
            )
        if spec.shutdown_date is not None and spec.shutdown_date < today and spec.status != "disabled":
            problems.append(
                ConfigProblem(
                    FILE, f"{path}.shutdown_date", f"'{spec.id}' was shut down {spec.shutdown_date}"
                )
            )
        problems.extend(_effort_problems(path, spec))
    problems.extend(
        ConfigProblem(FILE, f"models.{model_id}", "canonical model id collides with a deployment id")
        for model_id in config.models
        if model_id in specs
    )
    for name, group in config.groups.items():
        problems.extend(_chain_problems(f"groups.{name}", group, specs, expanded.canonical))
    for name, alias in config.aliases.items():
        if name in specs or name in config.models:
            problems.append(ConfigProblem(FILE, f"aliases.{name}", "alias shadows a deployment or model id"))
        problems.extend(_alias_problems(f"aliases.{name}", alias, config.groups, expanded.canonical))
    for pname, profile in config.profiles.items():
        groups = {**config.groups, **profile.groups}
        for name, group in profile.groups.items():
            problems.extend(
                _chain_problems(f"profiles.{pname}.groups.{name}", group, specs, expanded.canonical)
            )
        for name, alias in {**config.aliases, **profile.aliases}.items():
            problems.extend(
                _alias_problems(f"profiles.{pname}.aliases.{name}", alias, groups, expanded.canonical)
            )
        for i, provider in enumerate([*(profile.only_providers or []), *profile.disable_providers]):
            if provider not in config.providers:
                problems.append(
                    ConfigProblem(FILE, f"profiles.{pname}.providers.{i}", f"unknown provider '{provider}'")
                )
    return problems


def check_models_config(bundle: ConfigBundle) -> Iterable[ConfigProblem]:
    """cross validator for the composition root's load_bundle()"""
    config = bundle.optional(ModelsConfig)
    return validate_models(config) if config is not None else []


def _timeouts(config: ModelsConfig, provider: str, spec: DeploymentSpec) -> Timeouts:
    rel = config.reliability
    layers: list[Any] = [
        (rel.get("defaults") or {}),
        ((rel.get("providers") or {}).get(provider) or {}),
        spec.reliability,
    ]
    merged: dict[str, Any] = {}
    deadline: dict[str, Any] = {}
    for layer in layers:
        if isinstance(layer, dict):
            merged.update(layer.get("timeouts") or {})  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            deadline.update(layer.get("deadline_s") or {})  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    base = Timeouts()
    return Timeouts(
        connect_s=float(merged.get("connect_s", base.connect_s)),
        ttft_s=float(merged.get("ttft_s", base.ttft_s)),
        inter_chunk_s=float(merged.get("inter_chunk_s", merged.get("read_s", base.inter_chunk_s))),
        total_s=float(deadline.get("stream", base.total_s)),
    )


def _runtimes(
    config: ModelsConfig,
    settings: Settings,
    only: list[str] | None,
    disabled: list[str],
    available: Collection[str],
) -> tuple[dict[str, ProviderRuntime], list[ConfigProblem]]:
    runtimes: dict[str, ProviderRuntime] = {}
    problems: list[ConfigProblem] = []
    for name, spec in config.providers.items():
        if (
            not spec.enabled
            or spec.type not in available
            or name in disabled
            or (only is not None and name not in only)
        ):
            continue
        if spec.type == "mock" and settings.env == "prod":
            continue
        creds = settings.provider(name)
        base = creds.base_url or spec.base_url or (MOCK_BASE_URL if spec.type == "mock" else None)
        if base is None or (spec.requires_key and creds.api_key is None):
            continue
        base_url = base.rstrip("/") + spec.base_path
        quirks = resolve_quirks(config, spec.quirks)
        forbidden = [p for p in quirks.get("forbid_base_url_patterns", ()) if p in base_url]
        if forbidden:
            # subscription/coding-plan endpoints forbid proxying in their terms
            problems.append(
                ConfigProblem(FILE, f"providers.{name}.base_url", f"base url matches forbidden {forbidden}")
            )
            continue
        runtimes[name] = ProviderRuntime(
            name=name,
            type=spec.type,
            base_url=base_url,
            api_key=creds.api_key,
            auth=spec.auth,
            quirks=quirks,
            options=spec.options,
            max_in_flight=spec.max_in_flight,
            data_policy=spec.data_policy,
        )
    return runtimes, problems


def build_catalog(
    config: ModelsConfig,
    settings: Settings,
    *,
    profile: str | None = None,
    available_types: Collection[str] = DEFAULT_TYPES,
    quirk_models: Mapping[str, type[BaseModel]] = DEFAULT_QUIRK_MODELS,
    checker: CapabilityChecker | None = None,
    today: date | None = None,
) -> Catalog:
    """validated config + settings (credentials, profile) -> ModelCatalog; raises ConfigError"""
    today = today or datetime.now(UTC).date()
    problems = validate_models(config, quirk_models=quirk_models, today=today)
    profile_name = profile or settings.model_profile
    overlay = config.profiles.get(profile_name)
    if overlay is None and config.profiles and profile_name != "prod":
        problems.append(ConfigProblem(FILE, "profiles", f"unknown profile '{profile_name}'"))
    if problems:
        raise ConfigError(problems)
    groups_spec = {**config.groups, **(overlay.groups if overlay else {})}
    aliases = {**config.aliases, **(overlay.aliases if overlay else {})}
    runtimes, problems = _runtimes(
        config,
        settings,
        overlay.only_providers if overlay else None,
        overlay.disable_providers if overlay else [],
        available_types,
    )
    if problems:
        raise ConfigError(problems)

    expanded = expand(config, today)
    deployments: dict[str, Deployment] = {}
    for spec in expanded.deployments:
        provider = spec.provider_name
        billing = spec.billing or config.providers[provider].billing
        tags = set(spec.tags)
        if spec.id in expanded.quantization:
            tags.add("quant:" + expanded.quantization[spec.id])
        deployments[spec.id] = Deployment(
            id=spec.id,
            provider=provider,
            upstream_model=spec.upstream,
            capabilities=spec.capabilities.build(),
            pricing=PriceSchedule(tuple(p.build() for p in spec.pricing), billed=billing == "paid"),
            defaults=dict(spec.defaults),
            tier=spec.tier,
            timeouts=_timeouts(config, provider, spec),
            tags=frozenset(tags),
            status=spec.status,
            shutdown_date=spec.shutdown_date,
            enabled=provider in runtimes and spec.status != "disabled",
            canonical_model=next((m for m, ids in expanded.canonical.items() if spec.id in ids), None),
            quirks=dict(spec.quirks),
        )
    groups: dict[str, Group] = {}
    for name, ids in expanded.canonical.items():
        groups[name] = Group(tuple(ids), None)
    for name, group in groups_spec.items():
        groups[name] = Group(
            tuple(_flatten(group.chain, expanded.canonical)),
            tuple(_flatten(group.chain_with_tools, expanded.canonical)) if group.chain_with_tools else None,
        )
    public = {s.id for s in expanded.deployments if s.public}
    return Catalog(
        deployments=deployments,
        groups=groups,
        aliases=aliases,
        canonical=frozenset(expanded.canonical),
        public=frozenset(public),
        providers=runtimes,
        bare_names=config.bare_names,
        checker=checker or CapabilityChecker(estimate_prompt_tokens),
        catalog_hash=sha256_hex(f"{section_hash(config)}:{profile_name}")[:16],
    )


def _flatten(members: list[str], canonical: Mapping[str, list[str]]) -> list[str]:
    out: list[str] = []
    for member in members:
        for dep_id in canonical.get(member, [member]):
            if dep_id not in out:
                out.append(dep_id)
    return out
