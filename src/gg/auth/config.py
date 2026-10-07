from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import AwareDatetime, StringConstraints, ValidationError, model_validator

from gg.config.loader import ConfigError, ConfigProblem, load_file
from gg.core.keypolicy import (
    BudgetPolicy,
    CachePolicy,
    GuardrailOverrides,
    KeyFlags,
    KeyPolicy,
    RateLimitPolicy,
    RequestLimits,
    RoutingOverrides,
)
from gg.core.schema import StrictModel

type KeyHash = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]

IDENTITY_FIELDS = frozenset({"id", "name", "prefix", "created_at", "hash"})


class KeyDefaults(StrictModel):
    """every KeyPolicy field except identity; deep-merged into each key entry"""

    status: Literal["active", "disabled"] = "active"
    expires_at: AwareDatetime | None = None
    tags: frozenset[str] = frozenset()
    allowed_models: tuple[str, ...] = ("gg/*",)
    allowed_providers: tuple[str, ...] | None = None
    rate_limits: RateLimitPolicy = RateLimitPolicy()
    budget: BudgetPolicy = BudgetPolicy()
    routing: RoutingOverrides = RoutingOverrides()
    cache: CachePolicy = CachePolicy()
    guardrails: GuardrailOverrides = GuardrailOverrides()
    limits: RequestLimits = RequestLimits()
    flags: KeyFlags = KeyFlags()


class KeyEntry(KeyPolicy):
    hash: KeyHash
    prefix: Annotated[str, StringConstraints(pattern=r"^gg-(live|test)-[0-9A-Za-z]{4}$")]

    def policy(self) -> KeyPolicy:
        return KeyPolicy.model_validate(self.model_dump(exclude={"hash"}))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """mappings merge recursively; lists and scalars from override replace"""
    merged = dict(base)
    for name, value in override.items():
        current = merged.get(name)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[name] = deep_merge(current, value)  # pyright: ignore[reportUnknownArgumentType]
        else:
            merged[name] = value
    return merged


class KeysConfig(StrictModel):
    version: Literal[1] = 1
    defaults: KeyDefaults = KeyDefaults()
    keys: tuple[KeyEntry, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _merge_defaults(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        defaults: Any = data.get("defaults") or {}  # pyright: ignore[reportUnknownMemberType]
        keys: Any = data.get("keys")  # pyright: ignore[reportUnknownMemberType]
        if not isinstance(defaults, Mapping) or not isinstance(keys, list):
            return data
        merged = [
            deep_merge(defaults, entry) if isinstance(entry, Mapping) else entry  # pyright: ignore[reportUnknownArgumentType]
            for entry in keys  # pyright: ignore[reportUnknownVariableType]
        ]
        return {**data, "keys": merged}

    @model_validator(mode="after")
    def _check_unique(self) -> Self:
        seen_ids: set[str] = set()
        seen_hashes: dict[str, str] = {}
        for entry in self.keys:
            if entry.id in seen_ids:
                raise ValueError(f"duplicate key id '{entry.id}'")
            seen_ids.add(entry.id)
            if entry.hash in seen_hashes:
                raise ValueError(f"keys '{seen_hashes[entry.hash]}' and '{entry.id}' share a hash")
            seen_hashes[entry.hash] = entry.id
        return self

    def policies_by_hash(self) -> dict[str, KeyPolicy]:
        return {entry.hash: entry.policy() for entry in self.keys}


def load_keys(path: Path) -> KeysConfig:
    if not path.exists():
        hint = "create one with `gg keys create --id ID --name NAME` and add the printed entry"
        raise ConfigError([ConfigProblem(str(path), "", f"keys file not found; {hint}")])
    try:
        return load_file(path, KeysConfig)
    except ValidationError as exc:
        raise ConfigError(
            [
                ConfigProblem(str(path), ".".join(str(p) for p in err["loc"]), err["msg"])
                for err in exc.errors(include_input=False)
            ]
        ) from None
