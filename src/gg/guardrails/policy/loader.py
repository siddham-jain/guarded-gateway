"""loads config/policies/*.yaml at startup and resolves the effective policy per key and request"""

import copy
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError
from yaml import YAMLError

from gg.config.loader import ConfigError, ConfigProblem
from gg.config.yaml_loader import load_yaml_file
from gg.core.errors import InternalError
from gg.core.jsonutil import sha256_hex
from gg.core.keypolicy import KeyPolicy
from gg.core.schema import ChatRequest, GGGuardrailsExt
from gg.guardrails.base import Mode
from gg.guardrails.errors import GuardrailOverrideRejectedError
from gg.guardrails.policy.effective import EffectivePolicy, PolicyError, build_effective
from gg.guardrails.policy.schema import COMMON_FIELDS, GuardEntry, PolicyDoc
from gg.guardrails.registry import GuardDeps, GuardRegistry

_CACHE_SIZE = 256


@dataclass(frozen=True, slots=True)
class LoadedPolicy:
    doc: PolicyDoc
    raw: Mapping[str, Any]
    file: str
    file_hash: str


def _guard_index(guards: list[Any], name: str, path: str) -> int:
    for i, entry in enumerate(guards):
        if isinstance(entry, dict) and (entry.get("name") or entry.get("guard")) == name:  # pyright: ignore[reportUnknownMemberType]
            return i
    raise PolicyError(f"{path}: no guard named '{name}'")


def apply_patch(raw: dict[str, Any], path: str, value: Any) -> None:
    """sets a dotted path in the raw policy dict; the patched dict is re-validated by the caller"""
    parts = path.split(".")
    if len(parts) < 2 or parts[0] not in ("input", "output"):
        raise PolicyError(f"{path}: patch paths start with input. or output.")
    node: Any = raw.setdefault(parts[0], {})
    rest = parts[1:]
    if rest[0] == "guards":
        guards = node.get("guards")
        if not isinstance(guards, list) or len(rest) < 3:
            raise PolicyError(f"{path}: expected guards.<name>.<field>")
        entry: Any = guards[_guard_index(cast("list[Any]", guards), rest[1], path)]
        rest = rest[2:]
        node = entry["config"] if "config" in entry and rest[0] not in COMMON_FIELDS else entry
    for part in rest[:-1]:
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        if not isinstance(child, dict):
            raise PolicyError(f"{path}: '{part}' is not a mapping")
        node = child
    node[rest[-1]] = value


def tighten(doc: PolicyDoc, ext: GGGuardrailsExt) -> PolicyDoc:
    """request-level changes can only add enforcement; anything else is rejected"""
    if not doc.tightening.allow_request:
        raise GuardrailOverrideRejectedError(
            "Guardrail override rejected: this policy does not allow request changes."
        )
    default_mode = doc.defaults.mode

    def adjust(entries: tuple[GuardEntry, ...]) -> tuple[GuardEntry, ...]:
        out: list[GuardEntry] = []
        for entry in entries:
            mode = entry.mode or default_mode
            if entry.key in ext.enable or (ext.enforce_shadowed and mode is Mode.SHADOW):
                mode = Mode.ENFORCE
            out.append(
                entry if mode is (entry.mode or default_mode) else entry.model_copy(update={"mode": mode})
            )
        return tuple(out)

    known = {e.key for e in (*doc.input.guards, *doc.output.guards)}
    unknown = sorted(set(ext.enable) - known)
    if unknown:
        raise GuardrailOverrideRejectedError(
            f"Guardrail override rejected: unknown guardrail(s) {', '.join(unknown)}."
        )
    output = doc.output.model_copy(update={"guards": adjust(doc.output.guards)})
    if ext.output_stream_mode == "buffer":
        output = output.model_copy(
            update={"streaming": output.streaming.model_copy(update={"mode": "buffer"})}
        )
    return doc.model_copy(
        update={"input": doc.input.model_copy(update={"guards": adjust(doc.input.guards)}), "output": output}
    )


class PolicySet:
    """all base policies; effective policies are built lazily per (policy, matched overrides, tightening)"""

    def __init__(
        self, policies: Mapping[str, LoadedPolicy], registry: GuardRegistry, deps: GuardDeps
    ) -> None:
        self._policies = dict(policies)
        self._registry = registry
        self._deps = deps
        self._cache: OrderedDict[tuple[str, tuple[int, ...], GGGuardrailsExt | None], EffectivePolicy] = (
            OrderedDict()
        )

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self._policies)

    @property
    def hash(self) -> str:
        """base files + rule packs; per-request hashes are on the effective policies"""
        parts = [
            f"{pid}={p.file_hash}:{self.build(pid, ()).hash}" for pid, p in sorted(self._policies.items())
        ]
        return sha256_hex("|".join(parts))[:12]

    def build(
        self, policy_id: str, overrides: tuple[int, ...], ext: GGGuardrailsExt | None = None
    ) -> EffectivePolicy:
        cache_key = (policy_id, overrides, ext)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached
        loaded = self._policies[policy_id]
        raw = copy.deepcopy(dict(loaded.raw))
        for i in overrides:
            override = loaded.doc.overrides[i]
            for path, value in override.patch.items():
                apply_patch(raw, path, value)
        try:
            doc = PolicyDoc.model_validate(raw)
        except ValidationError as exc:
            raise PolicyError(f"after overrides {list(overrides)}: {exc}") from exc
        if ext is not None:
            doc = tighten(doc, ext)
        effective = build_effective(doc, self._registry, self._deps)
        self._cache[cache_key] = effective
        if len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return effective

    def prepare(self, keys: Iterable[KeyPolicy]) -> None:
        """builds every configured key's effective policy at startup; model-scoped overrides stay lazy"""
        problems: list[ConfigProblem] = []
        for key in keys:
            loaded = self._policies[key.policy_id]
            matched = tuple(
                i
                for i, o in enumerate(loaded.doc.overrides)
                if not o.match.models and o.match.matches(key, "")
            )
            try:
                self.build(key.policy_id, matched)
            except PolicyError as exc:
                problems.append(ConfigProblem(loaded.file, "", f"key '{key.id}': {exc}"))
        if problems:
            raise ConfigError(problems)

    def effective(self, key: KeyPolicy, request: ChatRequest) -> EffectivePolicy:
        loaded = self._policies.get(key.policy_id)
        if loaded is None:
            # startup checks every configured key; a miss here is a gateway fault, not the client's
            raise InternalError(f"guardrail policy '{key.policy_id}' is not loaded")
        matched = tuple(i for i, o in enumerate(loaded.doc.overrides) if o.match.matches(key, request.model))
        ext = request.gg.guardrails if request.gg is not None else None
        try:
            return self.build(key.policy_id, matched, ext)
        except PolicyError as exc:
            raise InternalError(f"guardrail policy '{key.policy_id}' failed to build: {exc}") from exc

    def check(self, key_policy_ids: Iterable[str]) -> list[ConfigProblem]:
        """startup validation: every override alone and all together must build; every key's policy exists"""
        problems: list[ConfigProblem] = []
        for pid, loaded in self._policies.items():
            n = len(loaded.doc.overrides)
            combos = [(), *((i,) for i in range(n)), tuple(range(n))] if n else [()]
            for combo in combos:
                try:
                    self.build(pid, combo)
                except PolicyError as exc:
                    problems.append(ConfigProblem(loaded.file, "", str(exc)))
                    break
        missing = sorted(set(key_policy_ids) - set(self._policies))
        problems.extend(
            ConfigProblem("keys", "guardrails.policy_id", f"unknown guardrail policy '{pid}'")
            for pid in missing
        )
        return problems


def load_policies(
    policy_dir: Path, registry: GuardRegistry, deps: GuardDeps, *, key_policy_ids: Iterable[str] = ()
) -> PolicySet:
    problems: list[ConfigProblem] = []
    policies: dict[str, LoadedPolicy] = {}
    files = sorted(policy_dir.glob("*.yaml")) if policy_dir.is_dir() else []
    if not files:
        raise ConfigError([ConfigProblem(str(policy_dir), "", "no guardrail policy files found")])
    for path in files:
        try:
            raw = load_yaml_file(path)
            doc = PolicyDoc.model_validate(raw)
        except ValidationError as exc:
            problems.extend(
                ConfigProblem(str(path), ".".join(str(p) for p in err["loc"]), err["msg"])
                for err in exc.errors(include_input=False)
            )
            continue
        except (YAMLError, ValueError, UnicodeDecodeError) as exc:
            problems.append(ConfigProblem(str(path), "", str(exc)))
            continue
        if doc.id in policies:
            problems.append(ConfigProblem(str(path), "id", f"duplicate policy id '{doc.id}'"))
            continue
        policies[doc.id] = LoadedPolicy(
            doc, cast("Mapping[str, Any]", raw), str(path), sha256_hex(path.read_bytes())
        )
    if problems:
        raise ConfigError(problems)
    policy_set = PolicySet(policies, registry, deps)
    problems = policy_set.check(key_policy_ids)
    if problems:
        raise ConfigError(problems)
    return policy_set
