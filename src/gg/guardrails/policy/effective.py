"""one policy after overrides and request tightening, with its guard chains built and its hash computed"""

from collections.abc import Iterable
from dataclasses import dataclass

from gg.core.jsonutil import canonical_json, sha256_hex
from gg.guardrails.base import GuardStage, Mode, PolicyRef, Redacting, Restorer
from gg.guardrails.engine import BoundGuard, GuardChain, GuardSettings
from gg.guardrails.policy.schema import Defaults, GuardEntry, PolicyDoc
from gg.guardrails.registry import GuardDeps, GuardRegistry
from gg.guardrails.rules import RulePackError
from gg.plugins.registry import RegistryError


class PolicyError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class EffectivePolicy:
    doc: PolicyDoc
    ref: PolicyRef
    input_chain: GuardChain
    output_chain: GuardChain

    @property
    def hash(self) -> str:
        return self.ref.hash

    @property
    def restorer(self) -> Restorer | None:
        return next((g.guard for g in self.output_chain.guards if isinstance(g.guard, Restorer)), None)

    def output_detectors(self, *, streaming: bool) -> GuardChain:
        # windowed guards vet stream windows; buffer guards need the whole text so they only run non-stream
        kinds = ("windowed",) if streaming else ("windowed", "buffer")
        return self.output_chain.select(
            lambda g: g.guard.streaming in kinds and not isinstance(g.guard, Restorer)
        )

    def posthoc(self) -> GuardChain:
        return self.output_chain.select(lambda g: g.guard.streaming == "post_hoc")


def _bind(
    entry: GuardEntry, stage: GuardStage, defaults: Defaults, registry: GuardRegistry, deps: GuardDeps
) -> BoundGuard:
    try:
        guard = registry.create(entry.guard, entry.config, deps)
    except (RegistryError, RulePackError, ValueError, OSError) as exc:
        raise PolicyError(f"{stage}.guards.{entry.key}: {exc}") from exc
    if guard.stage != stage:
        raise PolicyError(f"{stage}.guards.{entry.key}: '{entry.guard}' is an {guard.stage} guard")
    timeout_ms = entry.timeout_ms or defaults.timeout_ms
    settings = GuardSettings(
        name=entry.key,
        mode=entry.mode or defaults.mode,
        on_error=entry.on_error or defaults.on_error,
        timeout_s=timeout_ms / 1000,
        sample_rate=entry.sample_rate,
        when=entry.when.matches if entry.when is not None else None,
    )
    return BoundGuard(guard, settings)


def _chain(
    entries: Iterable[GuardEntry], stage: GuardStage, doc: PolicyDoc, registry: GuardRegistry, deps: GuardDeps
) -> GuardChain:
    bound = [
        _bind(e, stage, doc.defaults, registry, deps)
        for e in entries
        if (e.mode or doc.defaults.mode) is not Mode.OFF
    ]
    # guards whose backend is not configured (e.g. a remote detector without an api key) drop out entirely
    bound = [g for g in bound if getattr(g.guard, "available", True)]
    if stage == "input" and not doc.input.allow_unredacted_upstream:
        for g in bound:
            if isinstance(g.guard, Redacting) and g.settings.mode is not Mode.ENFORCE:
                raise PolicyError(
                    f"input.guards.{g.name}: mode {g.settings.mode} sends raw values upstream; "
                    "set input.allow_unredacted_upstream: true to allow it"
                )
    return GuardChain(bound)


def _pack_refs(doc: PolicyDoc) -> list[str]:
    return sorted(
        {
            str(e.config["pack"])
            for e in (*doc.input.guards, *doc.output.guards)
            if isinstance(e.config.get("pack"), str)
        }
    )


def build_effective(doc: PolicyDoc, registry: GuardRegistry, deps: GuardDeps) -> EffectivePolicy:
    input_chain = _chain(doc.input.guards, "input", doc, registry, deps)
    output_chain = _chain(doc.output.guards, "output", doc, registry, deps)
    for name in doc.output.streaming.tool_args_guards:
        if not any(e.key == name for e in doc.output.guards):
            raise PolicyError(f"output.streaming.tool_args_guards: unknown guard '{name}'")
    digests = "".join(deps.packs.digest(ref) for ref in _pack_refs(doc))
    policy_hash = sha256_hex(canonical_json(doc.model_dump(mode="json")) + b"\x00" + digests.encode())[:12]
    return EffectivePolicy(doc, PolicyRef(doc.id, doc.version, policy_hash), input_chain, output_chain)
