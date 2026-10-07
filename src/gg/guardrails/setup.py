"""what the composition root needs: load policies, build the engine and the two pipeline stages"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import structlog

from gg.cache.base import Embedder
from gg.core.aio import CpuExecutor, TaskSupervisor
from gg.core.clock import Clock
from gg.core.keypolicy import KeyPolicy
from gg.core.lifecycle import HealthCheck, HealthStatus
from gg.guardrails.base import GuardMetrics
from gg.guardrails.builtin import default_registry
from gg.guardrails.engine import GuardrailEngine
from gg.guardrails.ml.artefacts import ArtefactStore
from gg.guardrails.ml.runtime import MlRuntime
from gg.guardrails.output.stage import OutputGuardStage
from gg.guardrails.policy.loader import PolicySet, load_policies
from gg.guardrails.probe import GuardProbe
from gg.guardrails.registry import GuardDeps, GuardRegistry, RemoteClients
from gg.guardrails.rules import RulePackStore
from gg.guardrails.stages import InputGuardPreStage

log = structlog.get_logger("gg.guardrails")


class _NoModels:
    async def check(self) -> HealthStatus:
        return HealthStatus("guard_models", "ok", "ml guards disabled")


@dataclass(frozen=True, slots=True)
class Guardrails:
    policies: PolicySet
    engine: GuardrailEngine
    input_stage: InputGuardPreStage
    output_stage: OutputGuardStage
    # tier 2-3 input guards; add to the concurrent probes stage (precedence 10)
    tier2_probe: GuardProbe | None = None
    ml: MlRuntime | None = None

    @property
    def hash(self) -> str:
        return self.policies.hash

    @property
    def health(self) -> HealthCheck:
        """down until every non-lazy guard model is loaded and warm"""
        return self.ml if self.ml is not None else _NoModels()

    async def start(self) -> None:
        if self.ml is not None:
            await self.ml.start()


def build_guardrails(
    policy_dir: Path,
    *,
    clock: Clock,
    supervisor: TaskSupervisor | None = None,
    keys: Iterable[KeyPolicy] = (),
    metrics: GuardMetrics | None = None,
    registry: GuardRegistry | None = None,
    cpu: CpuExecutor | None = None,
    embedder: Embedder | None = None,
    models_dir: Path | None = None,
    download_models: bool = True,
    remote: RemoteClients | None = None,
) -> Guardrails:
    """raises gg.config.loader.ConfigError with every policy problem found.

    model-backed guards run only when models_dir (weights cache, e.g. .models) and cpu are given; otherwise
    they build but allow everything with reason ml_disabled.
    """
    keys = tuple(keys)
    ml = None
    if models_dir is not None:
        if cpu is None:
            raise ValueError("model-backed guards need the shared cpu executor")
        ml = MlRuntime(cpu, ArtefactStore(models_dir, download=download_models), embedder)
    else:
        log.warning("guardrails.ml_disabled", reason="no models_dir; model-backed guards allow everything")
    deps = GuardDeps(packs=RulePackStore(policy_dir), clock=clock, cpu=cpu, models=ml, remote=remote)
    policies = load_policies(
        policy_dir, registry or default_registry(), deps, key_policy_ids={k.policy_id for k in keys}
    )
    policies.prepare(keys)
    engine = GuardrailEngine(clock=clock, metrics=metrics, supervisor=supervisor)
    return Guardrails(
        policies=policies,
        engine=engine,
        input_stage=InputGuardPreStage(policies, engine),
        output_stage=OutputGuardStage(engine, clock=clock, supervisor=supervisor),
        tier2_probe=GuardProbe(engine) if ml is not None or remote is not None else None,
        ml=ml,
    )
