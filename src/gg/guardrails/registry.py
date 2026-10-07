"""guards are created by name from policy config; ml guards register here in part 2"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import httpx2
from pydantic import BaseModel

from gg.core.aio import CpuExecutor
from gg.core.clock import Clock
from gg.guardrails.base import Guardrail
from gg.guardrails.ml.runtime import MlRuntime
from gg.guardrails.rules import RulePackStore
from gg.plugins.registry import Registry


@dataclass(frozen=True, slots=True)
class RemoteClients:
    """pooled http clients and api keys for remote detectors; the composition root owns their lifecycle"""

    client: Callable[[str, str], httpx2.AsyncClient]
    api_keys: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class GuardDeps:
    """shared resources handed to every guard factory"""

    packs: RulePackStore
    clock: Clock
    cpu: CpuExecutor | None = None
    # none: model-backed guards build but allow everything with reason ml_disabled
    models: MlRuntime | None = None
    # none: remote detectors (promptguard) build but allow everything with reason <name>_disabled
    remote: RemoteClients | None = None


type GuardRegistry = Registry[Guardrail, GuardDeps]


def new_registry() -> GuardRegistry:
    return Registry[Guardrail, GuardDeps]("guardrail")


def register[C: BaseModel](
    registry: GuardRegistry, name: str, config_model: type[C], factory: Callable[[C, GuardDeps], Guardrail]
) -> None:
    """typed wrapper: the factory receives its validated config model"""
    registry.register(name, factory, config_model=config_model)
