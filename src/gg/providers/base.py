from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from gg.core.context import RequestContext
from gg.core.deployment import Capabilities, Deployment, PriceSchedule, Pricing
from gg.core.errors import ProviderError, ProviderErrorKind
from gg.core.schema import ChatChunk, ChatRequest

__all__ = [
    "AliasRef",
    "Capabilities",
    "Deployment",
    "DeploymentRef",
    "ModelCatalog",
    "PriceSchedule",
    "Pricing",
    "ProviderAdapter",
    "ProviderError",
    "ProviderErrorKind",
    "PublicModel",
    "Resolution",
]


class ProviderAdapter(Protocol):
    """translates canonical requests to one provider's wire format.

    stream() yields nothing until the first content-bearing chunk (commit gate) so the executor can still
    retry or fall back; the final chunk always carries usage. errors are ProviderError with committed set
    once anything was yielded.
    """

    name: str

    def stream(
        self, request: ChatRequest, deployment: Deployment, ctx: RequestContext, /
    ) -> AsyncIterator[ChatChunk]: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AliasRef:
    name: str
    kind: Literal["router", "group"]


@dataclass(frozen=True, slots=True)
class DeploymentRef:
    deployment: Deployment


type Resolution = AliasRef | DeploymentRef


@dataclass(frozen=True, slots=True)
class PublicModel:
    id: str
    owned_by: str
    created: int
    context_window: int | None = None
    shutdown_date: str | None = None


class ModelCatalog(Protocol):
    @property
    def hash(self) -> str: ...

    def resolve(self, model: str, /) -> Resolution | None: ...

    def get(self, deployment_id: str, /) -> Deployment: ...

    def chain(self, group: str, request: ChatRequest, /) -> list[Deployment]: ...

    def groups_of(self, alias: str, /) -> tuple[str, ...]: ...

    def deployments_for(self, model: str, /) -> list[Deployment]: ...

    def list_public(self, allowed: Callable[[str], bool], /) -> Sequence[PublicModel]: ...
