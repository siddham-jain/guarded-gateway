from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class HealthStatus:
    name: str
    status: Literal["ok", "degraded", "down"]
    detail: str = ""
    gating: bool = True


@runtime_checkable
class Lifecycle(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


@runtime_checkable
class HealthCheck(Protocol):
    async def check(self) -> HealthStatus: ...
