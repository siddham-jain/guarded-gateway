import asyncio
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from starlette.requests import HTTPConnection

from gg import __version__
from gg.auth.base import KeyResolver
from gg.config.settings import Settings
from gg.core.aio import TaskSupervisor
from gg.core.clock import Clock
from gg.core.context import RequestContext
from gg.core.lifecycle import HealthCheck
from gg.pipeline.stage import PipelineResult
from gg.providers.base import ModelCatalog

type RunPipeline = Callable[[RequestContext], Awaitable[PipelineResult]]
type MetricsRenderer = Callable[[], tuple[bytes, str]]
type RequestCompleteHook = Callable[[RequestContext, int], Awaitable[None]]

# gg body extensions whose consumer has landed; the rest are reported in x-gg-ignored-params
DEFAULT_CONSUMED_EXTENSIONS = frozenset({"fallback", "route_threshold", "guardrails"})


@dataclass(frozen=True, slots=True)
class BuildInfo:
    version: str = __version__
    git_sha: str | None = None
    image_digest: str | None = None
    built_at: str | None = None

    @classmethod
    def from_env(cls) -> "BuildInfo":
        return cls(
            git_sha=os.environ.get("GG_GIT_SHA") or None,
            image_digest=os.environ.get("GG_IMAGE_DIGEST") or None,
            built_at=os.environ.get("GG_BUILT_AT") or None,
        )


class ServerState:
    """readiness, drain flag and in-flight count; flipped by the composition root's lifespan"""

    def __init__(self) -> None:
        self.ready = False
        self.draining = False
        self.inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()

    def mark_ready(self) -> None:
        self.ready = True

    def begin_drain(self) -> None:
        self.draining = True

    def request_started(self) -> None:
        self.inflight += 1
        self._idle.clear()

    def request_finished(self) -> None:
        self.inflight -= 1
        if self.inflight <= 0:
            self.inflight = 0
            self._idle.set()

    async def wait_idle(self, timeout_s: float) -> bool:
        try:
            async with asyncio.timeout(timeout_s):
                await self._idle.wait()
        except TimeoutError:
            return False
        return True


@dataclass(frozen=True, slots=True, kw_only=True)
class ApiServices:
    """everything the api layer needs; built by the composition root and stored on app.state.services"""

    settings: Settings
    clock: Clock
    keys: KeyResolver
    catalog: ModelCatalog
    run_pipeline: RunPipeline
    config_hash: str
    supervisor: TaskSupervisor
    health_checks: Sequence[HealthCheck] = ()
    metrics_renderer: MetricsRenderer | None = None
    on_request_complete: RequestCompleteHook | None = None
    consumed_extensions: frozenset[str] = DEFAULT_CONSUMED_EXTENSIONS
    build_info: BuildInfo = field(default_factory=BuildInfo.from_env)
    state: ServerState = field(default_factory=ServerState)


def get_services(conn: HTTPConnection) -> ApiServices:
    services = getattr(conn.app.state, "services", None)
    if not isinstance(services, ApiServices):
        raise RuntimeError("app.state.services is not an ApiServices; the composition root must set it")
    return services
