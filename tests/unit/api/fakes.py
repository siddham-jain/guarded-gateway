import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI

from gg.api.deps import ApiServices
from gg.api.install import install_exception_handlers, install_middleware, install_routes
from gg.auth.config import load_keys
from gg.auth.resolver import CachingKeyResolver
from gg.auth.store import YamlKeyStore
from gg.config.settings import ServerSettings, Settings
from gg.core.aio import TaskSupervisor
from gg.core.clock import SystemClock
from gg.core.context import RequestContext
from gg.core.deployment import Deployment, PriceSchedule
from gg.core.schema import (
    AssistantMessage,
    ChatChunk,
    ChatRequest,
    ChatResponse,
    Choice,
    ChunkChoice,
    Delta,
    Usage,
)
from gg.pipeline.stage import PipelineResult
from gg.pipeline.streams import ChunkStream, synthesize_chunks
from gg.providers.base import AliasRef, DeploymentRef, PublicModel, Resolution

KEYS_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "keys"
KEYS_FILE = KEYS_DIR / "keys.yaml"
TOKENS: dict[str, str] = yaml.safe_load((KEYS_DIR / "tokens.yaml").read_text())

ECHO = Deployment(id="mock/echo", provider="mock", upstream_model="echo", tier="weak")
OTHER = Deployment(id="other/big", provider="other", upstream_model="big", tier="strong")
FREE = Deployment(
    id="mock/free", provider="mock", upstream_model="free", pricing=PriceSchedule(periods=(), billed=False)
)
PREMIUM = Deployment(id="mock/premium", provider="mock", upstream_model="premium", tier="premium")


class FakeCatalog:
    def __init__(self, deployments: Sequence[Deployment] = (ECHO, OTHER, FREE, PREMIUM)) -> None:
        self._deployments = {d.id: d for d in deployments}
        self._aliases = {"gg/auto": AliasRef("gg/auto", "router"), "gg/weak": AliasRef("gg/weak", "group")}

    @property
    def hash(self) -> str:
        return "cataloghash"

    def resolve(self, model: str, /) -> Resolution | None:
        if model in self._aliases:
            return self._aliases[model]
        deployment = self._deployments.get(model)
        return DeploymentRef(deployment) if deployment else None

    def get(self, deployment_id: str, /) -> Deployment:
        return self._deployments[deployment_id]

    def chain(self, group: str, request: ChatRequest, /) -> list[Deployment]:
        return [ECHO]

    def groups_of(self, alias: str, /) -> tuple[str, ...]:
        return ()

    def deployments_for(self, model: str, /) -> list[Deployment]:
        return [ECHO]

    def list_public(self, allowed: Callable[[str], bool], /) -> Sequence[PublicModel]:
        rows = [PublicModel(id=a, owned_by="gg", created=1_759_622_400) for a in self._aliases]
        rows += [
            PublicModel(id=d.id, owned_by=d.provider, created=1_759_622_400, context_window=128_000)
            for d in sorted(self._deployments.values(), key=lambda d: d.id)
        ]
        return [r for r in rows if allowed(r.id)]


def response_for(ctx: RequestContext, text: str = "hello there") -> ChatResponse:
    return ChatResponse(
        id="chatcmpl-test",
        created=1_759_622_400,
        model=ECHO.id,
        choices=(Choice(index=0, message=AssistantMessage(content=text), finish_reason="stop"),),
        usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


def chunks_for(text: str = "hello there") -> list[ChatChunk]:
    response = ChatResponse(
        id="chatcmpl-test",
        created=1_759_622_400,
        model=ECHO.id,
        choices=(Choice(index=0, message=AssistantMessage(content=text), finish_reason="stop"),),
        usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    return synthesize_chunks(response, include_usage=True, chunk_chars=4)


def chunk(content: str) -> ChatChunk:
    return ChatChunk(
        id="chatcmpl-test",
        created=1_759_622_400,
        model=ECHO.id,
        choices=(ChunkChoice(index=0, delta=Delta(content=content)),),
    )


@dataclass
class StreamProbe:
    """records how far a fake upstream stream got and whether it was closed"""

    yielded: int = 0
    closed: bool = False
    cancelled: bool = False


@dataclass
class FakePipeline:
    """stands in for the real pipeline: returns a canonical response or stream, or raises"""

    chunks: list[ChatChunk] = field(default_factory=chunks_for)
    text: str = "hello there"
    error: BaseException | None = None
    prime_delay_s: float = 0.0
    chunk_delay_s: float = 0.0
    fail_after: int | None = None
    fail_with: BaseException | None = None
    calls: list[RequestContext] = field(default_factory=lambda: [])
    probe: StreamProbe = field(default_factory=StreamProbe)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    finalizer_runs: list[tuple[str | None, str]] = field(default_factory=lambda: [])

    async def __call__(self, ctx: RequestContext) -> PipelineResult:
        self.calls.append(ctx)
        self.started.set()
        ctx.finalizers.defer("spy", lambda: self._spy(ctx), 10)
        if self.prime_delay_s:
            await asyncio.sleep(self.prime_delay_s)
        if self.error is not None:
            raise self.error
        ctx.served_by = ECHO
        ctx.response_headers["x-gg-route"] = "direct"
        if ctx.request.stream:
            return PipelineResult(source="upstream", stream=ChunkStream(self._stream()))
        return PipelineResult(source="upstream", response=response_for(ctx, self.text))

    async def _spy(self, ctx: RequestContext) -> None:
        self.finalizer_runs.append((ctx.outcome, ctx.request_id))

    async def _stream(self) -> AsyncIterator[ChatChunk]:
        try:
            for i, item in enumerate(self.chunks):
                if self.fail_after is not None and i == self.fail_after:
                    assert self.fail_with is not None
                    raise self.fail_with
                if self.chunk_delay_s:
                    await asyncio.sleep(self.chunk_delay_s)
                self.probe.yielded += 1
                yield item
        except asyncio.CancelledError:
            self.probe.cancelled = True
            raise
        finally:
            self.probe.closed = True


def make_settings(**server: Any) -> Settings:
    return Settings(env="test", server=ServerSettings(**server))


def make_services(
    pipeline: FakePipeline | None = None, *, settings: Settings | None = None, **overrides: Any
) -> ApiServices:
    settings = settings or make_settings()
    clock = SystemClock()
    resolver = CachingKeyResolver(
        YamlKeyStore(load_keys(KEYS_FILE)), clock, allow_test_keys=settings.env != "prod"
    )
    services = ApiServices(
        settings=settings,
        clock=clock,
        keys=resolver,
        catalog=FakeCatalog(),
        run_pipeline=pipeline or FakePipeline(),
        config_hash="0123456789abcdef",
        supervisor=TaskSupervisor(),
        **overrides,
    )
    services.state.mark_ready()
    return services


def build_app(services: ApiServices) -> FastAPI:
    app = FastAPI()
    app.state.services = services
    install_middleware(app, services.settings)
    install_exception_handlers(app)
    install_routes(app)
    return app
