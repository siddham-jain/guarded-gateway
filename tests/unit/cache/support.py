import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from gg.cache.base import CacheBackend, CachedResponse, PutResult
from gg.cache.codec import Codec
from gg.cache.config import CacheConfig
from gg.cache.embedders.hashing import HashingEmbedder
from gg.cache.failopen import FailOpenCache
from gg.cache.keys import CacheKeyBuilder
from gg.cache.memory import InMemoryResponseCache
from gg.cache.setup import BuiltCache, build_cache
from gg.cache.stages import ExactCacheStage, HitResponder
from gg.cache.writer import CacheWriter
from gg.core.aio import TaskSupervisor
from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.schema import (
    AssistantMessage,
    ChatChunk,
    ChatResponse,
    Choice,
    ChunkChoice,
    Delta,
    FinishReason,
    Usage,
)
from gg.core.usage import UsageRecord
from gg.pipeline.stage import Next, PipelineResult
from gg.pipeline.streams import ChunkStream
from tests.conftest import make_ctx, make_key, make_request

USAGE = Usage(prompt_tokens=12, completion_tokens=7, total_tokens=19)
USAGE_RECORD = UsageRecord(
    provider="mock", deployment_id="mock/echo", upstream_model="echo", input_tokens=12, output_tokens=7
)


def config(**exact: Any) -> CacheConfig:
    return CacheConfig.model_validate(
        {
            "exact": {"singleflight_wait_s": 0.2, "singleflight_poll_s": 0.01, **exact},
            "semantic": {"embedder": {"provider": "hashing", "name": "hashing", "dim": 64}},
        }
    )


def codec() -> Codec:
    return Codec(compress_over_bytes=1024, max_value_bytes=262_144)


@dataclass
class RecordingHooks:
    lookups: list[tuple[str, str]] = field(default_factory=lambda: [])
    stores: list[tuple[str, str, str]] = field(default_factory=lambda: [])
    bypasses: list[str] = field(default_factory=lambda: [])
    flights: list[str] = field(default_factory=lambda: [])
    saved: list[tuple[str, float]] = field(default_factory=lambda: [])
    tokens: list[tuple[str, str, int]] = field(default_factory=lambda: [])
    distances: list[tuple[str, float]] = field(default_factory=lambda: [])
    durations: list[str] = field(default_factory=lambda: [])

    def lookup(self, layer: str, result: str, /) -> None:
        self.lookups.append((layer, result))

    def lookup_duration(self, layer: str, seconds: float, /) -> None:
        self.durations.append(layer)

    def semantic_distance(self, result: str, distance: float, /) -> None:
        self.distances.append((result, distance))

    def store(self, layer: str, result: str, reason: str, /) -> None:
        self.stores.append((layer, result, reason))

    def bypass(self, reason: str, /) -> None:
        self.bypasses.append(reason)

    def singleflight(self, outcome: str, /) -> None:
        self.flights.append(outcome)

    def cost_saved(self, layer: str, usd: float, /) -> None:
        self.saved.append((layer, usd))

    def tokens_saved(self, layer: str, kind: str, tokens: int, /) -> None:
        self.tokens.append((layer, kind, tokens))


def built(
    cfg: CacheConfig | None = None,
    *,
    clock: FakeClock | None = None,
    hooks: RecordingHooks | None = None,
    embedder: HashingEmbedder | None = None,
    supervisor: TaskSupervisor | None = None,
    **kwargs: Any,
) -> BuiltCache:
    return build_cache(
        cfg or config(),
        redis=None,
        clock=clock or FakeClock(),
        supervisor=supervisor if supervisor is not None else TaskSupervisor(),
        embedder=embedder,
        metrics_hooks=hooks,
        **kwargs,
    )


def ctx_for(
    clock: FakeClock | None = None, *, key: dict[str, Any] | None = None, **request: Any
) -> RequestContext:
    payload: dict[str, Any] = {"temperature": 0, **request}
    ctx = make_ctx(clock or FakeClock(), make_request(**payload))
    if key is not None:
        ctx.key = make_key(**key)
    return ctx


def response(
    content: str | None = "Paris.", *, finish: FinishReason = "stop", tool_calls: Any = None
) -> ChatResponse:
    message = AssistantMessage(content=content, tool_calls=tool_calls)
    return ChatResponse(
        id="up-1",
        created=1,
        model="echo-1",
        choices=(Choice(index=0, message=message, finish_reason=finish),),
        usage=USAGE,
        system_fingerprint="fp_1",
    )


def chunks(parts: Sequence[str], *, finish: FinishReason = "stop", usage: bool = True) -> list[ChatChunk]:
    def chunk(delta: Delta, reason: FinishReason | None = None) -> ChatChunk:
        return ChatChunk(
            id="up-1",
            created=1,
            model="echo-1",
            choices=(ChunkChoice(index=0, delta=delta, finish_reason=reason),),
        )

    out = [chunk(Delta(role="assistant", content=""))]
    out += [chunk(Delta(content=p)) for p in parts]
    out.append(chunk(Delta(), finish))
    if usage:
        out.append(ChatChunk(id="up-1", created=1, model="echo-1", choices=(), usage=USAGE))
    return out


DEPLOYMENT = Deployment(id="mock/echo", provider="mock", upstream_model="echo")


class Upstream:
    """terminal handler: counts calls, serves a json reply or a stream like the executor would"""

    def __init__(
        self,
        reply: ChatResponse | None = None,
        *,
        parts: Sequence[str] = ("Par", "is."),
        fail: BaseException | None = None,
        delay: float = 0.0,
    ) -> None:
        self.reply = reply or response()
        self.parts = list(parts)
        self.fail = fail
        self.delay = delay
        self.calls = 0

    async def __call__(self, ctx: RequestContext) -> PipelineResult:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail is not None:
            raise self.fail
        ctx.usage = USAGE_RECORD
        ctx.served_by = DEPLOYMENT
        if ctx.request.stream:

            async def gen() -> AsyncIterator[ChatChunk]:
                for c in chunks(self.parts):
                    yield c

            return PipelineResult(source="upstream", stream=ChunkStream(gen()))
        return PipelineResult(source="upstream", response=self.reply)


async def run(cache: BuiltCache, ctx: RequestContext, upstream: Next) -> PipelineResult:
    """one request through the exact stage, consumed like the api would, then its finalizers"""
    result = await cache.exact_stage(ctx, upstream)
    if result.stream is not None:
        collected = [c async for c in result.stream]
        result = PipelineResult(source=result.source, stream=_replay(collected))
    await finish_request(ctx)
    return result


async def finish_request(ctx: RequestContext) -> None:
    ctx.outcome = ctx.outcome or "completed"
    await ctx.finalizers.run(timeout_s=5)


def _replay(items: list[ChatChunk]) -> ChunkStream:
    async def gen() -> AsyncIterator[ChatChunk]:
        for c in items:
            yield c

    return ChunkStream(gen())


async def drain(stream: ChunkStream | None) -> list[ChatChunk]:
    assert stream is not None
    return [c async for c in stream]


def text_of(result: PipelineResult) -> str:
    assert result.response is not None
    return result.response.choices[0].message.content or ""


class BrokenBackend(InMemoryResponseCache):
    """every call raises, like redis being down"""

    def __init__(self) -> None:
        super().__init__(codec(), FakeClock())
        self.calls = 0

    async def get(self, key: str, /) -> CachedResponse | None:
        self.calls += 1
        raise ConnectionError("redis down")

    async def put(
        self, key: str, value: CachedResponse, ttl_s: int, /, *, replace: bool = False
    ) -> PutResult:
        self.calls += 1
        raise ConnectionError("redis down")

    async def acquire(self, lock_key: str, ttl_ms: int, /) -> str | None:
        self.calls += 1
        raise ConnectionError("redis down")


def stage_over(
    backend: CacheBackend, hooks: RecordingHooks, clock: FakeClock, cfg: CacheConfig | None = None
) -> ExactCacheStage:
    """an exact stage over a backend the test keeps a handle on"""
    cfg = cfg or config()
    cache = FailOpenCache(backend, cfg.backend, clock)
    responder = HitResponder(clock, hooks, None)
    writer = CacheWriter(cfg, cache, clock=clock, supervisor=TaskSupervisor(), hooks=hooks)
    return ExactCacheStage(
        cfg, cache, keys=CacheKeyBuilder(), responder=responder, writer=writer, hooks=hooks, clock=clock
    )
