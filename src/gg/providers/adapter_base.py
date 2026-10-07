from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import aclosing
from dataclasses import dataclass
from typing import Self

import httpx2

from gg.core.aio import run_shielded
from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.errors import ProviderError
from gg.core.ids import completion_id
from gg.core.schema import ChatChunk, ChatRequest, ChatResponse
from gg.core.usage import UsageSource
from gg.pipeline.streams import StreamAssembler
from gg.providers.catalog.capabilities import AdjustPolicy
from gg.providers.commit import CommitGate
from gg.providers.errors import after_commit, capability_mismatch
from gg.providers.http import UpstreamRequest, open_stream, request_timeout
from gg.providers.meta import ResponseMeta
from gg.providers.observer import AttemptRecorder
from gg.providers.runtime import AdapterDeps, ProviderRuntime
from gg.providers.sse import SSEEvent, aiter_sse
from gg.providers.stream_base import StreamTranslator
from gg.providers.usage import TokenCounts, estimate_prompt_tokens, estimate_text_tokens

STATE_WRITE_TIMEOUT_S = 1.0


@dataclass(slots=True)
class Prepared:
    request: ChatRequest
    deployment: Deployment
    ctx: RequestContext
    meta: ResponseMeta

    @property
    def chunk_id(self) -> str:
        return completion_id(self.ctx.request_id)

    def note(self, *, ignored: tuple[str, ...] = (), adjustments: tuple[str, ...] = ()) -> None:
        self.ctx.ignored_params.update(ignored)
        self.meta = self.meta.but(
            ignored_params=self.meta.ignored_params
            + tuple(i for i in ignored if i not in self.meta.ignored_params),
            adjustments=self.meta.adjustments + adjustments,
        )


@dataclass(slots=True)
class _RunState:
    meta: ResponseMeta
    translator: StreamTranslator | None = None
    yielded: int = 0


class BaseHTTPAdapter(ABC):
    """template method: prepare -> build upstream -> stream -> translate -> commit gate -> usage -> persist.

    subclasses fill build_upstream, stream_translator and classify_error; everything else is shared.
    """

    def __init__(self, runtime: ProviderRuntime, deps: AdapterDeps) -> None:
        self.runtime = runtime
        self.name = runtime.name
        self.deps = deps
        self.client = deps.http.client(runtime.name, runtime.base_url)
        self._in_flight = 0

    @classmethod
    def from_runtime(cls, runtime: ProviderRuntime, deps: AdapterDeps) -> Self:
        return cls(runtime, deps)

    @abstractmethod
    async def build_upstream(self, prep: Prepared) -> UpstreamRequest: ...

    @abstractmethod
    def stream_translator(self, prep: Prepared) -> StreamTranslator: ...

    @abstractmethod
    def classify_error(self, status: int, body: bytes, headers: Mapping[str, str]) -> ProviderError: ...

    async def persist_state(self, prep: Prepared, translator: StreamTranslator) -> None:
        return None

    def response_meta(self, prep: Prepared, headers: Mapping[str, str]) -> ResponseMeta:
        return prep.meta

    def transport_kind(self) -> str:
        return "retryable"

    def iter_events(self, response: httpx2.Response) -> AsyncIterator[SSEEvent]:
        return aiter_sse(response.aiter_bytes())

    def prepare(self, request: ChatRequest, dep: Deployment, ctx: RequestContext) -> Prepared:
        checker = self.deps.checker
        result = checker.check(request, dep, AdjustPolicy())
        if result.rejects:
            raise capability_mismatch(self.name, dep.id, result.rejects)
        prep = Prepared(
            request=checker.apply(request, result),
            deployment=dep,
            ctx=ctx,
            meta=ResponseMeta(provider=self.name, deployment_id=dep.id, upstream_model=dep.upstream_model),
        )
        prep.note(ignored=result.ignored_params, adjustments=result.adjustments)
        return prep

    async def stream(
        self, request: ChatRequest, deployment: Deployment, ctx: RequestContext, /
    ) -> AsyncGenerator[ChatChunk]:
        prep = self.prepare(request, deployment, ctx)
        limit = self.runtime.max_in_flight
        if limit is not None and self._in_flight >= limit:
            # a busy local model must fail fast so the executor can fall back instead of queueing
            raise ProviderError(
                "fallback",
                provider=self.name,
                status=0,
                code="local_busy",
                message=f"{self.name} has {limit} requests in flight",
                deployment_id=deployment.id,
            )
        self._in_flight += 1
        recorder = self.deps.observer.attempt(ctx, deployment, stream=True)
        state = _RunState(prep.meta)
        try:
            async with aclosing(self._run(prep, recorder, state)) as chunks:
                async for chunk in chunks:
                    yield chunk
        except ProviderError as err:
            recorder.fail(err)
            raise
        finally:
            self._in_flight -= 1
            if state.meta.usage is None and state.translator is not None and state.yielded:
                # cancelled or failed after commit: the emitted part is still billed upstream
                counts, source = self._usage(prep, state.translator)
                record = counts.to_record(deployment, source=source, raw=state.translator.raw_usage)
                state.meta = self._meta(state.meta, state.translator).but(usage=record)
            recorder.finish(state.meta)

    async def complete(
        self, request: ChatRequest, deployment: Deployment, ctx: RequestContext
    ) -> ChatResponse:
        assembler = StreamAssembler()
        meta: object = None
        async with aclosing(self.stream(request, deployment, ctx)) as chunks:
            async for chunk in chunks:
                assembler.feed(chunk)
                meta = chunk.gg_meta or meta
        return assembler.result().with_meta(meta)

    async def aclose(self) -> None:
        # clients belong to the HttpClientFactory, which the composition root closes
        return None

    async def _run(
        self, prep: Prepared, recorder: AttemptRecorder, state: _RunState
    ) -> AsyncGenerator[ChatChunk]:
        dep = prep.deployment
        up = await self.build_upstream(prep)
        state.meta = prep.meta
        translator = self.stream_translator(prep)
        state.translator = translator
        gate = CommitGate()
        yielded = 0

        def stamp(err: ProviderError) -> ProviderError:
            err.deployment_id = err.deployment_id or dep.id
            return after_commit(err) if yielded else err

        try:
            async with open_stream(
                self.client,
                up,
                timeouts=request_timeout(dep.timeouts),
                classify=self.classify_error,
                provider=self.name,
                transport_kind=self.transport_kind(),
            ) as response:
                recorder.first_byte()
                state.meta = self.response_meta(prep, response.headers)
                async for event in self.iter_events(response):
                    for out in gate.push_many(translator.feed(event)):
                        if not yielded:
                            recorder.commit()
                            out.with_meta(self._meta(state.meta, translator))
                        yielded += 1
                        state.yielded = yielded
                        yield out
        except ProviderError as err:
            stamped = stamp(err)
            if stamped is err:
                raise
            raise stamped from err
        if not translator.finished:
            raise stamp(
                ProviderError(
                    "retryable",
                    provider=self.name,
                    status=200,
                    code="truncated",
                    message="upstream stream ended without a terminal event",
                )
            )
        counts, source = self._usage(prep, translator)
        record = counts.to_record(dep, source=source, raw=translator.raw_usage)
        state.meta = self._meta(state.meta, translator).but(usage=record)
        if gate.forced:
            state.meta = state.meta.but(flags=(*state.meta.flags, "forced_commit"))
        await run_shielded(
            lambda: self.persist_state(prep, translator),
            name=f"{self.name}.persist",
            timeout_s=STATE_WRITE_TIMEOUT_S,
        )
        tail = gate.close(translator.tail(counts.to_usage()))
        for i, out in enumerate(tail):
            if not yielded:
                recorder.commit()
            if not yielded or i == len(tail) - 1:
                out.with_meta(state.meta)
            yielded += 1
            state.yielded = yielded
            yield out

    def _meta(self, base: ResponseMeta, tr: StreamTranslator) -> ResponseMeta:
        return base.but(
            served_model=tr.served_model,
            upstream_request_id=base.upstream_request_id or tr.upstream_id,
            upstream_provider=tr.upstream_provider,
            provider_finish_reason=tr.provider_finish_reason,
            upstream_cost_usd=tr.cost_usd,
            flags=tuple(dict.fromkeys((*base.flags, *tr.flags))),
        )

    def _usage(self, prep: Prepared, tr: StreamTranslator) -> tuple[TokenCounts, UsageSource]:
        if tr.counts is not None:
            return tr.counts, "reported"
        prompt = estimate_prompt_tokens(prep.request)
        output = estimate_text_tokens(tr.output_text() + "".join(tr.reasoning_parts))
        return TokenCounts(input=prompt, output=output), "estimated"
