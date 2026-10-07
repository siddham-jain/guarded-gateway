import asyncio
from collections import OrderedDict
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from typing import Any, Self

import orjson
from pydantic import ValidationError

from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.errors import ProviderError
from gg.core.ids import completion_id
from gg.core.schema import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    ChunkChoice,
    Delta,
    FunctionCallDelta,
    NamedToolChoice,
    ToolCallDelta,
)
from gg.pipeline.streams import StreamAssembler
from gg.providers.catalog.capabilities import AdjustPolicy
from gg.providers.errors import after_commit, capability_mismatch
from gg.providers.meta import ResponseMeta
from gg.providers.mock.scenarios import MockScenario
from gg.providers.openai_compat.errors import classify_payload
from gg.providers.openai_compat.quirks import QuirkProfile
from gg.providers.runtime import AdapterDeps, ProviderRuntime
from gg.providers.usage import TokenCounts, estimate_prompt_tokens, estimate_text_tokens

_MAX_TRACKED_REQUESTS = 4096


class MockAdapter:
    """in-process deterministic upstream; faults go through the real openai-compat classifier"""

    def __init__(self, runtime: ProviderRuntime, deps: AdapterDeps) -> None:
        self.runtime = runtime
        self.name = runtime.name
        self.deps = deps
        self._quirks = QuirkProfile()
        self._attempts: OrderedDict[str, int] = OrderedDict()

    @classmethod
    def from_runtime(cls, runtime: ProviderRuntime, deps: AdapterDeps) -> Self:
        return cls(runtime, deps)

    async def aclose(self) -> None:
        return None

    def scenario(self, request: ChatRequest, dep: Deployment) -> MockScenario:
        raw: dict[str, Any] = dict(dep.defaults.get("mock") or {})
        override = (request.model_extra or {}).get("mock")
        if isinstance(override, dict) and self.deps.env != "prod":
            raw.update(override)  # pyright: ignore[reportUnknownArgumentType]
        try:
            return MockScenario.model_validate(raw)
        except ValidationError as exc:
            raise ProviderError(
                "client", provider=self.name, status=400, code="invalid_mock", message=str(exc)
            ) from exc

    def _attempt(self, request_id: str) -> int:
        count = self._attempts.get(request_id, 0) + 1
        self._attempts[request_id] = count
        self._attempts.move_to_end(request_id)
        while len(self._attempts) > _MAX_TRACKED_REQUESTS:
            self._attempts.popitem(last=False)
        return count

    def _fault(self, scenario: MockScenario, attempt: int) -> ProviderError | None:
        fault = scenario.fail
        if fault is None or (fault.first_n_attempts is not None and attempt > fault.first_n_attempts):
            return None
        return classify_payload(
            fault.status, fault.body, fault.headers, provider=self.name, quirks=self._quirks
        )

    async def stream(
        self, request: ChatRequest, deployment: Deployment, ctx: RequestContext, /
    ) -> AsyncGenerator[ChatChunk]:
        result = self.deps.checker.check(request, deployment, AdjustPolicy())
        if result.rejects:
            raise capability_mismatch(self.name, deployment.id, result.rejects)
        request = self.deps.checker.apply(request, result)
        ctx.ignored_params.update(result.ignored_params)
        scenario = self.scenario(request, deployment)
        fault = self._fault(scenario, self._attempt(ctx.request_id))
        meta = ResponseMeta(
            provider=self.name,
            deployment_id=deployment.id,
            upstream_model=deployment.upstream_model,
            served_model=deployment.upstream_model,
            upstream_request_id="mock-" + ctx.request_id,
            ignored_params=result.ignored_params,
            adjustments=result.adjustments,
        )
        if scenario.ttft_ms:
            await asyncio.sleep(scenario.ttft_ms / 1000)
        if fault is not None and scenario.fail is not None and scenario.fail.after_chunks is None:
            fault.deployment_id = deployment.id
            raise fault

        def chunk(choices: tuple[ChunkChoice, ...], **extra: Any) -> ChatChunk:
            return ChatChunk.model_construct(
                id=completion_id(ctx.request_id),
                created=ctx.received_unix,
                model=deployment.id,
                choices=choices,
                **extra,
            )

        bodies = self._bodies(request, scenario)
        emitted: list[str] = []
        delay = scenario.chunk_tokens / scenario.tokens_per_s if scenario.tokens_per_s else 0.0
        for i, (delta, finish) in enumerate(bodies):
            if fault is not None and scenario.fail is not None and i == scenario.fail.after_chunks:
                fault.deployment_id = deployment.id
                raise after_commit(fault) if i else fault
            if i and delay:
                await asyncio.sleep(delay)
            emitted.append(delta.content or "")
            out = chunk((ChunkChoice.model_construct(index=0, delta=delta, finish_reason=finish),))
            if i == 0:
                out.with_meta(meta)
            yield out
        counts = TokenCounts(
            input=estimate_prompt_tokens(request), output=estimate_text_tokens("".join(emitted)) or 1
        )
        final_meta = meta.but(usage=counts.to_record(deployment, source="reported"))
        yield chunk((), usage=counts.to_usage()).with_meta(final_meta)

    async def complete(
        self, request: ChatRequest, deployment: Deployment, ctx: RequestContext
    ) -> ChatResponse:
        assembler = StreamAssembler()
        meta: object = None
        async with aclosing(self.stream(request, deployment, ctx)) as chunks:
            async for c in chunks:
                assembler.feed(c)
                meta = c.gg_meta or meta
        return assembler.result().with_meta(meta)

    def _bodies(self, request: ChatRequest, scenario: MockScenario) -> list[tuple[Delta, Any]]:
        tool = self._tool_call(request, scenario)
        if tool is not None:
            name, arguments = tool
            call = ToolCallDelta.model_construct(
                index=0,
                id="call_mock_0",
                type="function",
                function=FunctionCallDelta.model_construct(name=name, arguments=arguments),
            )
            return [
                (Delta.model_construct(role="assistant", tool_calls=(call,)), None),
                (Delta.model_construct(), "tool_calls"),
            ]
        words = scenario.reply(request.last_user_text()).split(" ")
        size = scenario.chunk_tokens
        pieces = [" ".join(words[i : i + size]) for i in range(0, len(words), size)]
        pieces = [p if i == 0 else " " + p for i, p in enumerate(pieces)] or [""]
        out: list[tuple[Delta, Any]] = [
            (Delta.model_construct(role="assistant" if i == 0 else None, content=p), None)
            for i, p in enumerate(pieces)
        ]
        out.append((Delta.model_construct(), "stop"))
        return out

    def _tool_call(self, request: ChatRequest, scenario: MockScenario) -> tuple[str, str] | None:
        if not request.tools or request.tool_choice == "none":
            return None
        configured = scenario.tool_call
        name = configured.name if configured is not None else None
        if isinstance(request.tool_choice, NamedToolChoice):
            name = request.tool_choice.function.get("name", name)
        if name is None:
            first = request.tools[0]
            fn = getattr(first, "function", None)
            name = fn.name if fn is not None else "tool"
        arguments: Mapping[str, Any] = configured.arguments if configured is not None else {}
        return name, orjson.dumps(dict(arguments)).decode()
