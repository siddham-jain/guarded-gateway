from collections.abc import AsyncIterator

import pytest
from openai.types.chat import ChatCompletionChunk

from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.errors import GuardrailBlockedError, InternalError
from gg.core.schema import (
    AssistantMessage,
    ChatChunk,
    ChatResponse,
    Choice,
    ChunkChoice,
    Delta,
    FunctionCall,
    FunctionCallDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
    to_wire,
)
from gg.pipeline.runner import Pipeline
from gg.pipeline.stage import Handler, Next, PipelineResult, StageOutcome
from gg.pipeline.streams import (
    ChunkStream,
    StreamAssembler,
    collect_stream,
    synthesize_chunks,
    synthesize_stream,
)
from tests.conftest import make_ctx, make_request

RESPONSE = ChatResponse(
    id="chatcmpl-1",
    created=1,
    model="m",
    choices=(Choice(index=0, message=AssistantMessage(content="hello world"), finish_reason="stop"),),
    usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
)


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, StageOutcome]] = []

    def stage_finished(
        self, ctx: RequestContext, stage: str, exclusive_s: float, outcome: StageOutcome, /
    ) -> None:
        self.events.append((stage, outcome))


class Sleeper:
    def __init__(self, name: str, clock: FakeClock, before: float, log: list[str]) -> None:
        self.name = name
        self._clock = clock
        self._before = before
        self._log = log

    async def __call__(self, ctx: RequestContext, call_next: Next, /) -> PipelineResult:
        self._log.append(f"{self.name}:in")
        self._clock.advance(self._before)
        result = await call_next(ctx)
        self._log.append(f"{self.name}:out")
        return result


def terminal_for(clock: FakeClock, seconds: float = 0.02) -> Handler:
    async def terminal(ctx: RequestContext) -> PipelineResult:
        clock.advance(seconds)
        return PipelineResult(source="upstream", response=RESPONSE)

    return terminal


async def test_onion_order_and_exclusive_timing(clock: FakeClock) -> None:
    log: list[str] = []
    observer = Recorder()
    pipeline = Pipeline(
        [Sleeper("a", clock, 0.005, log), Sleeper("b", clock, 0.001, log)],
        terminal_for(clock),
        clock=clock,
        observer=observer,
    )
    ctx = make_ctx(clock)
    result = await pipeline.run(ctx)
    assert result.response == RESPONSE
    assert log == ["a:in", "b:in", "b:out", "a:out"]
    assert ctx.timings.durations["a"] == pytest.approx(0.005)
    assert ctx.timings.durations["b"] == pytest.approx(0.001)
    assert ctx.timings.durations["terminal"] == pytest.approx(0.02)
    assert observer.events == [("b", "continued"), ("a", "continued")]


async def test_short_circuit_skips_inner_stages(clock: FakeClock) -> None:
    class Hit:
        name = "cache"

        async def __call__(self, ctx: RequestContext, call_next: Next, /) -> PipelineResult:
            return PipelineResult(source="exact_cache", response=RESPONSE)

    log: list[str] = []
    observer = Recorder()
    pipeline = Pipeline(
        [Hit(), Sleeper("inner", clock, 0, log)], terminal_for(clock), clock=clock, observer=observer
    )
    result = await pipeline.run(make_ctx(clock))
    assert result.source == "exact_cache"
    assert log == []
    assert observer.events == [("cache", "short_circuit")]


async def test_rejection_is_reported_and_propagated(clock: FakeClock) -> None:
    class Block:
        name = "guard"

        async def __call__(self, ctx: RequestContext, call_next: Next, /) -> PipelineResult:
            raise GuardrailBlockedError("no")

    observer = Recorder()
    pipeline = Pipeline([Block()], terminal_for(clock), clock=clock, observer=observer)
    with pytest.raises(GuardrailBlockedError):
        await pipeline.run(make_ctx(clock))
    assert observer.events == [("guard", "rejected")]


async def test_mode_mismatch_is_an_internal_error(clock: FakeClock) -> None:
    pipeline = Pipeline([], terminal_for(clock), clock=clock)
    with pytest.raises(InternalError):
        await pipeline.run(make_ctx(clock, make_request(stream=True)))


async def test_stream_map_closes_source_on_early_stop() -> None:
    closed: list[bool] = []

    async def source() -> AsyncIterator[ChatChunk]:
        try:
            for chunk in synthesize_chunks(RESPONSE, include_usage=False, chunk_chars=2):
                yield chunk
        finally:
            closed.append(True)

    async def take_two(inner: AsyncIterator[ChatChunk]) -> AsyncIterator[ChatChunk]:
        n = 0
        async for chunk in inner:
            yield chunk
            n += 1
            if n == 2:
                return

    stream = ChunkStream(source()).map(take_two)
    got = [c async for c in stream]
    await stream.aclose()
    assert len(got) == 2
    assert closed == [True]


async def test_synthesized_stream_round_trips_and_validates() -> None:
    stream = synthesize_stream(RESPONSE, include_usage=True, chunk_chars=4)
    chunks = [c async for c in stream]
    for chunk in chunks:
        ChatCompletionChunk.model_validate(to_wire(chunk))
    assert chunks[-1].usage == RESPONSE.usage
    assembled = await collect_stream(synthesize_stream(RESPONSE, include_usage=True))
    assert assembled.choices[0].message.content == "hello world"
    assert assembled.choices[0].finish_reason == "stop"
    assert assembled.usage == RESPONSE.usage


def test_assembler_merges_tool_call_fragments() -> None:
    def chunk(*deltas: ToolCallDelta, finish: bool = False) -> ChatChunk:
        return ChatChunk(
            id="c",
            created=1,
            model="m",
            choices=(
                ChunkChoice(
                    index=0, delta=Delta(tool_calls=deltas), finish_reason="tool_calls" if finish else None
                ),
            ),
        )

    assembler = StreamAssembler()
    assembler.feed(
        chunk(
            ToolCallDelta(
                index=0, id="call_a", type="function", function=FunctionCallDelta(name="f", arguments="")
            ),
            ToolCallDelta(
                index=1, id="call_b", type="function", function=FunctionCallDelta(name="g", arguments='{"')
            ),
        )
    )
    assembler.feed(
        chunk(
            ToolCallDelta(index=0, function=FunctionCallDelta(arguments='{"x":')),
            ToolCallDelta(index=0, function=FunctionCallDelta(arguments="1}")),
        )
    )
    assembler.feed(chunk(ToolCallDelta(index=1, function=FunctionCallDelta(arguments='y":2}')), finish=True))
    message = assembler.result().choices[0].message
    assert message.tool_calls == (
        ToolCall(id="call_a", function=FunctionCall(name="f", arguments='{"x":1}')),
        ToolCall(id="call_b", function=FunctionCall(name="g", arguments='{"y":2}')),
    )
    assert assembler.result().choices[0].finish_reason == "tool_calls"
