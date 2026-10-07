from collections.abc import Sequence

from gg.core.clock import Clock
from gg.core.context import RequestContext
from gg.core.errors import GGError, InternalError
from gg.pipeline.stage import (
    Handler,
    NullObserver,
    PipelineObserver,
    PipelineResult,
    Stage,
    StageOutcome,
)


class Pipeline:
    """onion middleware; records each stage's exclusive time (its own work minus inner stages)"""

    def __init__(
        self,
        stages: Sequence[Stage],
        terminal: Handler,
        *,
        clock: Clock,
        observer: PipelineObserver | None = None,
    ) -> None:
        self._stages = tuple(stages)
        self._terminal = terminal
        self._clock = clock
        self._observer = observer or NullObserver()

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self._stages)

    async def run(self, ctx: RequestContext) -> PipelineResult:
        ctx.timings.mark("pipeline_start", self._clock.monotonic())
        result = await self._call(0, ctx)
        self._check_mode(ctx, result)
        return result

    async def _call(self, i: int, ctx: RequestContext) -> PipelineResult:
        if i == len(self._stages):
            with ctx.timings.measure("terminal"):
                return await self._terminal(ctx)
        stage = self._stages[i]
        inner = 0.0
        called = False

        async def call_next(c: RequestContext) -> PipelineResult:
            nonlocal inner, called
            called = True
            start = self._clock.monotonic()
            try:
                return await self._call(i + 1, c)
            finally:
                inner += self._clock.monotonic() - start

        start = self._clock.monotonic()
        ctx.timings.start(stage.name, start)
        outcome: StageOutcome = "continued"
        try:
            result = await stage(ctx, call_next)
            if not called:
                outcome = "short_circuit"
            if not isinstance(result, PipelineResult):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise InternalError(f"stage {stage.name} returned no result")
            return result
        except GGError:
            outcome = "rejected"
            raise
        except BaseException:
            outcome = "error"
            raise
        finally:
            exclusive = self._clock.monotonic() - start - inner
            ctx.timings.record(stage.name, exclusive)
            self._observer.stage_finished(ctx, stage.name, exclusive, outcome)

    @staticmethod
    def _check_mode(ctx: RequestContext, result: PipelineResult) -> None:
        if ctx.request.stream and result.stream is None:
            raise InternalError("streaming request produced a non-stream result")
        if not ctx.request.stream and result.response is None:
            raise InternalError("non-stream request produced a stream result")
