from typing import Protocol

import structlog
from structlog.typing import FilteringBoundLogger

from gg.core.clock import Clock
from gg.core.context import FinalizerOrder, RequestContext
from gg.observability.metrics import Metrics
from gg.observability.record import RECORD, Pricer, RequestRecord, StrongDeployment
from gg.pipeline.stage import Next, PipelineResult


class RequestTracer(Protocol):
    def submit(self, ctx: RequestContext, record: RequestRecord, /) -> None: ...


class ObservabilityStage:
    """outermost stage: tracks inflight and defers the METRICS, TRACE and LOG finalizers.

    all three read the same RequestRecord, built once when METRICS runs (after the last byte).
    """

    name = "observability"

    def __init__(
        self,
        metrics: Metrics,
        *,
        clock: Clock,
        pricer: Pricer | None = None,
        strong_deployment: StrongDeployment | None = None,
        tracer: RequestTracer | None = None,
        log: FilteringBoundLogger | None = None,
    ) -> None:
        self._metrics = metrics
        self._clock = clock
        self._pricer = pricer
        self._strong_deployment = strong_deployment
        self._tracer = tracer
        self._log: FilteringBoundLogger = log or structlog.get_logger("gg.request")

    async def __call__(self, ctx: RequestContext, call_next: Next, /) -> PipelineResult:
        ctx.timings.mark("received", ctx.received_at)
        stream = "true" if ctx.request.stream else "false"
        self._metrics.inflight.labels(stream).inc()

        async def finish_metrics() -> None:
            self._metrics.inflight.labels(stream).dec()
            try:
                self._metrics.record_request(self.record(ctx))
            except Exception:
                self._metrics.telemetry_error("metrics")
                raise

        async def finish_log() -> None:
            try:
                self._emit_log(self.record(ctx))
            except Exception:
                self._metrics.telemetry_error("log")
                raise

        ctx.finalizers.defer("metrics", finish_metrics, FinalizerOrder.METRICS)
        if self._tracer is not None:
            tracer = self._tracer

            async def finish_trace() -> None:
                try:
                    tracer.submit(ctx, self.record(ctx))
                except Exception:
                    self._metrics.telemetry_error("trace")
                    raise

            ctx.finalizers.defer("trace", finish_trace, FinalizerOrder.TRACE)
        ctx.finalizers.defer("request_log", finish_log, FinalizerOrder.LOG)
        return await call_next(ctx)

    def record(self, ctx: RequestContext) -> RequestRecord:
        cached = ctx.get(RECORD)
        if cached is not None:
            return cached
        ctx.timings.mark("completed")
        record = RequestRecord.from_ctx(
            ctx, now=self._clock.monotonic(), pricer=self._pricer, strong_deployment=self._strong_deployment
        )
        ctx.set(RECORD, record)
        return record

    def _emit_log(self, record: RequestRecord) -> None:
        fields = record.to_log()
        if record.status_class == "5xx":
            self._log.error("request.completed", **fields)
        else:
            self._log.info("request.completed", **fields)
