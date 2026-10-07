from gg.core.context import RequestContext
from gg.observability.metrics import Metrics
from gg.pipeline.stage import StageOutcome


class MetricsObserver:
    """PipelineObserver feeding gg_stage_duration_seconds from the runner's exclusive stage times"""

    def __init__(self, metrics: Metrics) -> None:
        self._metrics = metrics

    def stage_finished(
        self, ctx: RequestContext, stage: str, exclusive_s: float, outcome: StageOutcome, /
    ) -> None:
        self._metrics.observe_stage(stage, exclusive_s)
