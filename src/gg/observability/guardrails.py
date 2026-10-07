from gg.guardrails.base import ErrorKind, GuardStage
from gg.observability.metrics import Metrics


class GuardrailMetrics:
    """GuardMetrics implementation feeding the gg_guardrail_* and gg_pii_placeholders_total series"""

    def __init__(self, metrics: Metrics) -> None:
        self._metrics = metrics

    def decision(self, stage: GuardStage, guard: str, action: str, mode: str, /) -> None:
        m = self._metrics
        labels = m.labels
        m.guardrail_decisions.labels(
            stage, labels("guardrail", guard), labels("action", action), labels("mode", mode)
        ).inc()

    def duration(self, stage: GuardStage, guard: str, seconds: float, /) -> None:
        m = self._metrics
        m.guardrail_duration.labels(stage, m.labels("guardrail", guard)).observe(seconds)

    def error(self, stage: GuardStage, guard: str, kind: ErrorKind, /) -> None:
        m = self._metrics
        m.guardrail_errors.labels(stage, m.labels("guardrail", guard), kind).inc()

    def placeholders(self, direction: str, strategy: str, count: int, /) -> None:
        m = self._metrics
        if count > 0:
            m.pii_placeholders.labels(m.labels("direction", direction), m.labels("strategy", strategy)).inc(
                count
            )
