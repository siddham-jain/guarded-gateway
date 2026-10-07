from gg.limits.base import BackendOp, BudgetEvent, RejectionLimit
from gg.observability.metrics import Metrics


class LimitsMetrics:
    """LimitsHooks implementation feeding the gg_ratelimit_* and gg_budget_* series"""

    def __init__(self, metrics: Metrics) -> None:
        self._metrics = metrics

    def rejected(self, limit: RejectionLimit, /) -> None:
        self._metrics.ratelimit_rejections.labels(limit).inc()

    def budget_event(self, event: BudgetEvent, /) -> None:
        self._metrics.budget_events.labels(event).inc()

    def backend_error(self, op: BackendOp, /) -> None:
        self._metrics.ratelimit_backend_errors.labels(op).inc()

    def degraded(self, active: bool, /) -> None:
        self._metrics.ratelimit_degraded.set(1 if active else 0)

    def holds_expired(self, count: int, /) -> None:
        if count > 0:
            self._metrics.budget_hold_expired.inc(count)
