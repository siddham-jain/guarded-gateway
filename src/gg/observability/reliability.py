from collections.abc import Mapping

from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.errors import GGError, ProviderError
from gg.core.usage import AttemptRecord
from gg.observability.labels import OTHER
from gg.observability.metrics import Metrics

# gg_circuit_state values; reliability types are not importable here, so states are plain strings
CIRCUIT_STATE: Mapping[str, int] = {"closed": 0, "half_open": 1, "open": 2}


class ReliabilityMetrics:
    """ReliabilityHooks + BreakerListener implementation; attempts and fallbacks come from the record"""

    def __init__(self, metrics: Metrics) -> None:
        self._metrics = metrics

    def attempt_finished(
        self,
        ctx: RequestContext,
        deployment: Deployment,
        record: AttemptRecord,
        error: ProviderError | None,
        /,
    ) -> None:
        m = self._metrics
        if error is not None and error.kind == "auth":
            m.provider_credential_errors.labels(m.labels("provider", deployment.provider)).inc()

    def deployment_skipped(self, ctx: RequestContext, deployment: Deployment, reason: str, /) -> None:
        m = self._metrics
        m.deployment_skips.labels(m.labels("deployment", deployment.id), reason).inc()

    def stream_interrupted(
        self, ctx: RequestContext, deployment: Deployment, error: ProviderError, /
    ) -> None:
        m = self._metrics
        m.stream_errors.labels("mid_stream", m.labels("provider", deployment.provider)).inc()

    def request_failed(self, ctx: RequestContext, error: GGError, /) -> None:
        m = self._metrics
        m.exhausted.labels(str(error.status), m.labels("error_code", error.code)).inc()

    def breaker_transition(self, deployment_id: str, old: str, new: str, reason: str, /) -> None:
        m = self._metrics
        deployment = m.labels("deployment", deployment_id)
        m.circuit_transitions.labels(deployment, new, m.labels("breaker_reason", reason)).inc()
        if deployment != OTHER:
            m.circuit_state.labels(deployment).set(CIRCUIT_STATE.get(new, 0))
