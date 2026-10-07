from typing import Protocol

from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.errors import GGError, ProviderError
from gg.core.usage import AttemptRecord
from gg.reliability.breaker import BreakerListener, BreakerStateName
from gg.reliability.errors import SkipReason


class ReliabilityHooks(BreakerListener, Protocol):
    """metrics / tracing seam for the executor; implementations must be cheap and must not raise"""

    def attempt_finished(
        self,
        ctx: RequestContext,
        deployment: Deployment,
        record: AttemptRecord,
        error: ProviderError | None,
        /,
    ) -> None: ...

    def deployment_skipped(
        self, ctx: RequestContext, deployment: Deployment, reason: SkipReason, /
    ) -> None: ...

    def stream_interrupted(
        self, ctx: RequestContext, deployment: Deployment, error: ProviderError, /
    ) -> None: ...

    def request_failed(self, ctx: RequestContext, error: GGError, /) -> None: ...


class NullHooks:
    def attempt_finished(
        self,
        ctx: RequestContext,
        deployment: Deployment,
        record: AttemptRecord,
        error: ProviderError | None,
        /,
    ) -> None:
        return None

    def deployment_skipped(self, ctx: RequestContext, deployment: Deployment, reason: SkipReason, /) -> None:
        return None

    def stream_interrupted(
        self, ctx: RequestContext, deployment: Deployment, error: ProviderError, /
    ) -> None:
        return None

    def request_failed(self, ctx: RequestContext, error: GGError, /) -> None:
        return None

    def breaker_transition(
        self, deployment_id: str, old: BreakerStateName, new: BreakerStateName, reason: str, /
    ) -> None:
        return None
