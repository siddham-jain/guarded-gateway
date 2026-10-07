from typing import Any, Protocol

from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.errors import ProviderError
from gg.providers.meta import ResponseMeta


class AttemptRecorder(Protocol):
    def first_byte(self) -> None: ...
    def commit(self) -> None: ...
    def event(self, name: str, **attrs: Any) -> None: ...
    def finish(self, meta: ResponseMeta) -> None: ...
    def fail(self, err: ProviderError) -> None: ...


class ProviderObserver(Protocol):
    """telemetry seam; c10 implements it with metrics and spans, c3 never imports either"""

    def attempt(self, ctx: RequestContext, dep: Deployment, *, stream: bool) -> AttemptRecorder: ...


class NoopRecorder:
    def first_byte(self) -> None:
        pass

    def commit(self) -> None:
        pass

    def event(self, name: str, **attrs: Any) -> None:
        pass

    def finish(self, meta: ResponseMeta) -> None:
        pass

    def fail(self, err: ProviderError) -> None:
        pass


class NoopObserver:
    def attempt(self, ctx: RequestContext, dep: Deployment, *, stream: bool) -> AttemptRecorder:
        return NoopRecorder()
