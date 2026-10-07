from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

import anyio
import structlog
from starlette.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from gg.api.deps import ApiServices
from gg.core.context import ContextKey, FinalizerOrder, Outcome, RequestContext
from gg.core.errors import GGError, InternalError
from gg.core.jsonutil import dumps
from gg.observability.record import RESPONSE_STATUS, ResponseStatus

log = structlog.get_logger("gg.api")

_FINALIZED = ContextKey[bool]("api.finalized")


def json_response(body: Any, *, status: int = 200, headers: Mapping[str, str] | None = None) -> Response:
    return Response(
        dumps(body),
        status_code=status,
        headers={"cache-control": "no-store", **(headers or {})},
        media_type="application/json",
    )


def error_body(message: str, *, code: str | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": code}}


def error_outcome(error: GGError) -> Outcome:
    if isinstance(error, InternalError):
        return "internal_error"
    return "upstream_error" if error.status >= 500 else "rejected"


async def finalize_request(ctx: RequestContext, status: int, services: ApiServices) -> None:
    """runs the request's finalizers exactly once, shielded, whatever path ended the request"""
    if ctx.get(_FINALIZED):
        return
    ctx.set(_FINALIZED, True)
    if ctx.outcome is None:
        ctx.outcome = "completed" if status < 400 else "rejected"
    ctx.timings.mark("completed")
    if ctx.get(RESPONSE_STATUS) is None:
        ctx.set(RESPONSE_STATUS, ResponseStatus(status))
    hook = services.on_request_complete
    if hook is not None:
        ctx.finalizers.defer("request_complete", lambda: hook(ctx, status), FinalizerOrder.METRICS)
    with anyio.CancelScope(shield=True):
        await ctx.finalizers.run(timeout_s=services.settings.server.finalizer_timeout_s)


class FinalizingJSONResponse(Response):
    media_type = "application/json"

    def __init__(
        self,
        content: Any,
        *,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
        ctx: RequestContext,
        services: ApiServices,
    ) -> None:
        super().__init__(
            dumps(content), status_code=status, headers={"cache-control": "no-store", **(headers or {})}
        )
        self._ctx = ctx
        self._services = services

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await finalize_request(self._ctx, self.status_code, self._services)


class FinalizingStreamResponse(StreamingResponse):
    """sse body; on any exit (done, error, disconnect, cancel) runs cleanup then the finalizers"""

    def __init__(
        self,
        body: AsyncIterator[bytes],
        *,
        headers: Mapping[str, str],
        ctx: RequestContext,
        services: ApiServices,
        cleanup: Callable[[], Awaitable[None]],
    ) -> None:
        super().__init__(body, headers=headers, media_type="text/event-stream")
        self._ctx = ctx
        self._services = services
        self._cleanup = cleanup

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self._cleanup()
            await finalize_request(self._ctx, self.status_code, self._services)
