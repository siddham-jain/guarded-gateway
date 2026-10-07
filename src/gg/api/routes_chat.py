from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from gg.api.authenticate import authenticate
from gg.api.authorize import authorize_request
from gg.api.context_factory import create_context
from gg.api.deps import ApiServices, get_services
from gg.api.disconnect import ClientDisconnectedError, run_with_disconnect_watch
from gg.api.errors import to_gg_error
from gg.api.headers import build_response_headers
from gg.api.parsing import parse_chat_request, read_body_limited
from gg.api.responses import FinalizingJSONResponse, finalize_request
from gg.api.sse import DISCONNECTED_STATUS, SSEWriter
from gg.core.context import RequestContext, StageTimings
from gg.core.errors import InternalError, ServiceUnavailableError
from gg.core.normalize import Normalization
from gg.core.schema import ChatRequest, ChatResponse, to_wire


@dataclass(frozen=True, slots=True)
class ParsedChat:
    request: ChatRequest
    normalizations: tuple[Normalization, ...] = ()
    ignored_params: frozenset[str] = frozenset()


type ParseBody = Callable[[bytes], ParsedChat]


def _not_ready() -> ServiceUnavailableError:
    return ServiceUnavailableError(
        "The server is shutting down; retry shortly.",
        code="not_ready",
        retry_after_s=1,
        headers={"x-should-retry": "true", "connection": "close"},
    )


def _parse_openai(body: bytes) -> ParsedChat:
    request, normalizations = parse_chat_request(body)
    return ParsedChat(request, normalizations)


async def admit(
    request: Request, *, parse: ParseBody, allow_api_key_header: bool = False
) -> tuple[ApiServices, RequestContext]:
    """auth, body parse and per-key authorization shared by every chat-shaped ingress"""
    services = get_services(request)
    if services.state.draining:
        raise _not_ready()
    key = await authenticate(request, services, allow_api_key_header=allow_api_key_header)
    server = services.settings.server
    timings: StageTimings = request.state.timings
    with timings.measure("parse"):
        body = await read_body_limited(
            request,
            limit=min(server.max_body_bytes, key.limits.max_body_bytes),
            timeout_s=server.body_read_timeout_s,
        )
        parsed = parse(body)
    with timings.measure("authorize"):
        authorized = authorize_request(
            parsed.request,
            key,
            services.catalog,
            normalizations=parsed.normalizations,
            consumed_extensions=services.consumed_extensions,
        )
    ctx = create_context(
        request,
        services,
        key=key,
        original=parsed.request,
        authorized=authorized,
        normalizations=parsed.normalizations,
    )
    ctx.ignored_params.update(parsed.ignored_params)
    request.state.ctx = ctx
    return services, ctx


async def chat_completions(request: Request) -> Response:
    services, ctx = await admit(request, parse=_parse_openai)
    if ctx.original.stream:
        return await SSEWriter(services, ctx, request.receive).open()
    return await complete(request, services, ctx, render=to_wire)


async def complete(
    request: Request,
    services: ApiServices,
    ctx: RequestContext,
    *,
    render: Callable[[ChatResponse], Any],
) -> Response:
    try:
        result = await run_with_disconnect_watch(request.receive, services.run_pipeline(ctx))
    except ClientDisconnectedError:
        ctx.outcome = "client_disconnected"
        await finalize_request(ctx, DISCONNECTED_STATUS, services)
        return Response(status_code=DISCONNECTED_STATUS)
    except Exception as exc:
        raise to_gg_error(exc) from None
    if result.response is None:
        raise InternalError("Non-streaming request produced no response.")
    headers = build_response_headers(ctx, streaming=False)
    return FinalizingJSONResponse(render(result.response), headers=headers, ctx=ctx, services=services)
