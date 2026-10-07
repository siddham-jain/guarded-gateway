from starlette.requests import Request
from starlette.responses import Response

from gg.api.anthropic_ingress.request import translate_request
from gg.api.anthropic_ingress.response import error_body, to_message
from gg.api.anthropic_ingress.stream import AnthropicStreamEncoder
from gg.api.errors import render_error, to_gg_error
from gg.api.parsing import chat_request_from_dict, load_json_object
from gg.api.routes_chat import ParsedChat, admit, complete
from gg.api.sse import SSEWriter


def _parse_anthropic(body: bytes) -> ParsedChat:
    payload, ignored = translate_request(load_json_object(body))
    request, normalizations = chat_request_from_dict(payload)
    return ParsedChat(request, normalizations, ignored)


async def messages(request: Request) -> Response:
    """anthropic messages ingress over the same pipeline as /v1/chat/completions"""
    try:
        services, ctx = await admit(request, parse=_parse_anthropic, allow_api_key_header=True)
        if ctx.original.stream:
            encoder = AnthropicStreamEncoder(request_id=ctx.request_id, model=ctx.original.model)
            return await SSEWriter(services, ctx, request.receive, encoder).open()
        return await complete(
            request, services, ctx, render=lambda response: to_message(response, request_id=ctx.request_id)
        )
    except Exception as exc:
        error = to_gg_error(exc)
        request_id: str | None = getattr(request.state, "request_id", None)
        return render_error(request, error, to_body=lambda e: error_body(e, request_id=request_id))
