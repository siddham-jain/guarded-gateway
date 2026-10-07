from collections.abc import Callable, Mapping, Sequence
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from gg.api.deps import get_services
from gg.api.headers import SECURITY_HEADERS, build_response_headers
from gg.api.responses import FinalizingJSONResponse, error_body, error_outcome, json_response
from gg.core.errors import (
    AuthenticationError,
    GGError,
    InternalError,
    InvalidRequestError,
    PayloadTooLargeError,
    ProviderError,
    RequestTimeoutError,
    UpstreamError,
)

log = structlog.get_logger("gg.api")

_CODE_BY_TYPE = {"missing": "missing_required_parameter", "extra_forbidden": "unknown_parameter"}


def _param_from_loc(loc: Sequence[int | str], data: Any) -> str | None:
    # walk the input alongside the loc so union/tag labels pydantic inserts are dropped
    parts: list[str] = []
    node: Any = data
    for i, item in enumerate(loc):
        if isinstance(item, int) and isinstance(node, list) and 0 <= item < len(node):
            parts.append(f"[{item}]")
            node = node[item]
        elif isinstance(item, str) and isinstance(node, dict):
            if item in node:
                parts.append(f".{item}" if parts else item)
                node = node[item]
            elif i == len(loc) - 1:
                parts.append(f".{item}" if parts else item)
    return "".join(parts) or None


def _code_for(error_type: str) -> str:
    if error_type in _CODE_BY_TYPE:
        return _CODE_BY_TYPE[error_type]
    if error_type.endswith(("_type", "_parsing")):
        return "invalid_type"
    return "invalid_value"


def validation_to_error(exc: ValidationError | RequestValidationError, data: Any = None) -> GGError:
    errors = exc.errors(include_input=False) if isinstance(exc, ValidationError) else exc.errors()
    if not errors:
        return InvalidRequestError("Invalid request.")
    first = errors[0]
    param = _param_from_loc(first["loc"], data)
    code = _code_for(first["type"])
    if code == "missing_required_parameter":
        message = f"Missing required parameter: '{param}'."
    elif code == "unknown_parameter":
        message = f"Unrecognized request argument supplied: {param}"
    else:
        message = f"Invalid value for '{param}': {first['msg']}" if param else str(first["msg"])
    details: dict[str, Any] = {"more_errors": len(errors) - 1} if len(errors) > 1 else {}
    return InvalidRequestError(message, param=param, code=code, details=details)


def to_gg_error(exc: BaseException) -> GGError:
    """maps anything raised below the api into a client-safe error; unknown exceptions are logged"""
    if isinstance(exc, GGError):
        return exc
    if isinstance(exc, ProviderError):
        return UpstreamError("The upstream provider returned an error.")
    log.error("request.internal_error", exc_info=exc)
    return InternalError("The server had an error while processing your request.")


def error_headers(error: GGError) -> dict[str, str]:
    headers = error.response_headers()
    if isinstance(error, AuthenticationError):
        headers["www-authenticate"] = 'Bearer realm="gg"'
    if isinstance(error, (PayloadTooLargeError, RequestTimeoutError)):
        headers["connection"] = "close"
    return headers


def render_error(
    request: Request, error: GGError, *, to_body: Callable[[GGError], Any] | None = None
) -> Response:
    headers = error_headers(error)
    body = to_body(error) if to_body is not None else error.to_body()
    ctx = getattr(request.state, "ctx", None)
    if ctx is None:
        log.info("request.rejected", status=error.status, code=error.code, param=error.param)
        return json_response(body, status=error.status, headers=headers)
    if ctx.outcome is None:
        ctx.outcome = error_outcome(error)
    headers = {**build_response_headers(ctx, streaming=False), **headers}
    return FinalizingJSONResponse(
        body, status=error.status, headers=headers, ctx=ctx, services=get_services(request)
    )


async def _gg_error_handler(request: Request, exc: Exception) -> Response:
    return render_error(request, to_gg_error(exc))


async def _validation_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, RequestValidationError)
    return render_error(request, validation_to_error(exc))


async def _http_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, HTTPException)
    if exc.status_code in (404, 405):
        message = f"Invalid URL ({request.method} {request.url.path})"
    else:
        message = str(exc.detail)
    headers: Mapping[str, str] = {"x-should-retry": "false", **(exc.headers or {})}
    return json_response(error_body(message), status=exc.status_code, headers=headers)


async def _unhandled_handler(request: Request, exc: Exception) -> Response:
    # runs in starlette's outermost middleware, outside ours, so it adds the request headers itself
    error = to_gg_error(exc)
    response = render_error(request, error)
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        response.headers["x-request-id"] = request_id
    for name, value in SECURITY_HEADERS:
        response.headers[name] = value
    return response


def install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(GGError, _gg_error_handler)
    app.add_exception_handler(ProviderError, _gg_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(HTTPException, _http_handler)
    app.add_exception_handler(Exception, _unhandled_handler)
