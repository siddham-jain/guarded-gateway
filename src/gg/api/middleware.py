import structlog
from fastapi import FastAPI
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from gg.api.anthropic_ingress.response import error_body
from gg.api.deps import ApiServices
from gg.api.errors import error_headers
from gg.api.headers import EXPOSED_HEADERS, SECURITY_HEADERS
from gg.api.responses import json_response
from gg.auth.failures import AuthFailureLimiter, InMemoryAuthFailureLimiter
from gg.auth.keys import parse_bearer
from gg.config.settings import Settings
from gg.core.clock import SystemClock
from gg.core.context import StageTimings
from gg.core.errors import (
    AuthenticationError,
    GGError,
    PayloadTooLargeError,
    RateLimitedError,
    ServiceUnavailableError,
)
from gg.core.ids import new_request_id, new_trace_id

log = structlog.get_logger("gg.api")

_MAX_CLIENT_REQUEST_ID = 128
ADMIN_TAG = "admin"


def _services(scope: Scope) -> ApiServices:
    services = scope["app"].state.services
    assert isinstance(services, ApiServices)
    return services


def _client_request_id(headers: Headers) -> str | None:
    value = headers.get("x-client-request-id")
    if value and value.isascii() and value.isprintable() and len(value) <= _MAX_CLIENT_REQUEST_ID:
        return value
    return None


MESSAGES_PATH = "/v1/messages"


def _is_api_path(scope: Scope) -> bool:
    return str(scope.get("path", "")).startswith("/v1/")


def _client_ip(scope: Scope) -> str:
    # uvicorn has already applied x-forwarded-for from trusted proxies (FORWARDED_ALLOW_IPS)
    client = scope.get("client")
    return str(client[0]) if client else "unknown"


async def _send_error(error: GGError, scope: Scope, receive: Receive, send: Send) -> None:
    # anthropic-style clients on /v1/messages only parse their own error shape
    body = error_body(error, request_id=None) if scope.get("path") == MESSAGES_PATH else error.to_body()
    response = json_response(body, status=error.status, headers=error_headers(error))
    await response(scope, receive, send)


class RequestContextMiddleware:
    """request and trace ids, received mark, log context, in-flight count, id and security headers"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        services = _services(scope)
        request_id = new_request_id()
        # always fresh: an incoming traceparent from a public client is never trusted
        trace_id = new_trace_id()
        timings = StageTimings(services.clock)
        timings.mark("received")
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["trace_id"] = trace_id
        state["timings"] = timings

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["x-request-id"] = request_id
                headers["x-gg-trace-id"] = trace_id
                for name, value in SECURITY_HEADERS:
                    headers.setdefault(name, value)
            await send(message)

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            trace_id=trace_id,
            client_request_id=_client_request_id(Headers(scope=scope)),
        )
        services.state.request_started()
        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            services.state.request_finished()
            structlog.contextvars.clear_contextvars()


class BodyLimitMiddleware:
    """global body cap: content-length precheck plus a counting receive wrapper"""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        length = Headers(scope=scope).get("content-length")
        if length is not None and length.isdigit() and int(length) > self.max_bytes:
            await _send_error(self._error(), scope, receive, send)
            return
        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise self._error()
            return message

        await self.app(scope, counting_receive, send)

    def _error(self) -> PayloadTooLargeError:
        return PayloadTooLargeError(f"Request body exceeds {self.max_bytes} bytes.")


class AuthFailureLimitMiddleware:
    """counts 401s on /v1/* per client ip; over the limit the ip gets 429 before any key lookup"""

    def __init__(self, app: ASGIApp, *, limiter: AuthFailureLimiter) -> None:
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _is_api_path(scope):
            await self.app(scope, receive, send)
            return
        ip = _client_ip(scope)
        retry_after = await self.limiter.blocked_for(ip)
        if retry_after is not None:
            error = RateLimitedError(
                "Too many failed authentication attempts; try again later.",
                code="too_many_auth_failures",
                retry_after_s=retry_after,
            )
            await _send_error(error, scope, receive, send)
            return
        statuses: list[int] = []

        async def watch_status(message: Message) -> None:
            if message["type"] == "http.response.start":
                statuses.append(message["status"])
            await send(message)

        await self.app(scope, receive, watch_status)
        if statuses == [401]:
            await self.limiter.record_failure(ip)


class MaintenanceMiddleware:
    """GG_MAINTENANCE kill switch: /v1/* answers 503 unless the key is tagged admin"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or not _is_api_path(scope)
            or not _services(scope).settings.maintenance
            or await self._is_admin(scope)
        ):
            await self.app(scope, receive, send)
            return
        error = ServiceUnavailableError(
            "GG is in maintenance mode; try again later.", code="maintenance", retry_after_s=300
        )
        await _send_error(error, scope, receive, send)

    async def _is_admin(self, scope: Scope) -> bool:
        try:
            token = parse_bearer(Headers(scope=scope).getlist("authorization"))
            key = await _services(scope).keys.resolve(token)
        except AuthenticationError:
            return False
        return ADMIN_TAG in key.tags


def install_middleware(
    app: FastAPI, settings: Settings, *, auth_failures: AuthFailureLimiter | None = None
) -> None:
    app.add_middleware(MaintenanceMiddleware)
    limits = settings.auth_failures
    if limits.enabled:
        limiter = auth_failures or InMemoryAuthFailureLimiter(
            SystemClock(), max_failures=limits.max_failures, window_s=limits.window_s
        )
        app.add_middleware(AuthFailureLimitMiddleware, limiter=limiter)
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.server.max_body_bytes)
    origins = settings.server.cors_origins
    if origins:
        if "*" in origins:
            log.warning("cors.wildcard_origin")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            allow_credentials=False,
            expose_headers=list(EXPOSED_HEADERS),
            max_age=600,
        )
    app.add_middleware(RequestContextMiddleware)
