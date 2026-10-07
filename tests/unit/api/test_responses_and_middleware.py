from typing import Any

import pytest
from starlette.types import Message

from gg.api.middleware import BodyLimitMiddleware
from gg.api.responses import FinalizingJSONResponse, error_outcome, finalize_request
from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.errors import GuardrailBlockedError, InternalError, ServiceUnavailableError, UpstreamError
from tests.conftest import make_ctx
from tests.unit.api.fakes import make_services


async def test_finalize_runs_once_with_hook(clock: FakeClock) -> None:
    seen: list[tuple[str, int]] = []
    order: list[str] = []

    async def hook(ctx: RequestContext, status: int) -> None:
        order.append("hook")
        seen.append((ctx.request_id, status))

    services = make_services(on_request_complete=hook)
    ctx = make_ctx(clock)

    async def limits() -> None:
        order.append("limits")

    ctx.finalizers.defer("limits", limits, 10)
    await finalize_request(ctx, 200, services)
    await finalize_request(ctx, 200, services)
    assert seen == [("req_test", 200)]
    assert order == ["limits", "hook"]
    assert ctx.outcome == "completed"
    assert "completed" in ctx.timings.marks


async def test_finalize_survives_failing_finalizer(clock: FakeClock) -> None:
    ctx = make_ctx(clock)
    ran: list[str] = []

    async def broken() -> None:
        raise RuntimeError("boom")

    async def after() -> None:
        ran.append("after")

    ctx.finalizers.defer("broken", broken, 10)
    ctx.finalizers.defer("after", after, 20)
    await finalize_request(ctx, 500, make_services())
    assert ran == ["after"]
    assert ctx.outcome == "rejected"


async def test_json_response_finalizes_even_when_send_fails(clock: FakeClock) -> None:
    ctx = make_ctx(clock)
    response = FinalizingJSONResponse({"ok": True}, ctx=ctx, services=make_services())
    assert response.headers["cache-control"] == "no-store"
    assert response.body == b'{"ok":true}'

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        raise OSError("gone")

    with pytest.raises(OSError, match="gone"):
        await response({"type": "http"}, receive, send)
    assert ctx.outcome == "completed"


def test_error_outcome() -> None:
    assert error_outcome(GuardrailBlockedError("x")) == "rejected"
    assert error_outcome(UpstreamError("x")) == "upstream_error"
    assert error_outcome(ServiceUnavailableError("x")) == "upstream_error"
    assert error_outcome(InternalError("x")) == "internal_error"


async def test_body_limit_content_length_precheck() -> None:
    called: list[bool] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        called.append(True)

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {"type": "http", "headers": [(b"content-length", b"999")]}
    await BodyLimitMiddleware(app, max_bytes=10)(scope, receive, send)
    assert called == []
    assert sent[0]["status"] == 413
    assert b"request_too_large" in sent[1]["body"]
