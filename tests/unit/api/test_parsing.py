import asyncio
from collections.abc import Sequence

import pytest
from starlette.requests import Request
from starlette.types import Message

from gg.api.parsing import parse_chat_request, read_body_limited
from gg.core.errors import (
    GGError,
    InvalidRequestError,
    PayloadTooLargeError,
    RequestTimeoutError,
    UnsupportedMediaTypeError,
)


def _request(parts: Sequence[bytes], headers: dict[str, str] | None = None, *, delay_s: float = 0) -> Request:
    headers = {"content-type": "application/json", **(headers or {})}
    remaining = list(parts)
    pulled = {"bytes": 0}

    async def receive() -> Message:
        if delay_s:
            await asyncio.sleep(delay_s)
        body = remaining.pop(0) if remaining else b""
        pulled["bytes"] += len(body)
        return {"type": "http.request", "body": body, "more_body": bool(remaining)}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
        "pulled": pulled,
    }
    return Request(scope, receive)


async def test_reads_whole_body() -> None:
    body = await read_body_limited(_request([b'{"a":', b" 1}"]), limit=100, timeout_s=1)
    assert body == b'{"a": 1}'


async def test_content_length_precheck_reads_nothing() -> None:
    request = _request([b"x" * 10], {"content-length": "1000"})
    with pytest.raises(PayloadTooLargeError):
        await read_body_limited(request, limit=100, timeout_s=1)
    assert request.scope["pulled"]["bytes"] == 0


async def test_chunked_overflow_stops_early() -> None:
    request = _request([b"x" * 64] * 100)
    with pytest.raises(PayloadTooLargeError):
        await read_body_limited(request, limit=200, timeout_s=1)
    assert request.scope["pulled"]["bytes"] <= 200 + 64


async def test_slow_body_times_out() -> None:
    with pytest.raises(RequestTimeoutError):
        await read_body_limited(_request([b"{}"], delay_s=0.2), limit=100, timeout_s=0.02)


@pytest.mark.parametrize(
    ("headers", "code"),
    [
        ({"content-type": "text/plain"}, "unsupported_media_type"),
        ({"content-type": ""}, "unsupported_media_type"),
        ({"content-encoding": "gzip"}, "unsupported_content_encoding"),
    ],
)
async def test_media_checks(headers: dict[str, str], code: str) -> None:
    with pytest.raises(UnsupportedMediaTypeError) as info:
        await read_body_limited(_request([b"{}"], headers), limit=100, timeout_s=1)
    assert info.value.code == code


async def test_charset_suffix_accepted() -> None:
    request = _request([b"{}"], {"content-type": "Application/JSON; charset=utf-8"})
    assert await read_body_limited(request, limit=100, timeout_s=1) == b"{}"


def test_parse_normalizes_legacy_params() -> None:
    request, applied = parse_chat_request(
        b'{"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 5, "stop": "END"}'
    )
    assert request.max_completion_tokens == 5
    assert request.stop == ("END",)
    assert {n.rule for n in applied} == {"max_tokens", "stop_str"}


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b"{nope", "invalid_json"),
        (b"\xff\xfe", "invalid_json"),
        (b'"string"', "invalid_json"),
        (b"[" * 2000 + b"]" * 2000, "invalid_json"),
        (b'{"messages": [{"role": "user", "content": "x"}]}', "missing_required_parameter"),
    ],
)
def test_parse_errors(body: bytes, code: str) -> None:
    with pytest.raises(GGError) as info:
        parse_chat_request(body)
    assert info.value.code == code


def test_parse_legacy_conflict() -> None:
    with pytest.raises(InvalidRequestError) as info:
        parse_chat_request(
            b'{"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 5, '
            b'"max_completion_tokens": 6}'
        )
    assert info.value.param == "max_tokens"
