"""openai-compatible mock upstream for overhead, chaos and contract tests.

run: uvicorn mock_upstream.app:app --port 9000
knobs come from `x-mock-<name>` headers, then `?<name>=` query params, then MOCK_<NAME> env vars:
  ttft_ms, itl_ms (non-stream replies wait ttft + itl * tokens), output_tokens, text,
  error_status, error_rate, error_type,
  fail_after_chunks (drop the stream without a terminal event), error_after_chunks (in-band error event),
  stall_ms (sleep before the first token, to trigger client timeouts), usage (0 disables the usage chunk)
GET /_stats: per-process chat counts (requests, in_flight, peak_in_flight, cancelled); POST /_stats resets
"""

import asyncio
import os
import random
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any

import orjson
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI(title="GG mock upstream")

_ERROR_TYPES = {
    400: ("invalid_request_error", "invalid_request"),
    401: ("invalid_request_error", "invalid_api_key"),
    404: ("invalid_request_error", "model_not_found"),
    429: ("rate_limit_error", "rate_limit_exceeded"),
    500: ("server_error", "internal_error"),
    503: ("service_unavailable_error", "server_is_overloaded"),
    529: ("overloaded_error", "overloaded"),
}


_STATS = {"requests": 0, "in_flight": 0, "peak_in_flight": 0, "cancelled": 0}


def _enter() -> None:
    _STATS["requests"] += 1
    _STATS["in_flight"] += 1
    _STATS["peak_in_flight"] = max(_STATS["peak_in_flight"], _STATS["in_flight"])


async def _tracked(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    try:
        async for chunk in chunks:
            yield chunk
    except asyncio.CancelledError:
        _STATS["cancelled"] += 1
        raise
    finally:
        _STATS["in_flight"] -= 1


class Knobs:
    def __init__(self, headers: Mapping[str, str], query: Mapping[str, str]) -> None:
        self._headers = {k.lower(): v for k, v in headers.items()}
        self._query = query

    def raw(self, name: str) -> str | None:
        return (
            self._headers.get(f"x-mock-{name.replace('_', '-')}")
            or self._query.get(name)
            or os.environ.get(f"MOCK_{name.upper()}")
        )

    def num(self, name: str, default: float) -> float:
        value = self.raw(name)
        try:
            return float(value) if value is not None else default
        except ValueError:
            return default

    def opt_int(self, name: str) -> int | None:
        value = self.raw(name)
        return int(value) if value is not None and value.lstrip("-").isdigit() else None


def _error_body(status: int, kind: str | None = None) -> dict[str, Any]:
    type_, code = _ERROR_TYPES.get(status, ("server_error", "mock_error"))
    return {"error": {"message": f"mock {status}", "type": kind or type_, "param": None, "code": code}}


def _tokens(body: dict[str, Any], knobs: Knobs) -> list[str]:
    text = knobs.raw("text")
    if text is None:
        last_user = next((m for m in reversed(body.get("messages") or []) if m.get("role") == "user"), None)
        content = (last_user or {}).get("content")
        text = content if isinstance(content, str) else "mock reply"
    words = text.split(" ") or [""]
    count = int(knobs.num("output_tokens", len(words)))
    out = [words[i % len(words)] for i in range(max(count, 1))]
    return [w if i == 0 else " " + w for i, w in enumerate(out)]


def _usage(body: dict[str, Any], completion: int) -> dict[str, Any]:
    prompt = max(1, len(orjson.dumps(body.get("messages") or [])) // 4)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


def _sse(obj: Any) -> bytes:
    return b"data: " + orjson.dumps(obj) + b"\n\n"


async def _stream(body: dict[str, Any], knobs: Knobs, tokens: list[str]) -> AsyncIterator[bytes]:
    base = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": body.get("model"),
    }
    fail_after = knobs.opt_int("fail_after_chunks")
    error_after = knobs.opt_int("error_after_chunks")
    itl = knobs.num("itl_ms", 0) / 1000
    await asyncio.sleep(knobs.num("ttft_ms", 0) / 1000 + knobs.num("stall_ms", 0) / 1000)
    yield b": mock keep-alive\n\n"
    for i, token in enumerate(tokens):
        if fail_after is not None and i >= fail_after:
            return
        if error_after is not None and i >= error_after:
            yield _sse(_error_body(500))
            return
        delta: dict[str, Any] = {"content": token}
        if i == 0:
            delta["role"] = "assistant"
        yield _sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
        if itl:
            await asyncio.sleep(itl)
    yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    if include_usage and knobs.raw("usage") != "0":
        yield _sse({**base, "choices": [], "usage": _usage(body, len(tokens))})
    yield b"data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    _enter()
    streaming = False
    try:
        body: dict[str, Any] = orjson.loads(await request.body())
        knobs = Knobs(request.headers, request.query_params)
        status = knobs.opt_int("error_status")
        rate = knobs.num("error_rate", 0)
        if status is not None and (rate <= 0 or random.random() < rate):  # noqa: S311
            headers = {"retry-after": "1"} if status == 429 else {}
            return JSONResponse(
                _error_body(status, knobs.raw("error_type")), status_code=status, headers=headers
            )
        tokens = _tokens(body, knobs)
        if body.get("stream"):
            streaming = True
            return StreamingResponse(_tracked(_stream(body, knobs, tokens)), media_type="text/event-stream")
        # a non-stream reply arrives once the whole generation is done; same sleeps as the stream, same drift
        await asyncio.sleep(knobs.num("ttft_ms", 0) / 1000)
        itl = knobs.num("itl_ms", 0) / 1000
        for _ in tokens if itl else ():
            await asyncio.sleep(itl)
        return _completion(body, tokens)
    except asyncio.CancelledError:
        _STATS["cancelled"] += 1
        raise
    finally:
        if not streaming:
            _STATS["in_flight"] -= 1


def _completion(body: dict[str, Any], tokens: list[str]) -> JSONResponse:
    return JSONResponse(
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(tokens)},
                    "finish_reason": "stop",
                }
            ],
            "usage": _usage(body, len(tokens)),
        }
    )


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": "gpt-mock", "object": "model", "created": 0, "owned_by": "mock"}],
    }


@app.get("/_stats")
async def stats() -> dict[str, int]:
    return dict(_STATS)


@app.post("/_stats")
async def reset_stats() -> dict[str, int]:
    for name in _STATS:
        _STATS[name] = _STATS["in_flight"] if name in ("in_flight", "peak_in_flight") else 0
    return dict(_STATS)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
