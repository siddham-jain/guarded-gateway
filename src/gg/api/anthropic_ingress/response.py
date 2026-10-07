from collections.abc import Mapping
from typing import Any

import orjson

from gg.core.errors import GGError, UpstreamError
from gg.core.schema import ChatResponse, ToolCall, Usage

STOP_REASONS: Mapping[str | None, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
    None: "end_turn",
}

ERROR_TYPES: Mapping[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    402: "billing_error",
    403: "permission_error",
    404: "not_found_error",
    408: "timeout_error",
    413: "request_too_large",
    415: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "timeout_error",
}


def message_id(request_id: str) -> str:
    return "msg_" + request_id.removeprefix("req_")


def stop_reason(finish_reason: str | None) -> str:
    return STOP_REASONS.get(finish_reason, "end_turn")


def usage_block(usage: Usage | None) -> dict[str, int]:
    """openai prompt_tokens include cache reads and writes; anthropic input_tokens exclude both"""
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    details = usage.prompt_tokens_details
    cached = details.cached_tokens if details is not None else 0
    written = details.cache_write_tokens if details is not None else 0
    out = {
        "input_tokens": max(0, usage.prompt_tokens - cached - written),
        "output_tokens": usage.completion_tokens,
    }
    if cached or written:
        out["cache_read_input_tokens"] = cached
        out["cache_creation_input_tokens"] = written
    return out


def tool_input(arguments: str) -> dict[str, Any]:
    try:
        parsed: Any = orjson.loads(arguments or "{}")
    except orjson.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        raise UpstreamError("The upstream provider returned malformed tool arguments.")
    return parsed  # pyright: ignore[reportUnknownVariableType]


def tool_use_block(call: ToolCall) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": call.id,
        "name": call.function.name,
        "input": tool_input(call.function.arguments),
    }


def to_message(response: ChatResponse, *, request_id: str) -> dict[str, Any]:
    choice = response.choices[0] if response.choices else None
    content: list[dict[str, Any]] = []
    if choice is not None:
        message = choice.message
        text = message.content or message.refusal
        if text:
            content.append({"type": "text", "text": text})
        content.extend(tool_use_block(call) for call in message.tool_calls or ())
    return {
        "id": message_id(request_id),
        "type": "message",
        "role": "assistant",
        "model": response.model,
        "content": content,
        "stop_reason": stop_reason(choice.finish_reason if choice is not None else None),
        "stop_sequence": None,
        "usage": usage_block(response.usage),
    }


def error_type(error: GGError) -> str:
    if error.status in ERROR_TYPES:
        return ERROR_TYPES[error.status]
    return "api_error" if error.status >= 500 else "invalid_request_error"


def error_payload(error: GGError) -> dict[str, Any]:
    body: dict[str, Any] = {"type": error_type(error), "message": error.message}
    details: dict[str, Any] = {"code": error.code}
    if error.param is not None:
        details["param"] = error.param
    if error.details:
        details["gg"] = error.details
    body["details"] = details
    return {"type": "error", "error": body}


def error_body(error: GGError, *, request_id: str | None) -> dict[str, Any]:
    body = error_payload(error)
    if request_id:
        body["request_id"] = request_id
    return body
