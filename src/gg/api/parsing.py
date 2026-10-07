import asyncio
from typing import Any

import orjson
from pydantic import ValidationError
from starlette.requests import Request

from gg.api.errors import validation_to_error
from gg.core.errors import (
    InvalidRequestError,
    PayloadTooLargeError,
    RequestTimeoutError,
    UnsupportedMediaTypeError,
)
from gg.core.normalize import Normalization, normalize_request_dict
from gg.core.schema import ChatRequest


def _too_large(limit: int) -> PayloadTooLargeError:
    return PayloadTooLargeError(f"Request body exceeds {limit} bytes.")


async def read_body_limited(request: Request, *, limit: int, timeout_s: float) -> bytes:
    if not request.headers.get("content-type", "").lower().startswith("application/json"):
        raise UnsupportedMediaTypeError("Content-Type must be application/json.")
    if request.headers.get("content-encoding", "identity").lower() != "identity":
        raise UnsupportedMediaTypeError(
            "Compressed request bodies are not supported.", code="unsupported_content_encoding"
        )
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > limit:
        raise _too_large(limit)
    buf = bytearray()
    try:
        async with asyncio.timeout(timeout_s):
            async for part in request.stream():
                buf += part
                if len(buf) > limit:
                    raise _too_large(limit)
    except TimeoutError:
        raise RequestTimeoutError("Timed out reading the request body.") from None
    return bytes(buf)


def load_json_object(body: bytes) -> dict[str, Any]:
    try:
        raw: Any = orjson.loads(body)
    except orjson.JSONDecodeError:
        raise InvalidRequestError(
            "We could not parse the JSON body of your request.", code="invalid_json"
        ) from None
    if not isinstance(raw, dict):
        raise InvalidRequestError("Request body must be a JSON object.", code="invalid_json")
    return raw  # pyright: ignore[reportUnknownVariableType]


def chat_request_from_dict(raw: dict[str, Any]) -> tuple[ChatRequest, tuple[Normalization, ...]]:
    normalized = normalize_request_dict(raw)
    try:
        return ChatRequest.model_validate(normalized.payload), normalized.applied
    except ValidationError as exc:
        raise validation_to_error(exc, normalized.payload) from None


def parse_chat_request(body: bytes) -> tuple[ChatRequest, tuple[Normalization, ...]]:
    return chat_request_from_dict(load_json_object(body))
