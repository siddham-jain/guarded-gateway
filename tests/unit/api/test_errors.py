from typing import Any

import pytest
from pydantic import ValidationError

from gg.api.errors import error_headers, to_gg_error, validation_to_error
from gg.core.errors import (
    AuthenticationError,
    GGError,
    InternalError,
    PayloadTooLargeError,
    ProviderError,
    RateLimitedError,
    RequestTimeoutError,
    UpstreamError,
)
from gg.core.schema import ChatRequest

USER: dict[str, Any] = {"role": "user", "content": "private words"}


def _error_for(payload: dict[str, Any]) -> GGError:
    with pytest.raises(ValidationError) as info:
        ChatRequest.model_validate(payload)
    return validation_to_error(info.value, payload)


@pytest.mark.parametrize(
    ("payload", "param", "code"),
    [
        ({"messages": [USER]}, "model", "missing_required_parameter"),
        ({"model": "m"}, "messages", "missing_required_parameter"),
        ({"model": "m", "messages": [{"role": "user", "content": 5}]}, "messages[0].content", "invalid_type"),
        (
            {"model": "m", "messages": [{"role": "bogus", "content": "x"}]},
            "messages[0].role",
            "invalid_value",
        ),
        ({"model": "m", "messages": [{"role": "user"}]}, "messages[0]", "invalid_value"),
        (
            {
                "model": "m",
                "messages": [USER],
                "tools": [{"type": "function", "function": {"name": "bad name"}}],
            },
            "tools[0].function.name",
            "invalid_value",
        ),
        ({"model": "m", "messages": [USER], "gg": {"nope": 1}}, "gg.nope", "unknown_parameter"),
        (
            {"model": "m", "messages": [USER], "gg": {"guardrails": {"enable": 5}}},
            "gg.guardrails.enable",
            "invalid_type",
        ),
        ({"model": "m", "messages": [USER], "temperature": "hot"}, "temperature", "invalid_type"),
        ({"model": "m", "messages": [USER], "temperature": 5}, "temperature", "invalid_value"),
        ({"model": "m", "messages": [USER], "tool_choice": 5}, "tool_choice", "invalid_value"),
        ({"model": "m", "messages": "nope"}, "messages", "invalid_type"),
    ],
)
def test_validation_param_and_code(payload: dict[str, Any], param: str, code: str) -> None:
    error = _error_for(payload)
    assert error.status == 400
    assert error.param == param
    assert error.code == code
    assert "private words" not in error.message


def test_missing_message_text() -> None:
    error = _error_for({"messages": [USER]})
    assert error.message == "Missing required parameter: 'model'."


def test_more_errors_counted() -> None:
    error = _error_for({"model": "m", "messages": [USER], "temperature": 9, "top_p": 9})
    assert error.details["more_errors"] == 1


def test_root_validator_error_has_no_param() -> None:
    error = _error_for({"model": "m", "messages": [USER], "metadata": {str(i): "v" for i in range(20)}})
    assert error.param is None
    assert "16 pairs" in error.message


def test_to_gg_error_passthrough_and_mapping() -> None:
    original = RateLimitedError("x")
    assert to_gg_error(original) is original
    assert isinstance(
        to_gg_error(ProviderError("fallback", provider="p", status=500, message="raw body")), UpstreamError
    )
    internal = to_gg_error(RuntimeError("stack detail"))
    assert isinstance(internal, InternalError)
    assert "stack detail" not in internal.message


def test_error_headers() -> None:
    auth = error_headers(AuthenticationError("no"))
    assert auth["www-authenticate"] == 'Bearer realm="gg"'
    assert auth["x-should-retry"] == "false"
    assert error_headers(PayloadTooLargeError("big"))["connection"] == "close"
    assert error_headers(RequestTimeoutError("slow"))["connection"] == "close"
    limited = error_headers(RateLimitedError("x", retry_after_s=2.4))
    assert limited == {"x-should-retry": "true", "retry-after": "2"}
