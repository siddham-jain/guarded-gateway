from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from gg.core.errors import ProviderErrorKind
from gg.core.schema import StrictModel
from gg.providers.meta import DEFAULT_RATE_LIMIT_HEADERS

type ToolChoiceMode = Literal["auto", "none", "required", "named"]


class ErrorRule(StrictModel):
    kind: ProviderErrorKind
    code: str | None = None


class ToolChoiceQuirk(StrictModel):
    supported: tuple[ToolChoiceMode, ...] = ("auto", "none", "required", "named")
    # "none" outside `supported`: drop the tools instead; "auto" outside it: omit the field


class ResponseFormatQuirk(StrictModel):
    json_object: Literal["native", "unsupported"] = "native"
    json_schema: Literal["native", "as_json_object", "unsupported"] = "native"


class RequestQuirks(StrictModel):
    model_prefix: str = ""
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    role_map: dict[str, str] = Field(default_factory=dict)
    renames: dict[str, str] = Field(default_factory=dict)
    drop: tuple[str, ...] = ()
    allow_only: tuple[str, ...] | None = None
    reject: tuple[str, ...] = ()
    clamp: dict[str, tuple[float, float]] = Field(default_factory=dict)
    stop_max: int | None = None
    defaults: dict[str, Any] = Field(default_factory=dict)
    tool_choice: ToolChoiceQuirk = Field(default_factory=ToolChoiceQuirk)
    response_format: ResponseFormatQuirk = Field(default_factory=ResponseFormatQuirk)
    strict_tools: bool = True
    parallel_tool_calls: bool = True
    stream_usage: Literal["inject", "automatic", "unsupported"] = "inject"
    stream_options_extra: dict[str, Any] = Field(default_factory=dict)
    passthrough_unknown: bool = False
    user_field: str | None = None
    send_metadata: bool = False
    store_false: bool = False
    prompt_cache_key: bool = False
    strip_message_fields: tuple[str, ...] = ("reasoning_content", "reasoning", "extra_content")
    tool_call_id_format: Literal["any", "alnum9"] = "any"
    extra_body: dict[str, Any] = Field(default_factory=dict)


class ReasoningQuirks(StrictModel):
    field: Literal["reasoning_content", "reasoning", "content_array", "think_tags", "none"] = (
        "reasoning_content"
    )
    control: Literal["reasoning_effort", "none"] = "reasoning_effort"
    param: str = "reasoning_effort"
    # effort -> provider value (str), a body patch (dict) or null to omit
    effort_map: dict[str, Any] = Field(default_factory=dict)
    history: Literal["optional", "required", "required_with_tools", "forbidden"] = "optional"
    history_field: str = "reasoning_content"
    expose: bool = False


class BodyStatusCheck(StrictModel):
    path: str
    ok: Any = 0
    message_path: str | None = None


class ResponseQuirks(StrictModel):
    usage_roots: tuple[str, ...] = ("usage", "x_groq.usage", "choices.0.usage")
    usage_map: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    cost_path: str | None = None
    cost_scale: float = 1.0
    finish_reason_map: dict[str, str] = Field(default_factory=dict)
    finish_reason_errors: tuple[str, ...] = ("error",)
    passthrough_fields: tuple[str, ...] = ()
    check_body_status: BodyStatusCheck | None = None


class ErrorQuirks(StrictModel):
    code_map: dict[str, ErrorRule] = Field(default_factory=dict)
    status_map: dict[int, ErrorRule] = Field(default_factory=dict)
    transport_kind: Literal["retryable", "fallback"] = "retryable"


class QuirkProfile(StrictModel):
    """wire-translation differences of one openai-compatible host; model capabilities live in models.yaml"""

    extra_headers: dict[str, str] = Field(default_factory=dict)
    client_request_id_header: str | None = None
    request_id_headers: tuple[str, ...] = ("x-request-id", "request-id", "x-groq-id")
    rate_limit_headers: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_RATE_LIMIT_HEADERS))
    forbid_base_url_patterns: tuple[str, ...] = ()
    request: RequestQuirks = Field(default_factory=RequestQuirks)
    reasoning: ReasoningQuirks = Field(default_factory=ReasoningQuirks)
    response: ResponseQuirks = Field(default_factory=ResponseQuirks)
    errors: ErrorQuirks = Field(default_factory=ErrorQuirks)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """maps merge recursively, everything else (lists included) replaces"""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = out.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(current, value)  # pyright: ignore[reportUnknownArgumentType]
        else:
            out[key] = value
    return out
