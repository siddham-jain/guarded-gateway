from decimal import Decimal
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    PrivateAttr,
    StringConstraints,
    Tag,
    model_validator,
)


class WireModel(BaseModel):
    # wire-facing: unknown fields survive, prompts never echoed in validation errors
    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        populate_by_name=True,
        hide_input_in_errors=True,
        ser_json_inf_nan="null",
    )


class StrictModel(BaseModel):
    # gg-owned shapes (extensions, config): typos are errors
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


type Role = Literal["system", "developer", "user", "assistant", "tool"]
type FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]
type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

FINISH_REASONS: frozenset[str] = frozenset({"stop", "length", "tool_calls", "content_filter"})


class TextPart(WireModel):
    type: Literal["text"]
    text: str


class ImageURL(WireModel):
    url: str
    detail: Literal["auto", "low", "high", "original"] | None = None


class ImagePart(WireModel):
    type: Literal["image_url"]
    image_url: ImageURL


class InputAudioPart(WireModel):
    type: Literal["input_audio"]
    input_audio: dict[str, str]


class FilePart(WireModel):
    type: Literal["file"]
    file: dict[str, str]


class RefusalPart(WireModel):
    type: Literal["refusal"]
    refusal: str


class UnknownPart(WireModel):
    type: str


_KNOWN_PARTS = frozenset({"text", "image_url", "input_audio", "file", "refusal"})


def _part_tag(value: Any) -> str:
    tag = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    return tag if tag in _KNOWN_PARTS else "unknown"  # pyright: ignore[reportReturnType]


type ContentPart = Annotated[
    Annotated[TextPart, Tag("text")]
    | Annotated[ImagePart, Tag("image_url")]
    | Annotated[InputAudioPart, Tag("input_audio")]
    | Annotated[FilePart, Tag("file")]
    | Annotated[RefusalPart, Tag("refusal")]
    | Annotated[UnknownPart, Tag("unknown")],
    Discriminator(_part_tag),
]


class FunctionCall(WireModel):
    name: str
    arguments: str


class ToolCall(WireModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(WireModel):
    role: Role
    content: str | tuple[ContentPart, ...] | None = None
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] | None = None
    tool_call_id: str | None = None
    refusal: str | None = None

    @model_validator(mode="after")
    def _check_role_fields(self) -> Self:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role in ("system", "developer", "user") and self.content is None:
            raise ValueError(f"{self.role} messages require content")
        if self.role == "assistant" and (
            self.content is None and not self.tool_calls and self.refusal is None
        ):
            raise ValueError("assistant messages require content, tool_calls or refusal")
        return self

    def text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(p.text for p in self.content if isinstance(p, TextPart))


class FunctionDef(WireModel):
    name: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")]
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class FunctionTool(WireModel):
    type: Literal["function"]
    function: FunctionDef


class UnknownTool(WireModel):
    type: str


def _tool_tag(value: Any) -> str:
    tag = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    return "function" if tag == "function" else "unknown"


type Tool = Annotated[
    Annotated[FunctionTool, Tag("function")] | Annotated[UnknownTool, Tag("unknown")],
    Discriminator(_tool_tag),
]


class NamedToolChoice(WireModel):
    type: Literal["function"]
    function: dict[str, str]


type ToolChoice = Literal["none", "auto", "required"] | NamedToolChoice | dict[str, Any]


class JsonSchemaSpec(WireModel):
    name: str
    description: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None


class ResponseFormat(WireModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaSpec | None = None


class StreamOptions(WireModel):
    include_usage: bool | None = None
    include_obfuscation: bool | None = None


type _Tag = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._:-]{1,64}$")]


class GGGuardrailsExt(StrictModel):
    enable: tuple[str, ...] = ()
    enforce_shadowed: bool = False
    output_stream_mode: Literal["buffer"] | None = None


class GGExtensions(StrictModel):
    route_threshold: Annotated[float, Field(ge=0, le=1)] | None = None
    fallback: bool = True
    cache: Literal["default", "off", "no_store", "refresh"] = "default"
    cache_ttl_s: Annotated[int, Field(ge=1, le=604_800)] | None = None
    semantic_cache: bool = True
    guardrails: GGGuardrailsExt | None = None
    max_cost_usd: Annotated[Decimal, Field(gt=0, le=100)] | None = None
    cache_threshold: Annotated[float, Field(ge=0, le=1)] | None = None
    session_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._:-]{1,128}$")] | None = None
    tags: Annotated[tuple[_Tag, ...], Field(max_length=10)] = ()
    include_metadata: bool = False


class ChatRequest(WireModel):
    model: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    messages: Annotated[tuple[Message, ...], Field(min_length=1)]
    tools: tuple[Tool, ...] | None = None
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    response_format: ResponseFormat | None = None
    max_completion_tokens: Annotated[int, Field(ge=1)] | None = None
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    top_p: Annotated[float, Field(ge=0, le=1)] | None = None
    stop: Annotated[tuple[str, ...], Field(max_length=4)] | None = None
    n: Annotated[int, Field(ge=1, le=16)] = 1
    seed: int | None = None
    presence_penalty: Annotated[float, Field(ge=-2, le=2)] | None = None
    frequency_penalty: Annotated[float, Field(ge=-2, le=2)] | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    top_logprobs: Annotated[int, Field(ge=0, le=20)] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    reasoning_effort: ReasoningEffort | None = None
    user: str | None = None
    metadata: dict[str, str] | None = None
    gg: GGExtensions | None = None

    @model_validator(mode="after")
    def _check_metadata(self) -> Self:
        if self.metadata is not None:
            if len(self.metadata) > 16:
                raise ValueError("metadata allows at most 16 pairs")
            for k, v in self.metadata.items():
                if len(k) > 64 or len(v) > 512:
                    raise ValueError("metadata keys must be <= 64 chars and values <= 512")
        return self

    def upstream_payload(self) -> dict[str, Any]:
        # exclude_unset keeps explicit nulls and drops what the client never sent
        return self.model_dump(mode="json", by_alias=True, exclude_unset=True, exclude={"gg"})

    def last_user_text(self) -> str | None:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.text()
        return None

    def has_tools(self) -> bool:
        return bool(self.tools)

    def wants_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.include_usage)


class PromptTokensDetails(WireModel):
    cached_tokens: int = 0
    cache_write_tokens: int = 0


class CompletionTokensDetails(WireModel):
    reasoning_tokens: int = 0


class Usage(WireModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


class AssistantMessage(WireModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    refusal: str | None = None
    tool_calls: tuple[ToolCall, ...] | None = None


class Choice(WireModel):
    index: int
    message: AssistantMessage
    finish_reason: FinishReason | None
    logprobs: dict[str, Any] | None = None


class ChatResponse(WireModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: tuple[Choice, ...]
    usage: Usage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None
    # provider metadata (gg.providers.meta.ResponseMeta); never serialised
    _gg_meta: Any = PrivateAttr(default=None)

    @property
    def gg_meta(self) -> Any:
        return self._gg_meta

    def with_meta(self, meta: Any) -> Self:
        self._gg_meta = meta
        return self


class FunctionCallDelta(WireModel):
    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(WireModel):
    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: FunctionCallDelta | None = None


class Delta(WireModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    refusal: str | None = None
    tool_calls: tuple[ToolCallDelta, ...] | None = None


class ChunkChoice(WireModel):
    index: int
    delta: Delta
    finish_reason: FinishReason | None = None
    logprobs: dict[str, Any] | None = None


class ChatChunk(WireModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: tuple[ChunkChoice, ...]
    usage: Usage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None
    _gg_meta: Any = PrivateAttr(default=None)

    @property
    def gg_meta(self) -> Any:
        return self._gg_meta

    def with_meta(self, meta: Any) -> Self:
        self._gg_meta = meta
        return self

    def has_content(self) -> bool:
        # the content commit point: first chunk carrying text, refusal or a tool call
        for choice in self.choices:
            d = choice.delta
            if d.content or d.refusal or d.tool_calls:
                return True
        return False


def to_wire(model: BaseModel) -> dict[str, Any]:
    # responses and chunks: defaults like object="chat.completion" must be emitted, nulls need not be
    return model.model_dump(mode="json", by_alias=True, exclude_none=True)
