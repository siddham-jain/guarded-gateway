from typing import Annotated, Any, Literal

import orjson
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, ValidationError

from gg.api.errors import validation_to_error
from gg.core.errors import InvalidRequestError

# fields anthropic accepts that have no canonical meaning; dropped and reported in x-gg-ignored-params
IGNORED_FIELDS = frozenset({"top_k", "service_tier", "inference_geo", "cache_control"})
# features gg cannot serve through the canonical request; rejected instead of silently dropped
UNSUPPORTED_FIELDS = frozenset({"container", "mcp_servers", "context_management", "speed"})
THINKING_EFFORT: tuple[tuple[int, str], ...] = ((4096, "low"), (16384, "medium"))
MAX_STOP_SEQUENCES = 4


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class _Base64Source(_Model):
    type: Literal["base64"]
    media_type: str
    data: str


class _UrlSource(_Model):
    type: Literal["url"]
    url: str


class _TextSource(_Model):
    type: Literal["text"]
    media_type: str = "text/plain"
    data: str


class _OtherSource(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    type: str


def _source_tag(value: Any) -> str:
    tag = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    return tag if tag in ("base64", "url", "text") else "other"  # pyright: ignore[reportReturnType]


type _Source = Annotated[
    Annotated[_Base64Source, Tag("base64")]
    | Annotated[_UrlSource, Tag("url")]
    | Annotated[_TextSource, Tag("text")]
    | Annotated[_OtherSource, Tag("other")],
    Discriminator(_source_tag),
]


class TextBlock(_Model):
    type: Literal["text"]
    text: str
    cache_control: dict[str, Any] | None = None
    citations: Any = None


class ImageBlock(_Model):
    type: Literal["image"]
    source: _Source
    cache_control: dict[str, Any] | None = None


class DocumentBlock(_Model):
    type: Literal["document"]
    source: _Source
    title: str | None = None
    context: str | None = None
    citations: Any = None
    cache_control: dict[str, Any] | None = None


class ToolUseBlock(_Model):
    type: Literal["tool_use"]
    id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    input: dict[str, Any]
    cache_control: dict[str, Any] | None = None


class ToolResultBlock(_Model):
    type: Literal["tool_result"]
    tool_use_id: Annotated[str, Field(min_length=1)]
    content: str | list["Block"] | None = None
    is_error: bool | None = None
    cache_control: dict[str, Any] | None = None


class ThinkingBlock(BaseModel):
    # prior-turn thinking cannot ride the canonical request; the anthropic adapter keeps its own copy
    model_config = ConfigDict(extra="allow", frozen=True)
    type: Literal["thinking", "redacted_thinking"]


class UnknownBlock(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    type: str


_BLOCK_TYPES = frozenset({"text", "image", "document", "tool_use", "tool_result"})


def _block_tag(value: Any) -> str:
    tag = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    if tag in ("thinking", "redacted_thinking"):
        return "thinking"
    return tag if tag in _BLOCK_TYPES else "unknown"  # pyright: ignore[reportReturnType]


type Block = Annotated[
    Annotated[TextBlock, Tag("text")]
    | Annotated[ImageBlock, Tag("image")]
    | Annotated[DocumentBlock, Tag("document")]
    | Annotated[ToolUseBlock, Tag("tool_use")]
    | Annotated[ToolResultBlock, Tag("tool_result")]
    | Annotated[ThinkingBlock, Tag("thinking")]
    | Annotated[UnknownBlock, Tag("unknown")],
    Discriminator(_block_tag),
]
ToolResultBlock.model_rebuild()


class InputMessage(_Model):
    role: Literal["user", "assistant", "system"]
    content: str | list[Block]


class ToolDef(_Model):
    type: str | None = None
    name: Annotated[str, Field(min_length=1)]
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    strict: bool | None = None
    cache_control: dict[str, Any] | None = None
    input_examples: Any = None
    eager_input_streaming: bool | None = None


class ToolChoice(_Model):
    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None
    disable_parallel_tool_use: bool | None = None


class Thinking(_Model):
    type: Literal["enabled", "disabled", "adaptive", "between_tools"]
    budget_tokens: Annotated[int, Field(ge=1)] | None = None
    display: str | None = None


class OutputFormat(_Model):
    type: Literal["json_schema"]
    schema_: dict[str, Any] = Field(alias="schema")


class OutputConfig(_Model):
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    format: OutputFormat | None = None


class Metadata(_Model):
    user_id: Annotated[str, Field(max_length=512)] | None = None


class MessagesRequest(_Model):
    model: Annotated[str, Field(min_length=1, max_length=256)]
    max_tokens: Annotated[int, Field(ge=1)]
    messages: Annotated[list[InputMessage], Field(min_length=1)]
    system: str | list[TextBlock] | None = None
    tools: list[ToolDef] | None = None
    tool_choice: ToolChoice | None = None
    temperature: Annotated[float, Field(ge=0, le=1)] | None = None
    top_p: Annotated[float, Field(ge=0, le=1)] | None = None
    top_k: Annotated[int, Field(ge=0)] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    metadata: Metadata | None = None
    thinking: Thinking | None = None
    output_config: OutputConfig | None = None
    service_tier: str | None = None
    inference_geo: str | None = None
    cache_control: dict[str, Any] | None = None
    gg: dict[str, Any] | None = None


def _bad(message: str, param: str, code: str = "invalid_value") -> InvalidRequestError:
    return InvalidRequestError(message, param=param, code=code)


def _unsupported(what: str, param: str) -> InvalidRequestError:
    return _bad(f"{what} is not supported by this gateway.", param, "unsupported_feature")


class _Translator:
    def __init__(self) -> None:
        self.ignored: set[str] = set()

    def system(self, system: str | list[TextBlock] | None) -> list[dict[str, Any]]:
        if system is None:
            return []
        if isinstance(system, str):
            return [{"role": "system", "content": system}] if system else []
        out: list[dict[str, Any]] = []
        for block in system:
            if block.cache_control is not None:
                self.ignored.add("cache_control")
            if block.text:
                out.append({"role": "system", "content": block.text})
        return out

    def messages(self, messages: list[InputMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, message in enumerate(messages):
            param = f"messages[{i}]"
            if message.role == "assistant":
                assistant = self.assistant(message.content, param)
                if assistant is not None:
                    out.append(assistant)
            elif message.role == "system":
                out.extend(self.system_turn(message.content, param))
            else:
                out.extend(self.user(message.content, param))
        return out

    def system_turn(self, content: str | list[Block], param: str) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"role": "system", "content": content}]
        texts = self.parts(content, f"{param}.content", allow=("text",))
        return [{"role": "system", "content": texts}] if texts else []

    def user(self, content: str | list[Block], param: str) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        out: list[dict[str, Any]] = []
        pending: list[Block] = []

        def flush() -> None:
            if pending:
                parts = self.parts(pending, f"{param}.content", allow=("text", "image", "document"))
                if parts:
                    out.append({"role": "user", "content": parts})
                pending.clear()

        for j, block in enumerate(content):
            if isinstance(block, ToolResultBlock):
                flush()
                out.append(self.tool_result(block, f"{param}.content[{j}]"))
            else:
                pending.append(block)
        flush()
        return out

    def assistant(self, content: str | list[Block], param: str) -> dict[str, Any] | None:
        if isinstance(content, str):
            return {"role": "assistant", "content": content}
        texts: list[str] = []
        calls: list[dict[str, Any]] = []
        for j, block in enumerate(content):
            here = f"{param}.content[{j}]"
            self.note_cache_control(block)
            if isinstance(block, TextBlock):
                texts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": orjson.dumps(block.input).decode()},
                    }
                )
            elif isinstance(block, ThinkingBlock):
                self.ignored.add("thinking_blocks")
            else:
                raise _unsupported(f"Content block type '{block.type}' in an assistant turn", here)
        if not texts and not calls:
            return None
        message: dict[str, Any] = {"role": "assistant", "content": "".join(texts) if texts else None}
        if calls:
            message["tool_calls"] = calls
        return message

    def tool_result(self, block: ToolResultBlock, param: str) -> dict[str, Any]:
        content: str | list[dict[str, Any]]
        if block.content is None or isinstance(block.content, str):
            content = block.content or ""
        else:
            content = self.parts(block.content, f"{param}.content", allow=("text", "image"))
        if block.is_error:
            # canonical tool messages have no error flag, so the model is told in-band
            if isinstance(content, str):
                content = f"Error: {content}"
            else:
                content = [{"type": "text", "text": "Error:"}, *content]
        return {"role": "tool", "tool_call_id": block.tool_use_id, "content": content}

    def note_cache_control(self, block: Block) -> None:
        if getattr(block, "cache_control", None) is not None:
            self.ignored.add("cache_control")

    def parts(self, blocks: list[Block], param: str, *, allow: tuple[str, ...]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for j, block in enumerate(blocks):
            here = f"{param}[{j}]"
            self.note_cache_control(block)
            if block.type not in allow:
                raise _unsupported(f"Content block type '{block.type}' here", here)
            if isinstance(block, TextBlock):
                if block.citations is not None:
                    self.ignored.add("citations")
                out.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                out.append(self.image(block, here))
            elif isinstance(block, DocumentBlock):
                out.append(self.document(block, here))
        return out

    def image(self, block: ImageBlock, param: str) -> dict[str, Any]:
        source = block.source
        if isinstance(source, _Base64Source):
            url = f"data:{source.media_type};base64,{source.data}"
        elif isinstance(source, _UrlSource):
            url = source.url
        else:
            raise _unsupported(f"Image source type '{source.type}'", f"{param}.source")
        return {"type": "image_url", "image_url": {"url": url}}

    def document(self, block: DocumentBlock, param: str) -> dict[str, Any]:
        source = block.source
        if block.citations is not None:
            self.ignored.add("citations")
        if isinstance(source, _TextSource):
            return {"type": "text", "text": source.data}
        if isinstance(source, _Base64Source):
            file = {"file_data": f"data:{source.media_type};base64,{source.data}"}
            if block.title:
                file["filename"] = block.title
            return {"type": "file", "file": file}
        raise _unsupported(f"Document source type '{source.type}'", f"{param}.source")

    def tools(self, tools: list[ToolDef]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, tool in enumerate(tools):
            if tool.type not in (None, "custom"):
                raise _unsupported(f"Server tool '{tool.type}'", f"tools[{i}].type")
            if tool.cache_control is not None:
                self.ignored.add("cache_control")
            if tool.input_examples is not None or tool.eager_input_streaming is not None:
                self.ignored.add("tools.input_examples")
            function: dict[str, Any] = {
                "name": tool.name,
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            }
            if tool.description is not None:
                function["description"] = tool.description
            if tool.strict is not None:
                function["strict"] = tool.strict
            out.append({"type": "function", "function": function})
        return out


def _tool_choice(choice: ToolChoice, payload: dict[str, Any]) -> None:
    match choice.type:
        case "auto" | "none":
            payload["tool_choice"] = choice.type
        case "any":
            payload["tool_choice"] = "required"
        case "tool":
            if not choice.name:
                raise _bad("tool_choice of type 'tool' requires a name.", "tool_choice.name")
            payload["tool_choice"] = {"type": "function", "function": {"name": choice.name}}
    if choice.disable_parallel_tool_use:
        payload["parallel_tool_calls"] = False


def _reasoning_effort(req: MessagesRequest) -> str | None:
    if req.output_config is not None and req.output_config.effort is not None:
        return req.output_config.effort
    thinking = req.thinking
    if thinking is None or thinking.type != "enabled":
        return None
    if thinking.budget_tokens is None:
        raise _bad("thinking.budget_tokens is required when thinking is enabled.", "thinking.budget_tokens")
    for limit, effort in THINKING_EFFORT:
        if thinking.budget_tokens < limit:
            return effort
    return "high"


def _parse(raw: dict[str, Any]) -> MessagesRequest:
    for name in raw:
        if name in UNSUPPORTED_FIELDS:
            raise _unsupported(f"The '{name}' parameter", name)
    try:
        return MessagesRequest.model_validate(raw)
    except ValidationError as exc:
        raise validation_to_error(exc, raw) from None


def translate_request(raw: dict[str, Any]) -> tuple[dict[str, Any], frozenset[str]]:
    """anthropic messages body -> canonical (openai-shaped) chat payload plus ignored param names"""
    req = _parse(raw)
    t = _Translator()
    t.ignored.update(name for name in IGNORED_FIELDS if name in req.model_fields_set)
    payload: dict[str, Any] = {
        "model": req.model,
        "messages": t.system(req.system) + t.messages(req.messages),
        "max_completion_tokens": req.max_tokens,
        "stream": req.stream,
    }
    if req.tools:
        payload["tools"] = t.tools(req.tools)
    if req.tool_choice is not None:
        _tool_choice(req.tool_choice, payload)
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.top_p is not None:
        payload["top_p"] = req.top_p
    if req.stop_sequences:
        if len(req.stop_sequences) > MAX_STOP_SEQUENCES:
            raise _bad(f"At most {MAX_STOP_SEQUENCES} stop_sequences are supported.", "stop_sequences")
        payload["stop"] = req.stop_sequences
    if req.metadata is not None and req.metadata.user_id is not None:
        payload["user"] = req.metadata.user_id
    effort = _reasoning_effort(req)
    if effort is not None:
        payload["reasoning_effort"] = effort
    if req.output_config is not None and req.output_config.format is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "output", "schema": req.output_config.format.schema_, "strict": True},
        }
    if req.gg is not None:
        payload["gg"] = req.gg
    if not payload["messages"]:
        raise _bad("The request contains no message content.", "messages")
    return payload, frozenset(t.ignored)
