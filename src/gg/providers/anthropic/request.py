import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import orjson

from gg.core.deployment import Deployment
from gg.core.errors import ProviderError
from gg.core.schema import (
    ChatRequest,
    ContentPart,
    FilePart,
    FunctionTool,
    ImagePart,
    InputAudioPart,
    Message,
    NamedToolChoice,
    RefusalPart,
    TextPart,
)
from gg.providers.anthropic.quirks import AnthropicQuirks
from gg.providers.anthropic.schema_limits import schema_violations
from gg.providers.errors import capability_mismatch
from gg.providers.openai_compat.request import hashed_user
from gg.providers.state.thinking import ThinkingBlocks

MESSAGES_PATH = "/v1/messages"
JSON_OBJECT_INSTRUCTION = "Respond with a single JSON object."
IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MIN_THINKING_BUDGET = 1024
THINKING_HEADROOM = 256
# visual token cost is capped per image/page; good enough for the caching threshold
MEDIA_TOKEN_ESTIMATE = 1600

_TOOL_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_HANDLED = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "n",
        "logprobs",
        "top_logprobs",
        "stream",
        "stream_options",
        "reasoning_effort",
        "user",
    }
)


@dataclass(frozen=True, slots=True)
class RequestInfo:
    """per-request inputs the pure translator needs from the request context"""

    key_id: str


@dataclass(frozen=True, slots=True)
class Translated:
    body: dict[str, Any]
    ignored: tuple[str, ...]
    adjustments: tuple[str, ...]
    thinking_injected: int = 0


def sanitize_tool_id(tool_id: str) -> str:
    """anthropic needs ^[a-zA-Z0-9_-]+$; other providers' ids are mapped deterministically, never randomly"""
    if _TOOL_ID.match(tool_id):
        return tool_id
    return "gg_" + hashlib.sha1(tool_id.encode()).hexdigest()[:24]  # noqa: S324


def history_tool_call_ids(req: ChatRequest) -> list[str]:
    return [c.id for m in req.messages if m.role == "assistant" for c in m.tool_calls or ()]


def uses_thinking(dep: Deployment) -> bool:
    return dep.capabilities.thinking_mode in ("adaptive", "adaptive_always", "manual")


class _Builder:
    def __init__(self, req: ChatRequest, dep: Deployment, quirks: AnthropicQuirks) -> None:
        self.req = req
        self.dep = dep
        self.q = quirks
        self.ignored: list[str] = []
        self.adjustments: list[str] = []
        self.violations: list[str] = []
        self.injected = 0

    def ignore(self, name: str) -> None:
        if name not in self.ignored:
            self.ignored.append(name)

    def violate(self, name: str) -> None:
        if name not in self.violations:
            self.violations.append(name)

    def client_error(self, code: str, message: str) -> ProviderError:
        return ProviderError(
            "client",
            provider=self.dep.provider,
            status=400,
            code=code,
            message=message,
            deployment_id=self.dep.id,
        )


def build_body(
    req: ChatRequest,
    dep: Deployment,
    quirks: AnthropicQuirks,
    info: RequestInfo,
    *,
    thinking: Mapping[str, ThinkingBlocks] | None = None,
) -> Translated:
    """canonical request (already capability-checked) -> messages api body; pure and deterministic"""
    b = _Builder(req, dep, quirks)
    for name in req.upstream_payload():
        if name not in _HANDLED:
            b.ignore(name)
    max_tokens = req.max_completion_tokens or dep.capabilities.max_output
    tools = _tools(b)
    tool_choice, forced = _tool_choice(b, bool(tools))
    thinking_cfg, output_config = _reasoning(b, max_tokens, forced)
    _response_format(b, output_config)

    # stored blocks only go back when this request thinks; manual mode thinks only with a budget
    manual = dep.capabilities.thinking_mode == "manual"
    inject = thinking if thinking and uses_thinking(dep) and not (manual and thinking_cfg is None) else {}
    system, messages = _messages(b, inject)
    if (
        req.response_format is not None
        and "format" not in output_config
        and req.response_format.type != "text"
    ):
        system.append({"type": "text", "text": JSON_OBJECT_INSTRUCTION})

    body: dict[str, Any] = {"model": dep.upstream_model, "max_tokens": max_tokens}
    if system:
        body["system"] = system
    body["messages"] = messages
    if tools:
        body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
    if thinking_cfg is not None:
        body["thinking"] = thinking_cfg
    if output_config:
        body["output_config"] = output_config
    stops = _stop_sequences(b)
    if stops:
        body["stop_sequences"] = stops
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    body["metadata"] = {"user_id": hashed_user(info.key_id, req.user or "")}
    if _should_cache(b, system, tools, messages):
        body["cache_control"] = {"type": "ephemeral"}
    body["stream"] = True

    if b.violations:
        raise capability_mismatch(dep.provider, dep.id, tuple(b.violations))
    return Translated(body, tuple(b.ignored), tuple(b.adjustments), b.injected)


def _tools(b: _Builder) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in b.req.tools or ():
        if not isinstance(tool, FunctionTool):
            b.ignore(f"tools:{tool.type}")
            continue
        fn = tool.function
        if not _TOOL_NAME.match(fn.name):
            b.violate("tool_name")
        spec: dict[str, Any] = {"name": fn.name}
        if fn.description:
            spec["description"] = fn.description
        spec["input_schema"] = fn.parameters or {"type": "object", "properties": {}}
        if fn.strict is not None:
            spec["strict"] = fn.strict
        out.append(spec)
    return out


def _tool_choice(b: _Builder, has_tools: bool) -> tuple[dict[str, Any] | None, bool]:
    choice = b.req.tool_choice
    out: dict[str, Any] | None = None
    forced = False
    if choice in ("auto", "none"):
        out = {"type": choice}
    elif choice == "required":
        out, forced = {"type": "any"}, True
    elif isinstance(choice, NamedToolChoice) and choice.function.get("name"):
        out, forced = {"type": "tool", "name": choice.function["name"]}, True
    elif choice is not None:
        b.ignore("tool_choice")
    if b.req.parallel_tool_calls is False:
        out = {**(out or {"type": "auto"}), "disable_parallel_tool_use": True}
    if out is not None and not has_tools:
        b.ignore("tool_choice")
        return None, False
    return out, forced


def _reasoning(b: _Builder, max_tokens: int, forced: bool) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    mode = b.dep.capabilities.thinking_mode
    effort = b.req.reasoning_effort
    output_config: dict[str, Any] = {}
    if mode in ("adaptive", "adaptive_always"):
        if effort == "none":
            # sonnet 5.5: no up-front thinking; between_tools needs effort <= high
            output_config["effort"] = "low"
            return {"type": "between_tools"}, output_config
        if effort is not None:
            output_config["effort"] = effort
        return None, output_config
    if mode != "manual":
        if effort is not None:
            b.ignore("reasoning_effort")
        return None, output_config
    budget = b.q.thinking_budgets.get(effort) if effort else None
    if not budget:
        return None, output_config
    if forced:
        # haiku rejects forced tool choice while manual thinking is on
        b.adjustments.append("thinking:dropped_forced_tool_choice")
        return None, output_config
    if budget >= max_tokens:
        shrunk = max_tokens - THINKING_HEADROOM
        if shrunk < MIN_THINKING_BUDGET:
            b.adjustments.append("thinking:dropped_max_tokens")
            return None, output_config
        b.adjustments.append(f"thinking.budget_tokens:{budget}->{shrunk}")
        budget = shrunk
    return {"type": "enabled", "budget_tokens": budget}, output_config


def _response_format(b: _Builder, output_config: dict[str, Any]) -> None:
    fmt = b.req.response_format
    if fmt is None or fmt.type == "text":
        return
    schema = fmt.json_schema.schema_ if fmt.json_schema is not None else None
    if fmt.type == "json_schema" and schema is not None:
        for violation in schema_violations(schema):
            b.violate(f"json_schema:{violation}")
        output_config["format"] = {"type": "json_schema", "schema": schema}
        return
    b.adjustments.append(f"response_format:{fmt.type}->instruction")


def _stop_sequences(b: _Builder) -> list[str]:
    stops = list(b.req.stop or ())
    kept = [s for s in stops if s.strip()]
    if len(kept) != len(stops):
        b.adjustments.append("stop:dropped_whitespace")
    return kept


def _messages(
    b: _Builder, thinking: Mapping[str, ThinkingBlocks]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system: list[dict[str, Any]] = []
    turns: list[tuple[str, list[dict[str, Any]]]] = []
    calls_seen: set[str] = set()
    for message in b.req.messages:
        if message.role in ("system", "developer"):
            # hoisted in order, one block per message; never demoted to user (research §9.3.1)
            text = message.text()
            if text:
                system.append({"type": "text", "text": text})
            continue
        if message.role == "assistant":
            role, blocks = "assistant", _assistant_blocks(b, message, thinking)
            calls_seen.update(c.id for c in message.tool_calls or ())
        elif message.role == "tool":
            if message.tool_call_id not in calls_seen:
                raise b.client_error(
                    "orphan_tool_result", f"tool message '{message.tool_call_id}' has no matching tool call"
                )
            role, blocks = "user", [_tool_result(b, message)]
        else:
            role, blocks = "user", _content_blocks(b, message.content)
        if not blocks:
            continue
        if turns and turns[-1][0] == role:
            turns[-1][1].extend(blocks)
        else:
            turns.append((role, blocks))
    if b.violations:
        raise capability_mismatch(b.dep.provider, b.dep.id, tuple(b.violations))
    if not turns or turns[0][0] != "user":
        raise b.client_error("invalid_message_order", "the conversation must start with a user message")
    messages: list[dict[str, Any]] = []
    for role, blocks in turns:
        if role == "user":
            # tool_result blocks must lead the user turn that answers a tool_use
            results = [x for x in blocks if x["type"] == "tool_result"]
            blocks = results + [x for x in blocks if x["type"] != "tool_result"]
        messages.append({"role": role, "content": blocks})
    _trim_prefill(messages)
    return system, messages


def _trim_prefill(messages: list[dict[str, Any]]) -> None:
    last = messages[-1]
    if last["role"] != "assistant":
        return
    blocks: list[dict[str, Any]] = last["content"]
    if blocks and blocks[-1]["type"] == "text":
        # a final assistant turn may not end with whitespace
        text = str(blocks[-1]["text"]).rstrip()
        if text:
            blocks[-1] = {**blocks[-1], "text": text}
        else:
            blocks.pop()
    if not blocks:
        messages.pop()


def _assistant_blocks(
    b: _Builder, message: Message, thinking: Mapping[str, ThinkingBlocks]
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    text = message.text()
    if text:
        blocks.append({"type": "text", "text": text})
    for i, call in enumerate(message.tool_calls or ()):
        stored = thinking.get(call.id)
        if stored:
            # the first call's blocks open the turn; later ones sit right before their own tool_use
            b.injected += len(stored)
            if i == 0:
                blocks[0:0] = stored
            else:
                blocks.extend(stored)
        blocks.append(
            {
                "type": "tool_use",
                "id": sanitize_tool_id(call.id),
                "name": call.function.name,
                "input": _tool_input(b, call.function.arguments, call.id),
            }
        )
    return blocks


def _tool_input(b: _Builder, arguments: str, call_id: str) -> dict[str, Any]:
    try:
        parsed: Any = orjson.loads(arguments or "{}")
    except orjson.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        raise b.client_error(
            "invalid_tool_arguments", f"tool call '{call_id}' arguments must be a json object"
        )
    return parsed  # pyright: ignore[reportUnknownVariableType]


def _tool_result(b: _Builder, message: Message) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": sanitize_tool_id(message.tool_call_id or ""),
    }
    content = message.content
    if isinstance(content, str):
        if content:
            block["content"] = content
    elif content:
        parts = _content_blocks(b, content)
        if parts:
            block["content"] = parts
    return block


def _content_blocks(b: _Builder, content: str | Sequence[ContentPart] | None) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            if part.text:
                blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append(_image(b, part.image_url.url))
        elif isinstance(part, FilePart):
            document = _document(b, part.file)
            if document is not None:
                blocks.append(document)
        elif isinstance(part, InputAudioPart):
            b.violate("audio")
        elif not isinstance(part, RefusalPart):
            b.ignore(f"content:{part.type}")
    return blocks


def _data_url(b: _Builder, url: str, kind: str) -> tuple[str, str]:
    header, sep, data = url.partition(",")
    if not sep or not header.startswith("data:") or ";base64" not in header:
        raise b.client_error(f"invalid_{kind}", f"{kind} data urls must be base64 encoded")
    return header[5:].split(";", 1)[0].lower(), data


def _image(b: _Builder, url: str) -> dict[str, Any]:
    if url.startswith("data:"):
        media_type, data = _data_url(b, url, "image")
        if media_type not in IMAGE_TYPES:
            raise b.client_error("unsupported_image_type", f"image type '{media_type}' is not supported")
        if len(data) * 3 // 4 > MAX_IMAGE_BYTES:
            raise b.client_error("image_too_large", "images must be at most 10 MB")
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
    if url.startswith(("https://", "http://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    raise b.client_error("invalid_image_url", "image urls must be https or base64 data urls")


def _document(b: _Builder, file: Mapping[str, str]) -> dict[str, Any] | None:
    data_url = file.get("file_data")
    if data_url is None:
        if "file_id" in file:
            b.violate("file_id")
        return None
    media_type, data = _data_url(b, data_url, "file")
    if media_type != "application/pdf":
        raise b.client_error("unsupported_file_type", f"file type '{media_type}' is not supported")
    return {"type": "document", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _approx_tokens(value: Any) -> int:
    if isinstance(value, str):
        return (len(value) + 3) // 4
    if isinstance(value, list):
        return sum(_approx_tokens(v) for v in value)  # pyright: ignore[reportUnknownVariableType]
    if not isinstance(value, dict):
        return 0
    block: dict[str, Any] = value  # pyright: ignore[reportUnknownVariableType]
    kind = block.get("type")
    if kind in ("image", "document"):
        return MEDIA_TOKEN_ESTIMATE
    if kind == "tool_use":
        return _approx_tokens(orjson.dumps(block.get("input")).decode()) + _approx_tokens(block.get("name"))
    if "content" in block:
        return _approx_tokens(block["content"])
    if "input_schema" in block:
        return _approx_tokens(orjson.dumps(block).decode())
    return _approx_tokens(block.get("text") or block.get("thinking"))


def _should_cache(
    b: _Builder, system: list[dict[str, Any]], tools: list[dict[str, Any]], messages: list[dict[str, Any]]
) -> bool:
    """automatic caching pays off only when the prefix is reused: multi-turn, or a large static prefix"""
    minimum = b.dep.capabilities.cache_min_tokens
    if not b.q.cache.enabled or minimum is None:
        return False
    static = _approx_tokens(system) + _approx_tokens(tools)
    total = static + _approx_tokens(messages)
    return total >= minimum and (len(messages) > 1 or static >= b.q.cache.min_static_tokens)
