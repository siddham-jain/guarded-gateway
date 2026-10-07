import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import orjson

from gg.core.deployment import EFFORT_ORDER, Deployment
from gg.core.errors import ProviderError
from gg.core.schema import (
    ChatRequest,
    FilePart,
    FunctionTool,
    ImagePart,
    InputAudioPart,
    Message,
    NamedToolChoice,
    TextPart,
    ToolCall,
)
from gg.providers.errors import capability_mismatch
from gg.providers.gemini.quirks import GeminiQuirks
from gg.providers.state.signatures import DUMMY_SIGNATURE

SYNTHETIC_ID_PREFIX = "call_gg_"
THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})
_TOOL_CHOICE_MODES = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
_HANDLED = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "response_format",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "n",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
        "stream",
        "stream_options",
        "reasoning_effort",
        "user",
    }
)

type Part = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Translated:
    body: dict[str, Any]
    ignored: tuple[str, ...]
    adjustments: tuple[str, ...]
    dummy_signatures: int


def safety_identifier(key_id: str, user: str) -> str:
    return hashlib.sha256(f"{key_id}:{user}".encode()).hexdigest()[:48]


def client_signature(call: ToolCall) -> str | None:
    """google's openai-compat slot: tool_calls[].extra_content.google.thought_signature"""
    extra: Any = (call.model_extra or {}).get("extra_content")
    google: Any = extra.get("google") if isinstance(extra, dict) else None
    value: Any = google.get("thought_signature") if isinstance(google, dict) else None
    return value if isinstance(value, str) and value else None


def history_call_ids(request: ChatRequest) -> list[str]:
    ids: list[str] = []
    for message in request.messages:
        for call in message.tool_calls or ():
            if call.id not in ids:
                ids.append(call.id)
    return ids


def _with_id(call_id: str, body: dict[str, Any]) -> dict[str, Any]:
    # ids gg synthesised were never issued by gemini, so they are matched by order and name instead
    return body if call_id.startswith(SYNTHETIC_ID_PREFIX) else {"id": call_id, **body}


def _inline(url: str) -> Part | None:
    header, sep, data = url.partition(",")
    if not sep or not header.startswith("data:") or not header.endswith(";base64"):
        return None
    mime = header.removeprefix("data:").removesuffix(";base64") or "application/octet-stream"
    return {"inlineData": {"mimeType": mime, "data": data}}


def _tool_response(text: str) -> dict[str, Any]:
    try:
        value: Any = orjson.loads(text)
    except orjson.JSONDecodeError:
        return {"result": text}
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {"result": value}


class _Builder:
    def __init__(
        self,
        req: ChatRequest,
        dep: Deployment,
        quirks: GeminiQuirks,
        signatures: Mapping[str, str],
    ) -> None:
        self.req = req
        self.dep = dep
        self.q = quirks
        self.signatures = signatures
        self.ignored: list[str] = []
        self.adjustments: list[str] = []
        self.violations: list[str] = []
        self.dummies = 0
        self.names: dict[str, str] = {}
        self.positions: dict[str, int] = {}

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
            status=0,
            code=code,
            message=message,
            deployment_id=self.dep.id,
        )

    def contents(self) -> tuple[list[str], list[dict[str, Any]]]:
        system: list[str] = []
        contents: list[dict[str, Any]] = []
        results: list[tuple[int, Part]] = []

        def push(role: str, parts: list[Part]) -> None:
            if not parts:
                return
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})

        def flush_results() -> None:
            # gemini wants every functionResponse of a step in one user turn, in call order
            push("user", [part for _, part in sorted(results, key=lambda r: r[0])])
            results.clear()

        for message in self.req.messages:
            if message.role in ("system", "developer"):
                if text := message.text():
                    system.append(text)
            elif message.role == "tool":
                results.append(self._tool_result(message))
            else:
                flush_results()
                if message.role == "user":
                    push("user", self._user_parts(message))
                else:
                    push("model", self._model_parts(message))
        flush_results()
        if not self.violations and (not contents or contents[0]["role"] != "user"):
            raise self.client_error(
                "invalid_message_order", "gemini needs the conversation to start with a user message"
            )
        return system, contents

    def _user_parts(self, message: Message) -> list[Part]:
        content = message.content
        if isinstance(content, str) or content is None:
            return [{"text": content or ""}]
        parts: list[Part] = []
        for part in content:
            inline: Part | None = None
            if isinstance(part, TextPart):
                parts.append({"text": part.text})
                continue
            if isinstance(part, ImagePart):
                inline = _inline(part.image_url.url)
            elif isinstance(part, FilePart):
                inline = _inline(part.file.get("file_data", ""))
            elif isinstance(part, InputAudioPart):
                audio = part.input_audio
                data, fmt = audio.get("data"), audio.get("format")
                if data and fmt:
                    inline = {"inlineData": {"mimeType": f"audio/{fmt}", "data": data}}
            if inline is None:
                self.violate(f"content.{part.type}")
            else:
                parts.append(inline)
        return parts

    def _model_parts(self, message: Message) -> list[Part]:
        parts: list[Part] = []
        if text := message.text() or message.refusal:
            parts.append({"text": text})
        for position, call in enumerate(message.tool_calls or ()):
            raw = call.function.arguments.strip() or "{}"
            try:
                args: Any = orjson.loads(raw)
            except orjson.JSONDecodeError:
                args = None
            if not isinstance(args, dict):
                raise self.client_error(
                    "invalid_tool_arguments", f"tool call {call.id} arguments must be a json object"
                )
            self.names[call.id] = call.function.name
            self.positions[call.id] = position
            part: Part = {"functionCall": _with_id(call.id, {"name": call.function.name, "args": args})}
            signature = self.signatures.get(call.id) or client_signature(call)
            if signature is None and position == 0:
                # gemini 3 validates the first call of every step; a miss means expired or foreign history
                signature = DUMMY_SIGNATURE
                self.dummies += 1
            if signature is not None:
                part["thoughtSignature"] = signature
            parts.append(part)
        return parts

    def _tool_result(self, message: Message) -> tuple[int, Part]:
        call_id = message.tool_call_id or ""
        name = self.names.get(call_id)
        if name is None:
            raise self.client_error("orphan_tool_result", f"tool result {call_id} has no matching tool call")
        response = _tool_response(message.text())
        return self.positions[call_id], {
            "functionResponse": _with_id(call_id, {"name": name, "response": response})
        }

    def tools(self, body: dict[str, Any]) -> None:
        declarations: list[dict[str, Any]] = []
        for tool in self.req.tools or ():
            if not isinstance(tool, FunctionTool):
                self.violate(f"tools.{tool.type}")
                continue
            fn = tool.function
            declaration: dict[str, Any] = {"name": fn.name, "description": fn.description or ""}
            if fn.parameters is not None:
                declaration["parametersJsonSchema"] = fn.parameters
            if fn.strict is not None:
                self.ignore("tools.function.strict")
            declarations.append(declaration)
        if not declarations:
            if self.req.tool_choice is not None:
                self.ignore("tool_choice")
            return
        body["tools"] = [{"functionDeclarations": declarations}]
        choice = self.req.tool_choice
        if choice is None:
            return
        config: dict[str, Any]
        if isinstance(choice, str):
            config = {"mode": _TOOL_CHOICE_MODES[choice]}
        elif isinstance(choice, NamedToolChoice) and choice.function.get("name"):
            config = {"mode": "ANY", "allowedFunctionNames": [choice.function["name"]]}
        else:
            self.ignore("tool_choice")
            return
        body["toolConfig"] = {"functionCallingConfig": config}

    def generation_config(self) -> dict[str, Any]:
        req = self.req
        config: dict[str, Any] = {}
        if req.max_completion_tokens is not None:
            config["maxOutputTokens"] = req.max_completion_tokens
        if req.temperature is not None:
            config["temperature"] = req.temperature
        if req.top_p is not None:
            config["topP"] = req.top_p
        if req.stop:
            config["stopSequences"] = list(req.stop)
        if req.n > 1:
            config["candidateCount"] = req.n
        if req.seed is not None:
            config["seed"] = req.seed
        if req.presence_penalty is not None:
            config["presencePenalty"] = req.presence_penalty
        if req.frequency_penalty is not None:
            config["frequencyPenalty"] = req.frequency_penalty
        if req.logprobs:
            config["responseLogprobs"] = True
            if req.top_logprobs is not None:
                config["logprobs"] = req.top_logprobs
        self._response_format(config)
        self._thinking(config)
        return config

    def _response_format(self, config: dict[str, Any]) -> None:
        fmt = self.req.response_format
        if fmt is None or fmt.type == "text":
            return
        spec = fmt.json_schema if fmt.type == "json_schema" else None
        schema = spec.schema_ if spec is not None else None
        if spec is not None and spec.strict is not None:
            self.ignore("response_format.json_schema.strict")
        if self.q.structured_output == "mime_type":
            config["responseMimeType"] = "application/json"
            if schema is not None:
                config["responseJsonSchema"] = schema
            return
        text: dict[str, Any] = {"mimeType": "application/json"}
        if schema is not None:
            text["schema"] = schema
        config["responseFormat"] = {"text": text}

    def _thinking(self, config: dict[str, Any]) -> None:
        caps = self.dep.capabilities
        effort = self.req.reasoning_effort
        if caps.thinking_mode != "levels":
            if effort is not None:
                self.ignore("reasoning_effort")
            return
        limit = config.get("maxOutputTokens")
        floor = self.q.min_thinking_output
        if isinstance(limit, int) and limit < floor:
            # maxOutputTokens includes thinking: a small cap would be spent thinking and return nothing
            levels = caps.effort_levels & THINKING_LEVELS
            lowest = min(levels, key=EFFORT_ORDER.index) if levels else None
            if lowest is not None and effort != lowest:
                self.adjustments.append(f"reasoning_effort:{effort or 'default'}->{lowest}")
                effort = lowest
            raised = min(limit + floor, caps.max_output)
            if raised != limit:
                self.adjustments.append(f"max_output_tokens:{limit}->{raised}")
                config["maxOutputTokens"] = raised
        if effort is None:
            return
        if effort not in THINKING_LEVELS:
            self.ignore("reasoning_effort")
            return
        config["thinkingConfig"] = {"thinkingLevel": effort}


def build_body(
    req: ChatRequest,
    dep: Deployment,
    quirks: GeminiQuirks,
    *,
    key_id: str,
    signatures: Mapping[str, str] | None = None,
) -> Translated:
    """canonical request (already capability-checked) -> generateContent body; pure and deterministic"""
    b = _Builder(req, dep, quirks, signatures or {})
    for name, value in req.upstream_payload().items():
        if value is not None and name not in _HANDLED:
            b.ignore(name)
    system, contents = b.contents()
    body: dict[str, Any] = {"contents": contents}
    if system:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
    b.tools(body)
    if config := b.generation_config():
        body["generationConfig"] = config
    if quirks.safety_settings:
        body["safetySettings"] = [s.model_dump() for s in quirks.safety_settings]
    body["labels"] = {"safety_identifier": safety_identifier(key_id, req.user or "")}
    body["store"] = False
    if b.violations:
        raise capability_mismatch(dep.provider, dep.id, tuple(b.violations))
    return Translated(body, tuple(b.ignored), tuple(b.adjustments), b.dummies)
