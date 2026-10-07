import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gg.core.deployment import Deployment
from gg.core.errors import ProviderError
from gg.core.schema import ChatRequest
from gg.providers.errors import capability_mismatch
from gg.providers.openai_compat.quirks import QuirkProfile, deep_merge

KNOWN_PARAMS = frozenset(ChatRequest.model_fields) - {"gg"}
_ALNUM9 = re.compile(r"^[A-Za-z0-9]{9}$")
_ALWAYS_KEPT = frozenset({"model", "messages", "stream"})


@dataclass(frozen=True, slots=True)
class RequestInfo:
    """per-request inputs the pure translator needs from the request context"""

    request_id: str
    key_id: str
    received_unix: int
    allow_store: bool = False
    allow_service_tier: bool = False


@dataclass(frozen=True, slots=True)
class Translated:
    body: dict[str, Any]
    ignored: tuple[str, ...]
    adjustments: tuple[str, ...]


def hashed_user(key_id: str, user: str) -> str:
    return hashlib.sha256(f"{key_id}:{user}".encode()).hexdigest()


def history_key(message: Mapping[str, Any]) -> str | None:
    """stable key for an assistant turn: its first tool call id, else a hash of its text"""
    calls = message.get("tool_calls") or []
    if calls and isinstance(calls[0], Mapping) and calls[0].get("id"):  # pyright: ignore[reportUnknownMemberType]
        return "tc:" + str(calls[0]["id"])  # pyright: ignore[reportUnknownArgumentType]
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            str(p.get("text", ""))
            for p in content
            if isinstance(p, Mapping)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        )
    if isinstance(content, str) and content:
        return "txt:" + hashlib.sha256(content.encode()).hexdigest()[:32]
    return None


def history_keys(request: ChatRequest) -> list[str]:
    keys: list[str] = []
    for message in request.messages:
        if message.role == "assistant":
            key = history_key(message.model_dump(mode="json", exclude_none=True))
            if key is not None:
                keys.append(key)
    return keys


def _set_path(body: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = body
    for part in parts[:-1]:
        nxt = target.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            target[part] = nxt
        target = nxt  # pyright: ignore[reportUnknownVariableType]
    target[parts[-1]] = value


def _alnum9(tool_id: str) -> str:
    if _ALNUM9.match(tool_id):
        return tool_id
    return hashlib.sha256(tool_id.encode()).hexdigest()[:9]


def _tool_choice_mode(choice: Any) -> str:
    if isinstance(choice, str):
        return choice
    return "named"


class _Builder:
    def __init__(self, req: ChatRequest, dep: Deployment, quirks: QuirkProfile, info: RequestInfo) -> None:
        self.req = req
        self.dep = dep
        self.q = quirks
        self.info = info
        self.ignored: list[str] = []
        self.adjustments: list[str] = []
        self.violations: list[str] = []

    def ignore(self, name: str) -> None:
        if name not in self.ignored:
            self.ignored.append(name)


def build_body(
    req: ChatRequest,
    dep: Deployment,
    quirks: QuirkProfile,
    info: RequestInfo,
    *,
    history: Mapping[str, str] | None = None,
) -> Translated:
    """canonical request (already capability-checked) -> upstream json body; pure and deterministic"""
    b = _Builder(req, dep, quirks, info)
    rq = quirks.request
    payload = {k: v for k, v in req.upstream_payload().items() if v is not None}
    body: dict[str, Any] = {}
    for key, value in payload.items():
        policed = key == "service_tier" and not info.allow_service_tier
        if key in KNOWN_PARAMS or (rq.passthrough_unknown and not policed):
            body[key] = value
        else:
            b.ignore(key)

    body["model"] = rq.model_prefix + dep.upstream_model
    body["messages"] = _messages(b, body.get("messages", []), history or {})

    if (max_tokens := body.pop("max_completion_tokens", None)) is not None:
        body[rq.max_tokens_param] = max_tokens
    _reasoning(b, body)
    _identity(b, body)
    _tools(b, body)
    _response_format(b, body)

    body["stream"] = True
    body.pop("stream_options", None)
    if rq.stream_usage == "inject":
        body["stream_options"] = {"include_usage": True, **rq.stream_options_extra}

    for name, (low, high) in rq.clamp.items():
        value = body.get(name)
        if isinstance(value, (int, float)) and not low <= value <= high:
            clamped = min(max(value, low), high)
            body[name] = clamped
            b.adjustments.append(f"{name}:{value}->{clamped}")
    stop = body.get("stop")
    if rq.stop_max is not None and isinstance(stop, list) and len(stop) > rq.stop_max:  # pyright: ignore[reportUnknownArgumentType]
        b.violations.append("stop")
    for name, value in rq.defaults.items():
        if name not in body:
            body[name] = dep.defaults.get(name, value)

    for src, dst in rq.renames.items():
        if src in body:
            value = body.pop(src)
            _set_path(body, dst, value)
    for name in rq.drop:
        if name in body:
            del body[name]
            b.ignore(name)
    if rq.allow_only is not None:
        allowed = set(rq.allow_only) | _ALWAYS_KEPT
        for name in [k for k in body if k not in allowed]:
            del body[name]
            b.ignore(name)
    b.violations.extend(name for name in rq.reject if name in body)
    if rq.extra_body:
        body = deep_merge(body, rq.extra_body)

    if b.violations:
        raise capability_mismatch(dep.provider, dep.id, tuple(b.violations))
    return Translated(body, tuple(b.ignored), tuple(b.adjustments))


def _messages(
    b: _Builder, messages: list[dict[str, Any]], history: Mapping[str, str]
) -> list[dict[str, Any]]:
    rq = b.q.request
    hist = b.q.reasoning
    keep_reasoning = hist.history in ("required", "required_with_tools")
    out: list[dict[str, Any]] = []
    missing = False
    for original in messages:
        message = dict(original)
        role = message.get("role", "")
        client_reasoning = message.get("reasoning_content") or message.get("reasoning")
        for name in rq.strip_message_fields:
            message.pop(name, None)
        if role in rq.role_map:
            message["role"] = rq.role_map[role]
        if rq.tool_call_id_format == "alnum9":
            if role == "tool" and isinstance(message.get("tool_call_id"), str):
                message["tool_call_id"] = _alnum9(message["tool_call_id"])
            if message.get("tool_calls"):
                message["tool_calls"] = [
                    {**c, "id": _alnum9(str(c.get("id", "")))} for c in message["tool_calls"]
                ]
        if role == "assistant" and keep_reasoning:
            key = history_key(original)
            reasoning = client_reasoning or (history.get(key) if key else None)
            if reasoning:
                message[hist.history_field] = reasoning
            elif message.get("tool_calls"):
                missing = True
        out.append(message)
    if missing and hist.history == "required_with_tools" and b.req.tools:
        disable = hist.effort_map.get("none")
        if isinstance(disable, dict) and hist.control != "none":
            # thinking off is the only way to call this host without the lost reasoning
            b.adjustments.append("reasoning:disabled_missing_history")
            b.req = b.req.model_copy(update={"reasoning_effort": "none"})
        else:
            b.violations.append("reasoning_history")
    return out


def _reasoning(b: _Builder, body: dict[str, Any]) -> None:
    hist = b.q.reasoning
    effort = body.pop("reasoning_effort", None)
    if b.req.reasoning_effort == "none" and effort is not None:
        effort = "none"
    if effort is None:
        return
    if hist.control == "none":
        b.ignore("reasoning_effort")
        return
    mapped = hist.effort_map.get(effort, effort) if effort in hist.effort_map else effort
    if isinstance(mapped, dict):
        body.update(deep_merge(body, mapped))  # pyright: ignore[reportUnknownArgumentType]
    elif mapped is not None:
        _set_path(body, hist.param, mapped)


def _identity(b: _Builder, body: dict[str, Any]) -> None:
    rq = b.q.request
    user = body.pop("user", None)
    if rq.user_field is not None:
        # every end user gets a stable per-key hash; the bare key hash when the client sent none
        body[rq.user_field] = hashed_user(b.info.key_id, user if isinstance(user, str) else "")
    elif user is not None:
        b.ignore("user")
    if "metadata" in body and not rq.send_metadata:
        del body["metadata"]
        b.ignore("metadata")
    if rq.store_false and not (b.info.allow_store and body.get("store") is True):
        body["store"] = False
    if rq.prompt_cache_key and "prompt_cache_key" not in body:
        body["prompt_cache_key"] = hashlib.sha256(b.info.key_id.encode()).hexdigest()[:32]


def _tools(b: _Builder, body: dict[str, Any]) -> None:
    rq = b.q.request
    choice = body.get("tool_choice")
    if choice is not None:
        mode = _tool_choice_mode(choice)
        if mode not in rq.tool_choice.supported:
            del body["tool_choice"]
            if mode == "none":
                body.pop("tools", None)
                body.pop("parallel_tool_calls", None)
            elif mode != "auto":
                b.violations.append("forced_tool_choice")
    if "parallel_tool_calls" in body and not rq.parallel_tool_calls:
        del body["parallel_tool_calls"]
        b.ignore("parallel_tool_calls")
    tools = body.get("tools")
    if isinstance(tools, list) and not rq.strict_tools:
        cleaned: list[Any] = []
        for tool in tools:  # pyright: ignore[reportUnknownVariableType]
            fn = tool.get("function") if isinstance(tool, dict) else None  # pyright: ignore[reportUnknownMemberType]
            if isinstance(fn, dict) and "strict" in fn:
                tool = {**tool, "function": {k: v for k, v in fn.items() if k != "strict"}}  # pyright: ignore[reportUnknownVariableType]
                b.ignore("tools.function.strict")
            cleaned.append(tool)
        body["tools"] = cleaned


def _response_format(b: _Builder, body: dict[str, Any]) -> None:
    fmt = body.get("response_format")
    if not isinstance(fmt, dict):
        return
    kind = fmt.get("type")  # pyright: ignore[reportUnknownMemberType]
    rf = b.q.request.response_format
    if kind == "json_schema" and rf.json_schema == "as_json_object":
        body["response_format"] = {"type": "json_object"}
        b.adjustments.append("response_format:json_schema->json_object")
        kind = "json_object"
    elif kind == "json_schema" and rf.json_schema == "unsupported":
        b.violations.append("json_schema")
    if kind == "json_object" and rf.json_object == "unsupported":
        b.violations.append("json_object")


__all__ = [
    "KNOWN_PARAMS",
    "ProviderError",
    "RequestInfo",
    "Translated",
    "build_body",
    "hashed_user",
    "history_key",
    "history_keys",
]
