from dataclasses import dataclass
from typing import Any

from gg.core.errors import InvalidRequestError


@dataclass(frozen=True, slots=True)
class Normalization:
    rule: str
    param: str


@dataclass(frozen=True, slots=True)
class NormalizedRequest:
    payload: dict[str, Any]
    applied: tuple[Normalization, ...]


def normalize_request_dict(raw: dict[str, Any]) -> NormalizedRequest:
    """rewrite legacy openai shapes into the canonical request; pure and idempotent"""
    payload = dict(raw)
    applied: list[Normalization] = []

    if "max_tokens" in payload:
        legacy = payload.pop("max_tokens")
        current = payload.get("max_completion_tokens")
        if current is not None and legacy is not None and current != legacy:
            raise InvalidRequestError(
                "max_tokens and max_completion_tokens are both set with different values",
                param="max_tokens",
            )
        if legacy is not None:
            payload["max_completion_tokens"] = legacy
        applied.append(Normalization("max_tokens", "max_tokens"))

    if "functions" in payload:
        if payload.get("tools"):
            raise InvalidRequestError("use either functions or tools, not both", param="functions")
        functions = payload.pop("functions") or []
        if functions:
            payload["tools"] = [{"type": "function", "function": f} for f in functions]
        applied.append(Normalization("functions", "functions"))

    if "function_call" in payload:
        if payload.get("tool_choice") is not None:
            raise InvalidRequestError(
                "use either function_call or tool_choice, not both", param="function_call"
            )
        choice = payload.pop("function_call")
        if isinstance(choice, dict):
            payload["tool_choice"] = {"type": "function", "function": {"name": choice.get("name")}}
        elif choice is not None:
            payload["tool_choice"] = choice
        applied.append(Normalization("function_call", "function_call"))

    messages = payload.get("messages")
    if isinstance(messages, list):
        payload["messages"] = _normalize_messages(messages, applied)

    stop = payload.get("stop")
    if isinstance(stop, str):
        payload["stop"] = [stop]
        applied.append(Normalization("stop_str", "stop"))
    elif isinstance(stop, list) and not stop:
        payload.pop("stop")
        applied.append(Normalization("stop_str", "stop"))

    if "stream_options" in payload and not payload.get("stream"):
        payload.pop("stream_options")
        applied.append(Normalization("stream_options_dropped", "stream_options"))

    return NormalizedRequest(payload, tuple(applied))


def _normalize_messages(messages: list[Any], applied: list[Normalization]) -> list[Any]:
    out: list[Any] = []
    last_call_id: dict[str, str] = {}
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            out.append(message)
            continue
        m: dict[str, Any] = dict(message)
        if m.get("role") == "assistant" and isinstance(m.get("function_call"), dict):
            fc: dict[str, Any] = m.pop("function_call")
            call_id = f"call_fn_{i}"
            m["tool_calls"] = [{"id": call_id, "type": "function", "function": fc}]
            last_call_id[str(fc.get("name"))] = call_id
            applied.append(Normalization("assistant_function_call", f"messages[{i}]"))
        elif m.get("role") == "function":
            name = str(m.get("name"))
            call_id = last_call_id.get(name)
            if call_id is None:
                raise InvalidRequestError(
                    "function message has no matching function_call", param=f"messages[{i}]"
                )
            m = {"role": "tool", "tool_call_id": call_id, "content": m.get("content")}
            applied.append(Normalization("function_role", f"messages[{i}]"))
        out.append(m)
    return out
