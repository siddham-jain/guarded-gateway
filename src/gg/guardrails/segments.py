from collections.abc import Iterable, Sequence
from typing import Any

from gg.core.schema import ChatRequest, Message, Role, TextPart
from gg.guardrails.base import Segment

# system/developer text is operator-controlled and never inspected or rewritten
INSPECTED_ROLES: frozenset[Role] = frozenset({"user", "assistant", "tool"})

PLACEHOLDER_HINT = (
    "Tokens like [EMAIL_1] are opaque placeholders for redacted values; reproduce them verbatim."
)


def extract_segments(request: ChatRequest, roles: frozenset[Role] = INSPECTED_ROLES) -> tuple[Segment, ...]:
    out: list[Segment] = []

    def add(msg: int, role: Role, text: str, **where: Any) -> None:
        out.append(Segment(index=len(out), role=role, msg=msg, text=text, **where))

    for mi, message in enumerate(request.messages):
        role = message.role
        if role not in roles:
            continue
        kind = "tool_result" if role == "tool" else "content"
        if isinstance(message.content, str):
            add(mi, role, message.content, kind=kind)
        elif message.content is not None:
            for pi, part in enumerate(message.content):
                if isinstance(part, TextPart):
                    add(mi, role, part.text, kind=kind, part=pi)
        if message.refusal:
            add(mi, role, message.refusal, kind="refusal")
        for ti, call in enumerate(message.tool_calls or ()):
            add(mi, role, call.function.arguments, kind="tool_args", tool_call=ti)
    return tuple(out)


def _rewrite(message: Message, segments: Iterable[Segment]) -> Message:
    update: dict[str, Any] = {}
    parts = list(message.content) if isinstance(message.content, tuple) else None
    calls = list(message.tool_calls or ())
    for seg in segments:
        if seg.kind == "refusal":
            update["refusal"] = seg.text
        elif seg.kind == "tool_args":
            call = calls[seg.tool_call]
            calls[seg.tool_call] = call.model_copy(
                update={"function": call.function.model_copy(update={"arguments": seg.text})}
            )
            update["tool_calls"] = tuple(calls)
        elif parts is not None:
            part = parts[seg.part]
            if isinstance(part, TextPart):
                parts[seg.part] = part.model_copy(update={"text": seg.text})
            update["content"] = tuple(parts)
        else:
            update["content"] = seg.text
    return message.model_copy(update=update)


def write_back(request: ChatRequest, original: Sequence[Segment], segments: Sequence[Segment]) -> ChatRequest:
    """returns a new request with changed segment text; the same object when nothing changed"""
    changed: dict[int, list[Segment]] = {}
    for before, after in zip(original, segments, strict=True):
        if before.text != after.text:
            changed.setdefault(after.msg, []).append(after)
    if not changed:
        return request
    messages = list(request.messages)
    for mi, segs in changed.items():
        messages[mi] = _rewrite(messages[mi], segs)
    return request.model_copy(update={"messages": tuple(messages)})


def with_placeholder_hint(request: ChatRequest) -> ChatRequest:
    messages = list(request.messages)
    for i, message in enumerate(messages):
        if message.role in ("system", "developer") and isinstance(message.content, str):
            messages[i] = message.model_copy(update={"content": f"{message.content}\n\n{PLACEHOLDER_HINT}"})
            return request.model_copy(update={"messages": tuple(messages)})
    messages.insert(0, Message(role="system", content=PLACEHOLDER_HINT))
    return request.model_copy(update={"messages": tuple(messages)})


def scoped_texts(segments: Sequence[Segment], roles: frozenset[Role], history_user_turns: int) -> list[str]:
    """inspection views (and decoded payloads) of in-scope segments from the recent turns"""
    user_msgs = sorted({s.msg for s in segments if s.role == "user"})
    first = user_msgs[-history_user_turns] if len(user_msgs) >= history_user_turns else 0
    out: list[str] = []
    for seg in segments:
        if seg.role not in roles or seg.kind == "tool_args" or seg.msg < first:
            continue
        out.append(seg.view)
        out.extend(d.text for d in seg.decoded)
    return [t for t in out if t.strip()]
