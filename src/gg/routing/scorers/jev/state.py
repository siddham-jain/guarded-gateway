import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from gg.core.jsonutil import canonical_json, sha256_hex
from gg.core.schema import ImagePart, Message, TextPart
from gg.routing.base import RoutingRequest
from gg.routing.config import JevStateConfig

STATE_VERSION = "state-v1"
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class JevState:
    """the scrubbed, bucketed view of a request that is sent to jev; None when there is nothing to score"""

    state: dict[str, Any] | None
    canonical: bytes

    @property
    def unscorable(self) -> bool:
        return self.state is None

    def cache_key(self, scorer_version: str) -> str:
        return sha256_hex(scorer_version.encode() + b"\n" + self.canonical)


class StateBuilder:
    version = STATE_VERSION

    def __init__(self, cfg: JevStateConfig) -> None:
        self._cfg = cfg

    def build(self, req: RoutingRequest) -> JevState:
        cfg = self._cfg
        messages = req.messages
        user_indexes = [i for i, m in enumerate(messages) if m.role == "user"]
        if not user_indexes:
            return JevState(None, b"")
        last_user = user_indexes[-1]
        request = _clean(_message_text(messages[last_user]))
        if not request:
            return JevState(None, b"")
        state: dict[str, Any] = {"request": self._truncate_request(request)}

        earlier = [_clean(_message_text(messages[i])) for i in reversed(user_indexes[:-1])]
        turns = [_head(t, cfg.prior_user_turn_max_chars) for t in earlier if t][: cfg.prior_user_turns]
        if turns:
            state["recent_user_turns"] = turns

        if len(request.split()) < cfg.followup_max_words:
            assistant = _last_assistant(messages[:last_user])
            if assistant:
                state["last_assistant_message"] = _head(assistant, cfg.last_assistant_max_chars)

        context: dict[str, str] = {"conversation_depth": self._depth(len(user_indexes))}
        if cfg.include_system_excerpt:
            system = next((m for m in messages if m.role in ("system", "developer")), None)
            excerpt = _clean(_message_text(system)) if system is not None else ""
            if excerpt:
                context["system_prompt_excerpt"] = _head(excerpt, cfg.system_excerpt_max_chars)
        context["tools_offered"] = "some" if req.tools_present else "none"
        state["context"] = context
        return JevState(state, canonical_json(state))

    def _truncate_request(self, text: str) -> str:
        cfg = self._cfg
        if len(text) <= cfg.request_max_chars:
            return text
        # the ask usually comes last, so the tail survives truncation
        dropped = len(text) - cfg.request_head_chars - cfg.request_tail_chars
        head, tail = text[: cfg.request_head_chars], text[-cfg.request_tail_chars :]
        return f"{head} … [truncated {dropped} chars] … {tail}"

    def _depth(self, user_turns: int) -> str:
        if user_turns <= self._cfg.new_max_turns:
            return "new"
        if user_turns <= self._cfg.short_max_turns:
            return "short"
        return "long"


def _message_text(message: Message) -> str:
    content = message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append(part.text)
        elif isinstance(part, ImagePart):
            parts.append("[image attached]")
    return "\n".join(parts)


def _last_assistant(messages: tuple[Message, ...]) -> str:
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        text = _clean(_message_text(message))
        if text:
            return text
        if message.tool_calls:
            names = ", ".join(call.function.name for call in message.tool_calls)
            return f"(assistant called tools: {names})"
        return ""
    return ""


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def _head(text: str, limit: int) -> str:
    return text[:limit]
