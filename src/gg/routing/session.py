from dataclasses import dataclass

from gg.core.clock import Clock
from gg.core.context import RequestContext
from gg.core.jsonutil import canonical_json, sha256_hex
from gg.core.schema import ChatRequest
from gg.routing.base import Tier
from gg.routing.ttl_cache import TTLCache


@dataclass(frozen=True, slots=True)
class SessionRecord:
    tier: Tier
    score: float | None


class SessionStore:
    """last routing decision per session, in process; a redis-backed store can replace it behind this api"""

    def __init__(self, max_entries: int, ttl_s: float, clock: Clock) -> None:
        self._cache: TTLCache[SessionRecord] = TTLCache(max_entries, ttl_s, clock)

    def get(self, session_id: str) -> SessionRecord | None:
        return self._cache.get(session_id)

    def put(self, session_id: str, record: SessionRecord) -> None:
        self._cache.put(session_id, record)


def session_id(ctx: RequestContext) -> str:
    """explicit gg.session_id, else a hash of the key and the conversation's opening messages"""
    if ctx.request.gg is not None and ctx.request.gg.session_id:
        return f"explicit:{ctx.key.id}:{ctx.request.gg.session_id}"
    source = ctx.scrubbed or ctx.request
    system = next((m.text() for m in source.messages if m.role in ("system", "developer")), "")
    user = next((m.text() for m in source.messages if m.role == "user"), "")
    return "derived:" + sha256_hex(canonical_json([ctx.key.id, system, user]))


def is_tool_continuation(request: ChatRequest) -> bool:
    last = request.messages[-1]
    return last.role == "tool" or (last.role == "assistant" and bool(last.tool_calls))
