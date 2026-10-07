from collections.abc import Mapping, Sequence
from typing import Any

import orjson

from gg.providers.state.base import ProviderStateStore

NAMESPACE = "gg:think"
DEFAULT_TTL_S = 3600

type ThinkingBlocks = list[dict[str, Any]]


class ThinkingStore:
    """anthropic thinking blocks that preceded a tool_use, keyed by key, upstream model and tool call id.

    openai clients cannot round-trip signed thinking blocks; anthropic wants them back within a tool loop.
    scoped by virtual key (tenants cannot read each other's) and upstream model (blocks are model-bound).
    """

    def __init__(self, store: ProviderStateStore, *, ttl_s: int = DEFAULT_TTL_S) -> None:
        self._store = store
        self._ttl_s = ttl_s

    @staticmethod
    def _scope(key_id: str, model: str, tool_call_id: str) -> str:
        return f"{key_id}:{model}:{tool_call_id}"

    async def lookup(
        self, key_id: str, model: str, tool_call_ids: Sequence[str]
    ) -> dict[str, ThinkingBlocks]:
        if not tool_call_ids:
            return {}
        scoped = {self._scope(key_id, model, t): t for t in tool_call_ids}
        found = await self._store.get_many(NAMESPACE, list(scoped))
        out: dict[str, ThinkingBlocks] = {}
        for key, raw in found.items():
            blocks: Any = orjson.loads(raw)
            if isinstance(blocks, list) and blocks:
                out[scoped[key]] = blocks
        return out

    async def save(self, key_id: str, model: str, items: Mapping[str, ThinkingBlocks]) -> None:
        payload = {self._scope(key_id, model, t): orjson.dumps(b) for t, b in items.items() if b}
        if payload:
            await self._store.put_many(NAMESPACE, payload, self._ttl_s)
