from collections.abc import Mapping, Sequence

from gg.providers.state.base import ProviderStateStore

NAMESPACE = "gg:reason"
DEFAULT_TTL_S = 24 * 3600


class ReasoningStore:
    """reasoning text of assistant turns, keyed by virtual key, provider and turn (tool call id or text hash).

    lets hosts that need prior reasoning back (deepseek with tools, kimi k3, qwen 3.8, glm) get it even when
    the client's sdk dropped reasoning_content from history. key-scoped so tenants cannot read each other's.
    """

    def __init__(self, store: ProviderStateStore, *, ttl_s: int = DEFAULT_TTL_S) -> None:
        self._store = store
        self._ttl_s = ttl_s

    @staticmethod
    def _scope(key_id: str, provider: str, turn_key: str) -> str:
        return f"{key_id}:{provider}:{turn_key}"

    async def lookup(self, key_id: str, provider: str, turn_keys: Sequence[str]) -> dict[str, str]:
        if not turn_keys:
            return {}
        scoped = {self._scope(key_id, provider, k): k for k in turn_keys}
        found = await self._store.get_many(NAMESPACE, list(scoped))
        return {scoped[k]: v.decode() for k, v in found.items()}

    async def save(self, key_id: str, provider: str, items: Mapping[str, str]) -> None:
        if items:
            payload = {self._scope(key_id, provider, k): v.encode() for k, v in items.items()}
            await self._store.put_many(NAMESPACE, payload, self._ttl_s)
