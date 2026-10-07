from collections.abc import Mapping, Sequence

from gg.providers.state.base import ProviderStateStore

NAMESPACE = "gg:sig"
DEFAULT_TTL_S = 24 * 3600
# documented escape hatch for function calls gemini did not issue (expired store, cross-provider history)
DUMMY_SIGNATURE = "skip_thought_signature_validator"


class SignatureStore:
    """gemini thought signatures, keyed by virtual key and tool call id.

    gemini 3 rejects replayed function calls without the signature it issued, and openai-shaped clients
    drop it, so gg keeps it server-side. key-scoped so tenants cannot replay each other's signatures.
    """

    def __init__(self, store: ProviderStateStore, *, ttl_s: int = DEFAULT_TTL_S) -> None:
        self._store = store
        self._ttl_s = ttl_s

    @staticmethod
    def _scope(key_id: str, tool_call_id: str) -> str:
        return f"{key_id}:{tool_call_id}"

    async def lookup(self, key_id: str, tool_call_ids: Sequence[str]) -> dict[str, str]:
        if not tool_call_ids:
            return {}
        scoped = {self._scope(key_id, i): i for i in tool_call_ids}
        found = await self._store.get_many(NAMESPACE, list(scoped))
        return {scoped[k]: v.decode() for k, v in found.items()}

    async def save(self, key_id: str, signatures: Mapping[str, str]) -> None:
        if signatures:
            payload = {self._scope(key_id, i): s.encode() for i, s in signatures.items()}
            await self._store.put_many(NAMESPACE, payload, self._ttl_s)
