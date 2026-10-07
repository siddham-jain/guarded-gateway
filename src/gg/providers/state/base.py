from collections.abc import Mapping, Sequence
from typing import Protocol


class ProviderStateStore(Protocol):
    """byte store for provider round-trip state (reasoning, thinking blocks, thought signatures)"""

    async def get_many(self, namespace: str, keys: Sequence[str]) -> dict[str, bytes]: ...

    async def put_many(self, namespace: str, items: Mapping[str, bytes], ttl_s: int) -> None: ...
