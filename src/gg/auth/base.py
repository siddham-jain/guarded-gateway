from typing import Protocol

from gg.core.keypolicy import KeyPolicy


class KeyStore(Protocol):
    async def get(self, key_hash: str, /) -> KeyPolicy | None: ...


class KeyResolver(Protocol):
    async def resolve(self, token: str, /) -> KeyPolicy: ...
