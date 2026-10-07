from gg.auth.config import KeysConfig
from gg.core.keypolicy import KeyPolicy


class YamlKeyStore:
    """keys from keys.yaml, indexed by hash; changes take effect on restart"""

    def __init__(self, keys: KeysConfig) -> None:
        self._by_hash = keys.policies_by_hash()

    def __len__(self) -> int:
        return len(self._by_hash)

    async def get(self, key_hash: str, /) -> KeyPolicy | None:
        return self._by_hash.get(key_hash)
