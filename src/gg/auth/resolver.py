from collections import OrderedDict
from typing import Literal

from gg.auth.base import KeyStore
from gg.auth.keys import check_format, hash_key
from gg.core.clock import Clock
from gg.core.errors import AuthenticationError
from gg.core.keypolicy import KeyPolicy

type LookupResult = Literal["hit", "miss", "negative_hit"]


class CachingKeyResolver:
    """lru + ttl in front of a key store; status and expiry are checked on every request, never cached"""

    def __init__(
        self,
        store: KeyStore,
        clock: Clock,
        *,
        allow_test_keys: bool,
        maxsize: int = 10_000,
        ttl_s: float = 60,
        negative_ttl_s: float = 5,
    ) -> None:
        self._store = store
        self._clock = clock
        self._allow_test_keys = allow_test_keys
        self._maxsize = maxsize
        self._ttl_s = ttl_s
        self._negative_ttl_s = negative_ttl_s
        self._entries: OrderedDict[str, tuple[KeyPolicy | None, float]] = OrderedDict()
        self.lookups: dict[LookupResult, int] = {"hit": 0, "miss": 0, "negative_hit": 0}

    async def resolve(self, token: str, /) -> KeyPolicy:
        check_format(token, allow_test_keys=self._allow_test_keys)
        key_hash = hash_key(token)
        cached = self._get(key_hash)
        if cached is None:
            self.lookups["miss"] += 1
            policy = await self._store.get(key_hash)
            self._put(key_hash, policy)
        else:
            policy = cached[0]
            self.lookups["hit" if policy is not None else "negative_hit"] += 1
        if policy is None:
            raise AuthenticationError("Incorrect API key provided.", code="invalid_api_key")
        if policy.status == "disabled":
            raise AuthenticationError("This API key has been disabled.", code="api_key_disabled")
        if policy.expires_at is not None and policy.expires_at <= self._clock.now():
            raise AuthenticationError("This API key has expired.", code="api_key_expired")
        return policy

    def _get(self, key_hash: str) -> tuple[KeyPolicy | None] | None:
        entry = self._entries.get(key_hash)
        if entry is None:
            return None
        policy, expires_at = entry
        if expires_at <= self._clock.monotonic():
            del self._entries[key_hash]
            return None
        self._entries.move_to_end(key_hash)
        return (policy,)

    def _put(self, key_hash: str, policy: KeyPolicy | None) -> None:
        ttl = self._ttl_s if policy is not None else self._negative_ttl_s
        self._entries[key_hash] = (policy, self._clock.monotonic() + ttl)
        self._entries.move_to_end(key_hash)
        while len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
