"""process-local backends for tests, the eval and redis-less dev runs"""

import math
import secrets
from collections.abc import Sequence

import structlog

from gg.cache.base import CachedResponse, PutResult, SemanticMatch, SemanticTags
from gg.cache.codec import Codec, CorruptValueError
from gg.core.clock import Clock

log = structlog.get_logger("gg.cache.memory")


class InMemoryResponseCache:
    """same contract as the redis backend, values go through the codec too"""

    def __init__(self, codec: Codec, clock: Clock) -> None:
        self._codec = codec
        self._clock = clock
        self._values: dict[str, tuple[bytes, float]] = {}
        self._locks: dict[str, tuple[str, float]] = {}

    @property
    def available(self) -> bool:
        return True

    def __len__(self) -> int:
        return sum(1 for key in list(self._values) if self._live(key))

    def raw_values(self) -> list[bytes]:
        return [data for data, _ in self._values.values()]

    def _live(self, key: str) -> bool:
        entry = self._values.get(key)
        if entry is None:
            return False
        if entry[1] <= self._clock.monotonic():
            del self._values[key]
            return False
        return True

    async def get(self, key: str, /) -> CachedResponse | None:
        if not self._live(key):
            return None
        try:
            return self._codec.decode(self._values[key][0])
        except CorruptValueError:
            log.warning("cache.corrupt_value", backend="memory")
            del self._values[key]
            return None

    async def put(
        self, key: str, value: CachedResponse, ttl_s: int, /, *, replace: bool = False
    ) -> PutResult:
        data = self._codec.encode(value)
        if not self._codec.fits(data):
            return "too_large"
        if not replace and self._live(key):
            return "exists"
        self._values[key] = (data, self._clock.monotonic() + ttl_s)
        return "stored"

    async def acquire(self, lock_key: str, ttl_ms: int, /) -> str | None:
        if await self.held(lock_key):
            return None
        token = secrets.token_hex(8)
        self._locks[lock_key] = (token, self._clock.monotonic() + ttl_ms / 1000)
        return token

    async def release(self, lock_key: str, token: str, /) -> None:
        held = self._locks.get(lock_key)
        if held is not None and held[0] == token:
            del self._locks[lock_key]

    async def held(self, lock_key: str, /) -> bool:
        held = self._locks.get(lock_key)
        if held is None:
            return False
        if held[1] <= self._clock.monotonic():
            del self._locks[lock_key]
            return False
        return True


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0:
        return 1.0
    return 1.0 - dot / norm


class InMemoryIndex:
    """brute-force cosine knn with exact tag pre-filter; the eval sweep uses it too"""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._entries: dict[str, tuple[tuple[float, ...], SemanticTags, float, str]] = {}

    @property
    def available(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self._entries)

    async def search(self, vector: Sequence[float], tags: SemanticTags, /) -> SemanticMatch | None:
        now = self._clock.monotonic()
        best: SemanticMatch | None = None
        for exact_key, (stored, stored_tags, expires, text) in list(self._entries.items()):
            if expires <= now:
                del self._entries[exact_key]
                continue
            if stored_tags != tags:
                continue
            distance = cosine_distance(vector, stored)
            if best is None or distance < best.distance:
                best = SemanticMatch(exact_key=exact_key, distance=distance, text=text)
        return best

    async def add(
        self, vector: Sequence[float], tags: SemanticTags, exact_key: str, ttl_s: int, text: str, /
    ) -> None:
        self._entries[exact_key] = (tuple(vector), tags, self._clock.monotonic() + ttl_s, text)
